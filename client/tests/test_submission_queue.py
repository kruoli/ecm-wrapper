"""Tests for the SubmissionQueue persistent retry mechanism."""
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from lib.submission_queue import SubmissionQueue


@pytest.fixture
def queue_dir():
    """Create a temporary directory for queue testing."""
    tmpdir = tempfile.mkdtemp(prefix="ecm_queue_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def queue(queue_dir):
    """Create a SubmissionQueue with a temp directory."""
    return SubmissionQueue(queue_dir=queue_dir)


@pytest.fixture
def mock_api_client():
    """Create a mock APIClient."""
    client = Mock()
    client.submit_result.return_value = {"status": "ok", "attempt_id": 1}
    client.complete_work.return_value = True
    client.complete_residue.return_value = {"new_t_level": 35.0}
    client.upload_residue.return_value = {"residue_id": 42, "curve_count": 100}
    return client


class TestEnqueueResult:
    def test_enqueue_creates_file(self, queue):
        payload = {"composite": "123", "method": "ecm"}
        filepath = queue.enqueue_result(payload)
        assert filepath is not None
        assert filepath.exists()
        data = json.loads(filepath.read_text())
        assert data["type"] == "result"
        assert data["payload"] == payload
        assert data["attempts"] == 0

    def test_enqueue_with_context(self, queue):
        payload = {"composite": "123456789" * 10}
        context = {"composite": "123456789" * 10, "b1": 50000}
        filepath = queue.enqueue_result(payload, results_context=context)
        data = json.loads(filepath.read_text())
        assert "composite_preview" in data
        assert len(data["composite_preview"]) <= 50


class TestEnqueueResidueUpload:
    def test_enqueue_preserves_file(self, queue, queue_dir):
        # Create a fake residue file
        residue = Path(queue_dir) / "original_residue.txt"
        residue.write_text("METHOD=ECM; SIGMA=3:12345\n")

        filepath = queue.enqueue_residue_upload(
            residue_file=residue,
            client_id="test-client",
            stage1_attempt_id=42
        )
        assert filepath is not None

        data = json.loads(filepath.read_text())
        assert data["type"] == "residue_upload"
        assert data["payload"]["client_id"] == "test-client"
        assert data["payload"]["stage1_attempt_id"] == 42

        # Verify the preserved residue file exists
        preserved_path = Path(data["residue_file"])
        assert preserved_path.exists()
        assert preserved_path.read_text() == "METHOD=ECM; SIGMA=3:12345\n"

    def test_enqueue_missing_file(self, queue):
        filepath = queue.enqueue_residue_upload(
            residue_file=Path("/nonexistent/residue.txt"),
            client_id="test"
        )
        assert filepath is None


class TestEnqueueCompletions:
    def test_enqueue_work_completion(self, queue):
        filepath = queue.enqueue_work_completion("work-123", "test-client")
        assert filepath is not None
        data = json.loads(filepath.read_text())
        assert data["type"] == "work_complete"
        assert data["payload"]["work_id"] == "work-123"
        assert data["payload"]["client_id"] == "test-client"

    def test_enqueue_residue_completion(self, queue):
        filepath = queue.enqueue_residue_completion(
            residue_id=42, client_id="test-client", stage2_attempt_id=99
        )
        assert filepath is not None
        data = json.loads(filepath.read_text())
        assert data["type"] == "residue_complete"
        assert data["payload"]["residue_id"] == 42
        assert data["payload"]["stage2_attempt_id"] == 99


class TestCount:
    def test_empty_queue(self, queue):
        assert queue.count() == 0

    def test_count_after_enqueue(self, queue):
        queue.enqueue_result({"composite": "123"})
        queue.enqueue_work_completion("w1", "c1")
        assert queue.count() == 2


class TestDrain:
    def test_drain_empty_queue(self, queue, mock_api_client):
        success, fail = queue.drain(mock_api_client)
        assert success == 0
        assert fail == 0

    def test_drain_result_success(self, queue, mock_api_client):
        payload = {"composite": "123", "method": "ecm"}
        filepath = queue.enqueue_result(payload)
        assert queue.count() == 1

        success, fail = queue.drain(mock_api_client)
        assert success == 1
        assert fail == 0
        assert queue.count() == 0

        # Verify the API was called with the right payload
        mock_api_client.submit_result.assert_called_once_with(
            payload=payload, save_on_failure=False
        )

    def test_drain_result_failure_stays_in_queue(self, queue, mock_api_client):
        mock_api_client.submit_result.return_value = None  # Simulate failure

        queue.enqueue_result({"composite": "123"})
        success, fail = queue.drain(mock_api_client)
        assert success == 0
        assert fail == 1
        assert queue.count() == 1  # Item stays in queue

    def test_drain_work_completion_success(self, queue, mock_api_client):
        queue.enqueue_work_completion("work-1", "client-1")
        success, fail = queue.drain(mock_api_client)
        assert success == 1
        assert fail == 0
        mock_api_client.complete_work.assert_called_once_with(
            work_id="work-1", client_id="client-1"
        )

    def test_drain_residue_completion_success(self, queue, mock_api_client):
        queue.enqueue_residue_completion(42, "client-1", 99)
        success, fail = queue.drain(mock_api_client)
        assert success == 1
        assert fail == 0
        mock_api_client.complete_residue.assert_called_once_with(
            client_id="client-1", residue_id=42, stage2_attempt_id=99
        )

    def test_drain_residue_upload_success(self, queue, mock_api_client, queue_dir):
        # Create a fake residue file
        residue = Path(queue_dir) / "test_residue.txt"
        residue.write_text("residue data\n")

        queue.enqueue_residue_upload(residue, "client-1", stage1_attempt_id=10)
        success, fail = queue.drain(mock_api_client)
        assert success == 1
        assert fail == 0

        # Verify upload was called
        mock_api_client.upload_residue.assert_called_once()
        call_kwargs = mock_api_client.upload_residue.call_args
        assert call_kwargs[1]["client_id"] == "client-1"
        assert call_kwargs[1]["stage1_attempt_id"] == 10

    def test_drain_residue_upload_cleans_preserved_file(self, queue, mock_api_client, queue_dir):
        residue = Path(queue_dir) / "test_residue.txt"
        residue.write_text("residue data\n")

        filepath = queue.enqueue_residue_upload(residue, "client-1")
        data = json.loads(filepath.read_text())
        preserved_path = Path(data["residue_file"])
        assert preserved_path.exists()

        success, fail = queue.drain(mock_api_client)
        assert success == 1
        # Preserved file should be cleaned up after successful upload
        assert not preserved_path.exists()

    def test_drain_multiple_items_oldest_first(self, queue, mock_api_client):
        """Items should be processed oldest-first."""
        import time

        queue.enqueue_result({"composite": "first"})
        time.sleep(0.01)  # Ensure different mtime
        queue.enqueue_result({"composite": "second"})

        call_order = []
        def track_calls(**kwargs):
            payload = kwargs.get("payload", {})
            call_order.append(payload.get("composite"))
            return {"status": "ok"}

        mock_api_client.submit_result.side_effect = lambda **kwargs: track_calls(**kwargs)

        success, fail = queue.drain(mock_api_client)
        assert success == 2
        assert call_order == ["first", "second"]

    def test_drain_increments_attempt_count(self, queue, mock_api_client):
        mock_api_client.submit_result.return_value = None  # Always fail

        filepath = queue.enqueue_result({"composite": "123"})

        queue.drain(mock_api_client)
        data = json.loads(filepath.read_text())
        assert data["attempts"] == 1

        queue.drain(mock_api_client)
        data = json.loads(filepath.read_text())
        assert data["attempts"] == 2

    def test_drain_mixed_success_and_failure(self, queue, mock_api_client):
        queue.enqueue_result({"composite": "will_succeed"})
        queue.enqueue_work_completion("will_fail", "client")

        # First call succeeds, second fails
        mock_api_client.submit_result.return_value = {"status": "ok"}
        mock_api_client.complete_work.return_value = False

        success, fail = queue.drain(mock_api_client)
        assert success == 1
        assert fail == 1
        assert queue.count() == 1  # One item remains
