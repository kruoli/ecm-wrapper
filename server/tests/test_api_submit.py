"""
Tests for the /submit_result API endpoint.

Tests cover:
- Successful result submission
- Factor discovery and validation
- Duplicate detection
- Error handling for invalid submissions
"""
import pytest
from conftest import create_composite


class TestSubmitResultBasic:
    """Basic submission tests."""

    def test_submit_result_no_factor(self, client):
        """Test submitting a result with no factor found."""
        composite = create_composite("12345678901234567890", digit_length=20)

        response = client.post(
            "/api/v1/submit_result",
            json={
                "composite": composite["current_composite"],
                "client_id": "test-client",
                "method": "ecm",
                "program": "gmp-ecm",
                "parameters": {
                    "b1": 50000,
                    "b2": 5000000,
                    "curves": 100,
                },
                "results": {
                    "curves_completed": 100,
                    "execution_time": 10.5,
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["factor_status"] == "no_factor"
        assert data["attempt_id"] is not None
        assert data["composite_id"] == composite["id"]

    def test_submit_result_unknown_composite(self, client):
        """Test that submissions for unknown composites are rejected."""
        response = client.post(
            "/api/v1/submit_result",
            json={
                "composite": "99999999999999999",
                "client_id": "test-client",
                "method": "ecm",
                "program": "gmp-ecm",
                "parameters": {
                    "b1": 50000,
                    "curves": 100,
                },
                "results": {
                    "curves_completed": 100,
                },
            },
        )

        assert response.status_code == 404
        assert "not found in database" in response.json()["detail"]

    def test_submit_result_duplicate_detection(self, client):
        """Test that duplicate submissions return existing attempt."""
        composite = create_composite("12345678901234567890", digit_length=20)

        payload = {
            "composite": composite["current_composite"],
            "client_id": "test-client",
            "method": "ecm",
            "program": "gmp-ecm",
            "parameters": {
                "b1": 50000,
                "b2": 5000000,
                "curves": 100,
                "sigma": "3:12345",
            },
            "results": {
                "curves_completed": 100,
            },
        }

        # First submission
        response1 = client.post("/api/v1/submit_result", json=payload)
        assert response1.status_code == 200
        attempt_id_1 = response1.json()["attempt_id"]

        # Second identical submission
        response2 = client.post("/api/v1/submit_result", json=payload)
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["factor_status"] == "duplicate"
        assert data2["attempt_id"] == attempt_id_1


class TestSubmitResultWithFactor:
    """Tests for factor submission and validation."""

    def test_submit_valid_factor(self, client):
        """Test submitting a valid factor."""
        # Composite is 15 = 3 * 5
        composite = create_composite("15", digit_length=2)

        response = client.post(
            "/api/v1/submit_result",
            json={
                "composite": "15",
                "client_id": "test-client",
                "method": "ecm",
                "program": "gmp-ecm",
                "parameters": {
                    "b1": 1000,
                    "curves": 10,
                    "sigma": "3:12345",
                },
                "results": {
                    "curves_completed": 5,
                    "factor_found": "3",
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["factor_status"] == "new_factor"

    def test_submit_invalid_factor(self, client):
        """Test that invalid factors are rejected."""
        # 7 does not divide 15
        composite = create_composite("15", digit_length=2)

        response = client.post(
            "/api/v1/submit_result",
            json={
                "composite": "15",
                "client_id": "test-client",
                "method": "ecm",
                "program": "gmp-ecm",
                "parameters": {
                    "b1": 1000,
                    "curves": 10,
                },
                "results": {
                    "curves_completed": 5,
                    "factor_found": "7",
                },
            },
        )

        assert response.status_code == 400
        assert "does not divide" in response.json()["detail"]

    def test_submit_multiple_factors(self, client):
        """Test submitting multiple factors in one request."""
        # 30 = 2 * 3 * 5
        composite = create_composite("30", digit_length=2)

        response = client.post(
            "/api/v1/submit_result",
            json={
                "composite": "30",
                "client_id": "test-client",
                "method": "ecm",
                "program": "gmp-ecm",
                "parameters": {
                    "b1": 1000,
                    "curves": 10,
                },
                "results": {
                    "curves_completed": 5,
                    "factors_found": [
                        {"factor": "2", "sigma": "3:111"},
                        {"factor": "3", "sigma": "3:222"},
                    ],
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["factor_status"] == "new_factor"
        assert data["composite_id"] == composite["id"]


class TestSubmitResultParametrization:
    """Tests for parametrization handling."""

    def test_parametrization_from_sigma_string(self, client):
        """Test that parametrization is extracted from sigma string format."""
        composite = create_composite("12345678901234567890", digit_length=20)

        response = client.post(
            "/api/v1/submit_result",
            json={
                "composite": composite["current_composite"],
                "client_id": "test-client",
                "method": "ecm",
                "program": "gmp-ecm",
                "parameters": {
                    "b1": 50000,
                    "curves": 100,
                    "sigma": "1:98765",  # Parametrization 1
                },
                "results": {
                    "curves_completed": 100,
                },
            },
        )

        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_explicit_parametrization(self, client):
        """Test explicit parametrization parameter."""
        composite = create_composite("12345678901234567890", digit_length=20)

        response = client.post(
            "/api/v1/submit_result",
            json={
                "composite": composite["current_composite"],
                "client_id": "test-client",
                "method": "ecm",
                "program": "gmp-ecm",
                "parameters": {
                    "b1": 50000,
                    "curves": 100,
                    "parametrization": 3,
                    "sigma": "12345",  # No prefix, but explicit param
                },
                "results": {
                    "curves_completed": 100,
                },
            },
        )

        assert response.status_code == 200
        assert response.json()["status"] == "success"

    def test_invalid_parametrization(self, client):
        """Test that invalid parametrization values are rejected."""
        composite = create_composite("12345678901234567890", digit_length=20)

        response = client.post(
            "/api/v1/submit_result",
            json={
                "composite": composite["current_composite"],
                "client_id": "test-client",
                "method": "ecm",
                "program": "gmp-ecm",
                "parameters": {
                    "b1": 50000,
                    "curves": 100,
                    "parametrization": 5,  # Invalid - must be 0-3
                },
                "results": {
                    "curves_completed": 100,
                },
            },
        )

        # Should fail validation (Pydantic validates ge=0, le=3)
        assert response.status_code == 422


class TestSubmitResultValidation:
    """Tests for request validation."""

    def test_missing_required_fields(self, client):
        """Test that missing required fields return 422."""
        response = client.post(
            "/api/v1/submit_result",
            json={
                "composite": "12345",
                # Missing client_id, method, program, parameters, results
            },
        )

        assert response.status_code == 422

    def test_invalid_method(self, client):
        """Test that invalid method returns 422."""
        response = client.post(
            "/api/v1/submit_result",
            json={
                "composite": "12345",
                "client_id": "test",
                "method": "invalid_method",
                "program": "test",
                "parameters": {"b1": 1000, "curves": 10},
                "results": {"curves_completed": 10},
            },
        )

        assert response.status_code == 422
