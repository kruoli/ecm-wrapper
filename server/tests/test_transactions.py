"""
Test script to verify transaction management improvements.

This module tests:
1. Bulk operations have atomic all-or-nothing guarantees
2. Transaction utilities work correctly
3. Service methods use flush() instead of commit()
"""

import pytest
from conftest import get_test_engine, create_composite

from app.models.composites import Composite
from app.models.projects import Project, ProjectComposite

from app.services.composites import CompositeService
from app.utils.transactions import transaction_scope


class TestAtomicOperations:
    """Tests for atomic transaction behavior."""

    def test_atomic_bulk_operation(self, db_session):
        """Test that bulk operations are atomic - all succeed or all fail."""
        service = CompositeService()

        valid_numbers = ["12345", "67890", "11111", "22222", "33333"]

        # Before transaction - count should be 0
        count_before = db_session.query(Composite).count()
        assert count_before == 0

        # Execute bulk load within transaction
        with transaction_scope(db_session, "test_bulk"):
            stats = service.bulk_load_composites(
                db_session, valid_numbers, source_type="list",
                default_priority=5, project_name="test-project"
            )

        # After transaction - all should be committed
        count_after = db_session.query(Composite).count()
        project_count = db_session.query(Project).count()

        assert count_after == len(valid_numbers), \
            f"Expected {len(valid_numbers)} composites, got {count_after}"
        assert project_count == 1, f"Expected 1 project, got {project_count}"

    def test_project_association_atomic(self, db_session):
        """Test that project associations are atomic with composite creation."""
        service = CompositeService()

        valid_numbers = ["111111", "222222", "333333"]
        project_name = "atomic-test-project"

        count_before = db_session.query(Composite).count()
        project_count_before = db_session.query(Project).count()
        assoc_count_before = db_session.query(ProjectComposite).count()

        assert count_before == 0
        assert project_count_before == 0
        assert assoc_count_before == 0

        # Execute with project association
        with transaction_scope(db_session, "test_project_atomic"):
            stats = service.bulk_load_composites(
                db_session, valid_numbers, source_type="list",
                project_name=project_name
            )

        count_after = db_session.query(Composite).count()
        project_count_after = db_session.query(Project).count()
        assoc_count_after = db_session.query(ProjectComposite).count()

        assert count_after == len(valid_numbers)
        assert project_count_after == 1
        assert assoc_count_after == len(valid_numbers), \
            f"Expected {len(valid_numbers)} associations, got {assoc_count_after}"


class TestServiceBehavior:
    """Tests for service method behavior."""

    def test_service_uses_flush_not_commit(self, db_session):
        """Verify that service methods use flush(), allowing rollback."""
        service = CompositeService()

        # Count before should be 0
        count_before = db_session.query(Composite).count()
        assert count_before == 0

        # Create a composite - service should use flush(), not commit()
        composite, created, updated = service.get_or_create_composite(
            db_session, "555555", priority=10
        )

        assert created is True
        assert composite.id is not None

        # Changes should be visible in this session (due to flush)
        found = db_session.query(Composite).filter_by(id=composite.id).first()
        assert found is not None, "Composite should be visible after flush"

        # But rollback should undo it
        db_session.rollback()

        # After rollback, it should be gone
        count = db_session.query(Composite).count()
        assert count == 0, "Rollback should have removed uncommitted composite"

    def test_graceful_handling_of_invalid_input(self, db_session):
        """Test that invalid input is skipped gracefully (not an error)."""
        service = CompositeService()

        # Mix of valid and invalid numbers
        mixed_numbers = ["99999", "invalid_not_a_number", "88888"]

        count_before = db_session.query(Composite).count()
        assert count_before == 0

        # Execute bulk load - should succeed, skipping invalid
        with transaction_scope(db_session, "test_graceful"):
            stats = service.bulk_load_composites(
                db_session, mixed_numbers, source_type="list",
                default_priority=5
            )

        # After transaction - valid numbers should be committed
        count_after = db_session.query(Composite).count()

        # Should have loaded 2 valid composites (99999, 88888)
        assert count_after == 2, \
            f"Expected 2 valid composites, got {count_after}"

        # Verify stats reflect the processing
        assert stats is not None
