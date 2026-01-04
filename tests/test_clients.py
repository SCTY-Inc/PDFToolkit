"""Tests for pdftoolkit.clients module."""

import pytest

from pdftoolkit.clients import api_retry


class TestApiRetry:
    """Tests for the retry decorator."""

    def test_succeeds_on_first_try(self):
        """Function that succeeds should return normally."""
        call_count = 0

        @api_retry
        def always_succeeds():
            nonlocal call_count
            call_count += 1
            return "success"

        result = always_succeeds()
        assert result == "success"
        assert call_count == 1

    def test_retries_on_failure(self):
        """Function that fails then succeeds should retry."""
        call_count = 0

        @api_retry
        def fails_twice_then_succeeds():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("temporary failure")
            return "success"

        result = fails_twice_then_succeeds()
        assert result == "success"
        assert call_count == 3

    def test_raises_after_max_retries(self):
        """Function that always fails should raise after max retries."""
        call_count = 0

        @api_retry
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise Exception("permanent failure")

        with pytest.raises(Exception, match="permanent failure"):
            always_fails()

        assert call_count == 3  # Should have tried 3 times
