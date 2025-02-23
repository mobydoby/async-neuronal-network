import pytest
import logging


@pytest.fixture(autouse=True)
def setup_logging():
    """Configure root logger for all tests"""
    # Clear existing handlers
    root = logging.getLogger()
    root.handlers = []

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s", force=True
    )

    # Set package loggers
    logging.getLogger("neuron_net").setLevel(logging.DEBUG)
