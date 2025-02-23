import pytest


def test_importable():
    # Just check that this module does not raise an error when imported.
    try:
        from mutagrep import plan_search  # noqa: F401
    except ImportError:
        pytest.fail("Failed to import plan_search module")
