import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run tests marked slow (real-model inference)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="slow: run with --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
