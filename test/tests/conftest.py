import pytest

def pytest_addoption(parser):
    """
    This function allows us to add args to the pytest command.
    """
    parser.addoption(
        "--num_ranks",
        type=int,
        default=2,
        help="How many ranks to use when testing.",
    )


@pytest.fixture
def num_ranks(request):
    return request.config.getoption("--num_ranks")

