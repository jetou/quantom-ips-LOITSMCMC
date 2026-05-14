import pytest


# Here we define ALL arguments that we wish to make available for pytest:
def pytest_addoption(parser):
    parser.addoption("--n_events", action="store", type=int, default=1000000)
    parser.addoption("--batch_size", action="store", type=int, default=20)
    parser.addoption("--n_targets", action="store", type=int, default=2)
    parser.addoption("--grid_size", action="store", type=int, default=50)
    parser.addoption("--n_points", action="store", type=int, default=50)
    parser.addoption(
        "--result_loc_loits_2d",
        action="store",
        type=str,
        default="results_test_base_loits_2d",
    )
    parser.addoption(
        "--result_loc_loits_1d",
        action="store",
        type=str,
        default="results_test_base_loits_1d",
    )


@pytest.fixture
def n_events(request):
    return request.config.getoption("--n_events")


@pytest.fixture
def batch_size(request):
    return request.config.getoption("--batch_size")


@pytest.fixture
def n_targets(request):
    return request.config.getoption("--n_targets")


@pytest.fixture
def grid_size(request):
    return request.config.getoption("--grid_size")


@pytest.fixture
def n_points(request):
    return request.config.getoption("--n_points")


@pytest.fixture
def result_loc_loits_2d(request):
    return request.config.getoption("--result_loc_loits_2d")


@pytest.fixture
def result_loc_loits_1d(request):
    return request.config.getoption("--result_loc_loits_1d")
