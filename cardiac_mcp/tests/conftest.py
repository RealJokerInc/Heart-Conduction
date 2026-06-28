def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow end-to-end test (runs a real cardiac_core simulation)")
