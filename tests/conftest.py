def pytest_configure(config):
    config.addinivalue_line("markers", "gcd: checks only d from egcd; u and v are ignored")
