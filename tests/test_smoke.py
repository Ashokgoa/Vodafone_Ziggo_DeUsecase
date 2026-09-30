"""Smoke test: proves pytest runs and the `app` package is importable."""

import app


def test_package_is_importable() -> None:
    assert isinstance(app.__version__, str)
    assert app.__version__
