"""Smoke tests: every submodule imports cleanly and carries a docstring."""

import importlib


def test_package_imports():
    sp = importlib.import_module("superproton")
    assert sp.__version__ == "0.1.0"
    # All submodules listed in __all__ should be importable.
    for name in sp.__all__:
        importlib.import_module(f"superproton.{name}")


def test_submodules_have_docstrings():
    """Every submodule documents what it computes, for `help(superproton.x)`."""
    sp = importlib.import_module("superproton")
    for name in sp.__all__:
        mod = importlib.import_module(f"superproton.{name}")
        assert mod.__doc__, f"superproton.{name} is missing its module docstring"
