"""
tests/conftest.py — pytest configuration and shared fixtures
"""

import sys
import os

# Ensure the project root is on the path so imports like
# "from data.models import ..." work from any test runner location.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
