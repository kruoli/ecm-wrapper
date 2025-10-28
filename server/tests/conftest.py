"""
Pytest configuration for server tests.

This file adds the server directory to Python path so tests can import from 'app'.
"""
import sys
from pathlib import Path

# Add server directory to Python path
server_dir = Path(__file__).parent.parent
sys.path.insert(0, str(server_dir))
