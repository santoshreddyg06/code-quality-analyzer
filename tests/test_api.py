"""
tests/test_api.py
=================
Integration tests for the FastAPI application.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_analyze_code_endpoint():
    payload = {
        "old_code": "def hello():\n    return 'hello'\n",
        "new_code": "def hello():\n    # greeting\n    return 'hello world'\n",
        "filename": "hello.py",
    }
    response = client.post("/api/analyze/code", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "metrics" in data
    assert "features" in data
    assert data["metrics"]["old"]["loc"] > 0
    assert data["metrics"]["new"]["loc"] > 0


def test_analyze_diff_endpoint():
    patch = """@@ -1,3 +1,5 @@
 def hello():
-    return 'hello'
+    # greeting
+    return 'hello world'
"""
    payload = {"patch": patch, "filename": "hello.py"}
    response = client.post("/api/analyze/diff", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "metrics" in data
    assert data["metrics"]["diff"]["lines_added"] >= 1
