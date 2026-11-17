import os
import subprocess
import sys
import numpy as np
import pytest
from part3_model import rmse, baseline_and_model_error


def test_rmse():
    true = np.array([10.0, 12.0, 14.0])
    pred = np.array([10.0, 10.0, 14.0])
    expected = np.sqrt((0 + 4 + 0) / 3)
    assert np.isclose(rmse(true, pred), expected)


def test_baseline_and_model_error():
    # Simple synthetic linear data: y = 2*x1 + 3*x2 + 5
    X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]])
    y_train = 2 * X_train[:, 0] + 3 * X_train[:, 1] + 5

    X_test = np.array([[5.0, 2.0], [1.0, 5.0]])
    y_test = 2 * X_test[:, 0] + 3 * X_test[:, 1] + 5

    base_rmse, lin_rmse = baseline_and_model_error(X_train, y_train, X_test, y_test)
    assert lin_rmse < 1e-10       # Exact linear fit should have near-zero error
    assert base_rmse > lin_rmse   # Model should outperform mean baseline


def test_script_execution_and_output():
    # Ensure data file exists before running end-to-end test
    assert os.path.exists("data/auto-mpg.data"), "Missing data/auto-mpg.data"

    result = subprocess.run(
        [sys.executable, "part3_model.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Script failed with stderr:\n{result.stderr}"
    assert "Random split test RMSE:" in result.stdout
    assert "5-fold CV RMSE:" in result.stdout
    assert "Formula: L/100km =" in result.stdout
    assert "Heaviest 30% test RMSE:" in result.stdout
    assert os.path.exists("report/extrapolation.png"), "Plot was not created"
