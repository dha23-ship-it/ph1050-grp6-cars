"""Tests for Part 2: vehicle mass vs. fuel consumption."""

import numpy as np
import pandas as pd
import pytest

import part_2_code as p2


def sample_raw():
    return pd.DataFrame({
        "mpg": [20.0, 25.0, 10.0, np.nan, 30.0],
        "weight": [2000.0, 2500.0, 4000.0, 3000.0, np.nan],
        # Missing horsepower must not remove a row from Part 2.
        "horsepower": [100.0, np.nan, 150.0, 120.0, 90.0],
    })


def test_unit_conversions_and_cleaning():
    prepared = p2.prepare_data(sample_raw())
    assert len(prepared) == 3
    assert prepared.iloc[0]["fuel_consumption"] == pytest.approx(235.2 / 20.0)
    assert prepared.iloc[0]["weight_kg"] == pytest.approx(2000 * 0.45359237)
    assert prepared["horsepower"].isna().sum() == 1


def test_invalid_nonpositive_values_are_removed():
    raw = pd.DataFrame({
        "mpg": [20.0, 0.0, -5.0, 25.0],
        "weight": [2000.0, 2500.0, 3000.0, 0.0],
    })
    assert len(p2.prepare_data(raw)) == 1


def test_regression_and_confidence_interval():
    x = np.array([800., 1000., 1200., 1400., 1600.])
    y = 2.0 + 0.01 * x
    data = pd.DataFrame({"weight_kg": x, "fuel_consumption": y})
    fit = p2.fit_linear_regression(data)
    assert fit["n"] == 5
    assert fit["slope"] == pytest.approx(0.01)
    assert fit["intercept"] == pytest.approx(2.0)
    assert fit["r_squared"] == pytest.approx(1.0)
    assert fit["ci_lower"] <= fit["slope"] <= fit["ci_upper"]


def test_theoretical_slope_uses_crr_point_zero_one():
    expected = 0.010 * 9.81 * 100000 / (0.20 * 32e6)
    assert p2.CRR == pytest.approx(0.010)
    assert p2.theoretical_rolling_slope() == pytest.approx(expected)
    assert p2.theoretical_rolling_slope() * 1000 == pytest.approx(1.5328125)


def test_theoretical_slope_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        p2.theoretical_rolling_slope(efficiency=0)
    with pytest.raises(ValueError):
        p2.theoretical_rolling_slope(energy_per_litre=-1)


def test_load_auto_mpg_from_local_file(tmp_path):
    # Include quoted car names, matching the UCI whitespace-delimited format.
    content = (
        '20.0 4 100.0 90.0 2000.0 15.0 76 1 "sample car one"\n'
        '25.0 4 120.0 ? 2500.0 14.0 77 1 "sample car two"\n'
    )
    path = tmp_path / "auto-mpg.data"
    path.write_text(content)
    loaded = p2.load_auto_mpg(path, download_if_missing=False)
    assert len(loaded) == 2
    assert loaded.loc[1, "horsepower"] != loaded.loc[1, "horsepower"]  # NaN
    assert loaded.loc[0, "car_name"] == "sample car one"


def test_plot_files_are_created(tmp_path):
    x = np.array([800., 1000., 1200., 1400., 1600.])
    y = np.array([10., 11., 13., 14., 16.])
    data = pd.DataFrame({"weight_kg": x, "fuel_consumption": y})
    fit = p2.fit_linear_regression(data)
    paths = p2.make_plots(data, fit, tmp_path, p2.theoretical_rolling_slope())
    assert {path.name for path in paths} == {
        "part2_regression.png",
        "part2_residuals.png",
        "part2_slope_comparison.png",
    }
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
