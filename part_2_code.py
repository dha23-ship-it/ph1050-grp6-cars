"""
Part 2: Vehicle mass vs. fuel consumption.

Run from the repository root:
    python part_2_code.py

The script reads data/auto-mpg.data when available. If the file is absent,
it downloads the UCI Auto MPG data to the requested location.

Outputs:
    figures/part2_regression.png
    figures/part2_residuals.png
    figures/part2_slope_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib
matplotlib.use("Agg")  # Allow reproducible plot generation on headless systems.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


UCI_AUTO_MPG_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
)
COLUMNS = [
    "mpg", "cylinders", "displacement", "horsepower", "weight",
    "acceleration", "model_year", "origin", "car_name",
]

# Unit conversions and theoretical-model assumptions.
MPG_TO_L_PER_100KM = 235.2
LB_TO_KG = 0.45359237
CRR = 0.010
GRAVITY = 9.81                  # m s^-2
DISTANCE_M = 100_000.0          # 100 km, in metres
ENGINE_EFFICIENCY = 0.20        # Assumed constant efficiency
FUEL_ENERGY_J_PER_LITRE = 32e6  # Approximate energy content of petrol


def load_auto_mpg(data_path: str | Path = "data/auto-mpg.data",
                  download_if_missing: bool = True) -> pd.DataFrame:
    """Load the whitespace-delimited UCI data, preserving car names."""
    path = Path(data_path)
    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"Auto MPG data file not found: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Local data not found; downloading Auto MPG data to {path}")
        urlretrieve(UCI_AUTO_MPG_URL, path)

    return pd.read_csv(
        path,
        sep=r"\s+",
        names=COLUMNS,
        na_values="?",
        quotechar='"',
    )


def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Add SI/consumption units and retain rows with valid positive MPG and mass.

    Horsepower is not required for Part 2, so missing horsepower does not
    cause an observation to be discarded.
    """
    data = raw.copy()
    data["fuel_consumption"] = MPG_TO_L_PER_100KM / pd.to_numeric(
        data["mpg"], errors="coerce"
    )
    data["weight_kg"] = LB_TO_KG * pd.to_numeric(
        data["weight"], errors="coerce"
    )
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["fuel_consumption", "weight_kg"])
    data = data[
        (data["fuel_consumption"] > 0) & (data["weight_kg"] > 0)
    ].copy()
    return data


def fit_linear_regression(data: pd.DataFrame) -> dict[str, float]:
    """Fit fuel consumption (L/100 km) against mass (kg)."""
    if len(data) < 3:
        raise ValueError("At least three valid observations are required.")
    x = data["weight_kg"].to_numpy(dtype=float)
    y = data["fuel_consumption"].to_numpy(dtype=float)
    result = stats.linregress(x, y)
    if not np.isfinite(result.stderr):
        raise ValueError("Regression could not estimate a finite standard error.")

    df = len(x) - 2
    t_critical = stats.t.ppf(0.975, df)
    margin = t_critical * result.stderr
    return {
        "n": int(len(x)),
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "r_squared": float(result.rvalue ** 2),
        "p_value": float(result.pvalue),
        "std_error": float(result.stderr),
        "ci_lower": float(result.slope - margin),
        "ci_upper": float(result.slope + margin),
    }


def theoretical_rolling_slope(
    crr: float = CRR,
    gravity: float = GRAVITY,
    distance_m: float = DISTANCE_M,
    efficiency: float = ENGINE_EFFICIENCY,
    energy_per_litre: float = FUEL_ENERGY_J_PER_LITRE,
) -> float:
    """
    Return rolling-resistance-only slope in (L/100 km)/kg.

    From F_r = Crr*m*g, W = F_r*d, and fuel energy
    E_fuel = W/efficiency, fuel volume per distance is
    Crr*g*d/(efficiency*energy_per_litre) per kg.

    This is a simplified contribution, not a complete vehicle fuel model.
    """
    if crr < 0 or gravity <= 0 or distance_m <= 0:
        raise ValueError("Crr must be non-negative; g and distance must be positive.")
    if not 0 < efficiency <= 1:
        raise ValueError("Efficiency must be in the interval (0, 1].")
    if energy_per_litre <= 0:
        raise ValueError("Energy per litre must be positive.")
    return crr * gravity * distance_m / (efficiency * energy_per_litre)


def make_plots(data: pd.DataFrame, result: dict[str, float],
               output_dir: str | Path = "figures",
               theoretical_slope: float | None = None) -> list[Path]:
    """Save regression, residual, and slope-comparison figures."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    x = data["weight_kg"].to_numpy(dtype=float)
    y = data["fuel_consumption"].to_numpy(dtype=float)
    order = np.argsort(x)
    xs = x[order]
    predicted = result["intercept"] + result["slope"] * x
    residuals = y - predicted

    paths = []

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, alpha=0.6, label="Observed cars")
    ax.plot(xs, result["intercept"] + result["slope"] * xs,
            color="red", label="Least-squares fit")
    ax.set(xlabel="Vehicle mass (kg)",
           ylabel="Fuel consumption (L/100 km)",
           title="Fuel consumption vs. vehicle mass")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = out / "part2_regression.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(x, residuals, alpha=0.6)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xlabel="Vehicle mass (kg)",
           ylabel="Residual (observed - fitted, L/100 km)",
           title="Residuals of fuel-consumption regression")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out / "part2_residuals.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    if theoretical_slope is not None:
        # Anchor the theoretical line at the sample mean. This isolates a
        # comparison of slopes; it is not an absolute prediction curve.
        theoretical_line = theoretical_slope * xs + (
            np.mean(y) - theoretical_slope * np.mean(x)
        )
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(x, y, alpha=0.45, label="Observed cars")
        ax.plot(xs, result["intercept"] + result["slope"] * xs,
                color="red", label="Empirical regression")
        ax.plot(xs, theoretical_line, linestyle="--", color="green",
                label="Rolling-resistance slope (anchored at sample mean)")
        ax.set(xlabel="Vehicle mass (kg)",
               ylabel="Fuel consumption (L/100 km)",
               title="Empirical slope vs. rolling-resistance-only slope")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = out / "part2_slope_comparison.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    return paths


def run_analysis(data_path: str | Path = "data/auto-mpg.data",
                 output_dir: str | Path = "figures",
                 download_if_missing: bool = True) -> dict:
    raw = load_auto_mpg(data_path, download_if_missing=download_if_missing)
    data = prepare_data(raw)
    result = fit_linear_regression(data)
    theory = theoretical_rolling_slope()
    paths = make_plots(data, result, output_dir, theory)

    # Convert slope and CI to the more readable per-tonne unit.
    empirical_per_tonne = result["slope"] * 1000
    theory_per_tonne = theory * 1000
    ratio = result["slope"] / theory

    print("\nPART 2: VEHICLE MASS VS FUEL CONSUMPTION")
    print("----------------------------------------")
    print(f"Rows in source data:       {len(raw)}")
    print(f"Valid observations:        {result['n']}")
    print(f"Rows excluded:             {len(raw) - result['n']}")
    print(f"Slope:                     {result['slope']:.8f} L/100 km/kg")
    print(f"Intercept:                 {result['intercept']:.6f} L/100 km")
    print(f"Slope per tonne:           {empirical_per_tonne:.3f} L/100 km/tonne")
    print(f"95% CI per tonne:          "
          f"[{result['ci_lower']*1000:.3f}, {result['ci_upper']*1000:.3f}]")
    print(f"R-squared:                 {result['r_squared']:.4f}")
    print(f"P-value:                   {result['p_value']:.3e}")
    print(f"Standard error:            {result['std_error']:.8f}")
    print("\nROLLING-RESISTANCE COMPARISON")
    print("-----------------------------")
    print(f"Assumed Crr:               {CRR:.3f}")
    print(f"Theoretical slope:         {theory:.8f} L/100 km/kg")
    print(f"Theoretical per tonne:     {theory_per_tonne:.3f} L/100 km/tonne")
    print(f"Empirical/theoretical:     {ratio:.2f}")
    print("Note: the theoretical estimate includes rolling resistance only,")
    print("      and assumes constant engine efficiency.")
    print("\nSaved figures:")
    for path in paths:
        print(f"  {path}")
    return {"raw": raw, "data": data, "regression": result,
            "theoretical_slope": theory, "ratio": ratio, "figures": paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/auto-mpg.data",
                        help="Path to UCI Auto MPG data file")
    parser.add_argument("--output-dir", default="figures",
                        help="Directory for generated figures")
    parser.add_argument("--no-download", action="store_true",
                        help="Fail rather than downloading if the data file is absent")
    args = parser.parse_args()
    run_analysis(args.data, args.output_dir,
                 download_if_missing=not args.no_download)


if __name__ == "__main__":
    main()
