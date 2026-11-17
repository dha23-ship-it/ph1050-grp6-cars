
# PH1050 Final Project, Problem 4: Cars and Drag

Group 6 project for PH1050 


## Where the data came from

We use **real data**: the **Auto MPG dataset** from the UCI Machine Learning
Repository (Dua & Graff, https://archive.ics.uci.edu/ml). The dataset was
originally taken from the StatLib library at Carnegie Mellon University and was
used in the 1983 American Statistical Association Exposition. The donor/source
is J. R. Quinlan.

- It describes about 400 car models sold in the US, model years 1970-1982.
- Attributes: mpg (US miles per gallon), cylinders, displacement (cubic
  inches), horsepower, weight (lb), acceleration, model year, origin, and car
  name.
- Horsepower is missing for a handful of cars (marked `?` in `auto-mpg.data`).
  These rows are handled in the cleaning step (Part 2).
- Files in `data/`: `auto-mpg.data` (the cleaned UCI table), `auto-mpg.data-original`
  (the original version, with extra rows and missing values), `auto-mpg.names`
  (the description file) and `Index`.
- **File used for the analysis:** `auto-mpg.data` <!-- edit if the team decides otherwise -->
- Units: mpg is US mpg, so we convert with L/100 km = 235.2 / mpg. Weight is in
  pounds, so we convert with kg = lb * 0.45359237. Horsepower is converted with
  kW = hp * 0.7457.
- Limitation: these are American cars from 1970-1982, not modern or Indian
  models. Engines, aerodynamics and tyres have changed since, so numerical
  results are about this fleet only.

## Part 1: Physics model

Code: `carsim.py` (model and integrators), `part1.py` (experiments),
`test_part1.py` (tests).

Model: m dv/dt = F - (1/2) rho Cd A v^2 - Crr m g, with m = 1200 kg, F = 2000 N,
rho = 1.2 kg/m^3, Cd = 0.30, A = 2.2 m^2, Crr = 0.012, g = 9.81 m/s^2.
It is solved with forward Euler and classical RK4. The experiments are:

1. No drag: acceleration is constant, v = a t.
2. Drag only (Crr = 0): v(t) = v_t tanh(t/tau), v_t = sqrt(2F/(rho Cd A)),
   tau = m v_t / F.
3. Terminal speed from the simulation, compared with the formula, plus a
   force-balance check at that speed.
4. Convergence study: max error against step size on a log-log plot, with
   measured orders.
5. Power P = F_resist * v at 100 km/h, and the split between air drag and
   rolling resistance.

## How to run

## Part 3: Data-driven model

Code: `part3_model.py`. Run with `python part3_model.py` from the repo root (reads `data/auto-mpg.data`).

Model: linear regression predicting L/100 km from weight (kg) and engine power (kW). Baseline: the mean L/100 km of the training cars, predicted for every car. Error measure: RMSE. The experiments are:

1. Random 80/20 train/test split (RMSE: baseline 3.975, linear regression 1.755).
2. 5-fold cross-validation (RMSE: baseline 3.909 +/- 0.216, linear regression 1.689 +/- 0.045).
3. Fitted formula on all cars: L/100 km = -0.455 + 0.00587 * mass_kg + 0.04839 * power_kW.
4. Extrapolation test: train on the lightest 70% of cars, test on the heaviest 30% (RMSE: baseline 7.157, linear regression 2.037). The plot is in `report/extrapolation.png`.

Conclusion: linear regression cuts the error by more than half against the mean baseline on random splits and by about 72% on the heaviest cars. Its error on heavy cars (2.04) is only slightly above the cross-validated error (1.69), so the weight trend extrapolates well. It still underpredicts some of the thirstiest cars, so mass and power do not capture everything.

![Extrapolation test](report/extrapolation.png)

    pip install -r requirements.txt
    python part1.py          # prints results, saves plots to figures/
    pytest -q                # runs the tests
