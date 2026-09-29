# Active Vision Model

This repository contains a lightweight active-vision and tracking prototype built around camera distortion correction and a Gaussian Mixture Probability Hypothesis Density (GM-PHD) tracker.

## Project Overview

The project combines camera geometry, multi-target tracking, and active-view planning:

1. Camera calibration and distortion correction for image geometry.
2. Multi-target tracking using a simplified Gaussian-mixture PHD approach in polar coordinates.
3. Spatiotemporal neighbor beliefs and uncertainty-guided heading selection.

The overall workflow is:

- the camera model corrects projected image coordinates,
- the tracker predicts target motion in a 4D state space,
- measurements are converted from Cartesian to polar form,
- an extended Kalman filter updates each Gaussian component,
- weak/duplicate components are pruned and merged to keep the mixture manageable,
- the uncertainty model builds a grid, and a heading planner selects a candidate field of view with high remaining uncertainty.

## Architecture

### 1. Distortion Correction
The distortion helper code is intended to model camera lens distortion and recover corrected image coordinates. This is useful when working with real sensors that introduce radial or tangential distortion errors.

Core idea:

- estimate the camera intrinsics,
- project points through the distortion model,
- undo the distortion so downstream tracking uses physically consistent coordinates.

### 2. Bounding Box Utilities
The bounding-box logic is typically used to convert pixel-space detections into trackable observations or regions of interest. In active-vision systems, bounding boxes often serve as input for object association, filtering, or motion estimation.

### 3. GM-PHD Tracking
The file `GMHPHDtracker.py` implements a simplified Gaussian-mixture PHD tracker.

The tracker maintains a list of Gaussian components, each containing:

- `mean`: the 4D state estimate for position and velocity,
- `cov`: the state covariance,
- `weight`: the component weight used for pruning and mixture management.

The main stages are:

#### Prediction
The state transition matrix advances each Gaussian forward in time and adds process noise. This step models target motion between sensor updates.

#### Measurement update
Measurements in polar coordinates are converted to the expected observation model using a Jacobian-based EKF update. The residual is computed between the actual measurement and the predicted measurement, then used to update the mean and covariance.

#### Birth of new components
Unmatched measurements are initialized as new Gaussian components with larger uncertainty so they can be absorbed as newly observed targets.

#### Pruning and merging
The tracker removes weak components and merges nearby ones to keep the number of Gaussians bounded and avoid exponential growth.

### 4. Neighbor Beliefs and Uncertainty
`GPNeighborBelief.py` provides a spatiotemporal Gaussian-process belief using a Matérn 3/2 kernel. `GPNeighborBelief` stores labeled observations and predicts a mean and uncertainty at a queried position and time.

`UncertaintyQuantifier` creates a 2D coordinate grid and computes an uncertainty matrix from a list of GP-like objects. Its inputs implement `prune_old_data(t_now)` and `predict(x, y, t_now)`. The field-of-view mask uses a 90-degree cone and the configured sensor range.

### 5. Heading Selection
`tsp_alg.py` contains `TSPHeadingPlanner`. Despite the module name, this is a discrete heading search, not a traveling-salesperson route optimizer. It scores candidate headings by summing uncertainty values inside the corresponding field of view and returns the selected heading and score. It accepts the uncertainty matrix and coordinate grids produced by `UncertaintyQuantifier`.

Example:

```python
from GPNeighborBelief import UncertaintyQuantifier
from tsp_alg import TSPHeadingPlanner

quantifier = UncertaintyQuantifier(max_fov_range=8.0)
uncertainty = quantifier.compute_UNC_matrix(gp_models, uav_heading, t_now)
planner = TSPHeadingPlanner(max_fov_range=quantifier.max_fov_range)
next_heading, reward = planner.select_best_heading(
	uncertainty,
	quantifier.X_grid,
	quantifier.Y_grid,
	current_heading=uav_heading,
)
```

`gp_models` is a list of objects implementing the prediction and pruning methods described above. The heading planner prefers the current heading when candidate rewards tie, avoiding unnecessary heading changes.

## File structure

- `bounding_box.py` – helper functions related to region and object boundaries.
- `distortion_correction.py` – distortion compensation and camera coordinate utilities.
- `GMHPHDtracker.py` – Gaussian-mixture PHD tracking implementation.
- `GPNeighborBelief.py` – spatiotemporal GP belief and uncertainty-grid implementation.
- `tsp_alg.py` – uncertainty-scored candidate heading planner.
- `test_distortion_correction.py` – camera geometry tests.
- `test_gmphdtracker.py` – tracker update, merging, and input-validation tests.
- `test_tsp_alg.py` – heading selection and uncertainty-grid integration tests.
- `Mockgp.py` – standalone uncertainty-grid behavior tests using a mock GP.

## Setup and Tests

The project requires Python and NumPy. Install NumPy in the active environment with:

```sh
python -m pip install numpy
```

Run the discoverable tests with Python's built-in `unittest` runner:

```sh
python -m unittest discover -v
```

Run the standalone uncertainty-grid checks with:

```sh
python Mockgp.py
```

## Notes on the implementation

This code is intentionally educational and experimental. It demonstrates a practical pattern for combining:

- sensor calibration,
- observation modeling,
- EKF-based updates,
- Gaussian-mixture filtering,
- and mixture management for multi-target tracking.

It is a simplified prototype and is not a full production-grade multi-target tracking system.

## Credits and attribution

This project builds on ideas and patterns from the active perception and multi-target tracking literature, especially Gaussian-mixture PHD filtering and extended Kalman filtering for nonlinear observation models.

The original concepts and implementation principles are credited to the broader research community in target tracking and Bayesian filtering, especially the work on:

- Probability Hypothesis Density (PHD) filters,
- Gaussian Mixture PHD (GM-PHD) methods,
- Extended Kalman filtering for polar measurement models,
- and active vision / perception-driven tracking systems.

This repository is maintained as a practical adaptation and study implementation. Credit is due to the original authors and researchers whose methods inspired this code.

## Usage notes

This project is intended for experimentation and testing in a Python environment. It is useful as a baseline for:

- learning Gaussian-mixture filtering,
- testing sensor correction methods,
- exploring active-vision tracking pipelines,
- and extending the tracker for more advanced association and target management logic.
