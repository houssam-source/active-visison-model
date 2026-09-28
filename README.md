# Active Vision Model

This repository contains a lightweight active-vision and tracking prototype built around camera distortion correction and a Gaussian Mixture Probability Hypothesis Density (GM-PHD) tracker.

## Project Overview

The project combines two main responsibilities:

1. Camera calibration and distortion correction for image geometry.
2. Multi-target tracking using a simplified Gaussian-mixture PHD approach in polar coordinates.

The overall workflow is:

- the camera model corrects projected image coordinates,
- the tracker predicts target motion in a 4D state space,
- measurements are converted from Cartesian to polar form,
- an extended Kalman filter updates each Gaussian component,
- weak/duplicate components are pruned and merged to keep the mixture manageable.

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

## File structure

- `bounding_box.py` – helper functions related to region and object boundaries.
- `distortion_correction.py` – distortion compensation and camera coordinate utilities.
- `GMHPHDtracker.py` – Gaussian-mixture PHD tracking implementation.
- `test_distortion_correction.py` – validation tests for the distortion-correction logic.

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
