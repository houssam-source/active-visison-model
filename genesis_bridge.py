import os
from pathlib import Path

try:
    import genesis as gs
except Exception:  # pragma: no cover - simulator stack is optional at runtime
    gs = None

import cv2
import numpy as np

from GMHPHDtracker import GMPHDTracker
from GPNeighborBelief import GPNeighborBelief, UncertaintyQuantifier
from tsp_alg import TSPHeadingPlanner
from yolov3_backbone import YOLOv3TinyPerception, resolve_detector_model_path


DEFAULT_DETECTOR_PATH = Path(__file__).resolve().parent / "models" / "drone_yolov8n.pt"


def build_measurement_provider(measurement_provider=None, detector=None):
    if measurement_provider is not None:
        return measurement_provider
    if detector is not None:
        return detector
    return None


def resolve_detector_runtime_path(detector_model_path=None):
    default_paths = [
        Path(__file__).resolve().parent / "models" / "drone_yolov8n.pt",
        Path(__file__).resolve().parent / "models" / "drone_yolov8n_30ep.pt",
        Path(__file__).resolve().parent / "yolov8n.pt",
    ]
    return resolve_detector_model_path(
        model_path=detector_model_path,
        default_paths=default_paths,
        env_var="YOLO_MODEL_PATH",
    )


class GenesisSwarmBridge:
    def __init__(
        self,
        control_hz=10,
        measurement_provider=None,
        backend=None,
        show_viewer=False,
        camera_gui=False,
        detector=None,
        detector_model_path=None,
    ):
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if gs is None:
            raise ImportError(
                "Genesis is required for the simulator runtime. "
                "Install genesis-world and its dependencies first."
            )

        self.detector = detector
        self.measurement_provider = build_measurement_provider(
            measurement_provider=measurement_provider,
            detector=self.detector,
        )
        if self.measurement_provider is None and self.detector is None:
            model_path = resolve_detector_runtime_path(detector_model_path)
            self.detector = YOLOv3TinyPerception(
                model_path=str(model_path),
                res=(640, 480),
                fov_deg=45.0,
            )
            self.measurement_provider = self.detector

        gs.init(backend=gs.cpu if backend is None else backend)
        self.camera_gui = camera_gui
        self.camera_window_name = "Genesis CF2X Drone"
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.005),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())
        drone_file = Path(gs.__file__).parent / "assets" / "urdf" / "drones" / "cf2x.urdf"
        self.uav = self.scene.add_entity(
            gs.morphs.Drone(
                file=str(drone_file),
                pos=(0.0, 0.0, 1.0),
                model="CF2X",
                propellers_spin=(-1, 1, -1, 1),
            )
        )
        self.camera = self.scene.add_camera(
            res=(640, 480),
            pos=(0.7, -1.0, 1.5),
            lookat=(0.0, 0.0, 1.0),
            fov=45,
            GUI=False,
        )
        self.scene.build()

        self.tracker = GMPHDTracker(
            dt=1.0 / control_hz,
            process_noise_std=0.5,
            dist_noise_std=0.15,
            bear_noise_std=0.1,
        )
        self.gp = GPNeighborBelief(
            l_spatial=(2.0, 2.0),
            l_temporal=3.0,
            sigma_f=1.0,
            noise_std=0.1,
        )
        self.unc = UncertaintyQuantifier(resolution=1.0)
        self.tsp = TSPHeadingPlanner(
            num_candidates=15,
            max_fov_range=self.unc.max_fov_range,
        )

        self.control_hz = control_hz
        self.control_steps = max(1, round(1.0 / (self.scene.dt * control_hz)))
        self.uav_mass = float(self._as_numpy(self.uav.get_mass()))
        self.hover_rpm = np.sqrt(self.uav_mass * 9.81 / (4.0 * self.uav.KF))
        self.propeller_spin = np.array([-1.0, 1.0, -1.0, 1.0])
        self.measurement_provider = measurement_provider
        self.target_yaw = 0.0
        self.step_counter = 0

    @staticmethod
    def _as_numpy(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _get_yaw(quaternion):
        w, x, y, z = quaternion
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def _extract_polar_measurements(self, rgb, depth, uav_pos, uav_heading):
        if self.measurement_provider is not None:
            return self.measurement_provider(rgb, depth, uav_pos, uav_heading)
        return self.detector(rgb, depth)

    @staticmethod
    def _to_world_polar_measurements(measurements, uav_pos, uav_heading):
        world_measurements = []
        for distance, relative_bearing in measurements:
            world_bearing = uav_heading + relative_bearing
            x = uav_pos[0] + distance * np.cos(world_bearing)
            y = uav_pos[1] + distance * np.sin(world_bearing)
            world_measurements.append([np.hypot(x, y), np.arctan2(y, x)])
        return world_measurements

    def _apply_control(self):
        current_yaw = self._get_yaw(self._as_numpy(self.uav.get_quat()))
        error = (self.target_yaw - current_yaw + np.pi) % (2.0 * np.pi) - np.pi
        rpm_delta = np.clip(error * self.hover_rpm * 0.05, -self.hover_rpm * 0.2, self.hover_rpm * 0.2)
        self.uav.set_propellers_rpm(self.hover_rpm + self.propeller_spin * rpm_delta)

    def run(self, frames=None):
        if frames is None and not self.camera_gui:
            raise ValueError("frames must be set when camera_gui is disabled")
        if frames is not None and frames < 0:
            raise ValueError("frames must be nonnegative or None")

        t_now = 0.0
        try:
            while frames is None or self.step_counter < frames:
                self._apply_control()
                self.scene.step()
                self.step_counter += 1

                if self.step_counter % self.control_steps != 0:
                    continue

                t_now += 1.0 / self.control_hz
                uav_pos = self._as_numpy(self.uav.get_pos())
                uav_heading = self._get_yaw(self._as_numpy(self.uav.get_quat()))
                self.camera.set_pose(
                    pos=uav_pos + np.array([0.7, -1.0, 0.5]),
                    lookat=uav_pos,
                )
                rgb, depth, _, _ = self.camera.render(rgb=True, depth=True)
                if self.camera_gui:
                    rgb_image = self._as_numpy(rgb)
                    if rgb_image.dtype != np.uint8:
                        scale = 255.0 if rgb_image.max() <= 1.0 else 1.0
                        rgb_image = np.clip(rgb_image * scale, 0, 255).astype(np.uint8)
                    cv2.imshow(
                        self.camera_window_name,
                        cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR),
                    )
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27) or cv2.getWindowProperty(
                        self.camera_window_name, cv2.WND_PROP_VISIBLE
                    ) < 1:
                        break

                measurements = self._extract_polar_measurements(
                    rgb, depth, uav_pos, uav_heading
                )
                ego_velocity = self._as_numpy(
                    self.uav.get_links_vel(links_idx_local=[0])
                )[0, :2]
                self.tracker.predict(ego_velocity=ego_velocity)
                self.tracker.update(
                    self._to_world_polar_measurements(
                        measurements, uav_pos, uav_heading
                    )
                )

                if self.tracker.gaussians:
                    best = max(self.tracker.gaussians, key=lambda gaussian: gaussian["weight"])
                    self.gp.update(best["mean"][0], best["mean"][1], t_now, True)

                uncertainty = self.unc.compute_UNC_matrix(
                    [self.gp], uav_heading, t_now
                )
                self.target_yaw, _ = self.tsp.select_best_heading(
                    uncertainty,
                    self.unc.X_grid,
                    self.unc.Y_grid,
                    uav_heading,
                )
        finally:
            gs.destroy()
            if self.camera_gui:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    GenesisSwarmBridge(camera_gui=True).run()