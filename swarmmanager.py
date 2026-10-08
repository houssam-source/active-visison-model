import numpy as np
import cv2
from pathlib import Path

try:
    import genesis as gs
except Exception:  # pragma: no cover - optional runtime dependency
    gs = None

from GMHPHDtracker import GMPHDTracker
from GPNeighborBelief import GPNeighborBelief, UncertaintyQuantifier
from tsp_alg import TSPHeadingPlanner
from yolov3_backbone import YOLOv3TinyPerception, resolve_detector_model_path


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_SWARM_MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "drone_yolov8n_30ep.pt",
    PROJECT_ROOT / "models" / "drone_yolov8n.pt",
    PROJECT_ROOT / "yolov8n.pt",
]


class SyntheticSoftwarePerception:
    """Lightweight software-only sensor used for simulation-first runs before any real drone model is connected."""

    def __init__(self, res=(640, 480), heading_bias=0.0):
        self.res = tuple(res)
        self.heading_bias = float(heading_bias)

    def __call__(self, rgb, depth):
        height, width = self.res[1], self.res[0]
        center_x = width / 2.0
        center_y = height / 2.0
        t = (self.heading_bias + 0.35) % (2 * np.pi)
        radius = min(width, height) * 0.18
        sensor_offset = 0.3 + 0.5 * np.sin(t)
        return [[
            radius * (1.0 + sensor_offset),
            np.arctan2(center_y - (center_y + np.sin(t) * radius), center_x - (center_x + np.cos(t) * radius))
        ]]


class SwarmActiveVisionEnv:
    def __init__(
        self,
        n_drones=3,
        control_hz=10,
        gui=False,
        camera_resolution=(640, 480),
        learning_epochs=10,
        detector_model_path=None,
        software_simulation=False,
    ):
        self.camera_resolution = tuple(camera_resolution)
        self.software_simulation = bool(software_simulation)
        self.control_hz = float(control_hz)
        self.gui = bool(gui)
        self.viewer_closed = False
        self.step_counter = 0
        self.latest_frame = None
        self.latest_agent_state = None
        self.last_render_error = None
        self.scene = None
        self.scene_dt = 1.0 / max(1.0, self.control_hz)
        self.control_steps = max(1, int(round(self.control_hz / 2.0)))
        self.drones = []
        self.perceptions = []
        self.pipelines = []

        if detector_model_path is None and not self.software_simulation:
            detector_model_path = DEFAULT_SWARM_MODEL_CANDIDATES[-1]

        if detector_model_path:
            try:
                self.detector_model_path = resolve_detector_model_path(
                    detector_model_path,
                    default_paths=DEFAULT_SWARM_MODEL_CANDIDATES,
                )
            except FileNotFoundError:
                self.detector_model_path = None
                self.software_simulation = True
        else:
            self.detector_model_path = None
            self.software_simulation = True

        if self.software_simulation or gs is None:
            self._init_software_environment(n_drones)
            return

        try:
            backend = gs.gpu if hasattr(gs, "gpu") else gs.cpu
            gs.init(backend=backend, logging_level="error")
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=0.005),
                show_viewer=False,
            )
            self.scene.add_entity(gs.morphs.Plane())
            drone_file = Path(gs.__file__).parent / "assets" / "urdf" / "drones" / "cf2x.urdf"
            for i in range(max(1, int(n_drones))):
                drone = self.scene.add_entity(gs.morphs.URDF(file=str(drone_file)), material=gs.materials.Rigid())
                cam = self.scene.add_camera(res=self.camera_resolution, fov=90.0, GUI=False)
                self.drones.append({"entity": drone, "camera": cam, "heading": 0.0})
                if self.detector_model_path is None:
                    self.perceptions.append(SyntheticSoftwarePerception(res=self.camera_resolution, heading_bias=i * 0.8))
                else:
                    self.perceptions.append(
                        YOLOv3TinyPerception(
                            model_path=str(self.detector_model_path),
                            res=self.camera_resolution,
                            epochs=learning_epochs,
                        )
                    )
                self.pipelines.append({
                    "tracker": GMPHDTracker(dt=1.0/self.control_hz, process_noise_std=0.5, dist_noise_std=0.15, bear_noise_std=0.1),
                    "gp": GPNeighborBelief(l_spatial=(2.0, 2.0), l_temporal=3.0, sigma_f=1.0, noise_std=0.1),
                    "unc": UncertaintyQuantifier(resolution=1.0),
                    "tsp": TSPHeadingPlanner(num_candidates=15),
                })
            self.scene.build()
            self.control_steps = int(1.0 / (self.scene.dt * self.control_hz)) if getattr(self.scene, "dt", None) else self.control_steps
        except Exception:
            self._init_software_environment(n_drones)

    def _init_software_environment(self, n_drones):
        self.software_simulation = True
        self.scene = None
        self.drones = []
        self.perceptions = []
        self.pipelines = []
        for i in range(max(1, int(n_drones))):
            angle = (2 * np.pi * i) / max(1, int(n_drones))
            self.drones.append({
                "entity": {"position": [np.cos(angle) * 1.7, np.sin(angle) * 1.2, 0.1]},
                "camera": None,
                "heading": float(angle),
                "reward": 0.0,
                "target_yaw": 0.0,
                "measurement_count": 0,
                "uncertainty_mean": 0.0,
            })
            self.perceptions.append(SyntheticSoftwarePerception(res=self.camera_resolution, heading_bias=angle))
            self.pipelines.append({
                "tracker": GMPHDTracker(dt=1.0/self.control_hz, process_noise_std=0.5, dist_noise_std=0.15, bear_noise_std=0.1),
                "gp": GPNeighborBelief(l_spatial=(2.0, 2.0), l_temporal=3.0, sigma_f=1.0, noise_std=0.1),
                "unc": UncertaintyQuantifier(resolution=1.0),
                "tsp": TSPHeadingPlanner(num_candidates=15),
            })
        self.control_steps = max(1, int(round(self.control_hz / 2.0)))
        self.scene_dt = 1.0 / max(1.0, self.control_hz)

    def _render_camera_safe(self, agent):
        try:
            rgb, depth, _, _ = agent["camera"].render(rgb=True, depth=True)
            return np.asarray(rgb), np.asarray(depth), None, None
        except Exception as exc:
            height, width = self.camera_resolution[1], self.camera_resolution[0]
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            step = getattr(self, "step_counter", 0)
            heading = float(agent.get("heading", 0.0))

            for y in range(height):
                tint = 8 + int((y / max(1, height)) * 28)
                rgb[y, :, 0] = tint
                rgb[y, :, 1] = tint + 8
                rgb[y, :, 2] = tint + 15

            for x in range(0, width, 60):
                cv2.line(rgb, (x, 0), (x, height), (24, 32, 36), 1, cv2.LINE_AA)
            for y in range(0, height, 60):
                cv2.line(rgb, (0, y), (width, y), (24, 32, 36), 1, cv2.LINE_AA)

            cx, cy = width // 2, height // 2
            orbit = 120 + 25 * np.sin(step * 0.15)
            drone_x = int(cx + np.cos(heading) * orbit)
            drone_y = int(cy + np.sin(heading) * orbit * 0.6)
            target_x = int(cx + np.cos(heading) * 180)
            target_y = int(cy + np.sin(heading) * 120)

            cv2.circle(rgb, (cx, cy), 90, (45, 55, 62), 2, cv2.LINE_AA)
            cv2.line(rgb, (cx, cy), (target_x, target_y), (80, 220, 255), 3, cv2.LINE_AA)
            cv2.circle(rgb, (target_x, target_y), 12, (80, 220, 255), -1)

            for idx in range(3):
                phase = heading + idx * 2.2 + step * 0.08
                px = int(cx + np.cos(phase) * (100 + idx * 52) + np.sin(step * 0.12 + idx) * 24)
                py = int(cy + np.sin(phase) * (80 + idx * 40) + np.cos(step * 0.12 + idx) * 16)
                cv2.circle(rgb, (px, py), 18, (40 + idx * 35, 180 + idx * 18, 90), -1)
                cv2.line(rgb, (px, py), (px + int(np.cos(phase) * 28), py + int(np.sin(phase) * 28)), (255, 255, 255), 3, cv2.LINE_AA)

            cv2.circle(rgb, (drone_x, drone_y), 24, (255, 200, 50), -1)
            cv2.line(rgb, (drone_x, drone_y), (drone_x + int(np.cos(heading) * 40), drone_y + int(np.sin(heading) * 40)), (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(
                rgb,
                "Headless fallback",
                (18, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                rgb,
                f"yaw={heading:.2f}",
                (18, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            depth = np.zeros((height, width), dtype=np.float32)
            self.last_render_error = str(exc)
            return rgb, depth, None, None

    def _update_agent(self, agent_idx):
        agent = self.drones[agent_idx]
        if self.software_simulation or self.scene is None:
            step_phase = self.step_counter / max(1.0, self.control_hz)
            orbit = 1.3 + 0.35 * np.sin(step_phase * 2.0 + agent_idx)
            position = [
                np.cos(step_phase + agent_idx) * orbit,
                np.sin(step_phase * 1.6 + agent_idx) * orbit,
                0.1,
            ]
            agent["position"] = position
            agent["heading"] = float(agent.get("heading", 0.0) + 0.08 * np.sin(step_phase * 2.0 + agent_idx))
            rgb, depth, _, _ = self._render_camera_safe(agent)
            measurements = self.perceptions[agent_idx](rgb, depth)
            pipe = self.pipelines[agent_idx]
            ego_velocity = np.array([np.cos(agent["heading"]) * 0.5, np.sin(agent["heading"]) * 0.5], dtype=float)
            pipe["tracker"].predict(ego_velocity=ego_velocity)
            pipe["tracker"].update([np.array(m) for m in measurements])
            tracked = sorted(pipe["tracker"].gaussians, key=lambda g: g["weight"], reverse=True)
            if tracked:
                m = tracked[0]["mean"]
                pipe["gp"].update(m[0], m[1], self.step_counter * self.scene_dt, True)
            U = pipe["unc"].compute_UNC_matrix([pipe["gp"]], agent["heading"], self.step_counter * self.scene_dt)
            target_yaw, reward = pipe["tsp"].select_best_heading(U, pipe["unc"].X_grid, pipe["unc"].Y_grid, agent["heading"])
            error = (target_yaw - agent["heading"] + np.pi) % (2*np.pi) - np.pi
            new_heading = agent["heading"] + np.clip(error * 2.0, -1.5, 1.5) * (1.0/self.control_hz)
            agent["heading"] = float(new_heading)
            agent["reward"] = float(reward)
            agent["target_yaw"] = float(target_yaw)
            agent["measurement_count"] = int(len(measurements))
            agent["uncertainty_mean"] = float(np.mean(U))
            return reward, U, rgb, {
                "agent_index": agent_idx,
                "heading": float(agent["heading"]),
                "target_yaw": float(target_yaw),
                "reward": float(reward),
                "measurement_count": int(len(measurements)),
                "uncertainty_mean": float(np.mean(U)),
                "position": [float(v) for v in position],
            }

        position = agent["entity"].get_pos()
        agent["camera"].set_pose(
            pos=position,
            lookat=[
                position[0] + np.cos(agent["heading"]),
                position[1] + np.sin(agent["heading"]),
                position[2] + 0.1,
            ],
        )
        rgb, depth, _, _ = self._render_camera_safe(agent)
        measurements = self.perceptions[agent_idx](rgb, depth)
        pipe = self.pipelines[agent_idx]

        ego_velocity = agent["entity"].get_links_vel(links_idx_local=[0])[0, :2]
        if hasattr(ego_velocity, "detach"):
            ego_velocity = ego_velocity.detach().cpu().numpy()
        pipe["tracker"].predict(ego_velocity=ego_velocity)
        pipe["tracker"].update([np.array(m) for m in measurements])

        tracked = sorted(pipe["tracker"].gaussians, key=lambda g: g["weight"], reverse=True)
        if tracked:
            m = tracked[0]["mean"]
            pipe["gp"].update(m[0], m[1], self.step_counter * self.scene_dt, True)

        U = pipe["unc"].compute_UNC_matrix([pipe["gp"]], agent["heading"], self.step_counter * self.scene_dt)
        target_yaw, reward = pipe["tsp"].select_best_heading(U, pipe["unc"].X_grid, pipe["unc"].Y_grid, agent["heading"])

        error = (target_yaw - agent["heading"] + np.pi) % (2*np.pi) - np.pi
        new_heading = agent["heading"] + np.clip(error * 2.0, -1.5, 1.5) * (1.0/self.control_hz)
        agent["heading"] = float(new_heading)
        agent["reward"] = float(reward)
        agent["target_yaw"] = float(target_yaw)
        agent["measurement_count"] = int(len(measurements))
        agent["uncertainty_mean"] = float(np.mean(U))
        return reward, U, rgb, {
            "agent_index": agent_idx,
            "heading": float(agent["heading"]),
            "target_yaw": float(target_yaw),
            "reward": float(reward),
            "measurement_count": int(len(measurements)),
            "uncertainty_mean": float(np.mean(U)),
            "position": [float(v) for v in position],
        }

    def _show_camera_frame(self, frame):
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            scale = 255.0 if frame.size and frame.max() <= 1.0 else 1.0
            frame = np.clip(frame * scale, 0, 255).astype(np.uint8)

        cv2.imshow("Genesis Swarm Camera", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        try:
            window_open = cv2.getWindowProperty(
                "Genesis Swarm Camera", cv2.WND_PROP_VISIBLE
            ) >= 1
        except cv2.error:
            window_open = False
        self.viewer_closed = key in (ord("q"), 27) or not window_open

    def step(self):
        if self.scene is not None:
            self.scene.step()
        self.step_counter += 1
        rewards, unc_maps = [], []
        agent_states = []
        if self.step_counter % self.control_steps == 0:
            frames = []
            for agent_idx in range(len(self.drones)):
                r, u, frame, state = self._update_agent(agent_idx)
                rewards.append(r)
                unc_maps.append(u)
                frames.append(frame)
                agent_states.append(state)
            if frames:
                self.latest_frame = frames[0]
                self.latest_agent_state = agent_states[0] if agent_states else None
            if self.gui and frames:
                self._show_camera_frame(frames[0])
        return {
            "step": int(self.step_counter),
            "rewards": rewards,
            "uncertainty_maps": unc_maps,
            "agents": agent_states,
            "n_drones": len(self.drones),
            "control_steps": self.control_steps,
        }

    def run(self, frames=None):
        if frames is None and not self.gui:
            raise ValueError("frames must be set when the visualizer is disabled")
        if frames is not None and frames < 0:
            raise ValueError("frames must be nonnegative or None")

        try:
            while (frames is None or self.step_counter < frames) and not self.viewer_closed:
                self.step()
        finally:
            if self.gui:
                cv2.destroyAllWindows()


class EmbodiedSwarmRuntime:
    def __init__(
        self,
        n_drones=1,
        control_hz=10,
        gui=False,
        camera_resolution=(640, 480),
        learning_epochs=10,
        detector_model_path=None,
        skip_simulation=False,
        software_simulation=True,
    ):
        self.n_drones = max(1, int(n_drones))
        self.control_hz = float(control_hz)
        self.gui = bool(gui)
        self.camera_resolution = tuple(camera_resolution)
        self.learning_epochs = int(learning_epochs)
        self.software_simulation = bool(software_simulation)
        if detector_model_path is None and not self.software_simulation:
            detector_model_path = DEFAULT_SWARM_MODEL_CANDIDATES[-1]
        try:
            self.model_path = resolve_detector_model_path(
                detector_model_path,
                default_paths=DEFAULT_SWARM_MODEL_CANDIDATES,
                env_var="YOLO_MODEL_PATH",
            )
        except FileNotFoundError:
            self.model_path = Path(detector_model_path) if detector_model_path else PROJECT_ROOT / "models" / "drone_yolov8n.pt"
            self.software_simulation = True
        self.skip_simulation = bool(skip_simulation)
        self.env = None
        self.latest_frame = None
        self.latest_agent_state = None

        if not self.skip_simulation:
            self.env = SwarmActiveVisionEnv(
                n_drones=self.n_drones,
                control_hz=self.control_hz,
                gui=self.gui,
                camera_resolution=self.camera_resolution,
                learning_epochs=self.learning_epochs,
                detector_model_path=str(self.model_path) if self.model_path.is_file() else None,
                software_simulation=self.software_simulation,
            )

    def summary(self):
        return {
            "mode": "embodied-swarm-runtime",
            "n_drones": self.n_drones,
            "active_agents": self.n_drones,
            "control_hz": self.control_hz,
            "model_path": str(self.model_path),
            "skip_simulation": self.skip_simulation,
            "software_simulation": self.software_simulation,
            "simulator_ready": self.env is not None,
        }

    def step(self, steps=1):
        if self.skip_simulation or self.env is None:
            return {
                **self.summary(),
                "agents": [{"agent_index": i, "heading": 0.0, "reward": 0.0, "measurement_count": 0} for i in range(self.n_drones)],
                "rewards": [0.0 for _ in range(self.n_drones)],
                "uncertainty": [0.0 for _ in range(self.n_drones)],
            }
        result = []
        for _ in range(max(1, int(steps))):
            result.append(self.env.step())
        if getattr(self.env, "latest_frame", None) is not None:
            self.latest_frame = self.env.latest_frame
        if getattr(self.env, "latest_agent_state", None) is not None:
            self.latest_agent_state = self.env.latest_agent_state
        summary = self.summary()
        summary["history"] = result
        summary["rewards"] = [float(step["rewards"][idx]) if step["rewards"] else 0.0 for idx in range(self.n_drones) for step in result]
        summary["uncertainty"] = [float(np.mean(step["uncertainty_maps"][idx])) if step["uncertainty_maps"] else 0.0 for idx in range(self.n_drones) for step in result]
        summary["agents"] = [
            {
                "agent_index": idx,
                "heading": 0.0,
                "reward": 0.0,
                "measurement_count": 0,
            }
            for idx in range(self.n_drones)
        ]
        if self.latest_frame is not None:
            summary["latest_frame_shape"] = list(self.latest_frame.shape)
            summary["latest_agent_state"] = self.latest_agent_state
        return summary

    def preview(self):
        if self.skip_simulation or self.env is None or self.env.latest_frame is None:
            return {
                "available": False,
                "message": "No simulation frame available yet.",
            }
        frame = np.asarray(self.env.latest_frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))[1]
        import base64
        return {
            "available": True,
            "frame": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "shape": list(frame.shape),
            "agent_state": self.env.latest_agent_state,
            "message": getattr(self.env, "last_render_error", None) and "Headless render fallback active; continuing simulation loop." or "Live swarm preview active.",
        }

    def run(self, frames=None):
        if self.skip_simulation or self.env is None:
            return self.summary()
        result = self.env.run(frames=frames)
        return {
            **self.summary(),
            "result": result,
        }


if __name__ == "__main__":
    EmbodiedSwarmRuntime(gui=True).run()