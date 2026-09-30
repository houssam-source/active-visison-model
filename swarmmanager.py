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
from yolov3_backbone import YOLOv3TinyPerception


class SwarmActiveVisionEnv:
    def __init__(self, n_drones=3, control_hz=10, gui=False, camera_resolution=(640, 480), learning_epochs=10):
        if gs is None:
            raise ImportError("Genesis is required for SwarmActiveVisionEnv. Install genesis-world and its dependencies first.")

        self.camera_resolution = tuple(camera_resolution)

        backend = gs.gpu if hasattr(gs, "gpu") else gs.cpu
        gs.init(backend=backend, logging_level="error")
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.005),
            show_viewer=False,
        )
        self.scene.add_entity(gs.morphs.Plane())

        self.drones = []
        self.perceptions = []
        self.pipelines = []

        drone_file = Path(gs.__file__).parent / "assets" / "urdf" / "drones" / "cf2x.urdf"
        for i in range(n_drones):
            drone = self.scene.add_entity(gs.morphs.URDF(file=str(drone_file)), material=gs.materials.Rigid())
            cam = self.scene.add_camera(res=self.camera_resolution, fov=90.0, GUI=False)
            self.drones.append({"entity": drone, "camera": cam, "heading": 0.0})
            self.perceptions.append(YOLOv3TinyPerception(res=self.camera_resolution, epochs=learning_epochs))
            # Initialize per-drone GM-PHD + GP + UNC + TSP (use your existing classes)
            self.pipelines.append({
                "tracker": GMPHDTracker(dt=1.0/control_hz, process_noise_std=0.5, dist_noise_std=0.15, bear_noise_std=0.1),
                "gp": GPNeighborBelief(l_spatial=(2.0, 2.0), l_temporal=3.0, sigma_f=1.0, noise_std=0.1),
                "unc": UncertaintyQuantifier(resolution=1.0),
                "tsp": TSPHeadingPlanner(num_candidates=15)
            })
            
        self.scene.build()
        self.control_hz = control_hz
        self.gui = bool(gui)
        self.viewer_closed = False
        self.control_steps = int(1.0 / (self.scene.dt * control_hz))
        self.step_counter = 0

    def _update_agent(self, agent_idx):
        agent = self.drones[agent_idx]
        position = agent["entity"].get_pos()
        agent["camera"].set_pose(
            pos=position,
            lookat=[
                position[0] + np.cos(agent["heading"]),
                position[1] + np.sin(agent["heading"]),
                position[2] + 0.1,
            ],
        )
        rgb, depth, _, _ = agent["camera"].render(rgb=True, depth=True)
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
            pipe["gp"].update(m[0], m[1], self.step_counter * self.scene.dt, True)
            
        U = pipe["unc"].compute_UNC_matrix([pipe["gp"]], agent["heading"], self.step_counter * self.scene.dt)
        target_yaw, reward = pipe["tsp"].select_best_heading(U, pipe["unc"].X_grid, pipe["unc"].Y_grid, agent["heading"])
        
        # P-control yaw rate
        error = (target_yaw - agent["heading"] + np.pi) % (2*np.pi) - np.pi
        agent["heading"] += np.clip(error * 2.0, -1.5, 1.5) * (1.0/self.control_hz)
        return reward, U, rgb

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
        self.scene.step()
        self.step_counter += 1
        rewards, unc_maps = [], []
        if self.step_counter % self.control_steps == 0:
            frames = []
            for agent_idx in range(len(self.drones)):
                r, u, frame = self._update_agent(agent_idx)
                rewards.append(r)
                unc_maps.append(u)
                frames.append(frame)
            if self.gui and frames:
                self._show_camera_frame(frames[0])
        return rewards, unc_maps

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


if __name__ == "__main__":
    SwarmActiveVisionEnv(gui=True).run()