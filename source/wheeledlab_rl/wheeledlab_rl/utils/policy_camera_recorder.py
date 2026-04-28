"""Tiled-camera observation recorder.

Records the per-env tiled camera RGB during training rollouts and logs short
clips to disk + wandb. Useful when the viewer-perspective video is too
zoomed-out to see the agent (e.g. many envs scattered across a large plane);
this captures exactly what the policy is consuming.

Layered as a peer to CustomRecordVideo so they share the same step_trigger.
Reads from `env.unwrapped.scene.sensors[sensor_name].data.output["rgb"]`,
which is updated by Isaac Lab on every env.step() call.
"""

import os
from typing import Callable

import av
import gymnasium as gym
import torch
import wandb


class PolicyCameraRecorder(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        sensor_name: str = "camera",
        env_id: int = 0,
        step_trigger: Callable[[int], bool] | None = None,
        video_length: int = 200,
        fps: int = 10,
        name_prefix: str = "policy-camera",
        enable_wandb: bool = True,
        video_crf: int = 30,
        wandb_key: str = "PolicyCamera",
    ):
        super().__init__(env)
        if enable_wandb and (wandb.run is None or wandb.run.name is None):
            raise ValueError("wandb must be initialized before wrapping.")
        os.makedirs(video_folder, exist_ok=True)
        self.video_folder = video_folder
        self.sensor_name = sensor_name
        self.env_id = env_id
        self.step_trigger = step_trigger or (lambda s: False)
        self.video_length = video_length
        self.fps = fps
        self.name_prefix = name_prefix
        self.enable_wandb = enable_wandb
        self.video_crf = video_crf
        self.wandb_key = wandb_key

        self._step_count: int = 0
        self._frames: list = []  # list of (H, W, 3) uint8 numpy
        self._recording: bool = False
        self._video_name: str | None = None
        self._sensor_resolved: bool = False
        self._sensor = None

    # ------------------------------------------------------------------
    # Sensor lookup is deferred until the first step so the scene is built.
    # ------------------------------------------------------------------
    def _resolve_sensor(self):
        scene = self.env.unwrapped.scene
        sensor = scene.sensors.get(self.sensor_name) if hasattr(scene.sensors, "get") else None
        if sensor is None:
            try:
                sensor = scene.sensors[self.sensor_name]
            except (KeyError, AttributeError):
                sensor = None
        if sensor is None:
            print(
                f"[WARN] PolicyCameraRecorder: no sensor named '{self.sensor_name}' "
                f"in scene.sensors; recorder disabled."
            )
        self._sensor = sensor
        self._sensor_resolved = True

    def _capture_frame(self):
        if not self._sensor_resolved:
            self._resolve_sensor()
        if self._sensor is None:
            return None
        rgb = self._sensor.data.output.get("rgb")
        if rgb is None or self.env_id >= rgb.shape[0]:
            return None
        frame = rgb[self.env_id].detach().cpu()
        if frame.dtype != torch.uint8:
            # Camera may emit float [0, 1] depending on Isaac Sim version.
            frame = (frame.clamp(0.0, 1.0) * 255).to(torch.uint8)
        return frame.numpy()

    # ------------------------------------------------------------------
    # Recording lifecycle
    # ------------------------------------------------------------------
    def _start_recording(self):
        self._recording = True
        self._frames = []
        self._video_name = f"{self.name_prefix}-step-{self._step_count}"

    def _save_and_log(self):
        if not self._frames:
            self._recording = False
            self._video_name = None
            return
        path = os.path.join(self.video_folder, f"{self._video_name}.mp4")
        H, W, _ = self._frames[0].shape
        # libx264 / yuv420p needs even dims.
        out_w = W if W % 2 == 0 else W + 1
        out_h = H if H % 2 == 0 else H + 1

        output = av.open(path, "w")
        stream = output.add_stream("libx264", rate=self.fps)
        stream.width, stream.height = out_w, out_h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(self.video_crf), "preset": "veryfast"}
        for frame in self._frames:
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            if (out_w, out_h) != (W, H):
                video_frame = video_frame.reformat(width=out_w, height=out_h)
            for packet in stream.encode(video_frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
        output.close()

        if self.enable_wandb:
            wandb.log({self.wandb_key: wandb.Video(path)}, commit=False)

        self._frames = []
        self._recording = False
        self._video_name = None

    # ------------------------------------------------------------------
    # gym.Wrapper hooks
    # ------------------------------------------------------------------
    def step(self, action):
        if not self._recording and self.step_trigger(self._step_count):
            self._start_recording()

        result = self.env.step(action)
        self._step_count += 1

        if self._recording:
            frame = self._capture_frame()
            if frame is not None:
                self._frames.append(frame)
            if len(self._frames) >= self.video_length:
                self._save_and_log()

        return result

    def close(self):
        if self._recording:
            self._save_and_log()
        return super().close()
