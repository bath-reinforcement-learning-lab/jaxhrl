from typing import Any

import gymnasium
import numpy as np


class FireResetEnv(gymnasium.Wrapper):
    """
    Take action on reset for environments that are fixed until firing.

    :param env: Environment to wrap
    """

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs) -> tuple[Any, dict]:  # noqa: ANN003
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class ToggleableRecordVideo(gymnasium.wrappers.RecordVideo):
    """A RecordVideo wrapper that allows for toggling the recording on and off."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self._is_recording_enabled = True

    def toggle_recording(self, enabled: bool) -> None:
        """Enable or disable video recording."""
        self._is_recording_enabled = enabled

        # If we are disabling recording, and a video is currently being recorded, close it.
        if not enabled and self.recording:
            self.stop_recording()

    def step(self, action: int) -> tuple:
        """Step the environment and record a frame if recording is enabled."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_id += 1

        if terminated or truncated:
            self.toggle_recording(enabled=False)

        if self._is_recording_enabled and self.step_trigger and self.step_trigger(self.step_id):
            self.start_recording(f"{self.name_prefix}-step-{self.step_id}")
        if self._is_recording_enabled and self.recording:
            self._capture_frame()

            if len(self.recorded_frames) > self.video_length:
                self.stop_recording()

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> tuple:  # noqa: ANN003
        """Reset the environment and start a new recording if enabled."""
        if self.recording:
            self.stop_recording()

        obs, info = self.env.reset(**kwargs)

        if self._is_recording_enabled and self.episode_trigger and self.episode_trigger(self.episode_id):
            self.episode_id += 1
            self.start_recording(f"{self.name_prefix}-episode-{self.episode_id}")

        return obs, info

class TupleToArrayObservation(gymnasium.ObservationWrapper):
    """Converts a raw (row, col) tuple observation into a numpy array
    matching the declared MultiDiscrete observation space."""

    def observation(self, obs: tuple[int, int]) -> np.ndarray:
        return np.array(obs, dtype=self.observation_space.dtype)
