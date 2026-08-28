import json
import os
import shutil
import uuid
import warnings
from collections import defaultdict
from datetime import datetime

import mlflow
import yaml

import wandb


class Logger:
    def __init__(self, config: dict) -> None:
        self.data = defaultdict(lambda: list())
        self.config = config

        self.save_json = config.get("save_json", False)
        self.use_mlflow = config.get("use_mlflow", False)
        self.use_wandb = config.get("use_wandb", False)
        # Check if overwrite flag is explicitly requested
        self.overwrite = config.get("overwrite", False)

        if (experiment_name := self.config.get("experiment", None)) is None:
            raise RuntimeError("Experiment name must be specified in config. experiment: experiment_name")
        if not any((self.save_json, self.use_mlflow, self.use_wandb)):
            warnings.warn(f"No logging functionality has been enabled for experiment: {experiment_name}.", stacklevel=2)

        # Persistent local results directory. This is created unconditionally
        # (not just under save_json) because checkpoints and per-eval-stage
        # JSON files are written to disk regardless of which remote backends
        # (mlflow/wandb) are also enabled. It is intentionally distinct from
        # the wandb scratch directory below, which is deleted on close().
        self.experiment_path = os.path.join("results", experiment_name)
        self.config_path = os.path.join(self.experiment_path, "config.yaml")

        if os.path.isdir(self.experiment_path) and self.overwrite:
            shutil.rmtree(self.experiment_path)

        os.makedirs(self.experiment_path, exist_ok=True)

        if self.save_json:
            if os.path.isfile(self.config_path) and not self.overwrite:
                with open(self.config_path) as file:
                    existing_config = yaml.safe_load(file)
                    for k, v in existing_config.items():
                        if config[k] != v and k != "seed":
                            raise RuntimeError(
                                f"Existing config mismatch. Experiment {experiment_name} has already been ran with {k} as {v}. Conflicting value: {config[k]}."
                            )
            else:
                with open(self.config_path, "w") as outfile:
                    yaml.dump(config, outfile)

        if self.use_mlflow:
            mlflow.set_experiment(experiment_name)
            mlflow.start_run()
            mlflow.log_params(config)

        if self.use_wandb:
            entity = config.get("entity")  
            try:
                wandb.init(
                    entity=entity,
                    project=config.get("project"),
                    name=experiment_name,
                    config=config,
                )
            except Exception as e:
                # Catch 403 Forbidden or communication errors if the entity space isn't accessible
                if entity is not None:
                    print(f"\n[W&B Warning]: Could not initialize under entity '{entity}' (Likely permission denied).")
                    print("Falling back to your default personal workspace...\n")
                    wandb.init(
                        entity=None, 
                        project=config.get("project"),
                        name=experiment_name,
                        config=config,
                    )
                else:
                    raise e

            # Scratch dir for temporary video artifacts only -- kept separate
            # from self.experiment_path (which now always holds the durable
            # checkpoints/eval-json/metrics output) since this one is deleted
            # in close(). Previously this reassigned self.experiment_path
            # itself, which meant enabling both save_json and use_wandb
            # silently deleted the metrics JSON on close().
            self._wandb_scratch_path = os.path.join("results", experiment_name + "_wandb_" + uuid.uuid4().hex)
            os.makedirs(self._wandb_scratch_path, exist_ok=True)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        if self.save_json:
            for k, v in metrics.items():
                self.data[k].append(v)

        if self.use_mlflow:
            mlflow.log_metrics(metrics, step)

        if self.use_wandb:
            wandb.log(metrics, step=step)

    def log_metric(self, key: str, value: float | int, step: int | None = None) -> None:
        if self.save_json:
            self.data[key].append(value)

        if self.use_mlflow:
            mlflow.log_metric(key, value, step)

        if self.use_wandb:
            wandb.log({key: value}, step=step)

    def get_artifact_path(self) -> str:
        if self.save_json:
            return os.path.join(self.experiment_path, "artifacts")
        if self.use_mlflow:
            uri = mlflow.get_artifact_uri()
            if uri.startswith("file://"):
                return uri[len("file://") :]
            elif uri.startswith("/"):  # Check if it's already a local absolute path
                return uri
            else:
                raise RuntimeError(f"MLflow artifact URI is not a file path: {uri}. get_artifact_path() requires extending to handle this.")  # fmt: off
        if self.use_wandb:
            return self._wandb_scratch_path
        else:
            raise RuntimeError("No logging functionality has been enabled. Use either save_json or use_mlflow.")

    def close(self) -> None:
        if self.save_json:
            now = datetime.now()

            runs_path = os.path.join(self.experiment_path, "runs")
            os.makedirs(runs_path, exist_ok=True)
            filename = f"{now.strftime('%Y-%m-%d_%H-%M-%S-%f')}_{uuid.uuid4().hex[:8]}.json"
            with open(
                os.path.join(runs_path, filename),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(self.data, f, ensure_ascii=False, sort_keys=True, indent=4)

        if self.use_mlflow:
            mlflow.end_run()

        if self.use_wandb:
            wandb.finish()
            shutil.rmtree(self._wandb_scratch_path)  # clean up temporary video directory


    def save_checkpoint(self, params: dict, step: int) -> None:
        """Serializes JAX PyTree parameters to a local msgpack file under
        results/<experiment>/checkpoints/, and additionally uploads it as a
        W&B Artifact if use_wandb is enabled. Local saving happens
        unconditionally -- callers decide how often to invoke this (e.g. an
        interval-based checkpoint schedule), not this method."""
        import flax.serialization

        ckpt_dir = os.path.join(self.experiment_path, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{step}.msgpack")
        with open(ckpt_path, "wb") as f:
            f.write(flax.serialization.to_bytes(params))

        if not self.use_wandb:
            return

        artifact = wandb.Artifact(
            name=f"{self.config.get('experiment', 'run')}_model",
            type="model",
            metadata={"step": step}
        )
        artifact.add_file(ckpt_path)
        wandb.log_artifact(artifact)

    def save_eval_returns(self, step: int, returns: list) -> str:
        """For off-policy algorithms: writes the per-episode returns from one
        eval stage (e.g. 10 greedy rollouts) to results/<experiment>/evals/eval_<step>.json,
        and logs the mean/std/min/max to W&B."""
        returns = [float(r) for r in returns]
        mean = float(sum(returns) / len(returns)) if returns else 0.0
        payload = {
            "step": step,
            "num_runs": len(returns),
            "returns": returns,
            "mean": mean,
        }
        if self.use_wandb and returns:
            std = float((sum((r - mean) ** 2 for r in returns) / len(returns)) ** 0.5)
            wandb.log(
                {
                    "eval/return_mean": mean,
                    "eval/return_std": std,
                    "eval/return_min": min(returns),
                    "eval/return_max": max(returns),
                },
                step=step,
            )
        return self._write_eval_json(step, payload)

    def save_eval_metrics(self, step: int, metrics: dict) -> str:
        """For on-policy algorithms: writes a snapshot of the metrics already
        being tracked for this chunk (return mean, skill/option usage, etc.)
        to results/<experiment>/evals/eval_<step>.json, instead of running
        separate eval episodes."""
        payload = {"step": step, **metrics}
        return self._write_eval_json(step, payload)

    def _write_eval_json(self, step: int, payload: dict) -> str:
        eval_dir = os.path.join(self.experiment_path, "evals")
        os.makedirs(eval_dir, exist_ok=True)
        path = os.path.join(eval_dir, f"eval_{step}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, sort_keys=True, indent=4)
        return path

    def log_eval_trajectory(self, step: int, trajectory: dict, frames: list = None) -> None:
        """Logs a per-timestep trajectory table and optional video to W&B."""
        if not self.use_wandb:
            return
            
        # 1. Log the per-timestep Table
        table = wandb.Table(columns=["timestep", "reward", "cumulative_reward", "option_chosen"])
        
        cum_reward = 0.0
        for t in range(len(trajectory["reward"])):
            r = trajectory["reward"][t]
            opt = trajectory["option"][t]
            cum_reward += r
            table.add_data(t, r, cum_reward, opt)
            
        wandb.log({f"eval/trajectory_table": table}, step=step)
        
        # 2. Log Video 
        if frames and len(frames) > 0:
            import numpy as np
            frames_arr = np.array(frames)  # Expected: (T, H, W, C)
            
            # W&B Video requires channel-first format (T, C, H, W)
            if frames_arr.ndim == 4 and frames_arr.shape[-1] == 3:
                frames_arr = np.transpose(frames_arr, (0, 3, 1, 2))
                
            wandb.log({f"eval/video": wandb.Video(frames_arr, fps=15, format="mp4")}, step=step)
