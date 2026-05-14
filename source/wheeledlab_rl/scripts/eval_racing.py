"""
Evals -p policy on --num-tracks __  log to wandb with --wandb

spawns num tracks and records time for single lap completion on 
specific seeds. Does this for each policy and logs averages to 
wandb

Usage: see eval.sh in eval_confgis
Params default to racing/configs/eval_configs/racing_eval.yaml
"""

###################################
###### BEGIN ISAACLAB SPINUP ######
###################################

from wheeledlab_rl.startup import startup
import argparse

parser = argparse.ArgumentParser(description="Evaluate trained racing policies.")
parser.add_argument("-p", "--run-paths", nargs="+", required=True,
                    help="One or more run folders. Each must contain run_config.pkl and models/.")
parser.add_argument("--checkpoint", type=int, default=None,
                    help="Checkpoint index to load (default: latest). Applied to all policies.")
parser.add_argument("--num-tracks", type=int, default=64,
                    help="Number of tracks to evaluate on (= num_envs). Default 64.")
parser.add_argument("--oob-alpha", type=float, default=1.0,
                    help="Linear OOB penalty: adjusted = lap + alpha * oob_time. Default 1.0.")
parser.add_argument("--off-track-wheel-threshold", type=int, default=None,
                    help="Wheels-off threshold for OOB. Default: pull from racing config 'goals'.")
parser.add_argument("--seed", type=int, default=0,
                    help="Env seed used when --seeds is not given. Pinned across policies.")
parser.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="List of seeds for multi-seed averaging. Each seed is run on every "
                         "policy; the same seed gives every policy an identical track set, "
                         "spawn pose, and finish-line draw. Default: [--seed].")
parser.add_argument("--max-steps", type=int, default=None,
                    help="Hard cap on env steps. Default: 2 * episode_length_s / step_dt.")
parser.add_argument("--dnf-multiplier", type=float, default=2.0,
                    help="DNF penalty time = multiplier * episode_length_s. Default 2.0.")
parser.add_argument("--out", type=str, default="eval_results",
                    help="Output directory for CSVs and plots.")
# Wandb behavior is driven by the YAML's `logging:` section
# (no_wandb / wandb_project / wandb_group) — same pattern as train_rl.py.

simulation_app, args_cli = startup(parser=parser)

#######################
###### END SETUP ######
#######################

import os
import csv
import glob
import gymnasium as gym
import torch
from tqdm import tqdm
from rsl_rl.runners import OnPolicyRunner

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from isaaclab.utils.io import load_pickle
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from wheeledlab_rl.configs import RunConfig
from wheeledlab_rl.utils import ClipAction
from wheeledlab_tasks.racing.mdp.rewards import on_track_mask
from wheeledlab_tasks.racing.config import CONFIG as RACING_CONFIG


# ---------------------------------------------------------------------------
# Resolve task from the first run path. All policies must share the same task.
# ---------------------------------------------------------------------------
def _load_run_cfg(run_path: str) -> RunConfig:
    pkl = os.path.join(run_path, "run_config.pkl")
    if not os.path.isfile(pkl):
        raise FileNotFoundError(f"Missing run_config.pkl in {run_path}")
    return load_pickle(pkl)


def _resolve_checkpoint(run_path: str, checkpoint: int | None) -> str:
    import glob
    models_dir = os.path.join(os.path.abspath(run_path), "models")
    if checkpoint is not None:
        path = os.path.join(models_dir, f"model_{checkpoint}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint {checkpoint} not found in {models_dir}")
        return path
    files = glob.glob(os.path.join(models_dir, "model_*.pt"))
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {models_dir}")
    return max(files, key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split("_")[1]))


_first_run_cfg = _load_run_cfg(args_cli.run_paths[0])
TASK = _first_run_cfg.env_setup.task_name
print(f"[eval] task resolved from first run: {TASK}")

# Sanity: warn if other runs disagree on task
for rp in args_cli.run_paths[1:]:
    other = _load_run_cfg(rp).env_setup.task_name
    if other != TASK:
        raise ValueError(f"Task mismatch: {rp} uses {other}, expected {TASK}")


def _build_env(env_cfg: ManagerBasedRLEnvCfg, seed: int):
    """Build the racing env from scratch, fully seeded for fair comparison.

    Pins three RNG sources so every policy built with the same `seed` sees:
      - identical track geometry (env_cfg.seed → IsaacLab's terrain gen)
      - identical spawn poses (numpy global RNG → sample_poses_along_polylines)
      - identical loop goal-arc draws (torch global RNG → init_progress)
    """
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env_cfg.scene.num_envs = args_cli.num_tracks
    env_cfg.seed = seed
    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    env.action_space.low = -1.0
    env.action_space.high = 1.0
    env = ClipAction(env)
    env = RslRlVecEnvWrapper(env)
    return env


@hydra_task_config(TASK, None)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    # Resolve eval-loop constants once. These come from env_cfg so they don't
    # require a live env; per-policy env rebuilds use the same env_cfg.
    sim_dt = env_cfg.sim.dt
    decimation = env_cfg.decimation
    dt_step = sim_dt * decimation
    episode_length_s = env_cfg.episode_length_s
    N = args_cli.num_tracks
    max_steps = args_cli.max_steps or int(2 * episode_length_s / dt_step)
    dnf_time_s = args_cli.dnf_multiplier * episode_length_s

    off_track_thresh = args_cli.off_track_wheel_threshold
    if off_track_thresh is None:
        off_track_thresh = int(RACING_CONFIG.get("goals", {}).get("off_track_wheel_threshold", 3))

    print(f"[eval] N={N} tracks, max_steps={max_steps}, dt_step={dt_step:.4f}s, "
          f"episode_length_s={episode_length_s}, dnf_time_s={dnf_time_s}, "
          f"oob_alpha={args_cli.oob_alpha}, off_track_thresh={off_track_thresh}")

    # ---- Resolve seed list ----
    seeds = args_cli.seeds if args_cli.seeds is not None else [args_cli.seed]
    print(f"[eval] seeds: {seeds}  ({len(seeds)} env builds per policy)")

    # ---- Wandb (driven by YAML logging:, same pattern as train_rl.py) ----
    log_cfg = RACING_CONFIG.get("logging", {})
    wandb_run = None
    if not log_cfg.get("no_wandb", False):
        import wandb
        wandb_run = wandb.init(
            project=log_cfg.get("wandb_project", "WheeledLab"),
            group=log_cfg.get("wandb_group", "eval"),
            config={
                "num_tracks": N,
                "seeds": seeds,
                "oob_alpha": args_cli.oob_alpha,
                "off_track_thresh": off_track_thresh,
                "dnf_time_s": dnf_time_s,
                "task": TASK,
                "policies": args_cli.run_paths,
            },
        )

    os.makedirs(args_cli.out, exist_ok=True)

    # ---- Helper: run one (seed, policy) combo, append per-track rows ----
    def _eval_one(seed: int, run_path: str, per_track_rows: list):
        policy_name = os.path.basename(os.path.normpath(run_path))
        print(f"\n[eval] === seed={seed}  policy={policy_name} ===")

        run_cfg = _load_run_cfg(run_path)
        policy_agent_cfg = run_cfg.agent
        ckpt_path = _resolve_checkpoint(run_path, args_cli.checkpoint)
        print(f"[eval] checkpoint: {ckpt_path}")

        # Fresh env with this seed. _build_env pins numpy + torch + env_cfg.seed
        # so every policy in this iteration sees identical tracks/spawns/finishes.
        env = _build_env(env_cfg, seed)
        device = env.unwrapped.device

        runner = OnPolicyRunner(env, policy_agent_cfg.to_dict(), log_dir=None, device=device)
        runner.load(ckpt_path)
        policy = runner.get_inference_policy(device=device)

        step_count = torch.zeros(N, dtype=torch.long, device=device)
        oob_steps = torch.zeros(N, dtype=torch.long, device=device)
        done_first_time = torch.zeros(N, dtype=torch.bool, device=device)
        lap_steps = torch.zeros(N, dtype=torch.long, device=device)
        oob_at_done = torch.zeros(N, dtype=torch.long, device=device)
        outcomes = ["pending"] * N

        term_mgr = env.unwrapped.termination_manager
        try:
            term_names = list(term_mgr._term_dones.keys())
        except AttributeError:
            term_names = list(getattr(term_mgr, "active_terms", []))
        print(f"[eval] termination terms: {term_names}")

        obs, _ = env.get_observations()

        with torch.inference_mode():
            for _ in tqdm(range(max_steps), desc=f"seed={seed} {policy_name}"):
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)

                on_track = on_track_mask(env.unwrapped, off_track_wheel_threshold=off_track_thresh)
                oob_now = ~on_track
                active = ~done_first_time
                oob_steps += (oob_now & active).long()
                step_count += active.long()

                new_done = dones.bool() & active
                if new_done.any():
                    term_dones = term_mgr._term_dones
                    new_done_idx = new_done.nonzero(as_tuple=True)[0].tolist()
                    for env_id in new_done_idx:
                        lap_steps[env_id] = step_count[env_id]
                        oob_at_done[env_id] = oob_steps[env_id]
                        if term_dones.get("goal_reached", torch.zeros(N, dtype=torch.bool, device=device))[env_id]:
                            outcomes[env_id] = "finished"
                        elif term_dones.get("time_out", torch.zeros(N, dtype=torch.bool, device=device))[env_id]:
                            outcomes[env_id] = "timeout"
                        else:
                            outcomes[env_id] = "crashed"
                    done_first_time |= new_done

                if done_first_time.all():
                    break

        for env_id in (~done_first_time).nonzero(as_tuple=True)[0].tolist():
            outcomes[env_id] = "timeout"
            lap_steps[env_id] = step_count[env_id]
            oob_at_done[env_id] = oob_steps[env_id]

        lap_time_s = (lap_steps.float() * dt_step).cpu()
        oob_time_s = (oob_at_done.float() * dt_step).cpu()
        adjusted_time_s = lap_time_s + args_cli.oob_alpha * oob_time_s

        for i in range(N):
            per_track_rows.append({
                "policy": policy_name,
                "seed": seed,
                "track_id": i,
                "outcome": outcomes[i],
                "lap_time_s": float(lap_time_s[i]),
                "oob_time_s": float(oob_time_s[i]),
                "adjusted_time_s": float(adjusted_time_s[i]),
                "checkpoint": ckpt_path,
            })

        env.close()
        del env, runner, policy
        return policy_name

    # ---- Outer seed loop, inner policy loop ----
    per_track_rows = []  # one row per (policy, seed, track)
    policy_names_in_order = []
    for seed in seeds:
        for run_path in args_cli.run_paths:
            name = _eval_one(seed, run_path, per_track_rows)
            if name not in policy_names_in_order:
                policy_names_in_order.append(name)

    # ---- Per-policy aggregation across seeds × tracks ----
    summaries = []
    for name in policy_names_in_order:
        rows = [r for r in per_track_rows if r["policy"] == name]
        n_attempts = len(rows)
        finished = [r for r in rows if r["outcome"] == "finished"]
        n_finished = len(finished)
        finish_rate = n_finished / n_attempts if n_attempts else float("nan")

        if n_finished > 0:
            adj_fin = torch.tensor([r["adjusted_time_s"] for r in finished])
            lap_fin = torch.tensor([r["lap_time_s"] for r in finished])
            oob_fin = torch.tensor([r["oob_time_s"] for r in finished])
            mean_lap_finishers = lap_fin.mean().item()
            mean_oob_finishers = oob_fin.mean().item()
            mean_adj_finishers = adj_fin.mean().item()
            std_adj_finishers = adj_fin.std().item() if n_finished > 1 else 0.0
        else:
            mean_lap_finishers = mean_oob_finishers = mean_adj_finishers = float("nan")
            std_adj_finishers = float("nan")

        # Mean with DNF: assign dnf_time_s to non-finishers, mean over all attempts
        with_dnf = [r["adjusted_time_s"] if r["outcome"] == "finished" else dnf_time_s for r in rows]
        mean_with_dnf = float(sum(with_dnf) / len(with_dnf)) if with_dnf else float("nan")

        # Across-seed std of per-seed means (measures seed-to-seed reliability of headline)
        per_seed_means = []
        for s in seeds:
            sf = [r["adjusted_time_s"] for r in rows
                  if r["seed"] == s and r["outcome"] == "finished"]
            if sf:
                per_seed_means.append(sum(sf) / len(sf))
        if len(per_seed_means) > 1:
            std_across_seeds = torch.tensor(per_seed_means).std().item()
        else:
            std_across_seeds = 0.0 if per_seed_means else float("nan")

        summaries.append({
            "policy": name,
            "n_seeds": len(seeds),
            "n_tracks_per_seed": N,
            "n_attempts_total": n_attempts,
            "n_finished": n_finished,
            "finish_rate": finish_rate,
            "mean_lap_time_finishers_s": mean_lap_finishers,
            "mean_oob_time_finishers_s": mean_oob_finishers,
            "mean_adjusted_time_finishers_s": mean_adj_finishers,
            "std_adjusted_time_finishers_s": std_adj_finishers,
            "std_across_seeds_s": std_across_seeds,
            "mean_adjusted_with_dnf_s": mean_with_dnf,
        })
        print(f"[eval] {name}: finish={finish_rate:.2%}  "
              f"mean_adj_finishers={mean_adj_finishers:.2f}s "
              f"(±{std_across_seeds:.2f}s across seeds)  "
              f"mean_with_dnf={mean_with_dnf:.2f}s")

        if wandb_run is not None:
            wandb.log({
                f"{name}/finish_rate": finish_rate,
                f"{name}/mean_lap_time_finishers_s": mean_lap_finishers,
                f"{name}/mean_oob_time_finishers_s": mean_oob_finishers,
                f"{name}/mean_adjusted_time_finishers_s": mean_adj_finishers,
                f"{name}/std_across_seeds_s": std_across_seeds,
                f"{name}/mean_adjusted_with_dnf_s": mean_with_dnf,
            })

    # ---- Save CSVs ----
    summary_csv = os.path.join(args_cli.out, "summary.csv")
    track_csv = os.path.join(args_cli.out, "per_track.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    with open(track_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_track_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_track_rows)
    print(f"\n[eval] wrote {summary_csv}")
    print(f"[eval] wrote {track_csv}")

    # ---- Plots ----
    names = [s["policy"] for s in summaries]
    finish_rates = [s["finish_rate"] for s in summaries]
    mean_adj_fin = [s["mean_adjusted_time_finishers_s"] for s in summaries]
    # Error bars use across-seed std (seed-to-seed reliability of the headline);
    # falls back to 0 when only one seed was run.
    std_adj_fin = [s["std_across_seeds_s"] for s in summaries]
    mean_with_dnf = [s["mean_adjusted_with_dnf_s"] for s in summaries]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    err_label = "across-seed std" if len(seeds) > 1 else "single seed"
    axes[0].bar(names, mean_adj_fin, yerr=std_adj_fin, capsize=4, color="steelblue")
    axes[0].set_ylabel("seconds")
    axes[0].set_title(f"Mean adjusted lap time (finishers)\nadj = lap + {args_cli.oob_alpha} * oob_time  ({err_label})")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(names, finish_rates, color="seagreen")
    axes[1].set_ylabel("fraction")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Finish rate")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(names, mean_with_dnf, color="indianred")
    axes[2].set_ylabel("seconds")
    axes[2].set_title(f"Mean adjusted time w/ DNF penalty\n(DNF = {dnf_time_s:.0f}s)")
    axes[2].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    plot_path = os.path.join(args_cli.out, "summary.png")
    fig.savefig(plot_path, dpi=120)
    print(f"[eval] wrote {plot_path}")

    # Per-policy distribution (strip plot of adjusted times of finishers)
    fig2, ax = plt.subplots(figsize=(max(6, 1.2 * len(names) + 2), 4))
    for i, name in enumerate(names):
        rows = [r for r in per_track_rows if r["policy"] == name and r["outcome"] == "finished"]
        if not rows:
            continue
        ys = [r["adjusted_time_s"] for r in rows]
        xs = [i + 0.05 * (j % 5 - 2) for j in range(len(ys))]
        ax.scatter(xs, ys, alpha=0.6, s=20)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30)
    ax.set_ylabel("adjusted lap time (s)")
    ax.set_title("Per-track adjusted lap times (finishers only)")
    fig2.tight_layout()
    dist_path = os.path.join(args_cli.out, "distribution.png")
    fig2.savefig(dist_path, dpi=120)
    print(f"[eval] wrote {dist_path}")

    if wandb_run is not None:
        wandb.log({
            "eval/summary_plot": wandb.Image(plot_path),
            "eval/distribution_plot": wandb.Image(dist_path),
        })
        # wandb.Table for per-track data (now includes seed column)
        table_cols = ["policy", "seed", "track_id", "outcome",
                      "lap_time_s", "oob_time_s", "adjusted_time_s"]
        wandb_table = wandb.Table(columns=table_cols)
        for r in per_track_rows:
            wandb_table.add_data(*[r[c] for c in table_cols])
        wandb.log({"eval/per_track": wandb_table})
        # Native wandb bar charts (interactive in UI)
        summary_table = wandb.Table(
            columns=["policy", "finish_rate", "mean_adj_finishers_s", "mean_with_dnf_s"],
            data=[[s["policy"], s["finish_rate"],
                   s["mean_adjusted_time_finishers_s"], s["mean_adjusted_with_dnf_s"]]
                  for s in summaries],
        )
        wandb.log({
            "eval/finish_rate_chart": wandb.plot.bar(summary_table, "policy", "finish_rate",
                                                    title="Finish rate by policy"),
            "eval/mean_adj_chart": wandb.plot.bar(summary_table, "policy", "mean_adj_finishers_s",
                                                  title="Mean adjusted lap time (finishers)"),
            "eval/mean_with_dnf_chart": wandb.plot.bar(summary_table, "policy", "mean_with_dnf_s",
                                                       title="Mean adjusted time w/ DNF penalty"),
        })
        wandb_run.finish()

    print("[eval] done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
