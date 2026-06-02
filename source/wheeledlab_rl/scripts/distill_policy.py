"""Phase A of asymmetric-critic + distillation: DAgger-distill a privileged
teacher into the camera student.

This is the teacher->student step of the sequential recipe (see DEVELOPMENT.md
"Asymmetric critic + distillation"). It loads a trained PRIVILEGED policy
(RSS_RACING_PRIVILEGED_CONFIG: MLP over exact cone state) as the teacher, rolls
out the CAMERA student (CNN+GRU) in the asymmetric racing env, and regresses the
student's actions onto the teacher's via online DAgger.

The asym env (Isaac-MushrRacingAsymRL-v0) exposes BOTH groups every step:
  * `policy` group  -> camera + proprio   (what the student sees / acts on)
  * `critic` group  -> cones in car frame + proprio   (what the teacher sees)
Because the `critic` group layout is identical to the privileged policy's
training obs, it feeds the teacher directly — no second env needed.

The student is built/saved through the SAME runner used for RL training, so the
saved checkpoint is byte-compatible with `train_rl.py train.load_run=...`
(phase B: RL fine-tune with the privileged critic).

Usage:

  WHEELEDLAB_RACING_CONFIG=<abs path to racing_asym.yaml> \
    python source/wheeledlab_rl/scripts/distill_policy.py \
      --teacher-run logs/<privileged run folder> \
      --run-name racing_distill_v0

Recurrence note: the GRU student is trained with truncated BPTT of length 1 —
the hidden state is carried across env steps (so the policy is genuinely
recurrent at rollout/inference) but the gradient is taken per-step with the
hidden detached between steps. This is stable and matches how the rollout uses
the net; for full BPTT over rollout windows, accumulate the per-step losses
before calling backward (see the loop below) — left as a TODO.
"""

###################################
###### BEGIN ISAACLAB SPINUP ######
###################################

from wheeledlab_rl.startup import startup
import argparse

parser = argparse.ArgumentParser(description="DAgger-distill a privileged teacher into the camera student.")
# Teacher (privileged) source — a finished RSS_RACING_PRIVILEGED_CONFIG run folder.
parser.add_argument("--teacher-run", type=str, required=True,
                    help="Path to the privileged teacher's run folder (contains run_config.pkl + models/).")
parser.add_argument("--teacher-checkpoint", type=int, default=None,
                    help="Teacher checkpoint index to load (default: latest).")
# Student env / task.
parser.add_argument("--task", type=str, default="Isaac-MushrRacingAsymRL-v0",
                    help="Asymmetric racing task (camera policy group + privileged critic group).")
parser.add_argument("--num-envs", type=int, default=256, help="Parallel envs for DAgger rollouts.")
# DAgger schedule.
parser.add_argument("--dagger-iters", type=int, default=200, help="Number of DAgger iterations.")
parser.add_argument("--steps-per-iter", type=int, default=256, help="Env steps collected/trained per iteration.")
parser.add_argument("--lr", type=float, default=5.0e-4, help="Student BC learning rate.")
parser.add_argument("--beta-start", type=float, default=1.0, help="Initial prob. of executing the TEACHER action.")
parser.add_argument("--beta-end", type=float, default=0.0, help="Final prob. of executing the teacher action.")
parser.add_argument("--explore-std", type=float, default=0.0,
                    help="Std of Gaussian noise added to the EXECUTED action for state coverage (0 = deterministic).")
parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Grad-norm clip for the student update.")
# Logging / saving.
parser.add_argument("--run-name", type=str, default=None, help="Run folder name under the logs dir.")
parser.add_argument("--logs-dir", type=str, default=None,
                    help="Base logs directory the run folder is created under "
                         "(default: WHEELEDLAB_RL_LOGS_DIR). Set to match train_rl's "
                         "train.log.logs_dir so phase-B load_run can find the checkpoint.")
parser.add_argument("--save-interval", type=int, default=10, help="Save a student checkpoint every n DAgger iters.")
parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging of the BC loss / beta curves.")
parser.add_argument("--wandb-project", type=str, default=None,
                    help="wandb project (default: LogConfig.wandb_project).")
parser.add_argument("--device", type=str, default="cuda:0", help="Torch device.")

simulation_app, args_cli = startup(parser=parser)

#######################
###### END SETUP ######
#######################

import os
import torch
import gymnasium as gym
from tqdm import tqdm

from isaaclab.utils.io import load_pickle
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from rsl_rl.modules import ActorCritic
try:
    from rsl_rl.modules import EmpiricalNormalization
except Exception:  # pragma: no cover - version dependent
    EmpiricalNormalization = None

from wheeledlab_rl import WHEELEDLAB_RL_LOGS_DIR
from wheeledlab_rl.configs import LogConfig
from wheeledlab_rl.utils import OnPolicyRunner as ModifiedRslRunner, ClipAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_student_policy(runner):
    """rsl_rl renamed alg.actor_critic -> alg.policy across versions."""
    alg = runner.alg
    return getattr(alg, "actor_critic", None) or alg.policy


def _detach_hidden(net):
    """Cut the BPTT graph between env steps while keeping the rolled-forward
    hidden state. Handles GRU (tensor) and LSTM (tuple) memory."""
    mem = getattr(net, "memory_a", None)
    if mem is None:
        return
    h = mem.hidden_states
    if h is None:
        return
    mem.hidden_states = tuple(x.detach() for x in h) if isinstance(h, tuple) else h.detach()


def _load_teacher(teacher_run, checkpoint_idx, device):
    """Rebuild the privileged MLP teacher (ActorCritic) + its obs normalizer
    from a finished privileged run folder."""
    run_cfg = load_pickle(os.path.join(teacher_run, "run_config.pkl"))
    agent = run_cfg.agent.to_dict()
    pol = agent["policy"]
    emp_norm = bool(agent.get("empirical_normalization", False))

    chkpt = "model_"
    if checkpoint_idx is not None and checkpoint_idx > 0:
        chkpt = f"{chkpt}{checkpoint_idx}"
    fp = os.path.abspath(teacher_run)
    ckpt_path = get_checkpoint_path(
        log_path=os.path.dirname(fp), run_dir=os.path.basename(fp),
        other_dirs=["models"], checkpoint=f"{chkpt}.*",
    )
    print(f"[distill] Loading teacher checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt, pol, emp_norm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):

    device = args_cli.device

    # Size the scene to the requested env count (cfg was built with its default).
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.num_envs = args_cli.num_envs

    # Build env. The asym env yields camera `policy` obs (primary) and privileged
    # `critic` obs (in extras["observations"]["critic"]).
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env.action_space.low = -1.0
    env.action_space.high = 1.0
    env = ClipAction(env)
    env = RslRlVecEnvWrapper(env)

    #################
    #### STUDENT ####
    #################
    # Build the student through the training runner so its checkpoint format is
    # identical to an RL run (phase B can resume from it directly).
    run_name = args_cli.run_name or f"racing_distill-{agent_cfg.seed}"
    log_kwargs = dict(run_name=run_name, no_wandb=args_cli.no_wandb, no_log=False, video=False)
    if args_cli.logs_dir is not None:
        log_kwargs["logs_dir"] = args_cli.logs_dir
    log_cfg = LogConfig(**log_kwargs)
    os.makedirs(log_cfg.model_save_path, exist_ok=True)
    print(f"[distill] writing student checkpoints under: {log_cfg.run_log_dir}")

    wandb_run = None
    if not log_cfg.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args_cli.wandb_project or log_cfg.wandb_project,
            name=run_name,
            config={
                "phase": "distill",
                "task": args_cli.task,
                "teacher_run": args_cli.teacher_run,
                "num_envs": args_cli.num_envs,
                "dagger_iters": args_cli.dagger_iters,
                "steps_per_iter": args_cli.steps_per_iter,
                "lr": args_cli.lr,
                "beta_start": args_cli.beta_start,
                "beta_end": args_cli.beta_end,
            },
        )

    runner = ModifiedRslRunner(env, agent_cfg.to_dict(), log_cfg, device=device)
    student = _get_student_policy(runner)
    student.train()
    if not getattr(student, "privileged_critic", False):
        print("[distill][WARN] student policy was built with privileged_critic=False. "
              "Distillation still works, but phase-B asymmetric critic needs it True "
              "(set policy.privileged_critic: true in racing_asym.yaml).")
    optimizer = torch.optim.Adam(student.parameters(), lr=args_cli.lr)

    #################
    #### TEACHER ####
    #################
    obs, extras = env.get_observations()
    priv = extras["observations"]["critic"].to(device)
    priv_dim = priv.shape[-1]
    act_dim = env.num_actions

    ckpt, t_pol, t_emp_norm = _load_teacher(args_cli.teacher_run, args_cli.teacher_checkpoint, device)
    teacher = ActorCritic(
        num_actor_obs=priv_dim,
        num_critic_obs=priv_dim,
        num_actions=act_dim,
        actor_hidden_dims=list(t_pol["actor_hidden_dims"]),
        critic_hidden_dims=list(t_pol["critic_hidden_dims"]),
        activation=t_pol["activation"],
        init_noise_std=float(t_pol.get("init_noise_std", 1.0)),
    ).to(device)
    teacher.load_state_dict(ckpt["model_state_dict"])
    teacher.eval()

    if t_emp_norm and EmpiricalNormalization is not None and "obs_norm_state_dict" in ckpt:
        teacher_norm = EmpiricalNormalization(shape=[priv_dim]).to(device)
        teacher_norm.load_state_dict(ckpt["obs_norm_state_dict"])
        teacher_norm.eval()
        print(f"[distill] Teacher uses empirical obs normalization (dim {priv_dim}).")
    else:
        teacher_norm = torch.nn.Identity()
        if t_emp_norm:
            print("[distill][WARN] teacher trained with empirical_normalization but no "
                  "obs_norm_state_dict / EmpiricalNormalization available — using identity.")

    print(f"[distill] priv_dim={priv_dim}, act_dim={act_dim}, num_envs={args_cli.num_envs}")

    #################
    ##### TRAIN #####
    #################
    student.reset()
    mse = torch.nn.MSELoss()
    n_iters = args_cli.dagger_iters

    for it in tqdm(range(n_iters), desc="DAgger"):
        # Linear teacher-execution schedule beta_start -> beta_end.
        frac = it / max(1, n_iters - 1)
        beta = args_cli.beta_start + frac * (args_cli.beta_end - args_cli.beta_start)
        running_loss = 0.0

        for _ in range(args_cli.steps_per_iter):
            # Teacher action (deterministic, no grad) from privileged obs.
            with torch.no_grad():
                teacher_act = teacher.act_inference(teacher_norm(priv))

            # Student action (deterministic mean, grad on). Advances memory_a once.
            student_obs = runner.obs_normalizer(obs).to(device)
            student_mean = student.act_inference(student_obs)

            # BC: regress student mean onto teacher action.
            loss = mse(student_mean, teacher_act.detach())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), args_cli.max_grad_norm)
            optimizer.step()
            _detach_hidden(student)  # TBPTT length 1
            running_loss += loss.item()

            # Execute a DAgger mixture: teacher w.p. beta, else student.
            with torch.no_grad():
                use_teacher = (torch.rand(student_mean.shape[0], 1, device=device) < beta)
                executed = torch.where(use_teacher, teacher_act, student_mean.detach())
                if args_cli.explore_std > 0.0:
                    executed = executed + args_cli.explore_std * torch.randn_like(executed)

            obs, _, dones, infos = env.step(executed)
            obs = obs.to(device)
            priv = infos["observations"]["critic"].to(device)
            # Zero the GRU hidden for envs whose episode just ended.
            student.reset(dones.to(device))

        avg_loss = running_loss / args_cli.steps_per_iter
        print(f"[distill] iter {it:4d} | beta {beta:.3f} | bc_mse {avg_loss:.5f}")
        if wandb_run is not None:
            wandb_run.log({"distill/bc_mse": avg_loss, "distill/beta": beta}, step=it)

        if (it % args_cli.save_interval == 0) or (it == n_iters - 1):
            save_path = os.path.join(log_cfg.model_save_path, f"model_{it}.pt")
            runner.save(save_path)
            print(f"[distill] saved student checkpoint: {save_path}")

    print(f"[distill] Done. Resume phase-B RL fine-tune with:\n"
          f"  train_rl.py -r RSS_RACING_ASYM_CONFIG train.load_run={run_name}")
    if wandb_run is not None:
        wandb_run.finish()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
