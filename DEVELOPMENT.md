# Development Guide

See [README](README.md) for installation and quickstart. This doc walks through the codebase in the order you'd actually touch it when developing a new task or modifying an existing one.

We start from a concrete task's env config (racing), explain each piece, then widen out to the agent config, shared modules, RL training plumbing, and finally a conceptual overview of the runtime flow.

For per-module specifics, see each subpackage's `docs/README.md`.

## 1. Starting point: a task's env cfg

The natural entry point for development is the env cfg file for whichever task you're working on — e.g. [mushr_racing_env_cfg.py](source/wheeledlab_tasks/wheeledlab_tasks/racing/mushr_racing_env_cfg.py) for racing. This file declares the whole RL environment as a tree of `@configclass`-decorated dataclasses.

The top of the tree is a subclass of Isaac Lab's `ManagerBasedRLEnvCfg`:

```python
@configclass
class MushrRacingRLEnvCfg(ManagerBasedRLEnvCfg):
    seed: int = _ENV["seed"]
    num_envs: int = 512
    env_spacing: float = 0.

    events:       RacingEventsCfg       = RacingEventsRandomCfg()
    actions:      Mushr4WDActionCfg     = Mushr4WDActionCfg()
    observations: RacingObsCfg          = RacingObsCfg()
    rewards:      RacingRewardsCfg      = RacingRewardsCfg()
    terminations: RacingTerminationsCfg = RacingTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # sim-level + viewer settings
        self.sim.dt           = float(_ENV["sim_dt"])
        self.decimation       = int(_ENV["decimation"])
        self.episode_length_s = float(_ENV["episode_length_s"])
        # build the scene tree
        self.scene = MushrRacingSceneCfg(
            num_envs=self.num_envs, env_spacing=self.env_spacing,
        )
```

That's the whole top-level surface. The five manager cfgs (`events`, `actions`, `observations`, `rewards`, `terminations`) define the MDP, and `__post_init__` does cheap setup + wires in the scene cfg.

Each manager cfg lives next to the others under `<task>/mdp/`:

```
racing/
├── mushr_racing_env_cfg.py   # the file above
├── mdp/
│   ├── events.py
│   ├── observations.py
│   ├── rewards.py
│   └── terminations.py
├── config/                   # YAML + agent cfgs (see §2)
└── track/                    # racing-specific procedural terrain
```

(Racing also has `track/` with a `TrackCache` system that drives reward projection. Other tasks won't have this — it's task-specific. Treat it as an example of where task-specific runtime state lives, not as a general pattern.)

### Events

[mdp/events.py](source/wheeledlab_tasks/wheeledlab_tasks/racing/mdp/events.py). Events fire at discrete moments — episode reset, periodic intervals, or specific triggers — and mutate the scene. Typical contents:

- **Reset events**: spawn pose randomization, RNG seeding per env, per-episode curriculum updates.
- **Periodic events**: physics randomization (friction, mass), texture randomization, lighting randomization.

Each event is an `EventTerm(func=..., mode=..., params=...)` bound to a free function in the same file. Example:

```python
randomize_ground_texture = EventTerm(
    func=randomize_ground_texture,
    mode="interval",
    interval_range_s=(...),
    params={...},
)
```

The cfg dataclass (`RacingEventsCfg`) just lists the `EventTerm` members; subclasses like `RacingEventsRandomCfg` add domain-randomization terms on top.

### Actions

[mushr_racing_env_cfg.py](source/wheeledlab_tasks/wheeledlab_tasks/racing/mushr_racing_env_cfg.py) imports `Mushr4WDActionCfg` from [wheeledlab_tasks/common](source/wheeledlab_tasks/wheeledlab_tasks/common/) — see §3. The action cfg names the joints, action scaling, and the policy→joint mapping for a specific robot. Most tasks reuse the existing per-robot cfg rather than defining their own.

### Observations

[mdp/observations.py](source/wheeledlab_tasks/wheeledlab_tasks/racing/mdp/observations.py). Observations are organized into groups (e.g. `policy`, `critic`) and each group lists `ObsTerm` members. Each `ObsTerm` binds an observation function (returning a tensor of shape `(num_envs, dim)`) to optional noise / clipping / scaling configuration.

```python
@configclass
class PolicyCfg(ObsGroup):
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(...))
    last_action  = ObsTerm(func=mdp.last_action, clip=(-1., 1.))
    camera_rgb   = ObsTerm(func=mdp.camera_rgb)
```

At runtime, Isaac Lab concatenates terms within a group and exposes the group as the policy's observation.

### Rewards

[mdp/rewards.py](source/wheeledlab_tasks/wheeledlab_tasks/racing/mdp/rewards.py). The reward cfg lists `RewTerm` members, each binding a reward function to a weight. The total reward is the weighted sum across terms.

```python
@configclass
class RacingRewardsCfg:
    progress       = RewTerm(func=progress_reward,       weight=1.0)
    off_track      = RewTerm(func=outside_cones_penalty, weight=-0.5)
    goal_reached   = RewTerm(func=goal_reached,          weight=15.0)
```

Reward funcs receive `env` and return a `(num_envs,)` tensor. They typically read state via `env.scene[...]` or task-specific cached structures (in racing, the `TrackCache` carried on the terrain importer).

### Terminations

[mdp/terminations.py](source/wheeledlab_tasks/wheeledlab_tasks/racing/mdp/terminations.py). Same shape as rewards but each `DoneTerm` returns a `(num_envs,)` bool indicating whether that env's episode should end. Multiple terms are OR'd. Common terms: time-out, out-of-bounds, goal reached.

### `__post_init__` and `scene`

`__post_init__` does anything that can't be expressed as a class-attribute default — typically reading `sim.dt`, `decimation`, `episode_length_s` from YAML, then instantiating the scene cfg.

The scene cfg (`MushrRacingSceneCfg`, a subclass of `InteractiveSceneCfg`) declares the **entities** that live in the simulator:

- **Terrain** (`TerrainImporterCfg` subclass) — ground/obstacles, possibly procedurally generated.
- **Robot** (`ArticulationCfg`) — the agent. Reused from [wheeledlab_assets/mushr](source/wheeledlab_assets/wheeledlab_assets/mushr.py).
- **Sensors** (`TiledCameraCfg`, IMU cfg, etc.) — anything the robot perceives.
- **Lights** — fixed or randomized over time via events.
- **Ground plane** — flat collision plane underneath.

Each entity cfg has a `class_type` pointing to its runtime class. When `InteractiveScene` builds the scene, it instantiates each entity by calling `cfg.class_type(cfg)`. Heavy work (USD authoring, asset materialization) belongs in that runtime class init — not in the cfg's `__post_init__`.

## 2. Per-task agent cfgs: `config/agents/`

[racing/config/agents/mushr/rsl_rl_ppo_cfg.py](source/wheeledlab_tasks/wheeledlab_tasks/racing/config/agents/mushr/rsl_rl_ppo_cfg.py) defines the RL agent for this task + robot pairing. Two pieces:

1. **Policy cfgs** — one `@configclass` per network architecture (`MushrMLPPolicyCfg`, `MushrCNNPolicyCfg`, `MushrRNNPolicyCfg`, `MushrCNNGRUPolicyCfg`). Each sets `class_name` (the module class that `rsl_rl` will instantiate) plus hyperparameters (hidden dims, CNN channels, RNN type, etc.).
2. **Runner cfg** (`MushrPPORunnerCfg`) — a subclass of `RslRlOnPolicyRunnerCfg` with PPO hyperparameters (`num_steps_per_env`, `learning_rate`, `clip_param`, `entropy_coef`, …) read from YAML.

The runner cfg dispatches to the right policy cfg based on `policy.class_name` in YAML — so swapping architectures across runs is a YAML-only change.

YAML for the task lives next to this in `config/{train_configs,eval_configs}/*.yaml`. `config/__init__.py` exposes the loaded YAML as a single `CONFIG` dict that both the env cfg and the agent cfg read from (`_PPO = CONFIG["ppo"]`, `_TER = CONFIG["terrain"]`, etc.). **YAML is the single source of truth for tunable values.**

## 3. Shared MDP pieces: `wheeledlab_tasks/common/`

Code that's robot-specific but task-agnostic. Currently:

- [common/actions.py](source/wheeledlab_tasks/wheeledlab_tasks/common/actions.py) — per-robot action cfgs (`Mushr4WDActionCfg`, `MushrRWDActionCfg`, `F1Tenth4WDActionCfg`, etc.). Each declares the wheel/steering joint names, base geometry, scale factors, and binds to an action class in `wheeledlab.envs.mdp` (`RCCar4WDActionCfg`, `RCCarRWDActionCfg`). New robots add their cfg here; tasks just import.
- [common/observations.py](source/wheeledlab_tasks/wheeledlab_tasks/common/observations.py) — observation functions reused across tasks (e.g. base velocities, joint readings).

## 4. RL run configs: `wheeledlab_rl/configs/`

This is where a "run" gets defined as a top-level dataclass that bundles **everything** the trainer needs.

[common_cfg.py](source/wheeledlab_rl/wheeledlab_rl/configs/common_cfg.py) defines the building blocks:

```python
@configclass
class LogConfig:    ...   # wandb, video, checkpointing
@configclass
class TrainConfig:  ...   # seed, device, load_run, log
@configclass
class EnvSetup:     ...   # num_envs, task_name (gym registry id)
@configclass
class AgentSetup:   ...   # entry_point for agent cfg lookup

@configclass
class RunConfig:
    train:       TrainConfig
    env_setup:   EnvSetup
    agent_setup: AgentSetup
    env:   Any = MISSING   # filled at Hydra registration from task_name
    agent: Any = MISSING
```

[rl_cfg.py](source/wheeledlab_rl/wheeledlab_rl/configs/rl_cfg.py) specializes `RunConfig` for actual RL libraries:

- `RLTrainConfig` adds RL-specific train fields (`num_iterations`, `rl_algo_lib`, `rl_algo_class`).
- `RslRlRunConfig` — `RunConfig` preset for `rsl_rl` PPO. Default for most runs.
- `SB3RLRunConfig` — stub for Stable-Baselines3 PPO.

[runs/](source/wheeledlab_rl/wheeledlab_rl/configs/runs/) hosts named runs. `rss_cfgs.py` and `f1tenth_cfgs.py` declare concrete `@configclass` run configs (`RSS_DRIFT_CONFIG`, `RSS_RACING_CONFIG`, …) that fix `task_name`, `num_iterations`, etc. [runs/__init__.py](source/wheeledlab_rl/wheeledlab_rl/configs/runs/__init__.py) calls `register_run_to_hydra(name, node)` for each at import time, which is how `-r <NAME>` on the CLI finds them.

## 5. RL plumbing: `wheeledlab_rl/utils/` and `wheeledlab_rl/scripts/`

[utils/](source/wheeledlab_rl/wheeledlab_rl/utils/) holds the glue between Isaac Lab / rsl_rl and our trainer:

- **`hydra.py`** — `register_run_to_hydra`, `hydra_run_config` decorator, the OmegaConf↔dataclass plumbing. Hydra is the CLI-driven config composer (`-r NAME key.subkey=value …`).
- **`modified_rsl_rl_runner.py`** — subclass of rsl_rl's `OnPolicyRunner` with our logging/checkpoint hooks.
- **`actor_critic_cnn_gru.py`** — extra policy network classes (CNN+GRU) registered into rsl_rl's network factory.
- **`custom_video_recorder.py`** — viewer-camera mp4 recording with W&B integration.
- **`policy_camera_recorder.py`** — per-env tiled-camera recording (what the policy sees, useful for debugging visual policies).
- **`clip_action.py`** — Gym wrapper that clips actions to `[-1, 1]`.

[scripts/](source/wheeledlab_rl/scripts/):

- **`train_rl.py`** — main training entry point. Calls `startup()` (boots Isaac Sim), then a `@hydra_run_config`-decorated `main(run_cfg)` that creates the env via `gym.make(task_name, cfg=env_cfg)`, wraps it (ClipAction → video recorders → `RslRlVecEnvWrapper`), instantiates the runner, and calls `runner.learn(...)`.
- **`play_policy.py`** — load a checkpoint and run inference.
- **`eval_racing.py`** — racing-specific eval harness.

## 6. Core shared code: `wheeledlab/`

Briefly: shared utilities and base MDP implementations.

[wheeledlab/envs/mdp/actions/](source/wheeledlab/wheeledlab/envs/mdp/actions/) implements the base `RCCar4WDActionCfg`, `RCCarRWDActionCfg`, `AckermannActionCfg` — these are what the per-robot cfgs in `wheeledlab_tasks/common/actions.py` compose with their joint names. `wheeledlab/envs/mdp/curriculums.py` and `observations.py` host generic terms reused across tasks.

If you're writing a new robot, you probably touch `wheeledlab_assets/` (asset cfg) + `wheeledlab_tasks/common/actions.py` (joint binding). The base action implementations under `wheeledlab/envs/mdp/actions/` rarely need changes.

## 7. Robot assets: `wheeledlab_assets/`

Briefly: one Python file per robot ([mushr.py](source/wheeledlab_assets/wheeledlab_assets/mushr.py), [hound.py](source/wheeledlab_assets/wheeledlab_assets/hound.py), [f1tenth.py](source/wheeledlab_assets/wheeledlab_assets/f1tenth.py)) exposing an `ArticulationCfg` (e.g. `MUSHR_SUS_CFG`) that the scene cfg imports. USD/USDA files for terrain, cones, etc. live under `data/`.

## 8. Conceptual control flow at training time

A training run goes through three distinct phases:

```
python source/wheeledlab_rl/scripts/train_rl.py -r <RUN_CFG_NAME>
        │
        ├─ startup() → Isaac Sim app boot
        │
        ├─ import wheeledlab_rl.configs.runs    # ── config-parse phase
        │     └─ register_run_to_hydra(...)
        │         └─ <Task>EnvCfg().__post_init__()
        │             └─ <Task>SceneCfg(...).__post_init__()
        │                 └─ cheap sizing / path resolution
        │
        ├─ gym.make(task_name, cfg=env_cfg)     # ── scene-build phase
        │     └─ ManagerBasedRLEnv(cfg)
        │         └─ InteractiveScene
        │             └─ cfg.class_type(cfg) per entity
        │                 └─ heavy USD authoring, asset materialization
        │
        └─ Runner.learn(...)                    # ── training phase
```

**Config-parse phase.** Importing `wheeledlab_rl.configs.runs` registers every run config with Hydra. Registration instantiates each cfg dataclass to validate its schema, which triggers every `__post_init__` in the tree. *Every task pays this cost on every training run, even if you only asked for one task.* Keep `__post_init__` cheap — no I/O, no GPU calls, no USD authoring. Just math and field assignment.

**Scene-build phase.** `gym.make(...)` instantiates `ManagerBasedRLEnv`, which builds `InteractiveScene`, which walks each entity cfg and calls `cfg.class_type(cfg)`. This is where the actual simulator state gets populated: USD assets are loaded, prims are authored, physics is initialized, manager terms are bound to live tensors. Heavy work belongs here, in the runtime classes — not in the cfg `__post_init__`.

**Training phase.** `Runner.learn(...)` is rsl_rl's PPO loop: rollout → advantage → update → checkpoint. By this point all the cfg/scene/manager wiring is behind you; any issue from here on is about reward shaping, hyperparameters, or numerical stability — not setup.

The most common subtle bug in WheeledLab development is putting scene-build-phase work in config-parse-phase code, which fires for every task on every run. If `import wheeledlab_rl.configs.runs` starts taking tens of seconds, that's the smell.

## Asymmetric critic + distillation (closing the privileged→camera gap)

When a privileged policy (exact cone state, `Isaac-MushrRacingPrivilegedRL-v0`)
races well but the camera policy (`Isaac-MushrRacingRL-v0`) does not, the
bottleneck is perception, not control. Two complementary mechanisms close that
gap; they do **not** conflict and are applied **sequentially**.

Both share a third racing task, `Isaac-MushrRacingAsymRL-v0`, whose observation
([RacingAsymObsCfg](source/wheeledlab_tasks/wheeledlab_tasks/racing/mdp/observations.py))
exposes two groups every step:

- `policy` — camera + proprio (what the **actor** sees; identical to the normal
  camera task, so deployment is unchanged).
- `critic` — cones in car frame + proprio (the exact privileged state). Its
  layout is identical to the privileged policy's training obs, so it doubles as
  the **teacher's** input during distillation.

**Asymmetric critic.** With `policy.privileged_critic: true`, the CNN/CNNGRU
nets route the `critic` group straight into the critic's MLP/GRU, bypassing the
image encoder ([actor_critic_cnn_gru.py](source/wheeledlab_rl/wheeledlab_rl/utils/actor_critic_cnn_gru.py)).
The value function then sees flawless perception (low-variance value targets)
while the actor stays camera-only. `privileged_critic` defaults to `false`, so
every existing run is unchanged; the symmetric camera task still shares one CNN.
The modified runner already forwards the `critic` group
([modified_rsl_rl_runner.py](source/wheeledlab_rl/wheeledlab_rl/utils/modified_rsl_rl_runner.py)
reads `extras["observations"]["critic"]`).

**Distillation (DAgger).** [scripts/distill_policy.py](source/wheeledlab_rl/scripts/distill_policy.py)
loads the privileged MLP as a frozen teacher, rolls out the camera student in
the asym env, and regresses the student's actions onto the teacher's. The
student is built/saved through the normal training runner, so its checkpoint is
byte-compatible with `train_rl.py train.load_run=...`. (The GRU student uses
TBPTT-length-1: hidden state is carried across steps but the gradient is
per-step — see the script header for extending to full BPTT.)

The two phases (run after a privileged teacher already exists):

```bash
ASYM=source/wheeledlab_tasks/wheeledlab_tasks/racing/config/train_configs/racing_asym.yaml

# Phase A — distill teacher -> camera student
WHEELEDLAB_RACING_CONFIG=$ASYM python source/wheeledlab_rl/scripts/distill_policy.py \
  --teacher-run logs/<privileged run folder> --run-name racing_distill_v0

# Phase B — RL fine-tune the distilled student with the privileged critic
WHEELEDLAB_RACING_CONFIG=$ASYM python source/wheeledlab_rl/scripts/train_rl.py \
  -r RSS_RACING_ASYM_CONFIG train.load_run=racing_distill_v0
```

Distillation solves exploration/credit-assignment (why pixel-PPO crawls); the
asymmetric-critic RL fine-tune then closes the *imitation gap* — the teacher
acts on state the camera can't recover (occluded/behind cones), so the student
can't match it everywhere and RL lets it find the best camera-realizable policy.
`racing_asym.yaml` is shared by both phases so the student architecture is
byte-identical across the checkpoint hand-off. Note rsl_rl's single
`empirical_normalization` flag covers both actor and critic, so it stays `false`
here (the image actor obs should not be empirically normalized).

## Deployment

This repo is sim-only. Hardware drivers, real-robot integrations, and field-test code live in a separate repo: [RealLab](https://github.com/UWRobotLearning/RealLab). It currently has branches for HOUND, MuSHR, and (coming) F1Tenth. See its README for setup instructions.
