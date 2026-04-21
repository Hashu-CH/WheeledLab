This will be an evolving and poorly written guide/recap of development on the racing task -- so that others can navigate work clean and fast.

## racing/config: 

In this very aptly named file we specify training, optimization, and policy architecture parameters. Many of these pull from unmodified/modified versions of the rsl_rl implementations of many such policies/algorithms. 

## racing/mdp: 

These files define terms used in our config files. Called on event/reward/terminations all things defined by the ManagerRlEnv which we subclass. 

## racing/mdp_sensors

Same thing but for sensors. In this case, just specifies how we format camera inputs.

## racing/utils

All things to do with map generation, storing, and querying. The procedural track generation folder inside specifies a curriculum which serves as an access point for generation arbitrarily hard open chain curves or closed loop tracks.

## racing/mushr_racing_env_cfg

The file where we define all our configs that we actually run! This is where you will combine your MDP terms to create new configs that can be used in training/playing -- alongside a lot of hyperparameters.


## Changes from Visual Task

Track generation and querying is the skeleton of the racing task. Most of the changes are within the utils folder.

Related changes to rewards are made alongside some refactoring to make the mdp folder house all terms. Previously spread sporadically throughout the mdp folder and env config.

Changes to the policy architecture as well in rsl_rl config and the environment config.

## Forking implementation

If you want to create a new config (for some specialized version of the task), as a general reminder: 1) import it in the task_name's/init file. 2) Import and register under isaac gym inside tasks/init file. 3) Register it under wheeledlab_rl 2x/configs/runs/rss_cfgs. 4) And also within the runs init file so that it can be stored in Hydra's run config and called.