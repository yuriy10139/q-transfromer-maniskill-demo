## Q-transformer

Minimal demo of usage of Q-transformer <a href="https://github.com/lucidrains/q-transformer">Q-Transformer</a> on Maniskill 3 PushCube-v1 task.

## Install

```bash
$ git clone https://github.com/yuriy10139/q-transfromer-maniskill-demo
$ cd q-transfromer-maniskill-demo
$ python3 -m venv venv
$ source ./venv/bin/activate
$ python3 -m pip install -e .
$ python3 -m pip install --upgrade mani_skill
```

## Usage

```bash
# download and convert PushCube-v1 trajectories to have RGB observations
$ cd q_transformer
$ mkdir ./demos
$ python3 -m mani_skill.utils.download_demo 'PushCube-v1' --output_dir ./demos
$ python3 -m mani_skill.trajectory.replay_trajectory  --traj-path ./demos/PushCube-v1/motionplanning/trajectory.h5 -c pd_ee_delta_pose -o rgbd --save-traj --max-retry 10

# convert trajectories into Q-transformer format
$ python3 convert_maniskill_traj.py

# learning 
$ python3 learn_no_hist.py

# eval
$ python3 evaluate.py
```