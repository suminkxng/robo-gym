import gym
import robo_gym
from robo_gym.wrappers.exception_handling import ExceptionHandling
from stable_baselines3 import PPO
import os

# specify the ip of the machine running the robot-server
target_machine_ip = '127.0.0.1' # or other xxx.xxx.xxx.xxx

models_dir = "models/husky_obs_PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# initialize environment
env = gym.make('NoObstacleNavigationHusky_ur3_Rob-v0', ip=target_machine_ip)
#env NoObstacleNavigationMir100Sim / ObstacleAvoidanceMir100Sim

env.reset()
# add wrapper for automatic exception handlingz
env = ExceptionHandling(env)

# load learned model
#models_dir = "models/husky_nav3_PPO"
#model_path = f"{models_dir}/60000.zip"
#model = PPO.load(model_path, env=env)

# choose and run appropriate algorithm provided by stable-baselines
model = PPO("MlpPolicy", env, verbose=1,tensorboard_log="./logs")

TIMESTEPS = 1000

for i in range(1,250):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="husky_obs_ppo")
    model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()