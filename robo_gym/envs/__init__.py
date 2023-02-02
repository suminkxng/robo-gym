# Example
from robo_gym.envs.example.example_env import ExampleEnvSim, ExampleEnvRob

# MiR100
from robo_gym.envs.mir100.mir100 import NoObstacleNavigationMir100Sim, NoObstacleNavigationMir100Rob
from robo_gym.envs.mir100.mir100 import ObstacleAvoidanceMir100Sim, ObstacleAvoidanceMir100Rob

# UR
from robo_gym.envs.ur.ur_base_env import EmptyEnvironmentURSim, EmptyEnvironmentURRob
from robo_gym.envs.ur.ur_ee_positioning import EndEffectorPositioningURSim, EndEffectorPositioningURRob
from robo_gym.envs.ur.ur_avoidance_basic import BasicAvoidanceURSim, BasicAvoidanceURRob
from robo_gym.envs.ur.ur_avoidance_iros import AvoidanceIros2021URSim, AvoidanceIros2021URRob
from robo_gym.envs.ur.ur_avoidance_iros import AvoidanceIros2021TestURSim, AvoidanceIros2021TestURRob

# Husky_ur3
from robo_gym.envs.husky_ur3.husky_ur3 import NoObstacleNavigationHusky_ur3_Sim, NoObstacleNavigationHusky_ur3_Rob
from robo_gym.envs.husky_ur3.husky_ur3 import ObstacleAvoidanceHusky_ur3_Sim, ObstacleAvoidanceHusky_ur3_Rob

# Pedsim + Husky_UR3
from robo_gym.envs.husky_ur3.pedsim_husky_ur3 import PedsimWithHusky_ur3_Sim, PedsimWithHusky_ur3_Rob