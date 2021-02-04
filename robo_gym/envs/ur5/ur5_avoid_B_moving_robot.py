#!/usr/bin/env python3
from copy import deepcopy
import math, copy
import numpy as np
from scipy.spatial.transform import Rotation as R
import gym
from gym import spaces
from gym.utils import seeding
from robo_gym.utils import utils, ur_utils
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError
import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym.envs.simulation_wrapper import Simulation
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2

from robo_gym.envs.ur5.ur5_avoidance import MovingBox3DSplineTargetUR5

# ? Variant B - Nice Environment with the position that the robot should keep that is changing over time. 

DEBUG = True

class ObstacleAvoidanceVarB1Box1PointUR5(MovingBox3DSplineTargetUR5):
        # TODO: The only difference to MovingBoxTargetUR5 are the settings of the target right? If so maybe we can find a cleaner solution later so we dont have to have the same reset twice
    def _get_observation_space(self):
        """Get environment observation space.

        Returns:
            gym.spaces: Gym observation space object.

        """

        # Joint position range tolerance
        pos_tolerance = np.full(6,0.1)
        # Joint positions range used to determine if there is an error in the sensor readings
        max_joint_positions = np.add(np.full(6, 1.0), pos_tolerance)
        min_joint_positions = np.subtract(np.full(6, -1.0), pos_tolerance)
        # Target coordinates range
        target_range = np.full(3, np.inf)
        
        max_delta_start_positions = np.add(np.full(6, 1.0), pos_tolerance)
        min_delta_start_positions = np.subtract(np.full(6, -1.0), pos_tolerance)

        # Definition of environment observation_space
        max_obs = np.concatenate((target_range, max_joint_positions, max_delta_start_positions, max_joint_positions))
        min_obs = np.concatenate((-target_range, min_joint_positions, min_delta_start_positions, min_joint_positions))

        return spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def reset(self, initial_joint_positions = None, type='random'):
        """Environment reset.

        Args:
            initial_joint_positions (list[6] or np.array[6]): robot joint positions in radians.
            ee_target_pose (list[6] or np.array[6]): [x,y,z,r,p,y] target end effector pose.

        Returns:
            np.array: Environment state.

        """
        self.elapsed_steps = 0

        # Initialize environment state
        self.state = np.zeros(self._get_env_state_len())
        rs_state = np.zeros(self._get_robot_server_state_len())
        
        # NOTE: maybe we can find a cleaner version when we have the final envs (we could prob remove it for the avoidance task altogether)
        # Set initial robot joint positions
        if initial_joint_positions:
            assert len(initial_joint_positions) == 6
            self.initial_joint_positions = initial_joint_positions
        elif (len(self.last_position_on_success) != 0) and (type=='continue'):
            self.initial_joint_positions = self.last_position_on_success
        else:
            self.initial_joint_positions = self._get_initial_joint_positions()

        rs_state[6:12] = self.ur._ur_joint_list_to_ros_joint_list(self.initial_joint_positions)


        # TODO: We should create some kind of helper function depending on how dynamic these settings should be
        # Set initial state of the Robot Server
        n_sampling_points = int(np.random.default_rng().uniform(low= 4000, high=8000))
        
        string_params = {"object_0_function": "3d_spline"}

        r = np.random.uniform()

        if r <= 0.75:
            # object in front of the robot
            float_params = {"object_0_x_min": -0.7, "object_0_x_max": 0.7, "object_0_y_min": 0.2, "object_0_y_max": 1.0, \
                            "object_0_z_min": 0.1, "object_0_z_max": 1.0, "object_0_n_points": 10, \
                            "n_sampling_points": n_sampling_points}
        elif r <= 0.83:
            # object behind robot
            float_params = {"object_0_x_min": -0.7, "object_0_x_max": 0.7, "object_0_y_min": - 0.7, "object_0_y_max": -0.2, \
                            "object_0_z_min": 0.1, "object_0_z_max": 1.0, "object_0_n_points": 10, \
                            "n_sampling_points": n_sampling_points}
        elif r <= 0.91:
            # object on the left side of the  robot
            float_params = {"object_0_x_min": 0.3, "object_0_x_max": 0.7, "object_0_y_min": - 0.7, "object_0_y_max": 0.7, \
                            "object_0_z_min": 0.1, "object_0_z_max": 1.0, "object_0_n_points": 10, \
                            "n_sampling_points": n_sampling_points}
        else :
            # object on the right side of the  robot
            float_params = {"object_0_x_min": -0.2, "object_0_x_max": -0.7, "object_0_y_min": - 0.7, "object_0_y_max": 0.7, \
                            "object_0_z_min": 0.1, "object_0_z_max": 1.0, "object_0_n_points": 10, \
                            "n_sampling_points": n_sampling_points}


        state_msg = robot_server_pb2.State(state = rs_state.tolist(), float_params = float_params, string_params = string_params)
        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")

        # Get Robot Server state
        rs_state = copy.deepcopy(np.nan_to_num(np.array(self.client.get_state_msg().state)))
        self.prev_rs_state = copy.deepcopy(rs_state)

        # Check if the length of the Robot Server state received is correct
        if not len(rs_state)== self._get_robot_server_state_len():
            raise InvalidStateError("Robot Server state received has wrong length")

        # Convert the initial state from Robot Server format to environment format
        self.state = self._robot_server_state_to_env_state(rs_state)

        # save start position
        self.start_position = self.state[3:9]

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(self.state):
            raise InvalidStateError()
        
        # check if current position is in the range of the initial joint positions
        if (len(self.last_position_on_success) == 0) or (type=='random'):
            joint_positions = self.ur._ros_joint_list_to_ur_joint_list(rs_state[6:12])
            if not np.isclose(joint_positions, self.initial_joint_positions, atol=0.1).all():
                raise InvalidStateError('Reset joint positions are not within defined range')
            
        return self.state

    def _robot_server_state_to_env_state(self, rs_state):
        """Transform state from Robot Server to environment format.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            numpy.array: State in environment format.

        """
        # Convert to numpy array and remove NaN values
        rs_state = np.nan_to_num(np.array(rs_state))

        # Transform cartesian coordinates of target to polar coordinates 
        # with respect to the end effector frame
        target_coord = rs_state[0:3]
        
        ee_to_ref_frame_translation = np.array(rs_state[18:21])
        ee_to_ref_frame_quaternion = np.array(rs_state[21:25])
        ee_to_ref_frame_rotation = R.from_quat(ee_to_ref_frame_quaternion)
        ref_frame_to_ee_rotation = ee_to_ref_frame_rotation.inv()
        # to invert the homogeneous transformation
        # R' = R^-1
        ref_frame_to_ee_quaternion = ref_frame_to_ee_rotation.as_quat()
        # t' = - R^-1 * t
        ref_frame_to_ee_translation = -ref_frame_to_ee_rotation.apply(ee_to_ref_frame_translation)

        target_coord_ee_frame = utils.change_reference_frame(target_coord,ref_frame_to_ee_translation,ref_frame_to_ee_quaternion)
        target_polar = utils.cartesian_to_polar_3d(target_coord_ee_frame)

        # Transform joint positions and joint velocities from ROS indexing to
        # standard indexing
        ur_j_pos = self.ur._ros_joint_list_to_ur_joint_list(rs_state[6:12])
        ur_j_vel = self.ur._ros_joint_list_to_ur_joint_list(rs_state[12:18])

        # Normalize joint position values
        ur_j_pos_norm = self.ur.normalize_joint_values(joints=ur_j_pos)

        # start joint positions
        start_joints = self.ur.normalize_joint_values(self._get_initial_joint_positions())
        delta_joints = ur_j_pos_norm - start_joints

        # Compose environment state
        state = np.concatenate((target_polar, ur_j_pos_norm, delta_joints, start_joints))

        return state

    def _get_initial_joint_positions(self):
        """Get initial robot joint positions.

        Returns:
            np.array: Joint positions with standard indexing.

        """
        # Fixed initial joint positions
        if self.elapsed_steps < 250:
            joint_positions = np.array([-0.78,-1.31,-1.31,-2.18,1.57,0.0])
        elif self.elapsed_steps < 500:
            joint_positions = np.array([-1.3,-1.0,-1.7,-2.18,1.57,0.0])
        elif self.elapsed_steps < 750:
            joint_positions = np.array([0.0,-1.8,-1.0,-1.0,2.0,0.0])
        else:
            joint_positions = np.array([-1.7,-0.8,-2.0,-2.0,2.5,0.0])

        return joint_positions

    def print_state_action_info(self, rs_state, action):
        env_state = self._robot_server_state_to_env_state(rs_state)

        print('Action:', action)
        print('Last A:', self.last_action)
        print('Distance: {:.2f}'.format(env_state[0]))
        # print('Polar 1 (degree): {:.2f}'.format(env_state[1] * 180/math.pi))
        # print('Polar 2 (degree): {:.2f}'.format(env_state[2] * 180/math.pi))
        print('Joint Positions: [1]:{:.2e} [2]:{:.2e} [3]:{:.2e} [4]:{:.2e} [5]:{:.2e} [6]:{:.2e}'.format(*env_state[3:9]))
        print('Joint PosDeltas: [1]:{:.2e} [2]:{:.2e} [3]:{:.2e} [4]:{:.2e} [5]:{:.2e} [6]:{:.2e}'.format(*env_state[9:15]))
        print('Current Desired: [1]:{:.2e} [2]:{:.2e} [3]:{:.2e} [4]:{:.2e} [5]:{:.2e} [6]:{:.2e}'.format(*env_state[15:21]))
        print('Sum of Deltas: {:.2e}'.format(sum(abs(env_state[9:15]))))
        print('Square of Deltas: {:.2e}'.format(np.square(env_state[9:15]).sum()))
        print()

    def _reward(self, rs_state, action):
        # TODO: remove print when not needed anymore
        # print('action', action)
        env_state = self._robot_server_state_to_env_state(rs_state)

        reward = 0
        done = False
        info = {}

        # minimum and maximum distance the robot should keep to the obstacle
        minimum_distance = 0.3 # m
        maximum_distance = 0.6 # m
        
        distance_to_target = env_state[0]   
        delta_joint_pos = env_state[9:15]


        # reward for being in the defined interval of minimum_distance and maximum_distance
        dr = 0
        if abs(delta_joint_pos).sum() < 0.5:
            dr = 1 * (1 - (sum(abs(delta_joint_pos))/0.5)) * (1/1000)
            reward += dr
        
        # reward moving as less as possible
        act_r = 0 
        if abs(action).sum() <= action.size:
            act_r = 1.5 * (1 - (np.square(action).sum()/action.size)) * (1/1000)
            reward += act_r

        # punish big deltas in action
        act_delta = 0
        for i in range(len(action)):
            if abs(action[i] - self.last_action[i]) > 0.4:
                a_r = - 0.3 * (1/1000)
                act_delta += a_r
                reward += a_r
        
        dist_1 = 0
        if (distance_to_target < minimum_distance):
            dist_1 = -3 * (1/1000) # -2
            reward += dist_1

        # TODO: we could remove this if we do not need to punish failure or reward success
        # Check if robot is in collision
        collision = True if rs_state[25] == 1 else False
        if collision:
            # reward = -1
            done = True
            info['final_status'] = 'collision'

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'success'

        

        if DEBUG: self.print_state_action_info(rs_state, action)
        # ? DEBUG PRINT
        if DEBUG: print('reward composition:', 'dr =', round(dr, 5), 'no_act =', round(act_r, 5), 'min_dist_1 =', round(dist_1, 5), 'min_dist_2 =', 'delta_act', round(act_delta, 5))


        return reward, done, info

class ObstacleAvoidanceVarB1Box1PointUR5DoF3(ObstacleAvoidanceVarB1Box1PointUR5):
    def _get_action_space(self):
        return spaces.Box(low=np.full((3), -1.0), high=np.full((3), 1.0), dtype=np.float32)

class ObstacleAvoidanceVarB1Box1PointUR5DoF5(ObstacleAvoidanceVarB1Box1PointUR5):
    def _get_action_space(self):
        return spaces.Box(low=np.full((5), -1.0), high=np.full((5), 1.0), dtype=np.float32)

class ObstacleAvoidanceVarB1Box1PointUR5Sim(ObstacleAvoidanceVarB1Box1PointUR5, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        world_name:=box100.world \
        yaw:=-0.78\
        reference_frame:=world \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=false \
        gazebo_gui:=true \
        objects_controller:=true \
        target_mode:=moving \
        n_objects:=1.0 \
        object_0_model_name:=box100 \
        object_0_frame:=target"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        ObstacleAvoidanceVarB1Box1PointUR5.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class ObstacleAvoidanceVarB1Box1PointUR5Rob(ObstacleAvoidanceVarB1Box1PointUR5):
    real_robot = True

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=moving n_objects:=1.0 object_0_frame:=target"

class ObstacleAvoidanceVarB1Box1PointUR5DoF3Sim(ObstacleAvoidanceVarB1Box1PointUR5DoF3, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        world_name:=box100.world \
        yaw:=-0.78\
        reference_frame:=world \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=false \
        gazebo_gui:=true \
        objects_controller:=true \
        target_mode:=moving \
        n_objects:=1.0 \
        object_0_model_name:=box100 \
        object_0_frame:=target"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        ObstacleAvoidanceVarB1Box1PointUR5DoF3.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class ObstacleAvoidanceVarB1Box1PointUR5DoF3Rob(ObstacleAvoidanceVarB1Box1PointUR5DoF3):
    real_robot = True

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=moving n_objects:=1.0 object_0_frame:=target"

class ObstacleAvoidanceVarB1Box1PointUR5DoF5Sim(ObstacleAvoidanceVarB1Box1PointUR5DoF5, Simulation):
    cmd = "roslaunch ur_robot_server ur5_sim_robot_server.launch \
        world_name:=box100.world \
        yaw:=-0.78\
        reference_frame:=world \
        max_velocity_scale_factor:=0.2 \
        action_cycle_rate:=20 \
        rviz_gui:=false \
        gazebo_gui:=true \
        objects_controller:=true \
        target_mode:=moving \
        n_objects:=1.0 \
        object_0_model_name:=box100 \
        object_0_frame:=target"
    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, **kwargs):
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        ObstacleAvoidanceVarB1Box1PointUR5DoF5.__init__(self, rs_address=self.robot_server_ip, **kwargs)

class ObstacleAvoidanceVarB1Box1PointUR5DoF5Rob(ObstacleAvoidanceVarB1Box1PointUR5DoF5):
    real_robot = True

# roslaunch ur_robot_server ur5_real_robot_server.launch  gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 target_mode:=moving n_objects:=1.0 object_0_frame:=target"
