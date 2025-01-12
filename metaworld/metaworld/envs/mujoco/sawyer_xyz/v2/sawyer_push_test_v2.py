import numpy as np
from gym.spaces import Box
import pprint
import random
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set
# from utils import get_geom_ids

import os


class SawyerPushTestEnvV2(SawyerXYZEnv):
    """
    Motivation for V2:
        V1 was very difficult to solve because the observation didn't say where
        to move after reaching the puck.
    Changelog from V1 to V2:
        - (7/7/20) Removed 3 element vector. Replaced with 3 element position
            of the goal (for consistency with other environments)
        - (6/15/20) Added a 3 element vector to the observation. This vector
            points from the end effector to the goal coordinate.
            i.e. (self._target_pos - pos_hand)
        - (6/15/20) Separated reach-push-pick-place into 3 separate envs.
    """
    TARGET_RADIUS=0.05

    def __init__(self, cm_offset=None, cm_visible=False, **kwargs):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.02)
        obj_high = (0.1, 0.7, 0.02)
        # obj_low =  (0.3, 0.7, 0.02)
        # obj_high = (0.3, 0.7, 0.02)
        goal_low = (-0.1, 0.85, 0.015)
        goal_high = (0.1, 0.85, 0.015)
        cm_low = (-0.18, 0, 0)
        cm_high = (0.18, 0, 0)

        if cm_offset is not None:
            self.cm_offset = cm_offset
        else:
            self.cm_offset = 0.18 * (np.random.random() * 2 - 1)

        self.cm_visible = cm_visible

        # self.rseed = seed
        # self.rng = np.random.RandomState(seed)
        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': .3,
            'obj_init_pos': np.array([0., 0.6, 0.02]),
            'hand_init_pos': np.array([0., 0.6, 0.2]),
        }

        self.goal = np.array([0.1, 0.8, 0.02])

        self.obj_init_angle = self.init_config['obj_init_angle']
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.action_space = Box(
            np.array([-1, -1, -1, -1]).astype(np.float64),
            np.array([+1, +1, +1, +1]).astype(np.float64),
        )

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        self.num_resets = 0
    
    @property
    def model_name(self):
        pm = 0.9 * self.cm_offset
        self.pm = pm
        mstr = f"""
        <mujoco>
            <include file="./scene/basic_scene.xml"/>
            <include file="../objects/assets/block_dependencies.xml"/>
            <include file="../objects/assets/xyz_base_dependencies.xml"/>
            <worldbody>
            <include file="../objects/assets/xyz_base.xml"/>
            <body name="obj" pos="0 0.6 0.02">
                <joint name="objjoint" type="free" limited='false' damping="0." armature="0."/>
                <inertial pos="0 0 0" mass="5.0" diaginertia="8.80012e-04 8.80012e-04 8.80012e-04"/>
                    <geom name="rod" type="box" pos="{ pm } 0 0"
                        size="0.2 0.02 0.02" rgba="0.75 0.75 0.85 1"
                        conaffinity="1" condim="1"
                    />
                    <geom name="vis_point" type="box" pos="0 0 0"
                        size="{'0.0201 0.0201 0.0201'if self.cm_visible else '0.0199 0.0199 0.0199'}" rgba="0 0.75 0.85 1"
                        conaffinity="1" condim="1"
                    />
                    <geom name="objGeom" type="box" pos="0 0 0"
                        size="0.01 0.01 0.01" rgba="0 0 0 0"
                        conaffinity="1" condim="1"
                    />
            </body>
            <site name="goal" pos="0.1 0.8 0.02" size="0.02" rgba="0 0.8 0 1"/>
            </worldbody>

            <actuator>
                <position ctrllimited="true" ctrlrange="-1 1" joint="r_close" kp="400"  user="1"/>
                <position ctrllimited="true" ctrlrange="-1 1" joint="l_close" kp="400"  user="1"/>
            </actuator>
            <equality>
                <weld body1="mocap" body2="hand" solref="0.02 1"></weld>
            </equality>
        </mujoco>
        """
        exenv = os.path.join(os.path.dirname(__file__), 'push_test/sawyer_push_test_v2.xml')
        with open(exenv, 'w') as f:
            f.write(mstr)
        return exenv #full_v2_path_for('sawyer_xyz/sawyer_push_test_v2.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        obj = obs[4:7]

        (
            reward,
            tcp_to_obj,
            tcp_opened,
            target_to_obj,
            object_grasped,
            in_place
        ) = self.compute_reward(action, obs)

        info = {
            'success': float(target_to_obj <= self.TARGET_RADIUS),
            'near_object': float(tcp_to_obj <= 0.03),
            'grasp_success': float(
                self.touching_main_object and
                (tcp_opened > 0) and
                (obj[2] - 0.02 > self.obj_init_pos[2])
            ),
            'grasp_reward': object_grasped,
            'in_place_reward': in_place,
            'obj_to_target': target_to_obj,
            'unscaled_reward': reward,
        }

        return reward, info

    def _get_quat_objects(self):
        return Rotation.from_matrix(
            self.data.get_geom_xmat('objGeom')
        ).as_quat()

    def _get_pos_objects(self):
        return self.get_body_com('obj')

    def fix_extreme_obj_pos(self, orig_init_pos):
        # This is to account for meshes for the geom and object are not
        # aligned. If this is not done, the object could be initialized in an
        # extreme position
        diff = self.get_body_com('obj')[:2] - \
               self.get_body_com('obj')[:2]
        adjusted_pos = orig_init_pos[:2] + diff
        # The convention we follow is that body_com[2] is always 0,
        # and geom_pos[2] is the object height
        return [
            adjusted_pos[0],
            adjusted_pos[1],
            self.get_body_com('obj')[-1]
        ]

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = np.array(self.fix_extreme_obj_pos(self.init_config['obj_init_pos']))
        self.obj_init_angle = self.init_config['obj_init_angle']

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            self._target_pos = goal_pos[3:]
            while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
                goal_pos = self._get_state_rand_vec()
                self._target_pos = goal_pos[3:]
            self._target_pos = np.concatenate((goal_pos[-3:-1], [self.obj_init_pos[-1]]))
            self.obj_init_pos = np.concatenate(([goal_pos[0] - self.pm], [goal_pos[1]], [self.obj_init_pos[-1]]))

        self._set_obj_xyz(self.obj_init_pos)
        self.num_resets += 1

        return self._get_obs()

    def compute_reward(self, action, obs):
        obj = obs[4:7]
        tcp_opened = obs[3]
        tcp_to_obj = np.linalg.norm(obj - self.tcp_center)
        target_to_obj = np.linalg.norm(obj - self._target_pos)
        target_to_obj_init = np.linalg.norm(self.obj_init_pos - self._target_pos)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=target_to_obj_init,
            sigmoid='long_tail',
        )

        object_grasped = self._gripper_caging_reward(
            action,
            obj,
            object_reach_radius=0.01,
            obj_radius=0.015,
            pad_success_thresh=0.05,
            xz_thresh=0.005,
            high_density=True
        )
        reward = 2 * object_grasped

        if tcp_to_obj < 0.02 and tcp_opened > 0:
            reward += 1. + reward + 5. * in_place
        if target_to_obj < self.TARGET_RADIUS:
            reward = 10.

        return (
            reward,
            tcp_to_obj,
            tcp_opened,
            target_to_obj,
            object_grasped,
            in_place
        )
