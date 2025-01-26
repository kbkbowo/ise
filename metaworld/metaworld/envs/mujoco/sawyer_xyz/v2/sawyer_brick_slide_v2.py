import os
import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerBrickSlideEnvV2(SawyerXYZEnv):

    def __init__(self, friction=0.4):

        hand_low = (-0.5, 0.40, 0.1)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.10, 0.7, 0.02)
        obj_high = (0.10, 0.75, 0.02)
        goal_low = (-0.1, 0.7, 0.133)
        goal_high = (0.1, 0.8, 0.133)

        self.random_noise = np.random.uniform(-0.01, 0.01, size=(3,))
        self.random_noise[2] = 0.0

        # print("NOISE", self.random_noise)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )
        
        self.last_height = 0

        self.threshold = False
        
        self.random_init = True
        self.goal = np.array([-0.2, 0.7, 0.06])
        self.init_config = {
            'obj_init_angle': .3,
            'obj_init_pos': np.array([0, 0.0, 0.0], dtype=np.float32),
            'hand_init_pos': np.array((-0.6, 0.7, 0.1), dtype=np.float32),
        }
        self.goal = np.array([0.0, 0.75, 0.133])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle'] 
        self.hand_init_pos = self.init_config['hand_init_pos']

        self._target_to_obj_init = None

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([+1, +1, +1, +1]),
        )

        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        # self._random_reset_space = Box(
        #     np.hstack((obj_low, goal_low)),
        #     np.hstack((obj_high, goal_high)),
        # )
        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        # for i, fric in enumerate(self.sim.model.geom_friction):
            # print("INDEX", i, "FRICTION", fric)
            
        # friction = 0.01
        self.sim.model.geom_friction[31] = np.array([0, 0, 0])
        self.sim.model.geom_friction[33] = np.array([0, 0, 0])
        self.sim.model.geom_friction[36][0] = friction
        self.sim.model.geom_friction[37][0] = friction 
        self.sim.model.geom_friction[38][0] = friction
        # print(friction)
    @property
    def model_name(self):
        init_bar_pos = np.array([-0.2, 0.7, 0.06]) + self.random_noise
        mstr = f"""
        <mujoco>
            <include file="./scene/basic_scene.xml"/>
            <include file="../objects/assets/box_dependencies.xml"/>
            <include file="../objects/assets/xyz_base_dependencies.xml"/>
            <worldbody>
                <include file="../objects/assets/xyz_base.xml"/>

                <body name="boxbody" euler="0.0 0.0 0.0" pos="0.0 0.7 0.00">
                <!-- MAX FRIC 0.4 -->
                <!-- MIN FRIC 0.21 -->
                    <geom material="box_red" pos="-0.325 0.0 0.01375" size="0.07625 0.2 0.001" type="box" mass=".99"  friction="0.22 0.000 0.0000" euler="0.0 -0.181319774 0.0"/>
                    <geom material="box_metal" pos="-0.22 0.0 0.033" size="0.0305 0.2 0.001" type="box" mass=".99"  friction="0.22 0.000 0.0000" euler="0.0 -0.181319774 0.0"/>
                    <geom material="box_red" pos="-0.04 0.0 0.066" size="0.1525 0.2 0.001" type="box" mass=".99"  friction="0.22 0.000 0.0000" euler="0.0 -0.1812119774 0.0"/>
                    <geom material="box_wood" pos="0.710 0.0 0.2035" size="0.61 0.2 0.001" type="box" mass=".99"  friction="0.0 0.000 0.0000" euler="0.0 -0.181319774 0.0"/>
                </body>

                <body name="top_link" euler="0.0 0.0 0.0" pos="{ init_bar_pos[0] } { init_bar_pos[1] } { init_bar_pos[2] }">
                    <joint name="objjoint" type="free" limited='false' damping="0." armature="0."/>
                    <!-- <inertial pos="0 0 0" mass="100.75" diaginertia="8.80012e-04 8.80012e-04 8.80012e-04"/> -->
                    <geom material="box_blue" pos="0.0 0.0 0.0" size="0.05 0.10 0.05" type="box" mass="100.20" friction="0.000 0.000 0.0000" euler="0.0 0.0 0.0"/>
                    <geom rgba="0. 1. 0. 1." pos="0.0 -0.10 -0.047" size="0.02 0.02 0.001" type="box" mass="0.01" friction="0.000 0.000 0.0000" euler="0.0 0.0 0.0"/>
                </body>

                <!-- <site name="fix_point0" pos="0 0.4 0.05" size="0.02" rgba="0 1 1 0.3" type="sphere"/>
                <site name="fix_point1" pos="0 0.4 0.09" size="0.02" rgba="0 1 1 0.3" type="sphere"/>
                <site name="fix_point2" pos="0 0.4 0.13" size="0.02" rgba="0 1 1 0.3" type="sphere"/>
                <site name="fix_point3" pos="0 0.4 0.17" size="0.02" rgba="0 1 1 0.3" type="sphere"/> -->

                <!-- <site name="goal" pos="0. 0.7 0.05" size="0.02"
                    rgba="0 0.8 0 1"/> -->
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
        exenv = os.path.join(os.path.dirname(__file__), 'push_test/sawyer_brick_slide_rdn.xml')
        with open(exenv, 'w') as f:
            f.write(mstr)
        # return full_v2_path_for('sawyer_xyz/sawyer_brick_slide.xml')
        return exenv

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            reward_grab,
            reward_ready,
            reward_success,
            success
        ) = self.compute_reward(action, obs)

        info = {
            'success': float(success),
            'near_object': reward_ready,
            'grasp_success': reward_grab >= 0.5,
            'grasp_reward': reward_grab,
            'in_place_reward': reward_success,
            'obj_to_target': 0,
            'unscaled_reward': reward,
        }

        return reward, info

    @property
    def _target_site_config(self):
        return []

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id('top_link')

    def _get_pos_objects(self):
        # print("OBJECT", self.sim.data.get_body_xpos('top_link'))
        return self.get_body_com('top_link')

    def _get_quat_objects(self):
        return self.sim.data.get_body_xquat('top_link')

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = np.array(self.init_config['obj_init_pos'])
        self.obj_init_angle = self.init_config['obj_init_angle']

        # if self.random_init:
        #     goal_pos = self._get_state_rand_vec()
        #     print("GOAL_POS", goal_pos, self._freeze_rand_vec)
        #     print(goal_pos[:2] - self._target_pos[:2])
        #     self._target_pos = goal_pos[3:]
        #     while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
        #         goal_pos = self._get_state_rand_vec()
        #         self._target_pos = goal_pos[3:]
        #     self._target_pos = np.concatenate((goal_pos[-3:-1], [self.obj_init_pos[-1]]))
        #     self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))

        # self._set_obj_xyz(self.obj_init_pos)
        
        # self._set_obj_xyz(self.obj_init_pos)
        # self._target_pos[1] -= 0.20

        return self._get_obs()

    @staticmethod
    def _reward_grab_effort(actions):
        return (np.clip(actions[3], -1, 1) + 1.0) / 2.0

    @staticmethod
    def _reward_quat(obs):
        # Ideal upright lid has quat [.707, 0, 0, .707]
        # Rather than deal with an angle between quaternions, just approximate:
        ideal = np.array([0.707, 0, 0, 0.707])
        error = np.linalg.norm(obs[7:11] - ideal)
        return max(1.0 - error/0.2, 0.0)

    @staticmethod
    def _reward_pos(obs, target_pos):
        hand = obs[:3]
        lid = obs[4:7] + np.array([.0, .0, .02])

        threshold = 0.02
        # floor is a 3D funnel centered on the lid's handle
        radius = np.linalg.norm(hand[:2] - lid[:2])
        if radius <= threshold:
            floor = 0.0
        else:
            floor = 0.04 * np.log(radius - threshold) + 0.4
        # prevent the hand from running into the handle prematurely by keeping
        # it above the "floor"
        above_floor = 1.0 if hand[2] >= floor else reward_utils.tolerance(
            floor - hand[2],
            bounds=(0.0, 0.01),
            margin=floor / 2.0,
            sigmoid='long_tail',
        )
        # grab the lid's handle
        in_place = reward_utils.tolerance(
            np.linalg.norm(hand - lid),
            bounds=(0, 0.02),
            margin=0.5,
            sigmoid='long_tail',
        )
        ready_to_lift = reward_utils.hamacher_product(above_floor, in_place)

        # now actually put the lid on the box
        pos_error = target_pos - lid
        error_scale = np.array([1., 1., 3.])  # Emphasize Z error
        a = 0.2  # Relative importance of just *trying* to lift the lid at all
        b = 0.8  # Relative importance of placing the lid on the box
        lifted = a * float(lid[2] > 0.04) + b * reward_utils.tolerance(
            np.linalg.norm(pos_error * error_scale),
            bounds=(0, 0.05),
            margin=0.25,
            sigmoid='long_tail',
        )

        return ready_to_lift, lifted

    def compute_reward(self, actions, obs):
        reward_grab = SawyerBrickSlideEnvV2._reward_grab_effort(actions)
        reward_quat = SawyerBrickSlideEnvV2._reward_quat(obs)
        reward_steps = SawyerBrickSlideEnvV2._reward_pos(obs, self._target_pos)

        reward = sum((
            2.0 * reward_utils.hamacher_product(reward_grab, reward_steps[0]),
            8.0 * reward_steps[1],
        ))

        # Override reward on success
        # success = np.linalg.norm(obs[4:7] - self._target_pos) < 0.08
        success = False

        height = obs[6]
        
        if height > 0.14:
            self.threshold = True
        success = (height < 0.093 and height > 0.073) and self.threshold and (abs(self.last_height - height) < 0.00003)
        # print("HEIGHT", obs[6], "LST_HEIGHT", self.last_height, "\tTHRESHOLD", self.threshold, )
        # print(success)
        if success:
            reward = 10.0
            # success = False

        # STRONG emphasis on proper lid orientation to prevent reward hacking
        # (otherwise agent learns to kick-flip the lid onto the box)
        reward *= reward_quat
        self.last_height = height
        return (
            reward,
            reward_grab,
            *reward_steps,
            success,
        )
