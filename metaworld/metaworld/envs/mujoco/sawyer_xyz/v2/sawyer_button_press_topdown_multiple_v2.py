import numpy as np
from gym.spaces import Box
import os

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set

class SawyerButtonPressTopdownMultipleEnvV2(SawyerXYZEnv):

    def __init__(self, idx=0, isHorizontal=False):
        hand_randomness = np.concatenate([np.random.uniform(-0.3, 0.3, size=2), np.random.uniform(0, 0.1, size=1)])

        self.idx = idx
        self.isHorizontal = isHorizontal

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.8, 0.115)
        obj_high = (0.1, 0.9, 0.115)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.8, 0.115], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.4, 0.2], dtype=np.float32) + hand_randomness,
        }
        self.goal = np.array([0, 0.88, 0.1])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        arrange_array = [-0.2, 0.2, 0]
        if self.idx == 0:
            pass
        elif self.idx == 1:
            arrange_array = [0, 0.2, -0.2]
        elif self.idx == 2:
            arrange_array = [0.2, -0.2, 0]

        is_horizontal = self.isHorizontal

        vertical_pos = "0.7 -0.06"
        horizontal_pos = "0.9 0.12"

        vertical_euler = "-1.57 0 0"
        horizontal_euler = "0 0 0"
        
        mstr = f"""
            <mujoco>
                <include file="./scene/basic_scene.xml"/>
                <include file="../objects/assets/buttonbox_dependencies.xml"/>
                <include file="../objects/assets/xyz_base_dependencies.xml"/>

                <worldbody>
                <include file="../objects/assets/xyz_base.xml"/>

                    <body name="box" euler="{horizontal_euler if is_horizontal else vertical_euler}" pos="{arrange_array[0]} {horizontal_pos if is_horizontal else vertical_pos}">
                        <include file="../objects/assets/buttonbox.xml"/>
                        <site name="buttonStart" pos="0 -0.1935 0" size="0.005" rgba="0 0.8 0 0"/>
                        <site name="hole" pos="0 -0.1 0" size="0.005" rgba="0 0.8 0 0"/>
                    </body>

                    <body name="box_2" euler="{horizontal_euler if is_horizontal else vertical_euler}" pos="{arrange_array[1]} {horizontal_pos if is_horizontal else vertical_pos}">
                        <include file="../objects/assets/buttonbox2.xml"/>
                    </body>

                    <body name="box_3" euler="{horizontal_euler if is_horizontal else vertical_euler}" pos="{arrange_array[2]} {horizontal_pos if is_horizontal else vertical_pos}">
                        <include file="../objects/assets/buttonbox3.xml"/>
                    </body>

                </worldbody>

                <actuator>
                    <position ctrllimited="true" ctrlrange="-1 1" joint="r_close" kp="400"  user="1"/>
                    <position ctrllimited="true" ctrlrange="-1 1" joint="l_close" kp="400"  user="1"/>
                </actuator>
                <equality>
                    <weld body1="mocap" body2="hand" solref="0.02 1"/>
                </equality>
            </mujoco>
        """
        exenv = os.path.join(os.path.dirname(__file__), 'button_press/sawyer_button_press_topdown_multiple.xml')
        with open(exenv, 'w') as f:
            f.write(mstr)
        return exenv

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            near_button,
            button_pressed
        ) = self.compute_reward(action, obs)

        info = {
            'success': float(obj_to_target <= 0.02),
            'near_object': float(tcp_to_obj <= 0.05),
            'grasp_success': float(tcp_open > 0),
            'grasp_reward': near_button,
            'in_place_reward': button_pressed,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward,
        }

        return reward, info

    @property
    def _target_site_config(self):
        return []

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id('btnGeom')

    def _get_pos_objects(self):
        return self.get_body_com('button') + (np.array([.0, .0, .193]) if not self.isHorizontal else np.array([.0, -.193, .0]))

    def _get_quat_objects(self):
        return self.sim.data.get_body_xquat('button')

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        qvel[9] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()

        # if self.random_init:
        #     goal_pos = self._get_state_rand_vec()
        #     self.obj_init_pos = goal_pos

        # self.sim.model.body_pos[self.model.body_name2id('box')] = self.obj_init_pos
        self._target_pos = self._get_site_pos('hole')

        self._obj_to_target_init = abs(
            self._target_pos[2] - self._get_site_pos('buttonStart')[2]
        )

        return self._get_obs()

    def compute_reward(self, action, obs):
        del action
        obj = obs[4:7]
        tcp = self.tcp_center

        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(obj - self.init_tcp)
        obj_to_target = abs(self._target_pos[2] - obj[2]) if not self.isHorizontal else abs(self._target_pos[1] - obj[1])

        tcp_closed = 1 - obs[3] if not self.isHorizontal else max(obs[3], 0)
        near_button = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, 0.01),
            margin=tcp_to_obj_init,
            sigmoid='long_tail',
        )
        button_pressed = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, 0.005),
            margin=self._obj_to_target_init,
            sigmoid='long_tail',
        )

        reward = 5 * reward_utils.hamacher_product(tcp_closed, near_button)
        if tcp_to_obj <= 0.03:
            reward += 5 * button_pressed

        return (
            reward,
            tcp_to_obj,
            obs[3],
            obj_to_target,
            near_button,
            button_pressed
        )
