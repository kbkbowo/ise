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

import numpy as np
from PIL import Image, ImageDraw, ImageFont

EXTENDER_RATIO = (0.025, 0.015)

def center_of_mass(array: np.ndarray):
    total = array.sum()

    x_grid = np.arange(array.shape[0])
    x_sumx = np.sum(array @ x_grid.T)
    x_coord = x_sumx / total

    y_grid = np.arange(array.shape[1])
    y_sumy = np.sum(y_grid @ array)
    y_coord = y_sumy / total

    return x_coord, y_coord

def l2b(letter, font_size=20, font_path=None, image_size=(32, 32), cm_sample_seed=None):
    # Step 1: Create a blank image with a black background
    image = Image.new('L', image_size, color=0)  # 'L' mode is for grayscale
    draw = ImageDraw.Draw(image)

    # print(font_path)
    # print(os.path.exists(font_path))

    # Step 2: Load a font and draw the letter on the image
    if font_path is None:
        font = ImageFont.load_default() # Load default font
    else:
        font = ImageFont.truetype(font_path, font_size)
    # print(font_path)

    bbox = draw.textbbox((0, 0), letter, font=font)  # Get bounding box for the text   
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)
    draw.text(position, letter, fill=255, font=font)  # Draw the letter in white (255)

    # Step 3: Convert the image to a numpy array
    image_np = np.array(image)

    # Step 4: Convert the grayscale values to binary (0 or 1)
    binary_array = (image_np > 128).astype(int)  # Threshold at 128

    # rotate 90 left
    binary_array = np.rot90(binary_array, 1)
    binary_array = np.flip(binary_array, 1)

    # SAMPLE RANDOM ARRAY FOR MASS CALCULATION
    if cm_sample_seed is not None:
        np.random.seed(cm_sample_seed)
        # rng = np.random.RandomState(self.init_seed)

    rnd_array = np.random.randint(0, 10, size=image_size)
    # rnd_array = rng.randint(0, 10, size=image_size)
    rnd_array = np.exp(rnd_array)

    # Step 5: Calculate the dot product of the binary array and the random array
    mass_array = np.dot(binary_array, rnd_array) # eliminate 0 values
    mass_array = mass_array / np.sum(mass_array)
    mass_array = mass_array * 100

    cm_x, cm_y = center_of_mass(mass_array)
    # print(cm_x, cm_y)

    return binary_array, (cm_x, cm_y)

def pt2geom(x=0, y=0):
    base = f"""
    <geom name="pt_{x}_{y}" type="box" pos="{ x * EXTENDER_RATIO[0] } { y * EXTENDER_RATIO[1] } 0"
        size="{EXTENDER_RATIO[0]} {EXTENDER_RATIO[1]} 0.015" rgba="0.75 0.75 0.85 1"
        conaffinity="1" condim="1"
    />
    """
    return base

def cm2geom(x=0, y=0, visible=False):
    cmvis = f"""
    <geom name="cm_{x}_{y}" type="box" pos="{ x * EXTENDER_RATIO[0] } { y * EXTENDER_RATIO[1] } 0"
        size="{EXTENDER_RATIO[0]} {EXTENDER_RATIO[1]} 0.0151" rgba="0 0.75 0.85 1"
        conaffinity="1" condim="1"
    />
    """
    base = f"""
    <geom name="objGeom" type="box" pos="{ x * EXTENDER_RATIO[0] } { y * EXTENDER_RATIO[1] } 0"
        size="0.001 0.001 0.001" rgba="0 0 0 0"
        conaffinity="0" condim="1" contype="0"
    />
    """
    return base + (cmvis if visible else "")

def l2barr2geom(l2barr):
    base = ""
    for i in range(l2barr.shape[0]):
        for j in range(l2barr.shape[1]):
            if l2barr[i, j] == 1:
                base += pt2geom(i - l2barr.shape[0] // 2, j - l2barr.shape[1] // 2)
    return base

class SawyerPushAlphabetEnvV2(SawyerXYZEnv):
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

    def __init__(self, letter='A', cm_visible=False, font_path=None, font_size=20, init_seed=None, cm_sample_seed=None, **kwargs):
        self.cm_visible = cm_visible

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.02)
        obj_high = (0.1, 0.7, 0.02)
        goal_low = (-0.1, 0.85, 0.015)
        goal_high = (0.1, 0.85, 0.015)
        cm_low = (-0.18, 0, 0)
        cm_high = (0.18, 0, 0)

        self.cm_sample_seed = cm_sample_seed
        # print(f"CM Sample Seed: {self.cm_sample_seed}")
        self.alphabet, self.cmx = l2b(letter, font_path=font_path, font_size=font_size, cm_sample_seed=self.cm_sample_seed)
        # print(self.cmx[0] - self.alphabet.shape[0] // 2, self.cmx[1] - self.alphabet.shape[1] // 2)

        self.init_seed = init_seed
        np.random.seed(self.init_seed)
        self.stored_init = {
            'obj_init_pos': np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(0.6, 0.7), 0.02]),
            'goal_pos': np.array([np.random.uniform(-0.1, 0.1), 0.85, 0.015]),
        }

        # print(self.alphabet, self.cmx)

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
            np.array([-1, -1, -1, -1]),
            np.array([+1, +1, +1, +1]),
        )

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        self.num_resets = 0

    def cm_coords(self):
        half_side = self.alphabet.shape[0] // 2
        return ((self.cmx[0] - half_side) * EXTENDER_RATIO[0], (self.cmx[1] - half_side) * EXTENDER_RATIO[1])
    
    def x_offset(self):
        cm_coords = self.cm_coords()
        print(self._target_pos, self.obj_init_pos + np.array([cm_coords[0], cm_coords[1], 0]))
        delta_x = self._target_pos[0] - self.obj_init_pos[0] - cm_coords[0]
        delta_y = self._target_pos[1] - self.obj_init_pos[1] - cm_coords[1]

        x_y_ratio = delta_x / delta_y

        delta_y_pt = self.obj_init_pos[1] + cm_coords[1] - 0.42
        delta_x_pt = delta_y_pt * x_y_ratio

        return self.obj_init_pos[0] - delta_x_pt
    
    @property
    def model_name(self):
        self.ixcm = self.cm_coords()
        half_side = self.alphabet.shape[0] // 2
        basestr = l2barr2geom(self.alphabet) + cm2geom(self.cmx[0] - half_side, self.cmx[1] - half_side, self.cm_visible)
        mstr = f"""
        <mujoco>
            <include file="./scene/basic_scene.xml"/>
            <include file="../objects/assets/block_dependencies.xml"/>
            <include file="../objects/assets/xyz_base_dependencies.xml"/>
            <worldbody>
            <include file="../objects/assets/xyz_base.xml"/>
            <body name="obj" pos="0 0.6 0.02">
                <joint name="objjoint" type="free" limited='false' damping="0." armature="0."/>
                <inertial pos="{self.ixcm[0]} {self.ixcm[1]} 0" mass="5.0" diaginertia="8.80012e-04 8.80012e-04 8.80012e-04"/>
                {basestr}
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
        # self._target_pos = self.goal.copy()
        # self.obj_init_pos = np.array(self.fix_extreme_obj_pos(self.init_config['obj_init_pos']))
        # self.obj_init_angle = self.init_config['obj_init_angle']

        # print("Stored init")
        self._target_pos = self.stored_init['goal_pos']
        self.obj_init_pos = self.stored_init['obj_init_pos']

        # self.obj_init_pos = np.array([self.offset + x_delta_pt, raw_init_pos[1], raw_init_pos[2]])

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
