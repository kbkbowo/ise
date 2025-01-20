import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerBrickSlideV2Policy(Policy):
    def __init__(self):
        self.checkpoint=False

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        # print(obs)
        # print(obs.shape)
        parsed_obs = {
            'hand_pos': obs[:3],
            'gripper': obs[3],
            'lid_pos': obs[4:7],
            'extra_info_1': obs[7:-3],
            'box_pos': obs[-3:-1],
            'extra_info_2': obs[-1],
        }
        
        # print(parsed_obs['lid_pos'])
        
        # print(parsed_obs)
        # assert False
        return parsed_obs

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=5.)
        # print("DELTA_POS", action['delta_pos'])
        # action['grab_effort'] = self._grab_effort(o_d)
        action['grab_effort'] = 0.

        return action.array

    def _desired_pos(self, o_d):
        push_r = 0.08
        # print(o_d)
        pos_curr = o_d['hand_pos']
        pos_lid = o_d['lid_pos'] + np.array([.0, .0, .07])
        push_pos = pos_lid + np.array([0, push_r, 0])
        pos_box = np.array([*o_d['box_pos'], 0.15]) + np.array([.0, .0, .0])
        # 1. move to pos_lid + [.0 .1 .0]
        # 2. move to goal

        # return np.array([0.0, 0.5, 0.1])
        # print("HEIGHT", pos_lid[2], self.checkpoint)
        checkpoint = (pos_lid[2] - 0.07) > 0.22
        checkpoint = (pos_lid[2] - 0.07) > 0.14
        # print("CHECKPOINT", self.checkpoint, checkpoint, pos_lid[2])
        self.last_pos = pos_lid[2]
        
        if not self.checkpoint and not checkpoint:
            # print("ORIGIN")
            return o_d['lid_pos'] + np.array([0.2, 0.0, -0.1])
        elif checkpoint or self.checkpoint:
            self.checkpoint = True
            return o_d['lid_pos'] + np.array([-1.0, 0.0, 1.0])
        else:
            # print("hahaha")
            return o_d['lid_pos'] + np.array([0.0, -0.1, 0.0])

        # If error in the XY plane is greater than 0.02, place end effector above the puck
        if not self.checkpoint and np.linalg.norm(pos_curr[:2] - push_pos[:2]) > 0.05:
            return np.array([*push_pos[:2], 0.25])
        # Once XY error is low enough, drop end effector down on top of puck
        elif not self.checkpoint and abs(pos_curr[2] - pos_lid[2]) > 0.05:
            return push_pos
        # If not at the same Z height as the goal, move up to that plane
        else: # pos_curr[2] - pos_box[2] < -0.04:
            self.checkpoint=True
            direction = pos_box - pos_curr
            dir_norm = direction / np.linalg.norm(direction)
            return pos_curr + dir_norm * 0.025

    @staticmethod
    def _grab_effort(o_d):
        return 1.
