import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerPushTestV2Policy(Policy):

    def __init__(self, relative_offset=None, **kwargs):
        if relative_offset is not None:
            # print(f"POLICY: relative_offset = {relative_offset}")
            self.relative_offset = relative_offset
        else:
            self.relative_offset = 0

        super().__init__(**kwargs)

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_1': obs[3],
            'puck_pos': obs[4:7],
            'unused_2':  obs[7:-3],
            'goal_pos': obs[-3:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })
        o_d['puck_pos'] += np.array([-self.relative_offset, 0, 0])

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=10.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_puck = o_d['puck_pos'] + np.array([-0.01, 0, 0])
        pos_goal = o_d['goal_pos']

        direction = pos_goal[:2] - pos_puck[:2]
        dir_dist = np.linalg.norm(direction)
        direction /= dir_dist

        place = pos_curr[:2] - pos_puck[:2]
        distance = np.linalg.norm(place)
        place /= distance

        inner_product = (place[0] * direction[0]) + (place[1] * direction[1])

        target_place = pos_puck[:2] - direction * 0.07

        # max_y = pos_goal[1] - 0.15
        # if target_place[1] > max_y:
        #     target_place[1] = max_y

        # if inner_product > -0.95:
        #     return [target_place[0], target_place[1], pos_puck[2] + 0.15]

        if abs(pos_curr[2] - pos_puck[2]) > 0.04:
            return [target_place[0], target_place[1], pos_puck[2] + 0.03]

        # if pos_goal[1] > max_y:
        #     pos_goal[1] = max_y

        
        if dir_dist > 0.02:
            return pos_curr + 0.02 * np.array([direction[0], direction[1], pos_goal[2]])
        return pos_goal

    @staticmethod
    def _grab_effort(o_d):
        return 1.0
        # pos_curr = o_d['hand_pos']
        # pos_puck = o_d['puck_pos']

        # if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.02 or abs(pos_curr[2] - pos_puck[2]) > 0.10:
        #     return 0.
        # # While end effector is moving down toward the puck, begin closing the grabber
        # else:
        #     return 0.6
