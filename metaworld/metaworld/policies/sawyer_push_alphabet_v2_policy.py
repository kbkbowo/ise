import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerPushAlphabetV2Policy(Policy):
    def __init__(self, offset=0, **kwargs):
        self.relative_offset = offset
        self.init_move = False
        print(f"POLICY: relative_offset = {self.relative_offset}")
        super().__init__(**kwargs)

    def init_reset(self):
        self.init_move = False

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

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=10.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    # @staticmethod
    def _desired_pos(self, o_d):
        pos_curr = o_d['hand_pos']
        pos_puck = o_d['puck_pos']
        pos_goal = o_d['goal_pos']

        init_pos = np.array([self.relative_offset, 0.42, 0.04])

        # Move to init position offset
        if not self.init_move:
            line_dist = np.linalg.norm(pos_curr[:2] - init_pos[:2])
            high_dist = abs(pos_curr[2] - init_pos[2])
            # print(f"line_dist: {line_dist}, high_dist: {high_dist}")
            if line_dist > 0.01 or high_dist > 0.01:
                return init_pos
            else:
                self.init_move = True
                # print(f"Init move done! pos_goal: {pos_goal}, puck_pos: {pos_puck}")

        small_step = 0.03
        direction = (pos_goal - pos_curr) / np.linalg.norm(pos_goal - pos_curr)
        
        # print(f"pos_curr: {pos_curr}")

        return pos_curr + direction * small_step

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
