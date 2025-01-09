import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerLeverSideV2Policy(Policy):
    def __init__(self, l2r=False, **kwargs):
        self.l2r = l2r
        self.trigger = False
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

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=10.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    def _desired_pos(self, o_d):
        pos_curr = o_d['hand_pos']
        pos_puck = o_d['puck_pos']

        attack_point = np.array([0.1, 0.7, 0.4]) if self.l2r else np.array([-0.1, 0.7, 0.4])

        if not self.trigger:
            if np.linalg.norm(pos_curr - attack_point) < 0.05:
                self.trigger = True
            return attack_point

        return pos_puck + np.array([0., 0., 0.4])

    @staticmethod
    def _grab_effort(o_d):
        return 1.0

