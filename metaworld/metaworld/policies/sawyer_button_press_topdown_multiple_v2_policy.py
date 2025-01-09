import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerButtonPressTopdownMultipleV2Policy(Policy):
    def __init__(self, offset=0, isHorizontal=False):
        self.at_top = False
        self.offset = offset
        self.isHorizontal = isHorizontal
        super().__init__()

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'hand_closed': obs[3],
            'button_pos': obs[4:7],
            'unused_info': obs[7:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=25.)
        action['grab_effort'] = 1.

        return action.array
    
    def reset_policy(self, offset=None, isHorizontal=None):
        self.at_top = False
        if offset is not None:
            self.offset = offset
        if isHorizontal is not None:
            self.isHorizontal = isHorizontal
        else:
            self.isHorizontal = False

    def _desired_pos(self, o_d):
        pos_curr = o_d['hand_pos']

        # print('pos_curr:', pos_curr)
        # print('button_pos:', o_d['button_pos'])

        top_targ = np.array([self.offset, 0.6, 0.15]) if self.isHorizontal else np.array([self.offset, 0.7, 0.25])
        bot_targ = np.array([self.offset, 0.8, 0.15]) if self.isHorizontal else np.array([self.offset, 0.7, 0.05]) 

        if np.linalg.norm(pos_curr - top_targ) < 0.01:
            self.at_top = True
        
        if not self.at_top:
            return top_targ
        else:
            return bot_targ

