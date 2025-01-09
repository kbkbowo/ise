from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_faucet_close_v2 import SawyerFaucetCloseEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_faucet_open_v2 import SawyerFaucetOpenEnvV2

def SawyerFaucetIOEnvV2(type=0):
    if type == 0:
        env = SawyerFaucetOpenEnvV2()
    else:
        env = SawyerFaucetCloseEnvV2()
    return env