from BattletoadsEnv import BattletoadsEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from operator import itemgetter

# useful later for subprocesses
def make_env(env_conf):

    def _init():
        gb_path, debug, init_state, act_freq, headless = itemgetter('gb_path', 'debug', 'init_state', 'act_freq', 'headless')(env_conf)
        env = BattletoadsEnv(gb_path, debug, init_state, act_freq, headless)
        env.reset()

    return _init

def dummy_env(env_conf):
        gb_path, debug, init_state, act_freq, headless = itemgetter('gb_path', 'debug', 'init_state', 'act_freq', 'headless')(env_conf)
        return BattletoadsEnv(gb_path, debug, init_state, act_freq, headless)    


if __name__ == '__main__':

    env_config = {
        'gb_path': '../Battletoads.gb',
        'debug': False,
        'init_state': '../init.state',
        'act_freq': 24,
        'headless': False
    }

    env = dummy_env(env_config)

    # check_env(env)

    model = PPO("MlpPolicy", env).learn(total_timesteps=1)

    # play loop
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
