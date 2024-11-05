import numpy as np
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from gymnasium import Env, spaces

import memory_addr as ma

# hacky
matrix_shape = (16, 20)
game_area_observation_space = spaces.Box(low=0, high=255, shape=matrix_shape, dtype=np.uint8)


class BattletoadsEnv(Env):
    
    def __init__(self, gb_path, init_state, debug = False, act_freq = 24, headless=True):
        super().__init__()

        self.gb_path = gb_path
        self.debug = debug
        self.init_state = init_state
        self.act_freq = act_freq
        self.headless = headless

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START
        ]

        actions = [
            '','a', 'b', 'left', 'right', 'up', 'down', 'start', 'select'
        ]

        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = game_area_observation_space

        self.pyboy = PyBoy(
            gb_path,
            debug=self.debug,
            no_input=False,
            window_type="headless" if self.headless else "SDL2",
        )

        if not self.headless:
            self.pyboy.set_emulation_speed(6)
        elif not self.debug:
            self.pyboy.set_emulation_speed(0)

    def reset(self, seed):
        # todo -> get suitable init_state post credit screen
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.lives_lost = 0
        self.health_lost = 0
        self.continues_lost = 0
        self.curr_score = 0 # fitness
        # see https://datacrystal.tcrf.net/wiki/Battletoads_in_Ragnarok%27s_World/Notes#Internal_Level_IDs for internal level ids
        self.explore_screens = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # consts
        self._starting_lives = self.read_m(ma.START_LIVES_CONST_ADDR)
        self._starting_health = self.read_m(ma.START_HP_CONST_ADDR)
        self._starting_continues = self.read_m(ma.START_CONTINUES_CONST_ADDR)

        self.curr_lives = self._starting_lives
        self.curr_health = self._starting_health
        self.curr_continues = self._starting_continues
        

        return self._get_obs(), self._get_info()

    def _get_obs(self):
        
        observation=self.pyboy.game_area()

        return observation

    def _get_info(self):
        # todo if necessary
        return {}

    # from https://github.com/PWhiddy/PokemonRedExperiments/blob/master/baselines/red_gym_env_v3_minimal.py#L127
    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        if self.headless:
            self.pyboy._rendering(False)
        for i in range(self.act_freq):
            # release action, so they are stateless
            if i == 8:
                # release button
                self.pyboy.send_input(self.release_actions[action])
            if i == self.act_freq - 1:
                # rendering must be enabled on the tick before frame is needed
                self.pyboy._rendering(True)
            self.pyboy.tick()

        
    def step(self, action):
        print("step")
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # actions before computing new reward
        self.run_action_on_emulator(self.valid_actions[action])
        
        self.update_explore_screens()
        self.update_deltas()
        self.update_score()

        new_reward = 0
        done = self.game_over();

        # do new reward calcs here
        new_reward += self.curr_score // 10000
        new_reward -= 0.5 * self.continues_lost
        new_reward -= 0.1 * self.lives_lost
        new_reward -= 0.0125 * self.health_lost

        for visited in self.explore_screens:
            new_reward += 5 * visited

        truncated = False

        
        return self._get_obs(), new_reward, done, truncated, self._get_info()

    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)

    # updates explore_screens if the screen we are currently on is different
    def update_explore_screens(self):
        cached = self.read_m(ma.CURR_LEVEL_ID_ADDR)        
        if (self.explore_screens[cached]):
            self.explore_screens[cached] = 1

    def update_deltas(self):
        step_continues = self.read_m(ma.CURR_CONTINUES_ADDR)
        step_lives = self.read_m(ma.CURR_LIVES_ADDR)
        step_health = self.read_m(ma.CURR_HP_ADDR)

        # lost continues
        if (step_continues < self.curr_continues):
            self.health_lost += self.curr_health * (self.curr_lives * (self.curr_continues - step_continues))
            self.lives_lost += self.curr_lives * (self.curr_continues - step_continues)
            self.continues_lost += self.curr_continues - step_continues
        # lost lives
        elif (step_lives < self.curr_lives):
            self.health_lost += (self.curr_health) * (self.curr_lives - step_lives)
            self.lives_lost += self.curr_lives - step_lives
        elif (step_health < self.curr_health):
            self.health_lost += self.curr_health - step_health

        # update curr values
        self.curr_continues = step_continues
        self.curr_lives = step_lives
        self.curr_health = step_health

    def update_score(self):
        step_score = sum([
            map(lambda idx, addr : np.power(2, 8 * idx) * self.read_m(addr)) 
            for (idx, addr) in enumerate(ma.SCORE_ADDRS)
        ])

        self.curr_score = step_score

    def game_over(self):
        if (not self.curr_continues and not self.curr_lives and not self.curr_health):
            return 1
        return 0
            

                
        
