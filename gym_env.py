import gymnasium as gym
from gymnasium import spaces
import torch
import guerrilla_checkers
import random

class gym_env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "curses"], "render_fps": 4}

    def __init__(self, passed_game, player, render_mode=None):
        self.game = passed_game
        self.player = player
        # https://gymnasium.farama.org/api/spaces/
        self.observation_space = spaces.Discrete(82)
        # Limiting the action space for COIN may give a small advantage to performance
        if player == 0:
            # Is this aligned the right way?
            self.action_space = spaces.MultiDiscrete([8,4,8,4])
        else:
            self.action_space = spaces.MultiDiscrete([7,7,7,7])

        # In future, I may want to limit action space to possible moves
        # To make sure the moves will be in the correct order, use the order saved in rules.pickle
        # rules["all guerrilla moves"]
        # rules["all COIN moves"]
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        observation, player = self.game.get_current_state()
        return observation, player

    def _get_info(self):
        return guerrilla_checkers.decompress_board(self.game.board)
    
    def reset(self):
        observation, player  = self._get_obs()
        info = self.game.reset()
        if self.render_mode == "human":
            self._render_frame()
        return observation

    def get_acting_player(self):
        return int(self.game.guerrillas_turn)
    
    def get_valid_sample(self):
        # This will need to be modified if I want to train agents that can play either role
        acting_player = self.get_acting_player()
        valid_actions_dict = self.game.get_valid_actions(acting_player)
        # return value should be like tensor([[[2, 5, 4, 0]]]), but valid
        valid_actions_list = [k for k, v in valid_actions_dict.items() if v]
        if len (valid_actions_list) < 1:
            breakpoint()
        sample = random.choice(valid_actions_list)
        sample = list(sample)
        return sample

    def step(self, action, acting_player):
        # TODO: Catch invalid actions?
        board, reward, terminated = self.game.take_action(acting_player, tuple(action))
        if not terminated:
            reward = self.game.get_small_reward(acting_player)
        observation, acting_player = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info
    
    def _step(self, tensordict):
        # TODO get action frpm tensordict?
        pass
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        pass

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()