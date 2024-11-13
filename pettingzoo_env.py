import functools
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from gymnasium import spaces
import guerrilla_checkers
import copy
import random

class PettingZoo(AECEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, render_mode=None):
        self.possible_agents = ["COIN", "Guerrilla"]
        self._action_spaces = {
            "COIN" : spaces.MultiDiscrete([8,4,8,4]),
            "Guerrilla" : spaces.MultiDiscrete([7,7,7,7]),
            }
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self._observation_spaces = {
            agent: spaces.Discrete(82) for agent in self.possible_agents
        }
        self.render_mode = render_mode

        # I'm creating the game object in the env instead of passing it. I think this will be more efficient
        self.game = guerrilla_checkers.game()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Discrete(82)
    
    def _get_obs(self):
        observation, player = self.game.get_current_state()
        return observation, player

    def _get_info(self):
        return guerrilla_checkers.decompress_board(self.game.board)  
      
    def observe(self, agent):
        observation, acting_player = self.game.get_current_state()
        if agent == acting_player:
            return observation
        else:
            print("Wrong agent!")

    def close(self):
        pass

    def reset(self, seed=None, options=None):

        # Reset game object
        starting_observation = self.game.reset()

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: starting_observation for agent in self.agents}
        self.observations = {agent: starting_observation for agent in self.agents}

        """
        PettingZoo agent_selector utility allows easy cyclic stepping through the agents list.
        """
        # I don't think I'll use it unless can find out how to override it
        # self._agent_selector = agent_selector(self.agents)
        # self.agent_selection = self._agent_selector.next()
        self.agent_selection = int(self.game.guerrillas_turn)

        # Not sure if it's useful to return anything here
        return starting_observation

    def step(self, action, acting_player):
        # TODO: Catch invalid actions?
        # acting_player = self.agent_selection
        board, reward, terminated = self.game.take_action(acting_player, tuple(action))
        if not terminated:
            reward = self.game.get_small_reward(acting_player)
        observation, acting_player = self._get_obs()
        info = self._get_info()
        self.agent_selection = int(self.game.guerrillas_turn)
        return observation, reward, terminated, False, info
    
    def get_valid_sample(self):
        # This will need to be modified if I want to train agents that can play either role
        acting_player = self.agent_selection
        valid_actions_dict = self.game.get_valid_actions(acting_player)
        # return value should be like tensor([[[2, 5, 4, 0]]]), but valid
        valid_actions_list = [k for k, v in valid_actions_dict.items() if v]
        if len (valid_actions_list) < 1:
            breakpoint()
        sample = random.choice(valid_actions_list)
        sample = list(sample)
        return sample
    
    def get_acting_player(self):
        return int(self.game.guerrillas_turn)
