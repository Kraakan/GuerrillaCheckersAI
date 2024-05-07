from timeit import timeit
from gym_env import gym_env
import guerrilla_checkers
import random
import sys
import statistics
import datetime

episode_durations = []

def no_dqn_test(n_games):

    env = gym_env(guerrilla_checkers.game(), 1)
    n_actions = len(guerrilla_checkers.rules['all guerrilla moves'])
    action_list = list(guerrilla_checkers.rules['all guerrilla moves'].keys())
    state = env.reset()
    n_observations = len(state)

    for i_episode in range(n_games):
        state = env.reset()
        terminated = False
        if i_episode % 50 == 0:
            print("Running episode", i_episode+1)
        while not terminated:
            acting_player = env.get_acting_player()
            valid_action_indexes = env.game.get_valid_action_indexes(acting_player)
            if len(env.game.get_valid_action_indexes(acting_player)) < 1:
                terminated = True
                if acting_player == env.player:
                    reward = -1
                else:
                    reward = 1
            else:
                random_action = random.choice(valid_action_indexes)
                action = env.get_valid_sample()
                observation, reward, terminated, truncated, _ = env.step(action, acting_player)
            if terminated:
                reward = env.game.get_reward(env.player)
                episode_durations.append((66 - env.game.get_remaining_stones())/2)
                break
    print('Average game length', statistics.mean(episode_durations))
    return episode_durations

games_num = 100

if len(sys.argv) > 0:
    if  sys.argv[1] == 'dqn':
        exec_str = "no_dqn_test(100)"
        if len(sys.argv) > 1:
            games_num = sys.argv[2]
            if games_num.isdigit():
                exec_str = "no_dqn_test({})".format(games_num)
        complete_str = '''
from __main__ import no_dqn_test
''' + exec_str
        time = timeit(stmt=complete_str, number=1)
        print('Execution time:', str(time), 'seconds')

f = open("speedtests.txt", "a")
f.write('''
''' + sys.platform + " " +  str(datetime.datetime.now()) + '''
''' + str(games_num) + ' games played' + '''
''' + 'Average game length: ' + str(statistics.mean(episode_durations) ) + '''
''' + 'Execution time: ' + str(time) + 'seconds' + '''
''')
f.close()