from __future__ import division

import argparse

from jan_ken_pon import JanKenPon
from dqn_agent import DQNAgent


def game(step):
    global win, lose
    global state_t_1, reward_t, terminal

    if terminal:
        env.reset()

        # for log
        if reward_t == 1:
            print('WIN')
            win += 1
        elif reward_t <= -1:
            print('LOSE')
            lose += 1
        else:
            print('DRAW')

        # print("WIN: {:03d}/{:03d} ({:.1f}%)".format(win, win + lose, 100 * win / (win + lose)))

    else:
        state_t = state_t_1

        # execute action in environment
        action_t = agent.select_action(state_t, 0.0)
        env.execute_action(action_t)
        print('Opponent=%d(%s), Player=%d(%s)' % (env.opponent_hand, env.render_hand_shape(env.opponent_hand), env.player_hand, env.render_hand_shape(env.player_hand)))

    # observe environment
    state_t_1, reward_t, terminal = env.observe()

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", default='models/jan_ken_pon.npz')
    args = parser.parse_args()

    # environmet, agent
    env = JanKenPon()
    agent = DQNAgent(env.enable_actions, env.name)
    agent.load_model(args.model_path)

    # variables
    win, lose = 0, 0
    state_t_1, reward_t, terminal = env.observe()

    for i in range(20):
        game(i)
