import tensorflow as tf
import numpy as np
import gym

from Actor import Actor
from Critic import Critic
import json
import sys
import argparse

#learning rate
lr_critic = 0.05
lr_actor = 0.01
reward = -1
discount_factor = 0.9
#position and velocity
n_features = 2
#0,1,2
n_actions = 3
n_episodes = 1000

def parse_args(args):
	parser = argparse.ArgumentParser()
	parser.add_argument('configfile', nargs=1, type=str, help='')
	parser.add_argument('-game', default='game2', type=str, help='')
	return parser.parse_args(args)

def parse(filename):
	configfile = open(filename)
	jsonconfig = json.load(configfile)
	configfile.close()
	return jsonconfig

def run():
    # build environment using openai gym
    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    sess = tf.Session()
    # create an actor and critic
    actor = Actor(sess, n_actions=n_actions, n_features=n_features, lr=lr_actor)
    critic = Critic(sess, n_features=n_features, lr=lr_critic)
    # build the two networks
    actor.build_net()
    critic.build_net()

    sess.run(tf.global_variables_initializer())

    # tf.summary.FileWriter("",sess.graph)
    # count steps
    step = 0
    # env.render()
    for episode in range(n_episodes):
        s = env.reset()
        # comment the render() to speed up
        # env.render()
        # s returned by gym is a vector, we need to transform it into a matrix
        s = s[np.newaxis, :]
        a = actor.choose_action(s)
        while (True):
            step += 1
            # a new transition
            s_, r, done, info = env.step(a)
            # in order to let s_ add one rank(matrix)
            s_ = s_[np.newaxis, :]
            a_ = actor.choose_action(s_)
            # calculate td_error
            td_error = critic.learn(s, s_)
            actor.learn(s, a, td_error)
            s = s_

            if step % 500 == 0:
                print(step, s_)

            if done:
                print('arrive')
                print(s_)
                break


if __name__ == '__main__':
    #parse parameters
    args = parse_args(sys.argv[1:])
    print("args:", args)
    config = parse(args.configfile[0])
    info = config[args.game]
    lr_critic,lr_actor,reward,discount_factor,n_features,n_actions,n_episodes = \
        info['lr_critic'],info['lr_actor'],info['reward'],info['discount_factor'],info['n_features'],info['n_actions'],info['n_episodes']
    print(lr_actor,lr_critic,reward,discount_factor,n_features,n_actions,n_episodes)

    run()