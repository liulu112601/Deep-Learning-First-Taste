import tensorflow as tf
import numpy as np
import gym

from Actor import Actor
from Critic import Critic


#learning rate
lr_critic = 0.05
lr_actor = 0.01
reward = -1
discount_factor = 0.9
#position and velocity
n_features = 2
#0,1,2
n_actions = 3
n_episode = 1000

if __name__ == '__main__':
    #build environment using openai gym
    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    sess = tf.Session()
    #create an actor and critic
    actor = Actor(sess,n_actions=n_actions,n_features=n_features,lr=lr_actor)
    critic = Critic(sess,n_features=n_features,lr=lr_critic)
    #build the two networks
    actor.build_net()
    critic.build_net()

    sess.run(tf.global_variables_initializer())

    #tf.summary.FileWriter("",sess.graph)
    #count steps
    step = 0
    #env.render()
    for episode in range(n_episode):
        s = env.reset()
        #comment the render() to speed up
        #env.render()
        #s returned by gym is a vector, we need to transform it into a matrix
        s = s[np.newaxis, :]
        a = actor.choose_action(s)
        while(True):
            step += 1
            #a new transition
            s_, r, done, info = env.step(a)
            #in order to let s_ add one rank(matrix)
            s_ = s_[np.newaxis,:]
            a_ = actor.choose_action(s_)
            #calculate td_error
            td_error = critic.learn(s,s_)
            actor.learn(s,a,td_error)
            s =s_

            if step%500 == 0:
                print(step,s_)

            if done:
                print('arrive')
                print(s_)
                break
