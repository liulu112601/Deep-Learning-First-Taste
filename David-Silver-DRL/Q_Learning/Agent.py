import numpy as np
from Environment import Environment
import numpy.random as rdm

imm_reward = -1

class Agent (object):
    def __init__(self,total_actions):
        self.total_actions = total_actions

    # return the action 0/1/2/3 using epsilon greedy method
    def epsilon_greedy(self,line,col,q_net,epsilon):

        next_q = np.zeros((self.total_actions))
        for ele in range(self.total_actions):
            next_q[ele] = q_net[line, col, ele]
        # how many are the max
        max_index = np.where(next_q == max(next_q))[0]
        random_number = rdm.random_sample()
        #if all the values are the same, it is randomly select
        if (random_number < epsilon) or (np.size(max_index) == self.total_actions):  # randomly pick one action

            return rdm.randint(0, self.total_actions)
        else:

            #if there is only one max
            if np.size(max_index) == 1:
                return np.argmax(next_q) #pick the max
            #if there are two or above max,randomly choose from them
            else:
                return int(next_q[max_index[rdm.choice(np.size(max_index),1)]][0])

