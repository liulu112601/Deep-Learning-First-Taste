import numpy as np
from Environment import Environment

imm_reward = -1

class Agent (object):

    def __init__(self,total_lines,total_cols,total_actions,imm_reward,discount_factor):
        self.total_lines = total_lines
        self.total_cols = total_cols
        self.total_actions = total_actions
        self.imm_reward = imm_reward
        self.discount_factor = discount_factor
        #initialize a three dim array for action possibility for every state
        #policy net
        self.policy_net = np.full((total_lines,total_cols,total_actions),1/total_actions)

    def update_policy_net(self,value_net):
        self.policy_net = np.zeros((self.total_lines,self.total_cols,self.total_actions))
        env = Environment(self.total_lines, self.total_cols, self.total_actions,self.imm_reward,self.discount_factor)
        env.value_net = value_net
        for line in range(self.total_lines):
            for col in range(self.total_cols):
                zero = env.transition(line, col, 0)
                one = env.transition(line, col, 1)
                two = env.transition(line,col,2)
                three = env.transition(line, col, 3)
                next_four_states = np.array([env.get_value(zero[0],zero[1]),env.get_value(one[0],one[1]),\
                             env.get_value(two[0],two[1]),env.get_value(three[0],three[1])])
                #find the maximum value index, return a scalar
                max_index = np.where(next_four_states == max(next_four_states))[0]
                #update the action possibility for every grid
                for element in max_index:
                    self.policy_net[line,col,element] = 1/np.size(max_index)
        return self.policy_net
