import numpy as np
from Environment import Environment
import numpy.random as rdm

imm_reward = -1

class Agent (object):
    def __init__(self):
        self.action_list = [-1,0,1]

    # return the action 0/1/2/3 using epsilon greedy method
    def epsilon_greedy(self,p, v,epsilon,env):
        random_number = rdm.random_sample()
        if (random_number < epsilon) or (
                env.get_q_value(p, v, -1) == env.get_q_value(p, v, 0) == env.get_q_value(p, v, 1)):  # randomly pick one action
            '''
            print('pick action:',action_list[rdm.randint(0, 3)])
            print([q_value(p,v,-1),q_value(p,v,0),q_value(p,v,1)])
            '''
            return self.action_list[rdm.randint(0, 3)]
        else:
            q_list = [env.get_q_value(p, v, -1), env.get_q_value(p, v, 0), env.get_q_value(p, v, 1)]
            '''
            print(q_list)
            print(len(q_list))
            print('pick action:',action_list[q_list.index(max(q_list))])
            '''
            return self.action_list[q_list.index(max(q_list))]  # pick the biggest
