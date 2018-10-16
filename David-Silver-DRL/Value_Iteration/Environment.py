import numpy as np

class Environment (object):

    def __init__(self,total_lines,total_cols,total_actions,imm_reward,discount_factor):
        self.total_lines = total_lines
        self.total_cols = total_cols
        self.total_actions = total_actions
        self.imm_reward = imm_reward
        self.discount_factor = discount_factor

    #how to transit from (current state)current line and column to (next state) the next line and column based on the given action
    #in this example, the policy is a deterministic policy
    #input: action
    #output: next line and col (next state)
    def transition(self,line,col,action):
        self.col = col
        self.line = line
        self.action = action
        #if the current state is on the edge and the agent still wants to go towards the edge,then the agent stays the same state
        if (self.line==0 and self.action == 1) or (self.line==self.total_lines-1 and self.action == 3) or (self.col==0 and self.action == 0) or (self.col==self.total_cols-1 and self.action == 2):
            return [self.line, self.col]
        #if the agent is not on the edge
        else:
            #go left
            if self.action == 0:
                self.col += -1
                return [self.line,self.col]
            #go up
            elif self.action == 1:
                self.line += -1
                return [self.line,self.col]
            #go right
            elif self.action == 2:
                self.col += 1
                return [self.line,self.col]
            #go down
            else:
                self.line += 1
                return [self.line,self.col]

    def update_value_net(self,value_net):

        self.value_net = value_net
        new_value_net = np.zeros((self.total_lines,self.total_cols))

        for line in range(self.total_lines):
            for col in range(self.total_cols):
                left = self.transition(line, col, 0)
                up = self.transition(line, col, 1)
                right = self.transition(line, col, 2)
                down = self.transition(line, col, 3)
                #choose the maximum value of all the possible next four states
                new_value_net[line,col] = max((self.imm_reward + self.discount_factor*self.value_net[left[0],left[1]]),\
                                          (self.imm_reward + self.discount_factor*self.value_net[up[0], up[1]]),\
                                          (self.imm_reward + self.discount_factor*self.value_net[right[0], right[1]]),\
                                          (self.imm_reward + self.discount_factor*self.value_net[down[0], down[1]]))
        #terminal are zero
        new_value_net[0,0] = 0

        return new_value_net
