import numpy as np

class Environment (object):

    def __init__(self,total_lines,total_cols,step_reward,cliff_reward,done):
        self.total_lines = total_lines
        self.total_cols = total_cols
        self.step_reward = step_reward
        self.cliff_reward = cliff_reward
        self.done = done

    # action 0-3 : left up right down
    # return the next state, according to the current state and action
    def transition(self,line, col, action):

        self.col = col
        self.line = line
        self.action = action

        #if reach the terminal
        if line == self.total_lines-2 and col == self.total_cols-1 and action == 3:
            self.done = True
        else:
            self.done = False


        #if the current state is on the edge and the agent still wants to go towards the edge,then the agent stays the same state
        if (self.line==0 and self.action == 1) or (self.col==0 and self.action == 0) or (self.col==self.total_cols-1 and self.action == 2) or (self.line==self.total_lines-1 and self.action == 3):
            return [self.line, self.col,self.step_reward,self.done]
        #if the agent is on the cliff, back to start
        elif (self.line==self.total_lines-1 and self.col != 0) :
            return [self.total_lines-1, 0, self.cliff_reward,self.done]
        #if the agent is not on the edge
        else:
            #go left
            if self.action == 0:
                self.col += -1
                return [self.line,self.col,self.step_reward,self.done]
            #go up
            elif self.action == 1:
                self.line += -1
                return [self.line,self.col,self.step_reward,self.done]
            #go right
            elif self.action == 2:
                self.col += 1
                return [self.line,self.col,self.step_reward,self.done]
            #go down
            else:
                self.line += 1
                return [self.line,self.col,self.step_reward,self.done]
