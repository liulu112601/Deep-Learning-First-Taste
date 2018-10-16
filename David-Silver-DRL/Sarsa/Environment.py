import numpy as np

class Environment (object):

    def __init__(self,total_lines,total_cols,step_reward,cliff_reward,done,eligibility_trace,discount_factor,lamda):
        self.total_lines = total_lines
        self.total_cols = total_cols
        self.step_reward = step_reward
        self.cliff_reward = cliff_reward
        self.done = done
        self.eligibility_trace = eligibility_trace
        self.discount_factor = discount_factor
        self.lamda = lamda


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

    def get_eligibility_trace(self,line,col,action):
        return self.eligibility_trace[line,col,action]


    def update_q_net_e_trace(self,alpha,td_error,state_action_trace,q_net):
        for element in state_action_trace:
            line, col, action = element[0],element[1],element[2]
            #print(line,col,action,td_error)
            q_net[line, col, action] += alpha * td_error * self.get_eligibility_trace(line,col,action)
            self.eligibility_trace[line, col, action] = \
                self.eligibility_trace[line, col, action] * self.discount_factor * self.lamda

        return q_net





