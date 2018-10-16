import numpy.random as rdm
#a list to store q(s,a) ,E(s,a)
q_value = [[[0 for x in range(4)] for y in range(12)] for z in range(4)]
accumulated_reward_list = [0 for x in range(500)]
state_action_trace_list = []
cliff_reward = -100
step_reward = -1
epsilon = 0.05
gamma = 0.9
alpha = 0.1 # update step
lamda = 0.9


#action 0-3 : left up right down
#return the next state, according to the current state and action

def state_transition_reward (line,column,action):
    if (line==0 and action == 1) or (column ==0 and action == 0) or (column ==11 and action == 2) or (line == 3 and column == 0 and action == 3):
        return line,column,step_reward
    elif line ==3 and column < 11 and column > 0:
        return 3,0,cliff_reward

    else:
        if action == 0:
            return line,column-1,step_reward
        elif action == 1:
            return line-1,column,step_reward
        elif action == 2:
            return line,column+1,step_reward
        else:
            return line+1,column,step_reward

#return the action 0/1/2/3 and the chosen q(s,a)
#input is a q list of a state
def epsilon_greedy(q):
    random_number = rdm.random_sample()
    if (random_number < epsilon) or (q[0]==q[1]==q[2]==q[3]): #randomly pick one action
        return rdm.randint(0, 4),q[0]
    else:
        return q.index(max(q)),max(q)

def epsilon_greedy_decay(q,iteration):
    random_number = rdm.random_sample()
    if (random_number < epsilon*(0.6**iteration)) or (q[0]==q[1]==q[2]==q[3]): #randomly pick one action
        return rdm.randint(0, 4),q[0]
    else:
        return q.index(max(q)),max(q)


for it_for_av in range(10):
    for it in range(500):
        # initial state and action
        e_trace = [[[0 for x in range(4)] for y in range(12)] for z in range(4)]
        #print('e trace dimen', len(e_trace[0]))
        #print('node1------------')
        terminate = 0
        accumulated_reward = 0
        state_action_trace = []
        #state_action_trace is like [[1,2,0],[2,3,1],....,[3,3,2]]
        #state = [3, 0]
        line = 3
        column = 0
        action = 1
        state_action_trace.append([line,column,action])
        q = q_value[line][column][action]
        next_line,next_column,reward = state_transition_reward(line,column,action)
        accumulated_reward += reward
        #print('node2---------------')
        while(True):
            next_action,next_q = epsilon_greedy(q_value[next_line][next_column])
            #next_action, next_q = epsilon_greedy_decay(q_value[next_line][next_column],it)
            td_error = reward + gamma*next_q - q
            e_trace[line][column][action] +=1
            #print('node3------------------')
            for element in state_action_trace:
                q_value[element[0]][element[1]][element[2]] += alpha*td_error*e_trace[element[0]][element[1]][element[2]]
                e_trace[element[0]][element[1]][element[2]] = e_trace[element[0]][element[1]][element[2]]*gamma*lamda
                #print('node4-----------------------')

            #transition
            state_action_trace.append([next_line, next_column, next_action])
            line = next_line
            column = next_column
            action = next_action
            q = next_q
            next_line, next_column, reward = state_transition_reward(line, column, action)
            accumulated_reward += reward
            if line == 3 and column == 11:
                break
            #print('node5------------------------')

        accumulated_reward_list[it] += accumulated_reward
        state_action_trace_list.append(state_action_trace)


        print('iteration',it)
        print('state action trace',state_action_trace)
        print('accumulated reward',accumulated_reward)

    print('--')

a = [n for n in range(1,501)]
av_accumulated_reward_list = list(map(lambda i:i/5,accumulated_reward_list))

print(av_accumulated_reward_list)

import matplotlib.pyplot as plt
plt.scatter(a,av_accumulated_reward_list)
plt.title('epsilon = 0.05,gamma = 0.9,alpha = 0.1,lamda = 0.9,5*500 episodes')
plt.xlabel('episode')
plt.ylabel('average reward per episode')

plt.show()


