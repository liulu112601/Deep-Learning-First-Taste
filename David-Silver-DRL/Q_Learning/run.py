from Agent import Agent
from Environment import Environment
import numpy as np

TOTAL_LINES = 4
TOTAL_COLS = 12
TOTAL_ACTIONS = 4
START_LINE = 3
START_COL = 0
START_ACTION = 1
STEP_REWARD = -1
CLIFF_REWARD = -100
DISCOUNT_FACTOR = 0.9

ITERATION_AV = 5
ITERATION = 500
ALPHA = 0.1
EPSILON = 0.1
done = False
agent = Agent( TOTAL_ACTIONS)
environment = Environment(TOTAL_LINES,TOTAL_COLS,STEP_REWARD,CLIFF_REWARD,done)
accumulated_reward_list = np.zeros((ITERATION))
state_action_trace_list = []

def plot_reward():
    # x axis
    a = [n for n in range(1, ITERATION+1)]
    #y axis
    av_accumulated_reward_list = list(map(lambda i: i / ITERATION_AV, accumulated_reward_list))
    print(av_accumulated_reward_list)
    import matplotlib.pyplot as plt
    plt.scatter(a, av_accumulated_reward_list)
    plt.title('epsilon = 0.05,gamma = 0.9,alpha = 0.1,lamda = 0.9,5*500 episodes')
    plt.xlabel('episode')
    plt.ylabel('average reward per episode')

    plt.show()

#run
for it_for_av in range(ITERATION_AV):
    q_net = np.zeros((TOTAL_LINES,TOTAL_COLS,TOTAL_ACTIONS))
    for it in range(ITERATION):
        done = False
        # initial state and action
        accumulated_reward = 0
        state_action_trace = []
        #state_action_trace is like [[1,2,0],[2,3,1],....,[3,3,2]]
        line = START_LINE
        col = START_COL
        action = START_ACTION
        state_action_trace.append([line,col,action])

        while(not done):
            next_line, next_col, reward, done = environment.transition(line, col, action)
            #print(next_line,next_col,reward,done,action,q_net)
            #next_line,next_col,next_action,reward = observation[0],observation[1],observation[2],observation[3]

            accumulated_reward += reward
            q_net[line][col][action] = q_net[line][col][action] + \
                                         ALPHA * (reward+DISCOUNT_FACTOR*max(q_net[next_line][next_col])-q_net[line][col][action])

            #transition
            next_action = agent.epsilon_greedy(next_line,next_col,q_net,EPSILON)
            state_action_trace.append([next_line, next_col, next_action])
            line = next_line
            col = next_col
            action = next_action

        accumulated_reward_list[it] += accumulated_reward
        print('state action trace for iteration',it, state_action_trace)
        #print('accumulated reward', accumulated_reward)
        #print('q_net', q_net)

plot_reward()