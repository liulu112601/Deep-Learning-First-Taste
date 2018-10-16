'''

state is according to position(-1.2--0.5) and velocity(-0.07----0.07)

when the car reaches postition 0.5, the episode ends
when the car reaches position -1.2, the velocity was reset to 0
position(t+1) = bound[position(t) + velocity(t+1)]
velocity(t+1) = bound[velocity(t) + 0.001A(t) - 0.0025*cos(3*position(t))]

action : 1(full throttle forward), -1(full throttle backward) 0(zero throttle)
reward : -1

TileCoding is a method to get a X vector if u input the state and action

'''

import numpy as np
import numpy.random as rdm

import TileCoding as tc
from Agent import Agent
from Environment import Environment
#some parameters for mountain car enironment

max_velocity = 0.07
min_velocity = -0.07
max_position = 0.5
min_position = -1.2
reward = -1
action_list = [-1,0,1]#reverse,zero,forward

#some parameters for tile coding


#the index for each tiling should between 0 and max_size
max_size = 2048
# the first time to use iht, the distribution will be reset, so should put this ahead
iht = tc.IHT(max_size)
#each tilng will have one non-zero number, indicate the location of the tiling
num_tiling = 8
tile_factor_velocity = 8/(0.07-(-0.07))
tile_factor_position = 8 / (0.5 - (-1.2))
theta = np.zeros(max_size)# store parameter

#some parameters for training
iteration = 104
epsilon = 0.1
#alpha = 1/(10*num_tiling)
alpha = 0.5/8
discount_factor = 0.9
flag = 0#flag =1 when the current position is the left top

#plot parameters
position_interval = 0.1
velocity_interval = 0.01
position_num = 1+int(((max_position-min_position)/position_interval))
velocity_num = 1+int(((max_velocity-min_velocity)/velocity_interval))


def plot():
    # this matrix stores the minus max of q over three actions for every position velocity comb
    max_q_matrix = np.zeros((position_num, velocity_num))

    for it in range(max_q_matrix.shape[0]):
        for it2 in range(max_q_matrix.shape[1]):
            max_q_matrix[it, it2] = -max(environment.get_q_value(min_position + it * position_interval, \
                                                                 min_velocity + it2 * velocity_interval, \
                                                                 action_list[0]), \
                                         environment.get_q_value(min_position + it * position_interval, \
                                                                 min_velocity + it2 * velocity_interval, \
                                                                 action_list[1]), \
                                         environment.get_q_value(min_position + it * position_interval, \
                                                                 min_velocity + it2 * velocity_interval, \
                                                                 action_list[2]) \
                                         )

    print(max_q_matrix)

    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    X = np.linspace(min_position, max_position, num=position_num, endpoint=True)
    Y = np.linspace(min_velocity, max_velocity, num=velocity_num, endpoint=True)
    X, Y = np.meshgrid(X, Y)
    Z = np.transpose(max_q_matrix)

    plt.title('-max q over a')
    plt.xlabel('position')
    plt.ylabel('velocity')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.show()


agent = Agent()
environment = Environment(iht,num_tiling,tile_factor_position,tile_factor_velocity,min_position,max_velocity,min_velocity,max_position,theta)



for it in range(iteration):
    steps = 0 # store the total steps per episode
    position,velocity = environment.initialization()
    action = agent.epsilon_greedy(position, velocity,epsilon,environment)
    while(True):
        steps +=1
        #environment transition
        next_position,next_velocity,flag = environment.transition(position,velocity,action,flag)
        #the td error for the terminal
        td_error_terminal = reward - environment.get_q_value(position,velocity,action)
        if next_position== max_position:#if next state is terminal, end this episode
            for element in environment.get_tiles(position,velocity,action): #gradient(q over theta) is X, X is 1 only for the active tiles
                theta[element] += alpha*td_error_terminal
            break
        else:
            next_action = agent.epsilon_greedy(next_position,next_velocity,epsilon,environment)
            td_error = reward + discount_factor * environment.get_q_value(next_position,next_velocity,next_action)-environment.get_q_value(position,velocity,action)
            for element in environment.get_tiles(position, velocity,action):  # update every terminal action q
                theta[element] += alpha * td_error
            action = next_action
            position = next_position
            velocity = next_velocity
    print('episode',it)
    print('steps',steps)

plot()
