'''

state is according to position(-1.2--0.5) and velocity(-0.07----0.07)

when the car reaches postition 0.5, the episode ends
when the car reaches position -1.2, the velocity was reset to 0
position(t+1) = bound[position(t) + velocity(t+1)]
velocity(t+1) = bound[velocity(t) + 0.001A(t) - 0.0025*cos(3*position(t))]

action : 1(full throttle forward), -1(full throttle backward) 0(zero throttle)
reward : -1

TileCoding is a method to get a X vector if u input the state and aciton

'''

import numpy as np
import TileCoding as tc
import numpy.random as rdm

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
iteration = 1000
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

#return a list of numbers, length is num_tiling,one for each
def get_tiles(p,v,a):
    return tc.tiles(iht,num_tiling,[tile_factor_position*p,tile_factor_velocity*v],[a]) #a list containing 8 numbers for every tiling


def get_next_position_velocity(p,v,a,flag):
    #if the car reaches the left point, the velocity is reset to 0
    if (p == min_position) and (flag == 0):
        v_next = 0
        flag = 1
    else:
        v_next = max(min(v+0.001*a-0.0025*np.cos(3*p),max_velocity),min_velocity)
        flag = 0
    p_next = max(min(p+v_next,max_position),min_position)
    return p_next,v_next,flag

def q_value(p,v,a):
    # return a list of numbers, length is num_tiling,one for each
    tiles= get_tiles(p,v,a)
    #print('tiles info:(p,v,a,tiles)',p,v,a,tiles)
    if p == max_position:
        return 0
    else:
        #tiles is a numpy array, so it calculate the sum of the elements in the theta list
        #q = theta*x , x is 1 in the index points, 0 in other places, so we only need to consider the index points

        return np.sum(theta[tiles])

#we have a theta parameter to store information, so we do not have to store q all the time
def epsilon_greedy(p,v):
    random_number = rdm.random_sample()
    if (random_number < epsilon) or (q_value(p,v,-1)== q_value(p,v,0)==q_value(p,v,1)) :  # randomly pick one action
        '''
        print('pick action:',action_list[rdm.randint(0, 3)])
        print([q_value(p,v,-1),q_value(p,v,0),q_value(p,v,1)])
        '''
        return action_list[rdm.randint(0, 3)]
    else:
        q_list = [q_value(p,v,-1),q_value(p,v,0),q_value(p,v,1)]
        '''
        print(q_list)
        print(len(q_list))
        print('pick action:',action_list[q_list.index(max(q_list))])
        '''
        return action_list[q_list.index(max(q_list))] # pick the biggest

def initialization():
    return rdm.uniform(-0.6,-0.4), 0

for it in range(iteration):
    steps = 0 # store the total steps per episode
    position,velocity = initialization()
    action = epsilon_greedy(position, velocity)
    while(True):
        steps +=1
        #print(position,velocity)
        next_position,next_velocity,flag = get_next_position_velocity(position,velocity,action,flag)
        td_error_terminal = reward - q_value(position,velocity,action)
        if next_position==0.5 :#if next state is terminal, end this episode
            for element in get_tiles(position,velocity,action): #gradient(q over theta) is X, X is 1 only for the active tiles
                theta[element] += alpha*td_error_terminal
            break
        else:
            next_action = epsilon_greedy(next_position,next_velocity)
            td_error = reward + discount_factor * q_value(next_position,next_velocity,next_action)-q_value(position,velocity,action)
            for element in get_tiles(position, velocity,action):  # update every terminal action q
                theta[element] += alpha * td_error
            action = next_action
            position = next_position
            velocity = next_velocity
    print('episode',it)
    print('steps',steps)


#this matrix stores the minus max of q over three actions for every position velocity comb
max_q_matrix = np.zeros((position_num,velocity_num))

for it in range(max_q_matrix.shape[0]):
    for it2 in range(max_q_matrix.shape[1]):
        max_q_matrix[it,it2] = -max(q_value(min_position+it*position_interval,\
                                            min_velocity+it2*velocity_interval,\
                                            action_list[0]), \
                                    q_value(min_position + it * position_interval, \
                                            min_velocity + it2 * velocity_interval, \
                                            action_list[1]),\
                                    q_value(min_position+it*position_interval,\
                                            min_velocity+it2*velocity_interval,\
                                            action_list[2])\
                                    )


print(max_q_matrix)


import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

X = np.linspace(min_position,max_position,num=position_num,endpoint=True)
Y = np.linspace(min_velocity,max_velocity,num=velocity_num,endpoint=True)
X, Y = np.meshgrid(X, Y)
Z = np.transpose(max_q_matrix)

plt.title('-max q over a')
plt.xlabel('position')
plt.ylabel('velocity')

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show()
