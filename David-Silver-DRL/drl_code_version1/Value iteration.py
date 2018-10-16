'''
state 1-15 and a terminal state
action : up, down, left, right for every state 1-15
'''

import numpy as np
state_value = [[0 for x in range(4)] for y in range(4)]
state_value_tmp = [[0 for x in range(4)] for y in range(4)]
action_possibility = [[[0.25 for x in range(4)] for y in range(4)] for z in range(4)]# possibililty follow :  left up right down
#print(action_possibility)
theta = 0.001 # when delta < theta, stop
discount_factor = 1
immediate_reward = -1

for iteration in range(6) :
    #state_value_tmp = state_value # store the old values
    for it in range(15):
        label = it +1
        if  (label == 1) or (label == 2):
            state_value[label//4][label%4] = max(immediate_reward + discount_factor*state_value_tmp[label//4][label%4-1],
                                                immediate_reward + discount_factor *state_value_tmp[label//4][label%4],
                                                immediate_reward + discount_factor *state_value_tmp[label//4][label%4+1],
                                                immediate_reward + discount_factor *state_value_tmp[label // 4 + 1][label % 4])
            #print('update label',label)

        if  (label == 3):
            state_value[label//4][label%4] = max(immediate_reward + discount_factor*state_value_tmp[label // 4][label % 4 - 1],\
                                                 immediate_reward + discount_factor*state_value_tmp[label // 4][label % 4], \
                                                 immediate_reward + discount_factor*state_value_tmp[label // 4 + 1][label % 4])
            #print('update label',label)

        if  (label == 4) or (label == 8):
            state_value[label//4][label%4]= max(immediate_reward + discount_factor*state_value_tmp[label // 4][label % 4], \
                                                immediate_reward + discount_factor*state_value_tmp[label // 4 - 1][label % 4], \
                                                immediate_reward + discount_factor*state_value_tmp[label // 4][label % 4 + 1], \
                                                immediate_reward + discount_factor*state_value_tmp[label // 4 + 1][label % 4])  # left up right down
            #print('update label',label)

        if  (label == 5) or (label == 6) or (label==9) or (label==10):
            state_value[label//4][label%4]= max(immediate_reward + discount_factor*state_value_tmp[label//4][label%4-1], \
                                                immediate_reward + discount_factor*state_value_tmp[label//4-1][label%4], \
                                                immediate_reward + discount_factor*state_value_tmp[label//4][label%4+1], \
                                                immediate_reward + discount_factor*state_value_tmp[label//4+1][label%4])   # left up right down
                #print('update label',label)

        if  (label == 7) or (label == 11):
            state_value[label//4][label%4] = max(immediate_reward + discount_factor*state_value_tmp[label//4][label%4-1], \
                                             immediate_reward + discount_factor*state_value_tmp[label//4-1][label%4], \
                                             immediate_reward + discount_factor*state_value_tmp[label//4][label%4], \
                                             immediate_reward + discount_factor*state_value_tmp[label//4+1][label%4])   # left up right down
            #print('update label',label)

        if  (label == 12) :
            state_value[label//4][label%4] = max(immediate_reward + discount_factor*state_value_tmp[label//4][label%4], \
                                             immediate_reward + discount_factor*state_value_tmp[label//4-1][label%4], \
                                             immediate_reward + discount_factor*state_value_tmp[label//4][label%4+1])# left up right down
            #print('update label',label)

        if  (label == 13) or (label == 14):
            state_value[label//4][label%4] = max(immediate_reward + discount_factor*state_value_tmp[label//4][label%4-1], \
                                             immediate_reward + discount_factor*state_value_tmp[label//4-1][label%4], \
                                             immediate_reward + discount_factor*state_value_tmp[label//4][label%4+1], \
                                             immediate_reward + discount_factor*state_value_tmp[label//4][label%4]) # left up right down
            #print('update label',label)

        if  (label == 15) :
            state_value[label//4][label%4] = max(immediate_reward + discount_factor*state_value_tmp[label//4][label%4-1], \
                                             immediate_reward + discount_factor*state_value_tmp[label//4-1][label%4], \
                                             immediate_reward + discount_factor*state_value_tmp[label//4][label%4]) # left up right down
            #print('update label',label)

    for it in range(4):
        for it2 in range(4):
            state_value_tmp[it][it2] = state_value[it][it2]


    print('it time:',iteration)
    print('state value function:',state_value)

