from Agent import Agent
from Environment import Environment
import numpy as np
from plot import Plot
import random

'''
Here we only consider unusable Ace situation where all the Ace is treated as 1
'''

#a list to store q(s,a) 200*2
NUM_STATES = 200
NUM_ACTIONS = 2
q_value = [[0 for x in range(NUM_ACTIONS)] for y in range(NUM_STATES)]
accumulated_reward = [[0 for x in range(NUM_ACTIONS)] for y in range(NUM_STATES)]
#a list to store counting numbers for every state action pair
count_list = [[0 for x in range(NUM_ACTIONS)] for y in range(NUM_STATES)]
ace_usable = 0
discount_factor = 1
# if sum < this num, we automatically twist
AUTO_TWIST = 12
#if sum > this bound, bust and lose
BUST_UPBOUND = 21
ITERATION = 500000
WIN_REWARD = 1
LOSE_REWARD = -1
DRAW_REWARD = 0
TOTAL_CARD_CHOICE = 10


agent = Agent(LOSE_REWARD,DRAW_REWARD,WIN_REWARD)
env = Environment(LOSE_REWARD,DRAW_REWARD,WIN_REWARD)

for iteration in range(ITERATION):

    # new round initializes
    bust, sequence_state_action, player_terminate, dealer_terminate, \
    player_list, dealer_list, player_sum, dealer_sum, dealer_showing_card, \
    state_label, reward = env.start_new_round(ace_usable)

    # if <12 or natural 21 points, we automatically twist
    # natural 21 points, it can only happen when ace is usable
    if (player_sum < AUTO_TWIST) or (player_sum==BUST_UPBOUND) or (dealer_sum==BUST_UPBOUND):
        continue

    #the initial policy is also choose the biggest q
    #if q is the same, then act randomly

    while(player_terminate==0):


        player_terminate, dealer_terminate, bust, reward, q_value, state_label,\
        sequence_state_action,player_list, player_sum, dealer_showing_card, ace_usable = \
        agent.run_player_policy(env, q_value, state_label, sequence_state_action,
                            player_list, player_sum, dealer_showing_card, ace_usable)


    while(dealer_terminate==0):
        action = agent.dealer_policy(dealer_sum)
        if action == 0:
            dealer_terminate =1
        if action == 1:
            dealer_list.append(env.get_card())
            dealer_sum += dealer_list[-1]
            if dealer_sum>BUST_UPBOUND:
                dealer_terminate =1
                reward = WIN_REWARD
                bust = True

    #if dealer and player both < 21, no bust, then reward need to calculate accoding to cards sum
    if (not bust):
        reward = env.get_reward_for_not_bust(dealer_sum,player_sum)

    #according to sequence state action value , update the q value
    #each element is a state action pair

    accumulated_reward,count_list,q_value = env.update_q_value(sequence_state_action, reward,
                                    accumulated_reward,discount_factor, count_list, q_value)


# plot section
plot = Plot(ace_usable,q_value,count_list,AUTO_TWIST,BUST_UPBOUND,TOTAL_CARD_CHOICE)
#policy figures
plot.plot_figure1_2()
#value function figure
plot.plot_figure3()
plot.show_figure()

