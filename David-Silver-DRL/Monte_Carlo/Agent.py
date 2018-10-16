import numpy as np
from Environment import Environment
import random


class Agent (object):
    def __init__(self,lose_reward,draw_reward,win_reward):
        self.lose_reward = lose_reward
        self.draw_reward = draw_reward
        self.win_reward = win_reward

    def run_player_policy(self,env, q_value,state_label,sequence_state_action,player_list,player_sum,dealer_showing_card,ace_usable):
        dealer_terminate = 0
        player_terminate = 0
        bust = False
        reward = self.draw_reward

        if q_value[state_label - 1][0] == q_value[state_label - 1][1]:  # if equal , pick either one randomly
            action = random.randint(0, 1)
            sequence_state_action.append([state_label, action])  # store state action pair

            if action == 0:
                player_terminate = 1
            else:  # action == 1, get a new card
                player_list.append(env.get_card())
                player_sum += player_list[-1]
                if player_sum > 21:
                    reward = self.lose_reward
                    player_terminate = 1
                    dealer_terminate = 1  # if player bust, dealer does not have to get card
                    bust = True
                else:  # twist less than 21, state changes to s'
                    state_label = env.get_state_label(player_sum, dealer_showing_card, ace_usable)

        else:  # if not equal, pick the bigger one
            if q_value[state_label - 1][0] > q_value[state_label - 1][1]:
                action = 0
                player_terminate = 1
                sequence_state_action.append([state_label, action])
            else:
                action = 1
                player_list.append(env.get_card())
                player_sum += player_list[-1]
                sequence_state_action.append([state_label, action])
                if player_sum > 21:
                    reward = self.lose_reward
                    player_terminate = 1
                    dealer_terminate = 1  # if player bust, dealer does not have to get card
                    bust = True
                else:  # twist less than 21, state changes to s'
                    state_label = env.get_state_label(player_sum, dealer_showing_card, ace_usable)

        return player_terminate, dealer_terminate, bust, reward, q_value, state_label,\
        sequence_state_action,player_list, player_sum, dealer_showing_card, ace_usable



    def dealer_policy(self,dealer_sum):
        # dealer fixed policy : he sticks on any sum of 17 or greater, and hits otherwise.
        if dealer_sum > 16:
            return 0
        else:
            return 1