import numpy as np
import random

class Environment (object):
    def __init__(self,lose_reward,draw_reward,win_reward):
        self.lose_reward = lose_reward
        self.draw_reward = draw_reward
        self.win_reward = win_reward

        '''
        Here we only consider unusable Ace situation where all the Ace is treated as 1
        '''

    def get_state_label(self,sum, dealer_card, ace):
        # sum: 12-21; dealer_card: 1-10; ace:0/1
        # 200 states in total, return : 1-200
        return 20 * (sum - 12) + 10 * ace + dealer_card

    def get_card(self):
        #get a new card
        #suppose the card is with replacement, so the probability of getting every card is always the same
        a = random.randint(1,13)
        if a<10:
            return a
        else:
            return 10

    def start_new_round(self,ace_usable):
        bust = False
        sequence_state_action = []
        player_terminate = 0
        dealer_terminate = 0
        # simulate an episode, in this case, ace is 1, unusable ace
        player_list = [self.get_card(), self.get_card()]
        dealer_list = [self.get_card(), self.get_card()]
        player_sum = player_list[0] + player_list[1]
        dealer_sum = dealer_list[0] + dealer_list[1]
        dealer_showing_card = dealer_list[1]
        state_label = self.get_state_label(player_sum, dealer_showing_card, ace_usable)
        reward = self.draw_reward

        return bust,sequence_state_action,player_terminate,dealer_terminate,player_list,dealer_list,player_sum,dealer_sum,dealer_showing_card,state_label,reward

    def get_reward_for_not_bust(self,dealer_sum,player_sum):
        if int(dealer_sum) > int(player_sum):
            reward = self.lose_reward
        elif int(dealer_sum) < int(player_sum):
            reward = self.win_reward
        else:
            reward = self.draw_reward
        return reward

    def update_q_value(self,sequence_state_action,reward,accumulated_reward,discount_factor,count_list,q_value):
        length = len(sequence_state_action)  # the number of pairs
        for element in sequence_state_action:
            accumulated_reward[element[0] - 1][element[1]] += reward * (discount_factor ** (length - 1))
            # for this eg, reward only occurs at the end, otherwise we need to record reward for every step
            count_list[element[0] - 1][element[1]] += 1
            # for this eg, each state only occurs once, otherwise we need to record times for every encountered state
            q_value[element[0] - 1][element[1]] = (accumulated_reward[element[0] - 1][element[1]]) / (
            count_list[element[0] - 1][element[1]])
            length -= 1

        return accumulated_reward, count_list, q_value