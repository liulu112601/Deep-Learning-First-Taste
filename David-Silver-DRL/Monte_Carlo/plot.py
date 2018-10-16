from Agent import Agent
from Environment import Environment
import numpy as np

import random


# plot section

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

WIN_REWARD = 1
LOSE_REWARD = -1
DRAW_REWARD = 0


class Plot(object):
    def __init__(self,ace_usable,q_value,count_list,auto_twist,bust_upbound,total_card_choice):
        self.ace_usable = ace_usable
        self.q_value = q_value
        self.count_list = count_list
        self.auto_twist = auto_twist
        self.bust_upbound = bust_upbound
        self.total_card_choice = total_card_choice
        self.env = Environment(LOSE_REWARD,DRAW_REWARD,WIN_REWARD)

    def plot_figure1_2(self):
        #to test policy figure
        #x is sum, y is showing card
        x_twist = []
        y_twist = []
        x_stick = []
        y_stick = []

        for it1 in range(self.auto_twist,self.bust_upbound+1):
            for it2 in range(1,self.total_card_choice+1):
                state_for_plt = self.env.get_state_label(it1,it2,self.ace_usable)
                if self.q_value[state_for_plt - 1][0] > self.q_value[state_for_plt-1][1]:
                    y_stick.append(it1)
                    x_stick.append(it2)
                else:
                    y_twist.append(it1)
                    x_twist.append(it2)

        plt.figure(1)
        plt.xlabel('showing card')
        plt.ylabel('player sum')
        plt.scatter(x_twist,y_twist,c='g')
        plt.legend('twist')

        plt.figure(2)
        plt.scatter(x_stick,y_stick,c='r')
        plt.legend('stick')
        plt.xlabel('showing card')
        plt.ylabel('sum cards')

    def f(self,x,y):
        state_for_3d_plot = self.env.get_state_label(y,x,self.ace_usable)
        if (self.count_list[state_for_3d_plot-1][0]+self.count_list[state_for_3d_plot-1][1]) == 0:
            v_value = 0
        else:
            v_value = (self.q_value[state_for_3d_plot-1][0]*self.count_list[state_for_3d_plot-1][0]+self.q_value[state_for_3d_plot-1][1]*self.count_list[state_for_3d_plot-1][1])\
                      /(self.count_list[state_for_3d_plot-1][0]+self.count_list[state_for_3d_plot-1][1])
        return v_value

    def plot_figure3(self):

        v_value =  [[0 for x in range(self.total_card_choice)] for y in range(self.bust_upbound-self.auto_twist+1)]
        #it1 is showing card, it2 is sum cards

        for it1 in range(self.total_card_choice):
            for it2 in range(self.bust_upbound-self.auto_twist+1):
                v_value[it1][it2] = self.f(it1+1,it2+self.auto_twist)

        def get_value(x,y):
            return v_value[x-1][y-self.auto_twist]

        fig = plt.figure(3)
        ax = fig.add_subplot(111, projection='3d')
        seq = list(range(1,self.total_card_choice+1))
        X=[]
        for it in range(self.total_card_choice):
            X.append(seq)
        Y = np.asarray([[1 for x in range(self.total_card_choice)] for y in range(self.bust_upbound-self.auto_twist+1)])
        for it in range(self.bust_upbound-self.auto_twist+1):
            Y[it] = Y[it]*(it+self.auto_twist)
        Z = np.asarray(v_value)
        print('---------')
        print(Z)
        ax.plot_surface(X,Y,np.transpose(Z))

    def show_figure(self):
        plt.show()