import numpy as np

import TileCoding as tc

import numpy.random as rdm

class Environment (object):

    def __init__(self,iht,num_tiling,tile_factor_position,tile_factor_velocity,min_position,max_velocity,min_velocity,max_position,theta):
        self.iht = iht
        self.num_tiling = num_tiling
        self.tile_factor_position = tile_factor_position
        self.tile_factor_velocity = tile_factor_velocity
        self.min_position = min_position
        self.max_velocity = max_velocity
        self.min_velocity = min_velocity
        self.max_position = max_position
        self.theta = theta

    #return a list of numbers, length is num_tiling,one for each
    def get_tiles(self,p,v,a):
        return tc.tiles(self.iht,self.num_tiling,[self.tile_factor_position*p,self.tile_factor_velocity*v],[a]) #a list containing 8 numbers for every tiling

    def initialization(self):
        return rdm.uniform(-0.6, -0.4), 0


    def transition(self,p,v,a,flag):
        #if the car reaches the left point, the velocity is reset to 0
        if (p == self.min_position) and (flag == 0):
            v_next = 0
            flag = 1
        else:
            v_next = max(min(v+0.001*a-0.0025*np.cos(3*p),self.max_velocity),self.min_velocity)
            flag = 0
        p_next = max(min(p+v_next,self.max_position),self.min_position)
        return p_next,v_next,flag

    def get_q_value(self,p, v, a):
        # return a list of numbers, length is num_tiling,one for each
        tiles = self.get_tiles(p, v, a)
        # print('tiles info:(p,v,a,tiles)',p,v,a,tiles)
        if p == self.max_position:
            return 0
        else:
            # tiles is a numpy array, so it calculate the sum of the elements in the theta list
            # q = theta*x , x is 1 in the index points, 0 in other places, so we only need to consider the index points

            return np.sum(self.theta[tiles])
