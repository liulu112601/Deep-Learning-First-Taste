
from Environment import Environment
import numpy as np

TOTAL_LINES = 4
TOTAL_COLS = 4
TOTAL_ACTIONS = 4
IMM_REWARD = -1
ITERATION = 100
DISCOUNT_FACTOR = 1


environment = Environment(TOTAL_LINES,TOTAL_COLS,TOTAL_ACTIONS,IMM_REWARD,DISCOUNT_FACTOR)

value_net = np.zeros((TOTAL_LINES,TOTAL_COLS))

for it in range(ITERATION):

    value_net = environment.update_value_net(value_net)
    print(value_net)
