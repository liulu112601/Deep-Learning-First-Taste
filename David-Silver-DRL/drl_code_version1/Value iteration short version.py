def transiton(label):
    if label == 1 or label == 2:
        return [label - 1, label, label + 1, label + 4]
    if label == 3:
        return [label - 1, label, label, label + 4]
    if label == 4 or label == 8:
        return [label, label - 4, label + 1, label + 4]
    if label == 5 or label == 6 or label == 9 or label == 10:
        return [label - 1, label - 4, label + 1, label + 4]
    if label == 7 or label == 11:
        return [label - 1, label - 4, label, label + 4]
    if label == 12:
        return [label, label - 4, label + 1, label]
    if label == 13 or label == 14:
        return [label - 1, label - 4, label + 1, label]
    if label == 15:
        return [label - 1, label - 4, label, label]

state_value = [[0 for x in range(4)] for y in range(4)]
state_value_tmp = [[0 for x in range(4)] for y in range(4)]
discount_factor = 1
immediate_reward = -1

for iteration in range(6):
    # state_value_tmp = state_value # store the old values
    # for every label, the position is line : label//4, column: label%4
    for it in range(15):
        label = it + 1
        transition_list = transiton(label)
        print(transition_list)
        state_value[label // 4][label % 4] = max(
            immediate_reward + discount_factor * state_value_tmp[transition_list[0] // 4][transition_list[0] % 4],
            immediate_reward + discount_factor * state_value_tmp[transition_list[1] // 4][transition_list[1] % 4],
            immediate_reward + discount_factor * state_value_tmp[transition_list[2] // 4][transition_list[2] % 4],
            immediate_reward + discount_factor * state_value_tmp[transition_list[3] // 4][transition_list[3] % 4])

    for it in range(4):
        for it2 in range(4):
            state_value_tmp[it][it2] = state_value[it][it2]

    print('it time:', iteration)
    print('state value function:', state_value)