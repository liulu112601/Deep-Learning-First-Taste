import random
import numpy as np

def get_state_label(sum,dealer_card,ace):
    #sum: 12-21; dealer_card: 1-10; ace:0/1
    #200 states in total, return : 1-200
    return 20*(sum-12)+10*ace+dealer_card

def get_card():
    a = random.randint(1,13)
    if a<10:
        return a
    else:
        return 10

# dealer fixed policy : he sticks on any sum of 17 or greater, and hits otherwise.
def dealer_policy(dealer_sum):
    if dealer_sum > 16:
        return 0
    else:
        return 1
#a list to store q(s,a) 200*2
q_value = [[0 for x in range(2)] for y in range(200)]
accumulated_reward = [[0 for x in range(2)] for y in range(200)]
#a list to store counting numbers for every state action pair
count_list = [[0 for x in range(2)] for y in range(200)]
ace_usable = 0
discount_factor = 1

if ace_usable == 0:
    for iteration in range(500000):
        bust = False
        sequence_state_action = []
        player_terminate = 0
        dealer_terminate = 0
        # simulate an episode, in this case, ace is 1, unusable ace
        player_list = [get_card(),get_card()]
        dealer_list = [get_card(),get_card()]
        player_sum = player_list[0]+player_list[1]
        if player_sum < 12: # if <12 we automatically twist
            continue
        dealer_sum = dealer_list[0]+dealer_list[1]
        dealer_showing_card = dealer_list[1]
        state_label = get_state_label(player_sum,dealer_showing_card,ace_usable)
        reward = 0
        # natural 21 points, it can only happen when ace is usable
        if player_sum==21 or dealer_sum==21 :
            print('natural 21')
            print('player list',player_list)
            print('dealer list',dealer_list)
            print('reward',reward)
            continue
        #the initial policy is also choose the biggest q
        #if q is the same, then act randomly

        while(player_terminate==0):
            if q_value[state_label-1][0] == q_value[state_label-1][1]: #if equal , pick either one randomly
                action = random.randint(0,1)
                sequence_state_action.append([state_label, action])  # store state action pair

                if action == 0:
                    player_terminate =1
                else:#action == 1, get a new card
                    player_list.append(get_card())
                    player_sum += player_list[-1]
                    if player_sum>21:
                        reward = -1
                        player_terminate = 1
                        dealer_terminate = 1 # if player bust, dealer does not have to get card
                        bust = True
                    else:#twist less than 21, state changes to s'
                        state_label = get_state_label(player_sum,dealer_showing_card,ace_usable)

            else: #if not equal, pick the bigger one
                if q_value[state_label-1][0] > q_value[state_label-1][1]:
                    action = 0
                    player_terminate = 1
                    sequence_state_action.append([state_label, action])
                else:
                    action = 1
                    player_list.append(get_card())
                    player_sum += player_list[-1]
                    sequence_state_action.append([state_label, action])
                    if player_sum>21:
                        reward = -1
                        player_terminate = 1
                        dealer_terminate = 1# if player bust, dealer does not have to get card
                        bust = True
                    else:#twist less than 21, state changes to s'
                        state_label = get_state_label(player_sum,dealer_showing_card,ace_usable)




        while(dealer_terminate==0):
            action = dealer_policy(dealer_sum)
            if action == 0:
                dealer_terminate =1
            if action == 1:
                dealer_list.append(get_card())
                dealer_sum += dealer_list[-1]
                if dealer_sum>21:
                    dealer_terminate =1
                    reward = 1
                    bust = True


        #if dealer and player both < 21, no bust, then reward need to calculate accoding to cards sum
        if (not bust):
            if int(dealer_sum) > int(player_sum):
                reward = -1

            elif int(dealer_sum) < int(player_sum):
                reward = 1

            else:
                reward = 0


        #according to sequence state action value , update the q value
        #each element is a state action pair

        length = len(sequence_state_action)# the number of pairs
        for element in sequence_state_action:
            accumulated_reward[element[0]-1][element[1]] += reward*(discount_factor**(length-1))
            #for this eg, reward only occurs at the end, otherwise we need to record reward for every step
            count_list[element[0]-1][element[1]] += 1
            #for this eg, each state only occurs once, otherwise we need to record times for every encountered state
            q_value[element[0]-1][element[1]] = (accumulated_reward[element[0]-1][element[1]]) / (count_list[element[0]-1][element[1]])
            length -= 1

        ''''
        #test the rule
        print('bust is',bust)
        print('dealer sum',dealer_sum)
        print('player sum',player_sum)
        print('player list',player_list)
        print('dealer list',dealer_list)
        print('reward',reward)
        print('.................')
        '''

        '''
        #test q value
        if iteration < 100:
            #test the q value update
            print('reward',reward)
            print('accu reward',accumulated_reward)
            print('count list',count_list)
            print('q value',q_value)
            print('.................')
        '''

print('q value')
print(np.asarray(q_value))
print('-------')
print('count list')
print(np.array(count_list))

import matplotlib.pyplot as plt


#to test policy figure
#x is sum, y is showing card
x_twist = []
y_twist = []
x_stick = []
y_stick = []


for it1 in range(12,22):
    for it2 in range(1,11):
        state_for_plt = get_state_label(it1,it2,ace_usable)
        if q_value[state_for_plt - 1][0] > q_value[state_for_plt-1][1]:
            y_stick.append(it1)
            x_stick.append(it2)
        else:
            y_twist.append(it1)
            x_twist.append(it2)

print(x_stick)
print(y_stick)
print(x_twist)
print(y_twist)

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

def f(x,y):
    state_for_3d_plot = get_state_label(y,x,ace_usable)
    if (count_list[state_for_3d_plot-1][0]+count_list[state_for_3d_plot-1][1]) == 0:
        v_value = 0
    else:
        v_value = (q_value[state_for_3d_plot-1][0]*count_list[state_for_3d_plot-1][0]+q_value[state_for_3d_plot-1][1]*count_list[state_for_3d_plot-1][1])/(count_list[state_for_3d_plot-1][0]+count_list[state_for_3d_plot-1][1])
    return v_value

v_value =  [[0 for x in range(10)] for y in range(10)]
#it1 is showing card, it2 is sum cards

for it1 in range(10):
    for it2 in range(10):
        v_value[it1][it2] = f(it1+1,it2+12)

def get_value(x,y):
    return v_value[x-1][y-12]

fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')

seq = [1,2,3,4,5,6,7,8,9,10]
X=[]
for it in range(10):
    X.append(seq)

Y = np.asarray([[1 for x in range(10)] for y in range(10)])
for it in range(10):
    Y[it] = Y[it]*(it+12)

Z = np.asarray(v_value)
print('---------')
print(Z)
ax.plot_surface(X,Y,np.transpose(Z))

plt.show()