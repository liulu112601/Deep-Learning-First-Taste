import gym
import numpy as np
import numpy.random as rdm
import sklearn.neural_network as nn


env = gym.make('MountainCar-v0')
low = env.observation_space.low  # seq : [position,velocity]
high = env.observation_space.high
# env.action_space        # 0,1,2
reward = -1
discount_factor = 0.9
ex_replay_size = 2000 # the experience replay can store 2000 transitions
ex_replay_column = 5  # current position,current velocity,action,next position,next velocity
epsilon = 0.1 # possibility of exploring
action_list = [0,1,2] #backward,zero,forward
iteration_time = 2000

# NN Parameter, one hidden layer in this example
inner_neuro = 10 #the number of neurons of the hidden layer
batch_size = 32
activation_function = 'relu'
learning_rate = 0.1
# create an empty experience replay to store transitions
ex_replay = np.zeros((ex_replay_size, ex_replay_column))
# to record the NO. of experience
count = 0
#200 transitions and q new == q old
learning_interval = 200
#count how many transitions
step = 0

#create two NN
q_new_net = nn.MLPRegressor(warm_start=True,hidden_layer_sizes=inner_neuro,activation=activation_function,learning_rate_init=learning_rate,batch_size=batch_size)
q_old_net = nn.MLPRegressor(warm_start=True,hidden_layer_sizes=inner_neuro,activation=activation_function,learning_rate_init=learning_rate,batch_size=batch_size)
#initialize the weight of the NN
x=np.zeros([batch_size,3])
y=np.zeros([batch_size,1])
q_new_net.fit(x,y)
q_old_net.fit(x,y)


def epsilon_greedy(p,v):
    random_number = rdm.random_sample()
    if (random_number < epsilon):  # randomly pick one action
        return action_list[rdm.randint(0, 3)]
    else:
        q_list = [q_old_net.predict(np.asarray([p,v,0]).reshape(1,-1)),\
                  q_old_net.predict(np.asarray([p,v,1]).reshape(1,-1)),\
                  q_old_net.predict(np.asarray([p,v,2]).reshape(1,-1))]
        return action_list[q_list.index(max(q_list))] # pick the biggest

#create an experience replay, in this example, the replay is a random action pick
#u can use epsilon greedy over a given q value distribution to make it better
while (count < ex_replay_size):
    # a new round
    observation = env.reset()

    while (count < ex_replay_size):
        action = env.action_space.sample()  # pick an action randomly
        ex_replay[count, 0], ex_replay[count, 1] = observation[0], observation[1]
        # store position and velocity
        action = env.action_space.sample() # pick an action
        next_observation, reward, done, info = env.step(action)  # do the action and transit to the next observation
        # store action and next position and velocity
        ex_replay[count, 2], ex_replay[count, 3], ex_replay[count, 4] = action, next_observation[0], next_observation[1]
        count += 1
        observation = next_observation
        if observation[0] == high[0]:
            break

        '''
        if done:
            #print("Episode")
            break        
        '''
#print(ex_replay)

for it in range(iteration_time):
    # a new episode
    observation = env.reset()
    while(True):
        #take action a according to epsilon-greedy
        env.render()
        action = epsilon_greedy(observation[0],observation[1])
        next_observation, reward, done, info = env.step(action)
        ex_replay[step%ex_replay_size] = [observation[0],observation[1],action,next_observation[0],next_observation[1]]
        observation = next_observation
        #experience replay changed

        #generate a minibatch
        #generate minibach index， randomly choose
        minibatch_index = rdm.choice(ex_replay_size,batch_size,replace=False)
        minibatch = []
        for element in minibatch_index:
            minibatch.append(ex_replay[element])
        minibatch = np.asarray(minibatch)
        #q_old is [p,v,a]
        q_old = minibatch[:,0:3]
        #q_new is [p',v']
        q_new = minibatch[:,3:5]

        #initailize target q value to zero
        target_q=np.zeros([batch_size,1])
        # to count target q
        count = 0
        #find the max action over the fixed net
        for element in q_new:
            target_q[count] = reward+discount_factor*max(q_new_net.predict([np.append(element,action_list[0])]), \
                                                         q_new_net.predict([np.append(element,action_list[1])]), \
                                                         q_new_net.predict([np.append(element,action_list[2])]))
            count +=1

        #update the old q network (true net)
        target_q = target_q.ravel()
        q_old_net.fit(q_old,target_q)
        step += 1
        #if the old q network has been updated for interval times, then it is time for the new q network to update according to the old
        if step%learning_interval == learning_interval-1:
            q_new_net.coefs_ = q_old_net.coefs_
            q_new_net.intercepts_ = q_old_net.intercepts_
            print('step',step,'fixed net weight changes')
        #reach the right top
        if observation[0] == high[0]:
            break
        '''
        
        if done:
            break
        print('step',step)
        '''








