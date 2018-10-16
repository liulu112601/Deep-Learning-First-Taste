import tensorflow as tf
import numpy as np
reward = -1

class Actor(object):
    def __init__(self, sess,n_actions,n_features, lr):
        self.sess = sess
        self.n_features = n_features
        self.state = tf.placeholder(tf.float32, [1, n_features], name='state')  # 1
        self.n_actions = n_actions
        self.action = tf.placeholder(tf.int32, None, name='action')  # 2
        self.td_error = tf.placeholder(tf.float32,[1,1],name='td_error') #3
        self.reward = reward
        self.lr = lr

    # one hidden layer NN
    def build_net(self):
        with tf.variable_scope('actor'):
            l1 = tf.layers.dense(inputs=self.state, units=10, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0., .1),
                                 bias_initializer=tf.zeros_initializer(),
                                 name='l1'
                                 )
            # probability is 0-1, so use softmax activation function
            self.action_prob = tf.layers.dense(inputs=l1, units=self.n_actions, activation=tf.nn.softmax,
                                               kernel_initializer=tf.random_normal_initializer(0, 1),
                                               bias_initializer=tf.zeros_initializer(),
                                               name='action_prob')
        # loss function
        with tf.variable_scope('actor_loss'):
            #td_error as advantage function
            #[0,self.action] to choose the current state action pair,we only update weights related to the current s,a pair
            #we use gradient ascent, so add a minus at the front
            self.loss = -tf.log(self.action_prob[0,self.action])*self.td_error

        with tf.variable_scope('train'):
            self.train = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss=self.loss)

    def learn(self, state, action,td_error):

        run_train = self.sess.run(self.train,
                                  {self.state: state, self.action: action, self.td_error: td_error})
    #choose action according to the current policy network
    def choose_action(self,state):
        prob_actions = self.sess.run(self.action_prob,{self.state:state})
        return np.random.choice(3,1,p=prob_actions.ravel())[0]

