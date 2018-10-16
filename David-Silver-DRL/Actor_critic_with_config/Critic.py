import tensorflow as tf
reward = -1
discount_factor = 0.9

class Critic(object):
    def __init__(self, sess, n_features,lr):
        self.sess = sess
        self.n_features = n_features
        self.state = tf.placeholder(tf.float32,[1,n_features],name='state') #1
        self.next_state_value = tf.placeholder(tf.float32,[1,1],name='next_state') #2
        self.reward = reward
        self.lr = lr

    # one hidden layer NN
    def build_net(self):

        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(inputs=self.state,units=10,activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0., .1),bias_initializer=tf.zeros_initializer(),
                                 name = 'l1'
                                 )
            self.state_value = tf.layers.dense(inputs=l1,units=1,activation=None,
                                          kernel_initializer=tf.random_normal_initializer(0,1),bias_initializer=tf.zeros_initializer(),
                                          name='state_value')
        #loss is squared td_error
        with tf.variable_scope('critic_loss'):
            self.td_error = self.reward + discount_factor*self.next_state_value - self.state_value
            self.loss = tf.square(self.td_error)

        with tf.variable_scope('train'):
            self.train = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss=self.loss)

    def learn(self,state,next_state):
        next_state_value = self.sess.run(self.state_value,{self.state:next_state})
        td_error,run_train = self.sess.run([self.td_error,self.train],{self.state : state,self.next_state_value:next_state_value})

        return td_error
