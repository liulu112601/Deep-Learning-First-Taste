# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range


pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

batch_size = 128
hidden_units = 1024
regularization_factor = 0.01
num_steps = 3000

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(None, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    train = tf.placeholder(tf.bool)
    decay_rt = 0.96
    '''
    #low level API 1
    # Variables
    weights_hidden = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_units]))
    biases_hidden = tf.Variable(tf.zeros([hidden_units]))
    weights_output = tf.Variable(tf.truncated_normal([hidden_units, num_labels]))
    biases_output = tf.Variable(tf.zeros([num_labels]))
    output_layer = tf.matmul(tf.nn.relu(tf.matmul(tf_train_dataset, weights_hidden) + biases_hidden),weights_output)+biases_output

    #low level API 2
    #[784*hidden_units]
    weights_hidden = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_units]))
    biases_hidden = tf.Variable(tf.zeros([hidden_units]))
    #[hidden_units*num_labels]
    weights_output = tf.Variable(tf.truncated_normal([hidden_units, num_labels]))
    biases_ouput = tf.Variable(tf.zeros([num_labels]))
    # Training computation.
    hidden_layer_1 = tf.nn.relu(tf.matmul(tf_train_dataset,weights_hidden)+biases_hidden)
    output_layer = tf.matmul(hidden_layer_1,weights_output) + biases_ouput
    
    #low level with regularization
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=output_layer)) + \
           regularization_factor*(tf.nn.l2_loss(weights_hidden)+tf.nn.l2_loss(weights_output))
    '''
    #using high-level APIs in tsf
    hidden_layer_1 = tf.layers.dense(tf_train_dataset,hidden_units,activation=tf.nn.relu,use_bias=True,
                                     kernel_initializer=tf.truncated_normal_initializer(),
                                     bias_initializer=tf.zeros_initializer())  # [bs, hn]

    drop_out = tf.layers.dropout(hidden_layer_1,rate=0.5,training=train)

    output_layer = tf.layers.dense(drop_out,num_labels,None,use_bias=True,bias_initializer=
                                   tf.zeros_initializer(),
                                   kernel_initializer=tf.truncated_normal_initializer())  # [bs, ln]

    #high level with regularization
    # #var_list is [weight1,bias1,weight3,bias2]
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=output_layer)) + \
           regularization_factor*tf.nn.l2_loss(var_list[0])+ regularization_factor*tf.nn.l2_loss(var_list[2])

    #learning rate decay
    global_step = tf.Variable(0) #automatically add one,after the optimizer has beed operated
    learning_rate = tf.train.exponential_decay(0.5, global_step=global_step, decay_steps=num_steps,decay_rate=decay_rt,name='LR')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(output_layer)
    valid_prediction = tf.nn.softmax(output_layer)
    test_prediction = tf.nn.softmax(output_layer)

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict_train = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, train : True}
    feed_dict_valid = {tf_train_dataset : valid_dataset, tf_train_labels : valid_labels, train : False}
    feed_dict_test = {tf_train_dataset : test_dataset, tf_train_labels : test_labels, train : False}


    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict_train)

    if (step % 100 == 0):
      #print(session.run(var_list)[0],session.run(var_list)[2])
      predictions_valid = session.run(valid_prediction,feed_dict=feed_dict_valid)

      predictions_test = session.run(test_prediction,feed_dict=feed_dict_test)
      #print(predictions.shape,np.squeeze(predictions_valid,0).shape,np.squeeze(predictions_test,0).shape)
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
          predictions_valid, valid_labels))
      print("Test accuracy: %.1f%%" % accuracy(predictions_test, test_labels))
