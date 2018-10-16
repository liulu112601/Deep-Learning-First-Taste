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

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(None, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.

    # Training computation.
    hidden_layer_1 = tf.layers.dense(tf_train_dataset,hidden_units,activation=tf.nn.relu,use_bias=True,
                                     kernel_initializer=tf.truncated_normal_initializer(),
                                     bias_initializer=tf.zeros_initializer())
    output_layer = tf.layers.dense(hidden_layer_1,num_labels,None,use_bias=True,bias_initializer=
                                   tf.zeros_initializer(),
                                   kernel_initializer=tf.truncated_normal_initializer())
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=output_layer))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(output_layer)
    valid_prediction = tf.nn.softmax(output_layer)
    test_prediction = tf.nn.softmax(output_layer)

num_steps = 3001

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
    feed_dict_train = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    feed_dict_valid = {tf_train_dataset : valid_dataset, tf_train_labels : valid_labels}
    feed_dict_test = {tf_train_dataset : test_dataset, tf_train_labels : test_labels}

    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict_train)

    if (step % 500 == 0):

      predictions_valid = session.run([valid_prediction],feed_dict=feed_dict_valid)

      predictions_test = session.run([test_prediction],feed_dict=feed_dict_test)
      #print(predictions.shape,np.squeeze(predictions_valid,0).shape,np.squeeze(predictions_test,0).shape)
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
          np.squeeze(predictions_valid, 0), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(np.squeeze(predictions_test,0), test_labels))
