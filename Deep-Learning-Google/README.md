# Deep Learning Google

This is the homework implementation for the [google deep learning course](https://www.udacity.com/course/deep-learning--ud730)

* Regularization method: l2 loss, Dropout, Early stopping
* CNN: convolutional layer, max pooling layer, drop out layer, test accuracy : 92.3% (3000 episodes)
* Embeddin Word2Vec: 
	1. Skip-gram: one word to one word
	2. CBOW(continuous bagging of words):several words(context) to one word
	3. Using cosine distance of the embeddings as the standard for similarity between words
* LSTM:
	1. Avoid gradient vanishing and gradient exploding problem of the classic RNN
	2. Use gates: input gate, forget gate, output gate.
	3. Dropuout layer can be applied to input and ouput layers
