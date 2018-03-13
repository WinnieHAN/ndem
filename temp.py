from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import h5py, os


run_type = 'no_init'
if run_type == 'init':
    var = tf.Variable(tf.random_uniform([2, 3]), name="var")
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print(sess.run(var))
    print(sess.run(init))
if 1==0:
    def open_dataset(out_list, path, name, train_size, valid_size, test_size):
        usage = 'training'
        batch_size = 64
        sequence_length = 16

        # open dataset file
        _hdf5_file = h5py.File(os.path.join(path, name + '.h5'), 'r')
        _data_in_file = {
            data_name: _hdf5_file[usage][data_name] for data_name in out_list
        }
        limit = ({'training': train_size, 'validation': valid_size, 'test': test_size}[usage] or
                      _data_in_file['features'].shape[1])

        # fix shapes and datatypes
        input_seq_len = 1 if _data_in_file['features'].shape[0] == 1 else sequence_length
        shapes = {
            data_name: (input_seq_len, batch_size, 1) + _data_in_file[data_name].shape[-3:]
            for data_name, data in _data_in_file.items()
        }
        shapes['idx'] = ()
        _dtypes = {data_name: tf.float32 for data_name in out_list}
        _dtypes['idx'] = tf.int32

        # set up placeholders for inserting data into queue
        _data_in = {
            data_name: tf.placeholder(_dtypes[data_name], shape=shape)
            for data_name, shape in shapes.items()
        }

        k = get_feed_data(_data_in, _data_in_file, sequence_length, start_idx=0)

        print(k)

    def get_feed_data(_data_in, _data_in_file, sequence_length, start_idx):
        batch_size = 64
        feed_dict = {_data_in[data_name]: ds[:sequence_length, start_idx:start_idx + batch_size][:, :, None]
                     for data_name, ds in _data_in_file.items()}
        feed_dict[_data_in['idx']] = start_idx
        return feed_dict
    open_dataset(out_list= ('features', 'groups'), path = './data', name = 'shapes', train_size = None, valid_size=1000, test_size = None)

if 1==0:
    class InputPipeLine(object):
        def _open_dataset(self, out_list, path, name, train_size, valid_size, test_size):
            # open dataset file
            self._hdf5_file = h5py.File(os.path.join(path, name + '.h5'), 'r')
            self._data_in_file = {
                data_name: self._hdf5_file[self.usage][data_name] for data_name in out_list
            }
            self.limit = ({'training': train_size, 'validation': valid_size, 'test': test_size}[self.usage] or
                          self._data_in_file['features'].shape[1])

            # fix shapes and datatypes
            input_seq_len = 1 if self._data_in_file['features'].shape[0] == 1 else self.sequence_length
            self.shapes = {
                data_name: (input_seq_len, self.batch_size, 1) + self._data_in_file[data_name].shape[-3:]
                for data_name, data in self._data_in_file.items()
            }
            self.shapes['idx'] = ()
            self._dtypes = {data_name: tf.float32 for data_name in out_list}
            self._dtypes['idx'] = tf.int32

            # set up placeholders for inserting data into queue
            self._data_in = {
                data_name: tf.placeholder(self._dtypes[data_name], shape=shape)
                for data_name, shape in self.shapes.items()
            }


        def __init__(self, usage, shuffle, batch_size, sequence_length, queue_capacity, _rnd, out_list=('features', 'groups')):
            self.usage = usage
            self.shuffle = shuffle
            self.sequence_length = sequence_length
            self.batch_size = batch_size
            self._rnd = _rnd
            self.samples_cache = {}

            with tf.name_scope("{}_queue".format(usage[:5])):

                self._open_dataset(out_list)

                # set up queue
                self.queue = tf.FIFOQueue(capacity=queue_capacity,
                                          dtypes=[v for k, v in sorted(self._dtypes.items(), key=lambda x: x[0])],
                                          shapes=[v for k, v in sorted(self.shapes.items(), key=lambda x: x[0])],
                                          names=[k for k in sorted(self._dtypes)])

                self._enqueue_op = self.queue.enqueue(self._data_in)

                # set up outputs of queue (inputs for the model)
                self.output = self.queue.dequeue()
                if self.shapes['features'][0] == 1 and self.sequence_length > 1:
                    # if the dataset has sequence length 1 we need to repeat the data
                    reshaped_output = {data_name: tf.tile(self.output[data_name], [self.sequence_length, 1, 1, 1, 1, 1])
                                       for data_name in out_list}
                    reshaped_output['idx'] = self.output['idx']
                    self.output = reshaped_output

        def get_feed_data(self, start_idx):
            feed_dict = {self._data_in[data_name]: ds[:self.sequence_length, start_idx:start_idx + self.batch_size][:, :, None]
                         for data_name, ds in self._data_in_file.items()}
            feed_dict[self._data_in['idx']] = start_idx
            return feed_dict

        def get_debug_samples(self, samples_list, out_list=None):
            samples_key = tuple(samples_list)
            if samples_key in self.samples_cache:
                return self.samples_cache[samples_key]

            out_list = self._data_in_file.keys() if out_list is None else out_list
            results = {}
            for data_name in out_list:
                data = self._hdf5_file[self.usage][data_name][:, samples_list][:, :, None]
                if data.shape[0] == 1 and self.sequence_length > 1:
                    data = np.repeat(data, self.sequence_length, axis=0)
                elif data.shape[0] > self.sequence_length:
                    data = data[:self.sequence_length]
                results[data_name] = data

            self.samples_cache[samples_key] = results
            return results

        def get_batch_start_indices(self):
            idxs = np.arange(0, self.limit - self.batch_size, step=self.batch_size)
            if self.shuffle:
                self._rnd.shuffle(idxs)
            return 0, idxs

        def enqueue(self, session, coord):
            i, idxs = self.get_batch_start_indices()
            try:
                while not coord.should_stop():
                    if i >= len(idxs):
                        i, idxs = self.get_batch_start_indices()
                    session.run(self._enqueue_op, feed_dict=self.get_feed_data(idxs[i]))
                    i += 1
            except Exception as e:
                coord.request_stop(e)
            finally:
                self._hdf5_file.close()

        def get_n_batches(self):
            return self.limit // self.batch_size


if 1==0:
    features = {
        'sales' : [[5], [10], [8], [9]],
        'department': ['sports', 'sports', 'gardening', 'gardening']}

    department_column = tf.feature_column.categorical_column_with_vocabulary_list(
            'department', ['sports', 'gardening'])
    department_column = tf.feature_column.indicator_column(department_column)

    columns = [
        tf.feature_column.numeric_column('sales'),
        department_column
    ]

    inputs = tf.feature_column.input_layer(features, columns)

    var_init = tf.global_variables_initializer()
    table_init = tf.tables_initializer()
    sess = tf.Session()
    result = sess.run((var_init, table_init))
    result = sess.run(inputs)
    print(result)

if 1==0:
    x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
    y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

    # linear_model = tf.layers.dense(inputs=1, units=1)
    # y_pred = linear_model(x)
    y_pred = tf.layers.dense(inputs=x, units=1)
    loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    for i in range(100):
      _, loss_value = sess.run((train, loss))
      print(loss_value)

    print(sess.run(y_pred))

if 1==1:
    tf.logging.set_verbosity(tf.logging.INFO)


    def cnn_model_fn(features, labels, mode):
      """Model function for CNN."""
      # Input Layer
      # Reshape X to 4-D tensor: [batch_size, width, height, channels]
      # MNIST images are 28x28 pixels, and have one color channel
      input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

      # Convolutional Layer #1
      # Computes 32 features using a 5x5 filter with ReLU activation.
      # Padding is added to preserve width and height.
      # Input Tensor Shape: [batch_size, 28, 28, 1]
      # Output Tensor Shape: [batch_size, 28, 28, 32]
      conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)

      # Pooling Layer #1
      # First max pooling layer with a 2x2 filter and stride of 2
      # Input Tensor Shape: [batch_size, 28, 28, 32]
      # Output Tensor Shape: [batch_size, 14, 14, 32]
      pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

      # Convolutional Layer #2
      # Computes 64 features using a 5x5 filter.
      # Padding is added to preserve width and height.
      # Input Tensor Shape: [batch_size, 14, 14, 32]
      # Output Tensor Shape: [batch_size, 14, 14, 64]
      conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=64,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)

      # Pooling Layer #2
      # Second max pooling layer with a 2x2 filter and stride of 2
      # Input Tensor Shape: [batch_size, 14, 14, 64]
      # Output Tensor Shape: [batch_size, 7, 7, 64]
      pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

      # Flatten tensor into a batch of vectors
      # Input Tensor Shape: [batch_size, 7, 7, 64]
      # Output Tensor Shape: [batch_size, 7 * 7 * 64]
      pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

      # Dense Layer
      # Densely connected layer with 1024 neurons
      # Input Tensor Shape: [batch_size, 7 * 7 * 64]
      # Output Tensor Shape: [batch_size, 1024]
      dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

      # Add dropout operation; 0.6 probability that element will be kept
      dropout = tf.layers.dropout(
          inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

      # Logits layer
      # Input Tensor Shape: [batch_size, 1024]
      # Output Tensor Shape: [batch_size, 10]
      logits = tf.layers.dense(inputs=dropout, units=10)

      predictions = {
          # Generate predictions (for PREDICT and EVAL mode)
          "classes": tf.argmax(input=logits, axis=1),
          # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
          # `logging_hook`.
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
      }
      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

      # Calculate Loss (for both TRAIN and EVAL modes)
      loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

      # Configure the Training Op (for TRAIN mode)
      if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

      # Add evaluation metrics (for EVAL mode)
      eval_metric_ops = {
          "accuracy": tf.metrics.accuracy(
              labels=labels, predictions=predictions["classes"])}
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


    def main(unused_argv):
      # Load training and eval data
      # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
      # train_data = mnist.train.images  # Returns np.array
      # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
      # eval_data = mnist.test.images  # Returns np.array
      # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

      train_data = np.random.rand(3, 28*28).astype(np.float32)
      train_labels = np.random.randint(low=0, high=10, size=(3,))
      eval_data = train_data
      eval_labels = train_labels

      # Create the Estimator
      mnist_classifier = tf.estimator.Estimator(
          model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

      # Set up logging for predictions
      # Log the values in the "Softmax" tensor with label "probabilities"
      tensors_to_log = {"probabilities": "softmax_tensor"}
      logging_hook = tf.train.LoggingTensorHook(
          tensors=tensors_to_log, every_n_iter=50)

      # Train the model
      train_input_fn = tf.estimator.inputs.numpy_input_fn(
          x={"x": train_data},
          y=train_labels,
          batch_size=1,
          num_epochs=None,
          shuffle=True)
      mnist_classifier.train(
          input_fn=train_input_fn,
          steps=20,
          hooks=[logging_hook])

      # Evaluate the model and print results
      eval_input_fn = tf.estimator.inputs.numpy_input_fn(
          x={"x": eval_data},
          y=eval_labels,
          num_epochs=100,
          shuffle=False)
      eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
      print(eval_results)


    if __name__ == "__main__":
      tf.app.run()