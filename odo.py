import numpy as np
import os
import tensorflow as tf
import subprocess

from loading import DataClass
from loading import split_images

# PARAMETRE_NN-------------------------------------------
num_steps = 1000
batch_size = 16
info_freq = 10
session_log_name = 'go_2'

num_hidden = [120, 80]

keep_prob_fc = 0.5

chunk_size = 128
channels = 3

image_height, image_width = (192, 256)
# ------------------
# nacitanie dat
# url = "/home/andrej/tf/odo/"
url = "/home/marek/kody/ODO_reader"
# url = '/home/katarina/PycharmProjects/TensorFlowTut/ODO_loading'

train_data_size = 6000
num_classes = split_images(url, train_data_size)


# velkost obrazka po aplikovani conv vrstiev
def conv_output_size(padding, input_height, input_width, stride, kernel_height, kernel_width):
    output_height, output_width = (0, 0)
    if padding == "VALID":
        output_height = (input_height - kernel_height) / stride + 1
        output_width = (input_width - kernel_width) / stride + 1
        output_height = int(np.floor(output_height))
        output_width = int(np.floor(output_width))
    if padding == "SAME":
        output_height = input_height / stride
        output_width = input_width / stride
        output_height = int(np.ceil(output_height))
        output_width = int(np.ceil(output_width))
    return output_height, output_width
    # pouzitie
    # c1_output_size=conv_output_size("valid", 20, 10, 1, 5, 3)
    # c2_output_size=conv_output_size("valid", c1_output_size[0], c1_output_size[1],1, 3, 3)


def accuracy(predictions, labels):
    acc = sum([(labels[i, np.argmax(predictions[i, :])] == 1) for i in range(predictions.shape[0])]) \
                / predictions.shape[0]
    return acc

train_data = DataClass(os.path.join(url, 'train/'), batch_size, chunk_size, num_classes, data_use='train')
valid_data = DataClass(os.path.join(url, 'valid/'), batch_size, chunk_size, num_classes, data_use='valid')

###############################
# CONVOLUTION LAYERS SETTINGS #
###############################
conv_layer_names = ['conv1', 'conv2', 'conv3', 'conv4']

kernel_sizes = [
    (3, 3),
    (3, 3),
    (3, 3),
    (3, 3)
]
kernel_sizes = {name: kernel_sizes[i] for i, name in enumerate(conv_layer_names)}

num_filters = [
    16, 32, 48, 64
]
num_filters = {name: num_filters[i] for i, name in enumerate(conv_layer_names)}

strides = [

    (3, 3),
    (3, 3),
    (2, 2),
    (2, 2),
]
strides = {name: strides[i] for i, name in enumerate(conv_layer_names)}

paddings = [
    'VALID',
    'VALID',
    'VALID',
    'VALID'
]
paddings = {name: paddings[i] for i, name in enumerate(conv_layer_names)}

# DROPOUT
output_sizes = {

}

for i, layer in enumerate(conv_layer_names):
    if i == 0:
        output_sizes[layer] = conv_output_size(
            paddings[layer],
            image_height, image_width,
            strides[layer][0],
            kernel_sizes[layer][0], kernel_sizes[layer][1]
        )
    else:
        output_sizes[layer] = conv_output_size(
            paddings[layer],
            output_sizes[conv_layer_names[i - 1]][0], output_sizes[conv_layer_names[i - 1]][1],
            strides[layer][0],
            kernel_sizes[layer][0], kernel_sizes[layer][1]
        )

graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_dataset = tf.placeholder(
        tf.float32, shape=(None, image_height, image_width, channels))
    tf_labels = tf.placeholder(tf.int32, shape=(None, num_classes))

    # dropout
    tf_keep_prob_fc = tf.placeholder(tf.float32)

    weights = {
        'conv1': tf.Variable(tf.random_normal(
            [kernel_sizes['conv1'][0], kernel_sizes['conv1'][1], channels, num_filters['conv1']],
            stddev=np.sqrt(2 / (kernel_sizes['conv1'][0] * kernel_sizes['conv1'][1] * channels)))
        ),

        'fc1': tf.Variable(tf.truncated_normal(
            [output_sizes[conv_layer_names[-1]][0] * output_sizes[conv_layer_names[-1]][1] * num_filters[
                conv_layer_names[-1]],
             num_hidden[0]],
            stddev=np.sqrt(2 / (
                    output_sizes[conv_layer_names[-1]][0]
                    * output_sizes[conv_layer_names[-1]][1]
                    * num_filters[conv_layer_names[-1]]))
                )
        ),

        'fc2': tf.Variable(tf.truncated_normal(
                [num_hidden[0],
                 num_hidden[1]],
                stddev=np.sqrt(2 / (num_hidden[0])))
        ),

        'out': tf.Variable(tf.truncated_normal(
            [num_hidden[1], num_classes], stddev=np.sqrt(2 / (num_hidden[1]))))
    }

    # vytvor vahy pre ostatne konvolucne vrstvy
    for l, l_prev in zip(conv_layer_names[1:], conv_layer_names[:-1]):
        weights[l] = tf.Variable(tf.random_normal(
            [kernel_sizes[l][0], kernel_sizes[l][1], num_filters[l_prev], num_filters[l]],
            stddev=np.sqrt(2 / (kernel_sizes[l][0] * kernel_sizes[l][1] * num_filters[l_prev])))
        )

    biases = {
        'fc1': tf.Variable(tf.zeros([num_hidden[0]])),
        'fc2': tf.Variable(tf.zeros([num_hidden[1]])),
        'out': tf.Variable(tf.zeros([num_classes]))
    }

    # vytvor biasy pre ostatne konvolucne vrstvy
    for l in conv_layer_names:
        biases[l] = tf.Variable(tf.zeros([num_filters[l]])),

    # Model
    def model(data):
        # INPUT je teraz velkosti batch x h x w x ch
        print('input:', data.get_shape().as_list())
        out = data

        # ak chces menit konvolucne vrstvy, robi sa to hore pod settings
        # ------convolution-------
        for i, l in enumerate(conv_layer_names):
            out = tf.nn.conv2d(out, weights[l], [1, strides[l][0], strides[l][1], 1], padding=paddings[l])
            out = tf.nn.relu(out + biases[l])

            print(l, ':', out.get_shape().as_list())
        # -------------------------

        shape = out.get_shape().as_list()

        # reshape:
        out = tf.reshape(out, [-1, shape[1] * shape[2] * shape[3]])
        print('after reshape:', out.get_shape().as_list())

        # fully connected

        out = tf.matmul(out, weights['fc1']) + biases['fc1']
        out = tf.nn.dropout(out, tf_keep_prob_fc)
        out = tf.nn.relu(out)

        print('fc1:', out.get_shape().as_list())

        out = tf.matmul(out, weights['fc2']) + biases['fc2']
        out = tf.nn.dropout(out, tf_keep_prob_fc)
        out = tf.nn.relu(out)

        print('fc2:', out.get_shape().as_list())
        out = tf.nn.relu(out)

        return tf.matmul(out, weights['out']) + biases['out']


    # compute output activations and loss
    output = model(tf_dataset)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, tf_labels))

    # Optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(loss)

    prediction = tf.nn.softmax(output)

    saver = tf.train.Saver()
    initialization = tf.initialize_all_variables()


with tf.Session(graph=graph) as session:
    if os.path.isfile("abcdefghij/{}.ckpt".format(session_log_name)):
        saver.restore(session, "abcdefghij/{}.ckpt".format(session_log_name))
    else:
        session.run(initialization)

    print('------------------------')
    print('Training {}'.format(session_log_name))
    print('------------------------')

    batch_data_valid = valid_data.data
    batch_labels_valid = valid_data.labels

    step = -1
    continue_training = '1'
    while continue_training == '1':
        step += 1
        # offset = (step * batch_size) % (y_train_ctc.shape[0] - batch_size)
        batch_data, batch_labels = train_data.next_batch()

        feed_dict = {tf_dataset: batch_data, tf_labels: batch_labels,
                     tf_keep_prob_fc: keep_prob_fc}

        _, loss_value, predictions = session.run(
            [optimizer, loss, prediction], feed_dict=feed_dict)

        if step % info_freq == 0:
            print('Minibatch loss at step {}: {}'.format(step, loss_value))
            print('Minibatch accuracy:', 100 * accuracy(predictions, batch_labels))

            valid_predictions = session.run(
                prediction,
                feed_dict={tf_dataset: batch_data_valid, tf_labels: batch_labels_valid,
                           tf_keep_prob_fc: 1}
            )
            print('Validation accuraccy (batch-sized subset)',
                  100 * accuracy(valid_predictions, batch_labels_valid)
                  )
            print('------------------------')

        # if step == num_steps: pokracovat = 0
        if (step % num_steps) == 0:
            subprocess.call(['speech-dispatcher'])  # start speech dispatcher
            subprocess.call(['spd-say', '" process has finished"'])
            continue_training = (input('Continue? 1/0'))
            # continue_training = '0'

    save_path = saver.save(session, "abcdefghij/{}.ckpt".format(session_log_name))
    # print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

    results = []
    valid_labels = []
    for offset in range(0, valid_data.total_data_size - batch_size + 1, batch_size):
        data, lab = valid_data.next_batch()
        results.append((session.run(
            prediction,
            feed_dict={tf_dataset: data, tf_labels: lab, tf_keep_prob_fc: 1}
        )))
        valid_labels.append(lab)


results = np.array(results).reshape(-1, num_classes)
valid_labels = np.array(valid_labels).reshape(-1, num_classes)

print('(prediction, true label):', list(zip([np.argmax(r) for r in np.array(results)], [np.argmax(r) for r in np.array(valid_labels)])))
# print('lab', [np.argmax(r) for r in np.array(valid_labels)])

print('accuracy', accuracy(results, valid_labels))
