import numpy as np
import os
import tensorflow as tf
import subprocess
import time
import pandas as pd

from loading import DataClass
from loading import split_images

# PARAMETRE_NN-------------------------------------------
num_steps = 50
batch_size = 16
info_freq = 25
session_log_name = input('Name your baby... architecture!')

num_hidden = [120]

keep_prob_fc = 0.5

chunk_size = 128
channels = 3

image_height, image_width = (192, 256)
cut_height, cut_width = (int(np.floor(0.85*image_height)), int(np.floor(0.85*image_width)))
# ------------------
# nacitanie dat
url = "/home/andrej/tf/ODO_reader"
# url = "/home/marek/PycharmProjects/ODO_reader_/ODO_reader"
# url = '/home/katarina/PycharmProjects/TensorFlowTut/ODO_reader'

train_data_size = 6000
znacky = split_images(url, train_data_size, image_height, image_width)
print(znacky)
num_classes = len(znacky)
print(num_classes)


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

train_data = DataClass(os.path.join(url, 'train/'),
                       batch_size, chunk_size, num_classes,
                       image_height, image_width, cut_height, cut_width,
                       znacky, data_use='train')
valid_data = DataClass(os.path.join(url, 'valid/'),
                       batch_size, chunk_size, num_classes,
                       image_height, image_width, cut_height, cut_width,
                       znacky, data_use='valid')

image_height, image_width = (cut_height, cut_width)

###############################
# CONVOLUTION LAYERS SETTINGS #
###############################
conv_layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7']

kernel_sizes = [
    (4, 4),
    (3, 3),
    (3, 3),
    (3, 3),
    (3, 3),
    (3, 3),
    (3, 3)
]
kernel_sizes = {name: kernel_sizes[i] for i, name in enumerate(conv_layer_names)}

num_filters = [
    16, 32, 48, 64, 64, 64, 64
]
num_filters = {name: num_filters[i] for i, name in enumerate(conv_layer_names)}

strides = [

    (2, 2),
    (2, 2),
    (1, 1),
    (1, 1),
    (1, 1),
    (1, 1),
    (1, 1)
]
strides = {name: strides[i] for i, name in enumerate(conv_layer_names)}

paddings = [
    'VALID',
    'VALID',
    'VALID',
    'VALID',
    'VALID',
    'VALID',
    'VALID'
]
paddings = {name: paddings[i] for i, name in enumerate(conv_layer_names)}

# DROPOUT
output_sizes = {}

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

# DICTIONARY SIZE CHECK CONV LAYERS
assert (len(conv_layer_names)
        == len(paddings)
        == len(strides)
        == len(num_filters)
        == len(kernel_sizes)
        == len(output_sizes)), \
        print("Error: sizes of parameter dictionaries of conv layers dont't match. "
              "Printing len [conv_layer_names, paddings, strides,num_filers, kernel_sizes, output_sizes]",
              len(conv_layer_names), len(paddings), len(strides),
              len(num_filters), len(kernel_sizes), len(output_sizes)
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

        # 'fc2': tf.Variable(tf.truncated_normal(
        #        [num_hidden[0],
        #         num_hidden[1]],
        #        stddev=np.sqrt(2 / (num_hidden[0])))
        # ),

        'out': tf.Variable(tf.truncated_normal(
            [num_hidden[0], num_classes], stddev=np.sqrt(2 / (num_hidden[0]))))
    }

    # vytvor vahy pre ostatne konvolucne vrstvy
    for l, l_prev in zip(conv_layer_names[1:], conv_layer_names[:-1]):
        weights[l] = tf.Variable(tf.random_normal(
            [kernel_sizes[l][0], kernel_sizes[l][1], num_filters[l_prev], num_filters[l]],
            stddev=np.sqrt(2 / (kernel_sizes[l][0] * kernel_sizes[l][1] * num_filters[l_prev])))
        )

    biases = {
        'fc1': tf.Variable(tf.zeros([num_hidden[0]])),
        # 'fc2': tf.Variable(tf.zeros([num_hidden[1]])),
        'out': tf.Variable(tf.zeros([num_classes]))
    }

    # vytvor biasy pre ostatne konvolucne vrstvy
    for l in conv_layer_names:
        biases[l] = tf.Variable(tf.zeros([num_filters[l]])),

    # Model
    log = []

    def model(data):
        # INPUT je teraz velkosti batch x h x w x ch
        log.append('input: ' + str(data.get_shape().as_list()))
        out = data

        # ak chces menit konvolucne vrstvy, robi sa to hore pod settings
        # ------convolution-------
        for i, l in enumerate(conv_layer_names):
            out = tf.nn.conv2d(out, weights[l], [1, strides[l][0], strides[l][1], 1], padding=paddings[l])
            out = tf.nn.relu(out + biases[l])

            log.append('KERNEL=' + str(kernel_sizes[l]) + ' STRIDE=' + str(strides[l]) + ' ' + paddings[l])
            log.append(l + ': ' + str(out.get_shape().as_list()))

        # -------------------------

        shape = out.get_shape().as_list()

        # reshape:
        out = tf.reshape(out, [-1, shape[1] * shape[2] * shape[3]])
        log.append('after reshape: ' + str(out.get_shape().as_list()))

        # fully connected

        out = tf.matmul(out, weights['fc1']) + biases['fc1']
        out = tf.nn.dropout(out, tf_keep_prob_fc)
        out = tf.nn.relu(out)

        log.append('fc1: ' + str(out.get_shape().as_list()))

        # out = tf.matmul(out, weights['fc2']) + biases['fc2']
        out = tf.nn.dropout(out, tf_keep_prob_fc)
        out = tf.nn.relu(out)

        log.append('fc2: ' + str(out.get_shape().as_list()))
        out = tf.nn.relu(out)

        print('\n'.join(log))
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
    step = -1

    # logovanie vysledkov
    if os.path.isfile("logs/{}.ckpt".format(session_log_name)):
        try:
            saver.restore(session, "logs/{}.ckpt".format(session_log_name))
        except:
            print("You probably have changed the model architecture."
                  " Please change the 'session_log_name' variable, tooo.")
            session_log_name = input("Type new session_log_name:")
            saver.restore(session, "logs/{}.ckpt".format(session_log_name))
        logfile = open('logs/{}.txt'.format(session_log_name), 'r+')
        current_log = logfile.read().split('\n')
        step_0 = int(current_log[0])
        arch_size = int(current_log[1])
        current_log = current_log[arch_size+1:]
        current_log.reverse()
        logfile.close()
    else:
        session.run(initialization)
        logfile = open('logs/{}.txt'.format(session_log_name), 'w')
        logfile.close()
        current_log = []
        step_0 = 0

    print('------------------------')
    print('Training {}'.format(session_log_name))
    print('------------------------')

    (batch_data_valid, batch_labels_valid) = valid_data.next_batch()

    # Timer
    cas = time.time()  # casovac
    subprocess.call(['speech-dispatcher'])  # start speech dispatcher
    step_counter = 0
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
            print('Minibatch loss at step {}: {}'.format(step + step_0, loss_value))
            print('Minibatch accuracy:', 100 * accuracy(predictions, batch_labels))

            valid_predictions = session.run(
                prediction,
                feed_dict={tf_dataset: batch_data_valid,
                           tf_keep_prob_fc: 1}
            )
            print('Validation accuraccy (batch-sized subset)',
                  100 * accuracy(valid_predictions, batch_labels_valid)
                  )
            print('------------------------')

        # if step == num_steps: pokracovat = 0
        if (step % num_steps) == 0:
            print("{} steps took {} minutes.".format(num_steps, (time.time()-cas)/60))
            cas = time.time()
            subprocess.call(['spd-say', 'Oh yeah! Its over, baby! Step {}. Continue?'.format(num_steps*step_counter)])
            step_counter += 1
            continue_training = (input('Continue? 1/0'))
            # continue_training = '0'
            if step != 0:
                current_log.append('Minibatch loss at step {}: {}'.format(step + step_0, loss_value))
                current_log.append('Minibatch accuracy: '+str(100 * accuracy(predictions, batch_labels)))
                current_log.append('Validation accuracy (batch-sized subset): '
                                   + str(100 * accuracy(valid_predictions, batch_labels_valid)))
                current_log.append('------------------------------------------------------')

    save_path = saver.save(session, "{}/logs/{}.ckpt".format(url, session_log_name))

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

print('(prediction, true label):', list(zip([np.argmax(r) for r in np.array(results)],
                                            [np.argmax(r) for r in np.array(valid_labels)])))
# print('lab', [np.argmax(r) for r in np.array(valid_labels)])

df = pd.DataFrame(znacky)
df['pred'] = [[znacky[np.argmax(r)] for r in np.array(results)].count(zn) for zn in znacky]
df['pred_pc'] = np.array([[znacky[np.argmax(r)] for r in results].count(zn) for zn in znacky]) / results.shape[0]
df['tr_lbls'] = [train_data.all_labels.count(zn) for zn in znacky]
df['tr_lbls_pc'] = np.array([train_data.all_labels.count(zn) for zn in znacky])\
                            / len(train_data.all_labels)
df['val_lbls'] = [[znacky[np.argmax(r)] for r in valid_labels].count(zn) for zn in znacky]
df['val_lbls_pc'] = np.array([[znacky[np.argmax(r)] for r in valid_labels].count(zn) for zn in znacky])\
                            / valid_labels.shape[0]
print(df)

print('accuracy', accuracy(results, valid_labels))
current_log.append('Validation accuracy (full) after {} steps: '.format(step+step_0)
                   + str(accuracy(results, valid_labels)))
current_log.append('------------------------------------------------------')

current_log.reverse()
logfile = open('logs/{}.txt'.format(session_log_name), 'w')
logfile.write(str(step + step_0)+'\n')
logfile.write(str(len(log)+3)+'\n')
logfile.write('\n'.join(log) + '\n\n\n')
logfile.write('\n'.join(current_log))
logfile.close()
