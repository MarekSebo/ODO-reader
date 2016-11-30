from PIL import Image as pilimg
import numpy as np
import os

import tensorflow as tf

from data_class import data_class

#PARAMETRE_NN-------------------------------------------
num_steps = 100000
batch_size = 16
info_freq = 100
session_log_name = 'go1'

num_hidden = [120,80]
num_hidden_lstm = 64

keep_prob_fc = 0.5

#learning_rate = 0.04

chunk_size = 1024


#-------------------------------------------------------
#cesta k obrazkom
url = '/home/andrej/tf/auta/images'
#url = '/home/marek/kody/ODO ocr/images'



#konstanty
abeceda = ['0','1','2','3','4','5','6','7','8','9','.','-']
nr_classes = len(abeceda)+1

#nasekanie obrazkov
image_height = 30
channels = 3
cut_width = 20
stride = 2
#------------------



# load and prepare data
def max_width(data_path, target_h):
    max_width = 0
    for f in os.listdir(data_path):
        im = pilimg.open(os.path.join(data_path, f))
        wpercent = target_h / float(im.size[1])
        wsize = int((float(im.size[0]) * float(wpercent)))
        im = np.array(im.resize((wsize, target_h))).astype(float) / 255
        max_width = max(max_width, im.shape[1])
    return max_width


def cut_image(image_data):
    img_width = np.shape(image_data)[1]

    cuts = []
    for i in range(0, img_width - cut_width, stride):
        cuts.append(image_data[:, i:i + cut_width, :])
    return cuts


def onehot(y):
    result = np.zeros(nr_classes)
    result[abeceda.index(y)] = 1
    return result


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def draw(image):
    plt.imshow(image)
    plt.show()


def sparse(ys):
    # y's je list stringov
    indeces = []
    values = []
    # seqlen = []

    for i, batch in enumerate(ys):
        for j, c in enumerate(ys[i]):
            if c in abeceda:
                indeces.append([i, j])
                values.append(abeceda.index(c))

                # seqlen.append(j+1)

    return np.array(indeces), np.array(values)  # , np.array(seqlen)


# ako urobit vyslednu sekvenciu
# 1,1,2,12,5 -> 125
def output_seq(seq):
    last = None
    result = []
    for item in seq:
        if item == last:
            continue
        else:
            last = item
            if item != 12: result.append(item)
    return result


# velkost obrazka po aplikovani conv vrstiev
def conv_output_size(padding, input_height, input_width, stride, kernel_height, kernel_width):
    if (padding == "VALID"):
        output_height = (input_height - kernel_height) / stride + 1
        output_width = (input_width - kernel_width) / stride + 1
        output_height = int(np.floor(output_height))
        output_width = int(np.floor(output_width))
    if (padding == "SAME"):
        output_height = (input_height) / stride
        output_width = (input_width) / stride
        output_height = int(np.ceil(output_height))
        output_width = int(np.ceil(output_width))
    return output_height, output_width
    # pouzitie
    # c1_output_size=conv_output_size("valid", 20, 10, 1, 5, 3)
    # c2_output_size=conv_output_size("valid", c1_output_size[0], c1_output_size[1],1, 3, 3)


def accuracy(predictions, labels, seq_lengths):
    correct = 0
    for j in range(batch_size):
        result = []
        for i in range(seq_lengths[j]):
            result.append(np.argmax(predictions[i][j]))
        if ''.join(list(np.array(abeceda)[output_seq(result)])) == labels[j]: correct += 1
    return correct


maxi = (max_width(url,image_height)-cut_width)//stride +1
train_data = data_class(url, batch_size, chunk_size, image_height, cut_width, stride, maxi)
test_data = data_class(url, batch_size, chunk_size, image_height, cut_width, stride, maxi, data_use="test")




###############################
# CONVOLUTION LAYERS SETTINGS #
###############################
conv_layer_names = ['conv1', 'conv2', 'conv3', 'conv4']

kernel_sizes = [
    (5, 3),
    (3, 3),
    (2, 2),
    (2, 2)
]
kernel_sizes = {name: kernel_sizes[i] for i, name in enumerate(conv_layer_names)}

num_filters = [
    16, 24, 28, 32
]
num_filters = {name: num_filters[i] for i, name in enumerate(conv_layer_names)}

strides = [

    (1, 1),
    (1, 1),
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
keep_prob_conv = np.array([0.9, 0.8, 0.7, 0.6])
keep_prob_conv_ones = np.ones_like(keep_prob_conv)

output_sizes = {

}

for i, layer in enumerate(conv_layer_names):
    if i == 0:
        output_sizes[layer] = conv_output_size(
            paddings[layer],
            image_height, cut_width,
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
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(None, maxi, image_height, cut_width, channels))
    tf_train_labels_ctc = tf.placeholder(tf.string, shape=(batch_size))
    tf_seq_len = tf.placeholder(tf.int32, shape=(None))

    # tf_test_dataset = tf.constant(X_test,tf.float32)

    # dropout
    tf_keep_prob_conv = tf.placeholder(tf.float32, shape=len(conv_layer_names))
    tf_keep_prob_fc = tf.placeholder(tf.float32)

    # SPARSE VECTOR
    target_index = tf.placeholder(tf.int64)
    target_value = tf.placeholder(tf.int32)
    target_shape = tf.constant([batch_size, maxi], tf.int64)
    target_y = tf.SparseTensor(target_index, target_value, target_shape)

    tf_step = tf.placeholder(tf.int32)

    weights = {
        'conv1': tf.Variable(tf.random_normal(
            [1, kernel_sizes['conv1'][0], kernel_sizes['conv1'][1], channels, num_filters['conv1']],
            stddev=np.sqrt(2 / (kernel_sizes['conv1'][0] * kernel_sizes['conv1'][1] * channels)))
        ),

        'fc1': tf.Variable(tf.truncated_normal(
            [output_sizes[conv_layer_names[-1]][0] * output_sizes[conv_layer_names[-1]][1] * num_filters[
                conv_layer_names[-1]],
             num_hidden[0]],
            stddev=np.sqrt(2 / (
            output_sizes[conv_layer_names[-1]][0] * output_sizes[conv_layer_names[-1]][1] * num_filters[
                conv_layer_names[-1]])))
        ),

        'fc2': tf.Variable(tf.truncated_normal(
            [num_hidden[0],
             num_hidden[1]],
            stddev=np.sqrt(2 / (num_hidden[0])))
        ),

        'out': tf.Variable(tf.truncated_normal(
            [num_hidden_lstm, nr_classes], stddev=np.sqrt(2 / (num_hidden_lstm))))
    }

    # vytvor vahy pre ostatne konvolucne vrstvy
    for l, l_prev in zip(conv_layer_names[1:], conv_layer_names[:-1]):
        weights[l] = tf.Variable(tf.random_normal(
            [1, kernel_sizes[l][0], kernel_sizes[l][1], num_filters[l_prev], num_filters[l]],
            stddev=np.sqrt(2 / (kernel_sizes[l][0] * kernel_sizes[l][1] * num_filters[l_prev])))
        )

    biases = {
        'fc1': tf.Variable(tf.zeros([num_hidden[0]])),
        'fc2': tf.Variable(tf.zeros([num_hidden[1]])),
        'out': tf.Variable(tf.zeros([nr_classes]))
    }

    # vytvor biasy pre ostatne konvolucne vrstvy
    for l in conv_layer_names:
        biases[l] = tf.Variable(tf.zeros([num_filters[l]])),

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden_lstm, state_is_tuple=True)


    # Model
    def model(data):
        # INPUT je teraz velkosti batch x time x h x w x ch
        # ja chcem kazdy time prehnat conv,
        # aby som dostal vystup batch x time x h x w x kernels

        print('input:', data.get_shape().as_list())

        # CONVOLUTIONS
        # 3d convolution sa stara o casovy rozmer
        # kernel size v casovom rozmere je 1, cize je to ekvivalentne 2d conv sucasne na viac obrazkov
        out = data

        # ak chces menit konvolucne vrstvy, robi sa to hore pod settings
        # ------convolution-------
        for i, l in enumerate(conv_layer_names):
            out = tf.nn.conv3d(out, weights[l], [1, 1, strides[l][0], strides[l][1], 1], padding=paddings[l])
            #out = tf.nn.dropout(out, tf_keep_prob_conv[i])
            out = tf.nn.relu(out + biases[l])

            print(l, ':', out.get_shape().as_list())
        # -------------------------

        shape = out.get_shape().as_list()

        # reshape:
        out = tf.reshape(out, [-1, shape[1], shape[2] * shape[3] * shape[4]])
        print('after reshape:', out.get_shape().as_list())

        # fully connected

        out = tf.einsum('ijk,lk->ijl', out, tf.transpose(weights['fc1'])) + biases['fc1']
        out = tf.nn.dropout(out, tf_keep_prob_fc)
        out = tf.nn.relu(out)

        print('fc1:', out.get_shape().as_list())

        out = tf.einsum('ijk,lk->ijl', out, tf.transpose(weights['fc2'])) + biases['fc2']
        out = tf.nn.dropout(out, tf_keep_prob_fc)
        out = tf.nn.relu(out)

        print('fc2:', out.get_shape().as_list())

        # LSTM
        out, state_lstm = tf.nn.dynamic_rnn(
            cell=lstm_cell, inputs=out, dtype=tf.float32, sequence_length=tf_seq_len)

        print('LSTM:', out.get_shape().as_list())

        return [tf.matmul(out[:, i, :], weights['out']) + biases['out'] for i in range(maxi)]


    # compute output activations and loss
    output = model(tf_train_dataset)

    loss = tf.reduce_mean(
        tf.nn.ctc_loss(output,
                       target_y,
                       tf_seq_len,
                       # time_major=True
                       ))

    # Optimizer.
    optimizer = tf.train.RMSPropOptimizer(0.0001, decay=0.9, momentum=0, epsilon=1e-10).minimize(loss)

    prediction = [tf.nn.softmax(output[i]) for i in range(maxi)]

    saver = tf.train.Saver()
    initialization = tf.initialize_all_variables()


with tf.Session(graph=graph) as session:
    if os.path.isfile("abcdefghij/{}.ckpt".format(session_log_name)):
        saver.restore(session, "abcdefghij/{}.ckpt".format(session_log_name))
    else:
        session.run(initialization)


    print('------------------------')
    print('Training...')
    print('------------------------')

    batch_data_test, batch_labels_test, batch_seq_len_test = test_data.next_batch()

    step = -1
    pokracovat = '1'
    while pokracovat == '1':
        step += 1
        # offset = (step * batch_size) % (y_train_ctc.shape[0] - batch_size)
        batch_data, batch_labels, batch_seq_len = train_data.next_batch()
        batch_target_index, batch_target_value = sparse(batch_labels)

        feed_dict = {tf_train_dataset: batch_data, tf_seq_len: batch_seq_len,
                     target_index: batch_target_index, target_value: batch_target_value,
                     tf_keep_prob_conv: keep_prob_conv, tf_keep_prob_fc: keep_prob_fc}
        session.run(
            [target_y], feed_dict=feed_dict)

        _, l, predictions = session.run(
            [optimizer, loss, prediction], feed_dict=feed_dict)



        if (step % info_freq == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy:', 100 * accuracy(predictions, batch_labels, batch_seq_len) / batch_size)

            # test accuracy



            test_predictions = session.run(
                prediction,
                feed_dict={tf_train_dataset: batch_data_test,
                           tf_seq_len: batch_seq_len_test,
                           tf_keep_prob_conv: keep_prob_conv_ones, tf_keep_prob_fc: 1}
            )
            print('Test accuraccy (batch-sized subset)',
                  100 * accuracy(test_predictions, batch_labels_test, batch_seq_len_test) / batch_size)
            print('------------------------')

        if (step % num_steps) == 0:
            pokracovat = (input('Continue? 1/0'))

    save_path = saver.save(session, "abcdefghij/{}.ckpt".format(session_log_name))
    # print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
    results = []
    seq_test = []
    lab_test = []
    for offset in range(0, test_data.size - batch_size + 1, batch_size):
        data, lab, seq_len = test_data.next_batch()
        results.append((session.run(
            prediction,
            feed_dict={tf_train_dataset: data,
                       tf_seq_len: seq_len,
                       tf_keep_prob_conv: keep_prob_conv_ones, tf_keep_prob_fc: 1}
        )))
        seq_test.append(seq_len)
        lab_test.append(lab)



seq_test = np.array(seq_test).reshape(-1)
lab_test = np.array(lab_test).reshape(-1)
results = np.transpose(np.array(results),[1,0,2,3]).reshape((maxi,-1,nr_classes))


print('---------------')
print('test accuracy')
print('---------------')
print('EXAMPLES')
print('---------------')

correct=0
for j in range(results.shape[1]):
    result=[]
    for i in range(seq_test[j]):
        result.append(np.argmax(results[i][j]))
    if j<10:
        print('predict', ''.join(list(np.array(abeceda)[output_seq(result)])))
        print('correct',lab_test[j])
        print('-----------')
    if ''.join(list(np.array(abeceda)[output_seq(result)]))==lab_test[j]: correct+=1
print('overall accuracy',correct/(j+1),'(',correct,'/',j+1,')')

print('-------------------')
print('mini-batch accuracy')
print('-------------------')

correct = 0
for j in range(batch_size):
    result = []
    for i in range(batch_seq_len[j]):
        result.append(np.argmax(predictions[i][j]))

    print('predict', ''.join(list(np.array(abeceda)[output_seq(result)])))
    print('correct', batch_labels[j])
    print('-----------')
    if ''.join(list(np.array(abeceda)[output_seq(result)])) == batch_labels[j]:
        correct += 1
print('accuracy', correct / (j + 1), '(', correct, '/', j + 1, ')')