from PIL import Image as pilimg
import numpy as np
#import matplotlib.pyplot as plt

import os

import tensorflow as tf


#const
abeceda = ['0','1','2','3','4','5','6','7','8','9','.','-']
image_height = 20

#nasekanie obrazkov
channels = 3
cut_width = 10
stride = 3

conv_filters = 24
kernel_size = 3

batch_size = 1
nr_classes = len(abeceda)+1


# load and prepare data

def transform_example(data_path, target_h):
    imgs = []
    label = []
    for f in os.listdir(data_path):
        im = pilimg.open(os.path.join(data_path, f))
        wpercent = target_h / float(im.size[1])
        wsize = int((float(im.size[0]) * float(wpercent)))
        im = np.array(im.resize((wsize, target_h))).astype(float) / 255
        # sc.imsave('new' + f, im)
        imgs.append(im)
        target_str = ''.join(f.split('.', 1))[:-4].split('_')[-1]
        label.append(target_str)
    return imgs, label


def cut_image(image_data):
    img_width = np.shape(image_data)[1]

    cuts = []
    for i in range(0, img_width - cut_width, stride):
        cuts.append(image_data[:, i:i + cut_width, :])
    return cuts


def cut_and_pad(image_data, num_frames):
    img_width = np.shape(image_data)[1]

    cuts = []
    for i in range(0, img_width - cut_width, stride):
        cuts.append(image_data[:, i:i + cut_width, :])

    cuts = np.array(cuts)
    seq_len = cuts.shape[0]
    add = num_frames - seq_len
    cuts = np.pad(cuts, [(0, add), (0, 0), (0, 0), (0, 0)], mode='constant')
    return cuts, seq_len


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
            indeces.append([i, j])
            values.append(abeceda.index(c))

            # seqlen.append(j+1)

    return np.array(indeces), np.array(values)  # , np.array(seqlen)


# ako urobit vyslednu sekvenciu
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


#all_images = nr_images x height x width x channels
all_images,labels = transform_example('/home/andrej/tf/auta/images',image_height)

nr_images = len(all_images)

#zober len prve vyrezy (rovnaky shape ako all_images)
data_vyrezy = np.array([len(cut_image(i)) for i in all_images])
#print('max', max(data_vyrezy))
#maximalny pocet vyrezov:
maxi=max(data_vyrezy)




#labels su stale len prve
label_vyrezy = np.array([s[0] for s in labels])

data_vyrezy=[]
seq_len = []
for img in all_images:
    a,b = cut_and_pad(img,maxi)
    data_vyrezy.append(a)
    seq_len.append(b)

#images x time_frames x h x w x ch
data_vyrezy = np.array(data_vyrezy)
seq_len = np.array(seq_len)












nr_train_data = 1  #len(all_images)*4//5

X_train = data_vyrezy[:nr_train_data]
X_test = data_vyrezy[nr_train_data:]

y_train = np.array([onehot(label_vyrezy[i]) for i in range(nr_train_data)])
#y_test = np.array([onehot(label_vyrezy[i]) for i in range(nr_train_data,nr_images)])
y_train_ctc = np.array(labels[:nr_train_data])
y_test = np.array(labels[nr_train_data:])

y_test_ctc = np.array(labels[nr_train_data:])

seq_train = seq_len[:nr_train_data]
seq_test = seq_len[nr_train_data:]

test_data_size = len(seq_test)





























#BUILD GRAPH
num_hidden = 30
num_hidden_lstm = 40

graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, maxi, image_height, cut_width, channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, nr_classes))
    tf_train_labels_ctc = tf.placeholder(tf.string, shape=(batch_size))
    # tf_valid_dataset = tf.constant(X_test)
    tf_test_dataset = tf.constant(X_test, tf.float32)
    tf_seq_len = tf.placeholder(tf.int32, shape=(None))

    # SPARSE VECTOR
    target_index = tf.placeholder(tf.int64)
    target_value = tf.placeholder(tf.int32)
    target_shape = tf.constant([batch_size, maxi], tf.int64)
    target_y = tf.SparseTensor(target_index, target_value, target_shape)

    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal(
        [1, kernel_size, kernel_size, channels, conv_filters], stddev=np.sqrt(2 / (kernel_size ** 2 * channels))))
    layer1_biases = tf.Variable(tf.zeros([conv_filters]))
    layer2_weights = tf.Variable(tf.truncated_normal(
        [1, kernel_size, kernel_size, conv_filters, conv_filters],
        stddev=np.sqrt(2 / (kernel_size ** 2 * conv_filters))))
    layer2_biases = tf.Variable(tf.constant(0.0, shape=[conv_filters]))

    # kedze mame stride 2 v oboch konvolucnych vrstvach, tak sa vysledok zmensi na stvrtinu
    layer3_weights = tf.Variable(tf.truncated_normal(
        [1, 1, 1, conv_filters, 10], stddev=np.sqrt(2 / (conv_filters))))
    # layer2_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))


    layer4_weights = tf.Variable(tf.truncated_normal(
        [num_hidden_lstm, nr_classes], stddev=np.sqrt(2 / (num_hidden_lstm))))
    layer4_biases = tf.Variable(tf.constant(0.0, shape=[nr_classes]))

    layerFC_weights = tf.Variable(tf.truncated_normal(
        [(image_height // 2 // 2) * (cut_width // 2 // 2) * conv_filters, num_hidden],
        stddev=np.sqrt(2 / (image_height // 2 // 2) * (cut_width // 2 // 2) * conv_filters)))
    layerFC_biases = tf.Variable(tf.constant(0.0, shape=[num_hidden]))

    # o vahy je postarane, LSTM cell ma zjavne svoje vlastne osefene
    with tf.variable_scope("lstm_cell2", reuse=True):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden_lstm, state_is_tuple=True)


        # Model


    def model(data):
        # INPUT je teraz velkosti batch x time x h x w x ch
        # ja chcem kazdy time prehnat conv,
        # aby som dostal vystup batch x time x h x w x kernels

        # data = tf.transpose(data,[1,0,2,3,4])
        # conv1 = tf.zeros([batch_size, maxi, image_height, cut_width, channels], tf.int32)

        # for time in range(maxi):
        #    conv1[:,time,:,:,:] = tf.nn.conv2d(data[:,time,:,:,:], layer1_weights, [1, 1, 1, 1], padding='SAME')


        # conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        # print(conv1.get_shape().as_list())
        conv = tf.nn.conv3d(data, layer1_weights, [1, 1, 2, 2, 1], padding='VALID')
        conv = tf.nn.relu(conv + layer1_biases)

        shape = conv.get_shape().as_list()
        print(shape)

        conv = tf.nn.conv3d(conv, layer2_weights, [1, 1, 2, 2, 1], padding='SAME')
        conv = tf.nn.relu(conv)

        shape = conv.get_shape().as_list()
        print(shape)

        # conv = tf.nn.conv3d(conv, layer3_weights, [1, 1, 1, 1, 1], padding='SAME')
        # conv = tf.nn.relu(conv)

        # shape = conv.get_shape().as_list()
        # print(shape)

        # reshape:
        conv = tf.reshape(conv, [-1, shape[1], shape[2] * shape[3] * shape[4]])
        shape = conv.get_shape().as_list()
        print(shape)

        print('vahy', layerFC_weights.get_shape().as_list())

        # fully connected
        conv = tf.einsum('ijk,lk->ijl', conv, tf.transpose(layerFC_weights)) + layerFC_biases
        conv = tf.nn.relu(conv)

        # with tf.variable_scope("lstm4", reuse = True) as scope:
        output_lstm, state_lstm = tf.nn.dynamic_rnn(
            cell=lstm_cell, inputs=conv, dtype=tf.float32, sequence_length=tf_seq_len)  # tf_seq_len

        shape = output_lstm.get_shape().as_list()

        print(shape)
        # hidden = tf.nn.relu(conv + layer1_biases)
        # shape = hidden.get_shape().as_list()
        # print(shape)
        # shape 0 bude batch index
        # zvysne sa rozvektorizuju tak ako sa to robilo aj s obrazkom (28*28->784), az na to ze teraz je to kocka
        # reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        # hidden = tf.nn.relu(tf.matmul(reshape, layer2_weights) + layer2_biases)
        return [tf.matmul(output_lstm[:, i, :], layer4_weights) + layer4_biases for i in range(maxi)]


    # Training computation.
    # logits = model(tf_train_dataset)[1]
    # print('logit shape',logits.get_shape().as_list())
    # loss = tf.reduce_mean(
    #  tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # inputs je zatial list, uvidime co sa stane
    output = model(tf_train_dataset)

    loss = tf.reduce_mean(
        tf.nn.ctc_loss(output,
                       target_y,
                       tf_seq_len,
                       # time_major=True
                       ))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.04).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = [tf.nn.softmax(output[i]) for i in range(maxi)]
    # valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    # output_test = tf.nn.softmax(model(tf_test_dataset))
    # test_prediction = [output[i] for i in range(maxi)]
    print('Initialized')

num_steps = 100
with tf.Session(graph=graph) as session:
    print('Initialized')
    tf.initialize_all_variables().run()
    print('Initialized')

    for step in range(num_steps):
        offset =  0#(step * batch_size) % (y_train.shape[0] - batch_size)
        batch_data = X_train[offset:(offset + batch_size), :maxi, :, :, :]
        batch_labels = y_train[offset:(offset + batch_size), :]

        batch_labels_not_onehot = y_train_ctc[offset:(offset + batch_size)]
        batch_seq_len = seq_train[offset:(offset + batch_size)]

        batch_target_index, batch_target_value = sparse(batch_labels_not_onehot)

        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, tf_seq_len: batch_seq_len,
                     target_index: batch_target_index, target_value: batch_target_value}
        _, l = session.run(
            [optimizer, loss], feed_dict=feed_dict)
        #session.run(optimizer)

        if (step % 100 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            # print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))

    # print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

    results = session.run(
        train_prediction, feed_dict={tf_train_dataset: X_test, tf_seq_len: seq_test})

#summary_writer = tf.train.SummaryWriter('/logs', sess.graph)


for j in range(batch_size):
    result=[]
    for i in range(maxi):
        result.append(np.argmax(predictions[i][j]))
        print(np.argmax(predictions[i][j]))
    print('predict', ''.join(list(np.array(abeceda)[output_seq(result)])))
    print('correct',batch_labels_not_onehot[j])