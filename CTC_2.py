from PIL import Image as pilimg
import numpy as np
import matplotlib.pyplot as plt

import os

import tensorflow as tf


#PARAMETRE_NN-------------------------------------------
num_steps = 1000
batch_size = 16

num_hidden = 100
num_hidden_lstm = 50

learning_rate = 0.04
#-------------------------------------------------------
url = '/home/marek/kody/ODO ocr/images'
    #'/home/andrej/tf/auta/images'

#konstanty
abeceda = ['0','1','2','3','4','5','6','7','8','9','.','-']
nr_classes = len(abeceda)+1

#nasekanie obrazkov
image_height = 20
channels = 3
cut_width = 10
stride = 3
#------------------




#FUNCTION DEFINITIONS

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




#nacitanie datasetov---------------------------
#all_images = nr_images x height x width x channels
all_images,labels = transform_example(url,image_height)

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


nr_train_data = len(all_images)*4//5

X_train = data_vyrezy[:nr_train_data]
X_test = data_vyrezy[nr_train_data:]

#y_train = np.array([onehot(label_vyrezy[i]) for i in range(nr_train_data)])
#y_test = np.array([onehot(label_vyrezy[i]) for i in range(nr_train_data,nr_images)])
y_train_ctc = np.array(labels[:nr_train_data])
#y_test = np.array(labels[nr_train_data:])

y_test_ctc = np.array(labels[nr_train_data:])

seq_train = seq_len[:nr_train_data]
seq_test = seq_len[nr_train_data:]

test_data_size = len(seq_test)





#NN------------------------------------------------------------------------

conv_layer_names = ['conv1', 'conv2', 'conv3', 'conv4']

kernel_sizes = [
    (5, 3),
    (3, 3),
    (2, 2),
    (2, 2)
]
kernel_sizes = {name: kernel_sizes[i] for i, name in enumerate(conv_layer_names)}

num_filters = [
    16,
    24,
    28,
    32
]
num_filters = {name: num_filters[i] for i, name in enumerate(conv_layer_names)}

strides = [
    (1, 1),
    (2, 2),
    (1, 1),
    (1, 1)
]
strides = {name: strides[i] for i, name in enumerate(conv_layer_names)}

paddings = [
    'VALID',
    'VALID',
    'VALID',
    'VALID'
]
paddings = {name: paddings[i] for i, name in enumerate(conv_layer_names)}

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


#TF graf
graph = tf.Graph()

with graph.as_default():
    # Input data.
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(None, maxi, image_height, cut_width, channels))
    tf_train_labels_ctc = tf.placeholder(tf.string, shape=(batch_size))
    tf_seq_len = tf.placeholder(tf.int32, shape=(None))

    # tf_test_dataset = tf.constant(X_test,tf.float32)

    # SPARSE VECTOR
    target_index = tf.placeholder(tf.int64)
    target_value = tf.placeholder(tf.int32)
    target_shape = tf.constant([batch_size, maxi], tf.int64)
    target_y = tf.SparseTensor(target_index, target_value, target_shape)

    weights = {
        'conv1': tf.Variable(tf.random_normal(
            [1, kernel_sizes['conv1'][0], kernel_sizes['conv1'][1], channels, num_filters['conv1']],
            stddev=np.sqrt(2 / (kernel_sizes['conv1'][0] * kernel_sizes['conv1'][1] * channels)))
        ),

        'conv2': tf.Variable(tf.random_normal(
            [1, kernel_sizes['conv2'][0], kernel_sizes['conv2'][1], num_filters['conv1'], num_filters['conv2']],
            stddev=np.sqrt(2 / (kernel_sizes['conv2'][0] * kernel_sizes['conv2'][1] * num_filters['conv1'])))
        ),

        'conv3': tf.Variable(tf.random_normal(
            [1, kernel_sizes['conv3'][0], kernel_sizes['conv3'][1], num_filters['conv2'], num_filters['conv3']],
            stddev=np.sqrt(2 / (kernel_sizes['conv3'][0] * kernel_sizes['conv3'][1] * num_filters['conv2'])))
        ),

        'conv4': tf.Variable(tf.random_normal(
            [1, kernel_sizes['conv4'][0], kernel_sizes['conv4'][1], num_filters['conv3'], num_filters['conv4']],
            stddev=np.sqrt(2 / (kernel_sizes['conv4'][0] * kernel_sizes['conv4'][1] * num_filters['conv3'])))
        ),

        'fc1': tf.Variable(tf.truncated_normal(
            [output_sizes['conv4'][0] * output_sizes['conv4'][1] * num_filters['conv4'], num_hidden],
            stddev=np.sqrt(2 / (output_sizes['conv4'][0] * output_sizes['conv4'][1] * num_filters['conv4'])))
        ),

        'out': tf.Variable(tf.truncated_normal(
            [num_hidden_lstm, nr_classes], stddev=np.sqrt(2 / (num_hidden_lstm))))
    }

    biases = {
        'conv1': tf.Variable(tf.zeros([num_filters['conv1']])),
        'conv2': tf.Variable(tf.zeros([num_filters['conv2']])),
        'conv3': tf.Variable(tf.zeros([num_filters['conv3']])),
        'conv4': tf.Variable(tf.zeros([num_filters['conv4']])),
        'fc1': tf.Variable(tf.zeros([num_hidden])),
        'out': tf.Variable(tf.zeros([nr_classes]))
    }

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
        l = 'conv1'
        out = tf.nn.conv3d(data, weights[l], [1, 1, strides[l][0], strides[l][1], 1], padding=paddings[l])
        out = tf.nn.relu(out + biases[l])

        print('conv1:', out.get_shape().as_list())

        l = 'conv2'
        out = tf.nn.conv3d(out, weights[l], [1, 1, strides[l][0], strides[l][1], 1], padding=paddings[l])
        out = tf.nn.relu(out + biases[l])

        shape = out.get_shape().as_list()
        print('conv2:', shape)

        l = 'conv3'
        out = tf.nn.conv3d(out, weights[l], [1, 1, strides[l][0], strides[l][1], 1], padding=paddings[l])
        out = tf.nn.relu(out + biases[l])

        shape = out.get_shape().as_list()
        print('conv3:', shape)

        l = 'conv4'
        out = tf.nn.conv3d(out, weights[l], [1, 1, strides[l][0], strides[l][1], 1], padding=paddings[l])
        out = tf.nn.relu(out + biases[l])

        shape = out.get_shape().as_list()
        print('conv4:', shape)

        # reshape:
        out = tf.reshape(out, [-1, shape[1], shape[2] * shape[3] * shape[4]])
        print('after reshape:', out.get_shape().as_list())

        # fully connected
        out = tf.einsum('ijk,lk->ijl', out, tf.transpose(weights['fc1'])) + biases['fc1']
        out = tf.nn.relu(out)

        print('fc1:', out.get_shape().as_list())

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
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    train_prediction = [tf.nn.softmax(output[i]) for i in range(maxi)]



#TRENOVANIE-----------------------------------------------------

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')

    for step in range(num_steps):
        offset = 0  # (step * batch_size) % (y_train_ctc.shape[0] - batch_size)
        batch_data = X_train[offset:(offset + batch_size), :maxi, :, :, :]

        batch_labels_not_onehot = y_train_ctc[offset:(offset + batch_size)]
        batch_seq_len = seq_train[offset:(offset + batch_size)]

        batch_target_index, batch_target_value = sparse(batch_labels_not_onehot)

        feed_dict = {tf_train_dataset: batch_data, tf_seq_len: batch_seq_len,
                     target_index: batch_target_index, target_value: batch_target_value}

        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        # _, l = session.run(
        #    [optimizer, loss], feed_dict=feed_dict)


        if (step % 100 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            # print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))

    # print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
    results = []
    for offset in range(0, test_data_size - batch_size + 1, batch_size):
        results.append((session.run(
            train_prediction,
            feed_dict={tf_train_dataset: X_test[offset:offset + batch_size, :, :, :, :],
                       tf_seq_len: seq_test[offset:offset + batch_size]}
        )))


results=np.transpose(np.array(results),[1,0,2,3]).reshape((maxi,-1,nr_classes))

print('TESTOVACIE DATA')
for j in range(results.shape[1]):
    result=[]
    for i in range(seq_test[j]):
        result.append(np.argmax(results[i][j]))
    print('predict', output_seq(result))
    print('correct',y_test_ctc[j])

print('last batch trenovacie data')
for j in range(batch_size):
    result=[]
    for i in range(batch_seq_len[j]):
        result.append(np.argmax(predictions[i][j]))
        print(np.argmax(predictions[i][j]))
    print('predict', ''.join(list(np.array(abeceda)[output_seq(result)])))
    print('correct',batch_labels_not_onehot[j])