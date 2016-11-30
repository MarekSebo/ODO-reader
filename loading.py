import numpy as np
import os
from PIL import Image as pilimg
import random

path = "/home/andrej/tf/odo/"
all_img = os.listdir(os.path.join(path, 'images/'))
print([f.split("_")[1] for f in all_img])


for f in all_img[:6]:
    im = pilimg.open(os.path.join(path, 'images/', f))
    im = im.resize((256, 192))
    im.save(os.path.join(path, 'train/', f))

for f in all_img[6:8]:
    im = pilimg.open(os.path.join(path, 'images/', f))
    im = im.resize((256, 192))
    im.save(os.path.join(path, 'valid/', f))


class DataClass(object):
    """
    POUZITIE
    -vytvor instanciu meno = DataClass(args)
    -vypytaj si novy batch: meno.next_batch()

    BACKEND
    -data sa nacitavaju v chunkoch zadanej velkosti (kvoli RAM)
    -batche sa vzdy beru zaradom z aktualneho chunku
    -ked sa minie chunk (dojde sa na koniec), nacita sa novy chunk
    -ked sa minu chunky, premiesaju sa data a zacne sa znova

    TODO
    -zatial sa obrazky pri novom chunku vzdy nacitavaju, reshapuju, padduju
    -mohlo by byt vyhodne pouzit jeden chunk viackrat, ked uz ho mam nacitany
    -spravit podclasses na test, train a validation sety
    -idealne ich ulozit do zvlast priecinkov
    """
    def __init__(self, path, batch_size, chunk_size, num_class, data_use="train"):
        self.data = None
        self.labels = None

        self.path = path
        self.data_use = data_use
        self.num_class = num_class
        self.file_names = self.load_filenames()
        self.total_data_size = len(self.file_names)

        self.batch_size = batch_size
        self.batch_cursor = 0              # pozicia batchu vramci chunku

        self.chunk_size = chunk_size        # (chunk_size // batch_size) * batch_size
        self.chunk_cursor = 0           # pozicia chunku vramci datasetu

        self.next_chunk()


    def load_filenames(self):
        all_img = os.listdir(self.path)
        names = all_img
        self.car_makes = list(set([f.split("_")[1] for f in all_img]))
        print(self.car_makes)
        return names

    def shuffle(self):
        random.shuffle(self.file_names)

    def load_chunk(self):
        chunk_imgs = []
        chunk_labels = []

        for f in self.file_names[self.chunk_cursor:self.chunk_cursor + self.chunk_size]:
            im = pilimg.open(os.path.join(self.path, f))
            im = np.array(im).astype(float) / 255

            chunk_imgs.append(im)
            chunk_labels.append(self.labels_to_onehot(f.split("_")[1]))
        self.chunk_cursor = (self.chunk_cursor + self.chunk_size)

        self.current_chunk_size = len(chunk_imgs)

        # docasne riesenie
        if self.chunk_cursor + self.chunk_size > self.total_data_size:
            print('last chunk of the epoch')
            self.chunk_cursor = 0
            self.shuffle()

        return np.array(chunk_imgs), np.array(chunk_labels)

    def next_chunk(self):
        print('Getting new chunk')
        self.data, self.labels = self.load_chunk()
        print('Got it')

    def next_batch(self):
        data = self.data[self.batch_cursor:self.batch_cursor + self.batch_size]
        labels = self.labels[self.batch_cursor:self.batch_cursor + self.batch_size]

        self.batch_cursor += self.batch_size
        if self.batch_cursor + self.batch_size > self.current_chunk_size:
            self.batch_cursor = 0
            self.next_chunk()

        # ak z nejakeho dovodu nie je dost dat do  batchu (napr. malo suborov) tak to sposobi errory
        # tiez len docasne riesenie
        if len(labels) < self.batch_size:
            self.next_batch()

        return data, labels

    def labels_to_onehot(self, lab):
        onehots = np.zeros(self.num_class)
        onehots[self.car_makes.index(lab)]=1
        return list(onehots)
