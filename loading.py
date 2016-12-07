import numpy as np
import os
import pandas as pd
from PIL import Image as pilimg
from numpy import random


def split_images(url, split, h, w):
    all_img = os.listdir(os.path.join(url, 'images/'))
    if len(os.listdir(os.path.join(url, 'train/'))) != split:
        for f in all_img[:split]:
            im = pilimg.open(os.path.join(url, 'images/', f))
            im = im.resize((w, h))
            im.save(os.path.join(url, 'train/', f))

        for f in all_img[split:]:
            im = pilimg.open(os.path.join(url, 'images/', f))
            im = im.resize((w, h))
            im.save(os.path.join(url, 'valid/', f))

    znacky = (list(set([f.split("_")[1] for f in all_img])))
    znacky.remove("S▌Мkoda")
    znacky.sort()

    return znacky


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
    """
    def __init__(self, path, batch_size, chunk_size, num_class, h, w, cut_h, cut_w, car_makes, data_use="train"):
        self.data = None
        self.labels = None

        self.path = path
        self.data_use = data_use
        self.num_class = num_class
        self.car_makes = car_makes

        self.h = h
        self.w = w
        self.cut_h = cut_h
        self.cut_w = cut_w

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
        self.all_labels = [f.split("_")[1] for f in all_img]
        self.all_labels = [l if l != "S▌Мkoda" else "Škoda" for l in self.all_labels]
        return names

    def shuffle(self):
        random.shuffle(self.file_names)

    def load_chunk(self):
        chunk_imgs = []
        chunk_labels = []

        for f in self.file_names[self.chunk_cursor:self.chunk_cursor + self.chunk_size]:
            im = pilimg.open(os.path.join(self.path, f))

            if self.data_use == "train":
                # random crop
                x = random.randint(self.w - self.cut_w)
                y = random.randint(self.h - self.cut_h)
            else:
                # central crop
                x = (self.w - self.cut_w) // 2
                y = (self.h - self.cut_h) // 2

            im = im.crop((x, y, x + self.cut_w, y + self.cut_h))

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
        #print('Getting new chunk')
        self.data, self.labels = self.load_chunk()
        #print('Got it')

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
        if lab=="S▌Мkoda": lab = "Škoda"
        onehots[self.car_makes.index(lab)] = 1
        return list(onehots)
