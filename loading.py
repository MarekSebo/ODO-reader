import numpy as np
import os
import pandas as pd
from PIL import Image as pilimg
from numpy import random

def file_to_model(f):
    if f.split("_")[1] == "Alfa":  # or f.split("_")[1] == "Land":
        model = f.split("_")[1] + '_' + f.split('_')[4].split('-')[0] + '_' + f.split('_')[4].split('-')[1]
    else:
        model = f.split("_")[1] + '_' + f.split('_')[4].split('-')[0]
    return model

def zoznam_tried(url, trieda):
    # trieda = {'znacky', 'modely'}
    all_img = os.listdir(os.path.join(url, 'images/'))

    if(trieda == 'znacky'):
        triedy = [f.split("_")[1] for f in all_img]
    elif(trieda == 'modely'):
        triedy = [file_to_model(f) for f in all_img]
    else:
        print("Error: zle zadanie triedy aut. oprav v kode ty dilino!")
        return
    triedy = list(set(triedy))
    triedy.sort()

    return triedy

def split_images(url, train_perc, h, w):
    # h, w - height, width obrazkov (ak chceme resize)

    # vytvor foldre ak nie su
    dir_train=os.path.join(url, 'train/')
    if not os.path.exists(dir_train):
        os.mkdir(dir_train)
    dir_valid = os.path.join(url, 'valid/')
    if not os.path.exists(dir_valid):
        os.mkdir(dir_valid)

    # rozdel a resizni obrazky
    all_img = os.listdir(os.path.join(url, 'images/'))
    print("I have found {} images in directory {}. I will allocate {} percent of them in train dataset".format(
        len(all_img), url, train_perc * 100))
    # index posledneho train obrazku
    split = int(np.floor(train_perc *len(all_img)))
    if len(os.listdir(os.path.join(url, 'train/'))) != split:
        for f in all_img[:split]:
            im = pilimg.open(os.path.join(url, 'images/', f))
            im = im.resize((w, h))
            im.save(os.path.join(url, 'train/', f))

        for f in all_img[split:]:
            im = pilimg.open(os.path.join(url, 'images/', f))
            im = im.resize((w, h))
            im.save(os.path.join(url, 'valid/', f))


def split_images_equal(url, train_perc, h, w, trieda):
    #h, w - height, width obrazkov (ak chceme resize)
    # trieda = {'znacky', 'modely'}

    # vytvor foldre ak nie su
    dir_train = os.path.join(url, 'train_equal_'+trieda+'/')
    if not os.path.exists(dir_train):
        os.mkdir(dir_train)
    dir_valid = os.path.join(url, 'valid_equal_'+trieda+'/')
    if not os.path.exists(dir_valid):
        os.mkdir(dir_valid)

    all_img = os.listdir(os.path.join(url, 'images/'))
    print("I have found {} images in directory {}. I will try to allocate {} percent of them in train dataset with equal distribution.".format(
        len(all_img), url, train_perc * 100))
    # list znaciek

    if trieda == 'znacky':
        triedy_all = [f.split("_")[1] for f in all_img]
    elif trieda == 'modely':
        triedy_all = [f.split("_")[1] + '_' + f.split('_')[4].split('-')[0] for f in all_img]
    else:
        print("Zle si zadal nazov triedy, ty dilino!")
        return
    triedy = (list(set(triedy_all)))
    triedy_all = list(triedy_all)
    print("Celkom je {} unikátnych tried typu: {}".format(len(triedy), trieda))

    # zisti ich pocty
    triedy_poc = [triedy_all.count(tr) for tr in triedy]
    triedy_cutoffs = [int(np.ceil(train_perc * i)) for i in triedy_poc]

    # list kde element su indexy, na ktorych su obrazky danej triedy auta
    indexy_znaciek = [[i for i in range(len(triedy_all)) if triedy_all[i] == tr] for tr in triedy]

    valid_indices = []
    for i in range(len(triedy_cutoffs)):
        valid_indices = valid_indices + indexy_znaciek[i][triedy_cutoffs[i]:]
    train_indices = np.array(list(set(range(len(all_img))) - set(list(np.array(valid_indices)))), dtype='int')
    valid_indices = np.array(list(valid_indices), dtype='int')

    all_img = np.array(all_img)
    # v kazdom najdi train_cut_index
    print()
    if len(os.listdir(dir_train)) != len(train_indices):
        for f in all_img[train_indices]:
            im = pilimg.open(os.path.join(url, 'images/', f))
            im = im.resize((w, h))
            im.save(os.path.join(dir_train, f))

        for f in all_img[valid_indices]:
            im = pilimg.open(os.path.join(url, 'images/', f))
            im = im.resize((w, h))
            im.save(os.path.join(dir_valid, f))

    print("V train_equal directory je {} percent dat.".format(
        100 * len(os.listdir(dir_train)) / len(all_img)))


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
        self.all_labels = [file_to_model(f) for f in all_img]
        return np.array(names)

    def shuffle(self):
        random.shuffle(self.file_names)

    def load_chunk(self):
        chunk_imgs = []
        chunk_labels = []

        for f in self.file_names[self.chunk_cursor:self.chunk_cursor + self.chunk_size]:
            im = pilimg.open(os.path.join(self.path, f))

            if self.data_use == "train":
                # random crop or reshape
                if random.randint(0, 2) == 0:
                    x = random.randint(self.w - self.cut_w)
                    y = random.randint(self.h - self.cut_h)
                    im = im.crop((x, y, x + self.cut_w, y + self.cut_h))
                else:
                    im = im.resize((self.cut_w, self.cut_h))
            else:  #validacne data
                # no crop
                im = im.resize((self.cut_w, self.cut_h))

            im = np.array(im).astype(float) / 255
            chunk_imgs.append(im)
            chunk_labels.append(self.labels_to_onehot(file_to_model(f)))
            print(f, file_to_model(f))
        self.chunk_cursor = (self.chunk_cursor + self.chunk_size)

        self.current_chunk_size = len(chunk_imgs)

        # docasne riesenie
        if self.chunk_cursor + self.chunk_size > self.total_data_size:
            print('last chunk of the epoch')
            self.chunk_cursor = 0
            self.shuffle()

        return np.array(chunk_imgs), np.array(chunk_labels)

    def next_chunk(self):
        # print('Getting new chunk')
        self.data, self.labels = self.load_chunk()
        # print('Got it')

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
        onehots[self.car_makes.index(lab)] = 1
        return list(onehots)
