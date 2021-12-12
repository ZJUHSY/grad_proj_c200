import os
import json
import re
import random
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import fasttext
import pandas as pd
from tqdm import tqdm


CLIENT_BATCH_SIZE = 4096
WORD_NUM = 64
MIN_SEN_LEN = 5

fasttext_model = fasttext.load_model('../../data/model/0.5_45_3.bin')

class CustomDataset(Dataset):
    def __init__(self, path, word_num = 50):
        if os.path.isfile(path + '.dat'):
            self.data = torch.load(path + '.dat')
            self.label = torch.load(path + '.lab')
            self.start = torch.load(path + '.sta')
            self.end = torch.load(path + '.end')
            self.word_num = word_num
            # print(self.data.shape)
            return 
        self.word_num = word_num

        self.data = None
        self.start = []
        self.end = []
        self.label = []

        pbar = tqdm(total=100)
        with open(path) as f:
            cnt = 0
            for l in f:
                cnt += 1
                if cnt % 100 == 0:
                    print('-----------------' + str(cnt) + '-----------------')
                cur_json = json.loads(l.rstrip())
                sentences = cur_json['Sentences']
                for sentence in sentences:
                    if len(sentence) > 2 * self.word_num:
                        sentence = sentence[0 : 2 * self.word_num]
                    sentence_vec = torch.FloatTensor([fasttext_model.get_word_vector(word) for word in sentence])
                    if self.data is None:
                        self.start += [0]
                        self.data = sentence_vec
                    else:
                        self.start += [self.data.shape[0]]
                        self.data = torch.cat((self.data, sentence_vec), dim = 0)

                    self.end += [self.data.shape[0]]
                    self.label += [cur_json['ratings']]
                    pbar.update(10)
        pbar.close()

        torch.save(torch.FloatTensor(self.data), path + ".dat")
        torch.save(self.label, path + ".lab")
        torch.save(self.start, path + ".sta")
        torch.save(self.end, path + ".end")
        



    def __getitem__(self, index):
        if self.end[index] - self.start[index] <= self.word_num:
            words = self.data[self.start[index] : self.end[index]]
            length = self.end[index] - self.start[index]
            para = torch.cat((words, torch.zeros((self.word_num - (self.end[index] - self.start[index]), 100))), dim=0)
            #print(para.shape)
        else:
            start = random.randint(0, self.end[index] - self.word_num)
            end = start + self.word_num
            length = self.word_num
            para = self.data[start : end]
            # print(para.shape, length, self.label[index])
        return para, length, self.label[index]

    def __len__(self):
        return len(self.label)


if __name__ == '__main__':
    data = CustomDataset('./data/train_raw_data.json', './data/train_data.json')
    # train_loader = Data.DataLoader(dataset = CustomDataset(path="train.json", balance=False), batch_size = BATCH_SIZE, shuffle = True)
