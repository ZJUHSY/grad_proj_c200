import sys
import getopt
import os
from tqdm import tqdm
import json
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F

from database import CustomDataset
from model import TextCNN
from config import Config
import argparse
import fasttext
import numpy as np

def get_f1(precision, recall):
    return  2.0 * precision * recall / (precision + recall)

#write word vectors
def prepare_test_data(test_path, word_num=50):
    test_write_path = test_path + '.dat'
    if not os.path.isfile(test_write_path):
        pbar = tqdm(total=100)
        fasttext_model = fasttext.load_model('./data/model/0.3_45_3.bin')
        with open(test_path) as f:
            with open(test_write_path, 'a') as outf:
                cnt = 0
                for l in f:
                    cnt += 1
                    if  cnt % 100 == 0:
                        print('-----------------' + str(cnt) + '-----------------')
                    cur_json = json.loads(l.strip())
                    sentences = cur_json['Sentences']
                    para_vec = []
                    for sentence in sentences:
                        if len(sentence) > 2 * word_num:
                            sentence = sentence[0 : 2 * word_num]
                        sentence_vec = torch.FloatTensor([fasttext_model.get_word_vector(word) for word in sentence])
                        para_vec.append(sentence_vec)
                    cur_json['encodings'] = para_vec
                    del cur_json['reviewText']
                    del cur_json['Sentences']
                    json.dump(cur_json, outf)
                    outf.write('\n')

                    pbar.update(10)
        pbar.close()
    else:
        return 

def test(cnn, test_path):
    right, total = 0, 0
    prepare_test_data(test_path)
    
    with open(test_path + '.dat') as inp:
        cnt = 0
        for line in inp:
            cur_json = json.loads(line.strip())
            para_vec, label = torch.FloatTensor(cur_json['encodings']), int(cur_json['ratings'])
            label = label.to(dtype=torch.int64)
            #print(vec.shape)
            pred_v = []
            for sen_vec in para_vec:
                
                output = cnn(sen_vec)
                pred = torch.max(output, 1)[1]
                pred_v.append(pred)
            pred_score = np.mean(pred_v)
            overall_pred = round(pred_score)  #average mean for a para

            if cnt == 0:
                labels = label
                targ = overall_pred
            else:
                targ = torch.cat((targ, overall_pred), dim=0)
                labels = torch.cat((labels, label), dim=0)
            cnt += 1
            
            right += label[targ == label].size(0)
            total += label.size(0)
            
    pred = targ
    #Get metrics for each kind of labels
    for tag in range(1, 6):
        print('For Label ' + str(tag) + ': ')
        true_pos = labels[(pred == tag) & (labels == pred)].size(0)
        pred_pos = labels[pred == tag].size(0)
        label_pos = labels[labels == tag].size(0)
        precision = true_pos / pred_pos
        recall =  true_pos / label_pos
        F1 = get_f1(precision, recall)
        print("precision: {} ".format(precision), end="")
        print("recall: {} ".format(recall), end="")
        print("F1: {} ".format(F1), end="")

    accuracy = float(right) / total
    print("accuracy: {} ".format(accuracy), end="")

if __name__ == "__main__":  
    # Hyper Parameters
    torch.manual_seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out_channel', type=int, default=2)
    parser.add_argument('--label_num', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    
    torch.manual_seed(args.seed)
    config = Config(sentence_max_size=50,
                    batch_size=args.batch_size,
                    word_num=11000,
                    label_num=args.label_num,
                    learning_rate=args.lr,
                    cuda=args.gpu,
                    epoch=args.epoch,
                    out_channel=args.out_channel)
    #use CUDA to speed up
    use_cuda = torch.cuda.is_available() and config.cuda

    #get data
    train_loader = Data.DataLoader(dataset = CustomDataset(path="./data/train.json"), batch_size = config.batch_size, shuffle = True)
    # test_loader = Data.DataLoader(dataset = CustomDataset(path="./data/test.json"), batch_size = config.batch_size, shuffle = False)

    #initialize model
    cnn = TextCNN(config)

    if use_cuda:
        cnn = cnn.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr = config.lr, weight_decay=0.0005)

    #train
    for epoch in range(config.epoch):
        print("epoch :")
        if epoch % 5 == 0:
            test(cnn, './data/test.json')
        for step, data in enumerate(train_loader):
            vec, lens, label = data
            #print(vec.shape)
            if use_cuda:
                vec = vec.cuda()
                label = label.cuda()
            output = cnn(vec)
            label = label.to(dtype=torch.int64)
            loss = F.cross_entropy(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
   
            #output process every 1000 batch
            if step % 1000 == 0:
                pred = torch.max(output, 1)[1]
                accuracy = float(label[pred == label].size(0)) / float(label.size(0))
                print('Epoch:', epoch, '|| Loss:%.4f' % loss, '|| Accuracy:%.3f' % accuracy)
        cnn.save('checkpoints/epoch{}.ckpt'.format(epoch))

    test(cnn, './data/test.json')
