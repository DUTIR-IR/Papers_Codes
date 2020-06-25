########################----Pytorch
import torch
import torch.nn as nn #用于搭建模型
import torch.optim as optim #用于生成优化函数
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
########################----NLP
from gensim.test.utils import datapath, get_tmpfile
import gensim as gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import LineSentence
from torchtext.vocab import Vectors #用于载入预训练词向量
from torchtext.data import TabularDataset
from torchtext import data #用于生成数据集
from torchtext.data import Iterator, BucketIterator #用于生成训练和测试所用的迭代器
#######################----image模型包
import pretrainedmodels
import pretrainedmodels.utils as utils
#######################----常用包
import numpy as np
import pandas as pd
import csv
import pkuseg
from tqdm import tqdm #用于绘制进度条
import os
#######################----自制程序
from datasetWeibo import DatasetWeibo_Unlabled, DatasetWeibo_Labled  # 请自行将"datasetWeibo数据集建立demo.py"这个文件改成“datasetWeibo.py”
import datasetWeibo
import utils_myself


def funcLoadImgModel(model_name = 'resnet18'):
    print(pretrainedmodels.model_names)
    print(pretrainedmodels.pretrained_settings[model_name])
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model.eval()
    return model
    
    
Resnet = funcLoadImgModel('resnet18')


path_json_weibo_labled = "./data/labled_data/json_weibo_labled.json"
path_json_weibo_unlabled = "./data/json_weibo_unlabled.json"
unlabledset = DatasetWeibo_Unlabled('./data/', data_type='unlabled')
labledset = DatasetWeibo_Labled('./data/', data_type='labled')


def get_dataloader(unlabledset, labledset, num_training=116+71,num_labeled=71,batch_size=10):
#     num_training=50000,num_labeled=3000,batch_size=200
    dataloader={}
    dataloader['labeled'] = torch.utils.data.DataLoader(labledset,
                                                        batch_size=batch_size, shuffle=True,
                                                        num_workers=4)

    dataloader['unlabeled'] = torch.utils.data.DataLoader(unlabledset,
                                                          batch_size=batch_size, shuffle=True,
                                                          num_workers=4)
#     dataloader['test'] = DataLoader(TensorDataset(dataset['test_data'],dataset['test_label']),
#                                     batch_size=500,shuffle=False,num_workers=4)
    return dataloader
    

dataloader_weibo = get_dataloader(unlabledset, labledset, num_training=116+71,num_labeled=71,batch_size=10)



# 加载词向量
word_vectors = KeyedVectors.load_word2vec_format('sgns.weibo.bigram-char', binary=False, unicode_errors='ignore')

# 处理词向量
len(word_vectors.vocab)
tensor_unk = torch.zeros(1,300) # 初始化<unk>词向量
weight_matrix = torch.FloatTensor(word_vectors.vectors) # 构建权重矩阵
weight_matrix = torch.cat((tensor_unk, weight_matrix), dim=0) # 拼接权重矩阵
print(weight_matrix.shape)
vocab_list = [word for word, Vocab in word_vectors.vocab.items()]# 存储 所有的 词语 构建词典库
print(len(vocab_list))


# 给每个单词编码，也就是用数字来表示每个单词，这样才能够传入word embeding得到词向量。
word_to_idx = {'<unk>': 0} # 初始化 `[word : token]` ，后期 tokenize 语料库就是用该词典。使用前必须添加一个索引0.
word_to_idx = {word: i+1 for i, word in enumerate(vocab_list)}
# word_to_vector = {} # 初始化`[word : vector]`字典

idx_to_word = {i+1: word for i, word in enumerate(vocab_list)}
idx_to_word[0] = '<unk>'


def word2idx(word, word_to_idx):
    if(word in word_to_idx.keys()):
        return word_to_idx[word]
    else:
        # word_to_idx['<unk>']
        return 0
        
        
def preprocess_text_every(text_for_img, max_len=64):
    def pad(x):
        return x[:max_len] if len(x) > max_len else x + [0] * (max_len - len(x))
    ##############################################################################
    list_text_for_img = []
    for i, text_every in enumerate(text_for_img):
        tokenized_text_every_for_img = utils_myself.tokenize(text_every)
        idx_text_for_img = pad([word2idx(word, word_to_idx) for word in tokenized_text_every_for_img])
        list_text_for_img.append(idx_text_for_img)
    idx_text_for_img = torch.tensor(list_text_for_img)
    return idx_text_for_img


# 此处仅仅是个演示，具体需要根据你自己的模型修改代码。
epoch_bar = tqdm(range(20))
for epoch in epoch_bar:
    Loss = 0
    L_loss = 0
    U_loss = 0
    S_loss = 0
    model_multi_semi_Modal.train()
    lr.step()
    batch_bar = tqdm(zip(dataloader_weibo['labeled'], dataloader_weibo['unlabeled']))
    for label_batch, unlabel_batch in batch_bar:
        l_x_text_1, l_x_img_1, l_y = label_batch
        l_y = l_y.float()
        l_x_text_1 = preprocess_text_every(l_x_text_1)
        l_x_text_1 = l_x_text_1.cuda()
        l_x_img_1 = l_x_img_1.cuda()
        l_y = l_y.cuda()

        u_x_text_1, u_x_img_1 = unlabel_batch
        u_x_text_1 = preprocess_text_every(u_x_text_1)
        u_x_text_1 = u_x_text_1.cuda()
        u_x_img_1 = u_x_img_1.cuda()


        loss, L_loss_mean, U_loss_mean, S_loss_ = model_multi_semi_Modal(l_x_text_1, l_x_img_1, l_y, u_x_text_1, u_x_img_1)

