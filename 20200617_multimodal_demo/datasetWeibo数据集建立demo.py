import os
import numpy as np
import pandas as pd
import csv
import torch
from torch.utils.data import Dataset
import itertools
import re
from torchvision import transforms
from PIL import Image
import utils_myself
import oriDataOp




class DatasetWeibo_Unlabled(Dataset):
    def __init__(self, pathRoot='./data/', data_type='unlabled', key_split = False):
        self.img_fileList =os.listdir(pathRoot + 'images_final_224/')
        pathJson = pathRoot + 'json_weibo_unlabled.json'
        self.json_weibo = utils_myself.read_json(pathJson)


    def __getitem__(self, index):
        entry = self.json_weibo[index]
        image_name_1 = str(entry['image_id']) + '.jpg'
        image_path_1 = entry['image_path']
        if(self.key_split):
            text_for_img = utils_myself.tokenize(entry['text_for_img'])
        else:
            text_for_img = entry['text_for_img']
#         历史博文样本，用于丰富语料
#         text_sample = entry['text_sample']
        lable = entry['happiness']
#         情绪标签
#         sentiment = entry['sentiment']
        if image_name_1 in self.img_fileList:
            img1 = Image.open(image_path_1).convert('RGB')
        else:
            print("%s 不存在"%(image_path_1))
            img1 = Image.new('RGB', (150, 150), (255, 255, 255))
        transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        img1 = transform(img1)
        return text_for_img, img1


    def __len__(self):
        return len(self.json_weibo)
    
    
    
    
class DatasetWeibo_Labled(Dataset):
    def __init__(self, pathRoot='./data/'):
        self.img_fileList =os.listdir(pathRoot + 'images_final_224/')
        pathJson = pathRoot + 'labled_data/' + 'json_weibo_labled.json'
        self.json_weibo = utils_myself.read_json(pathJson)



    def __getitem__(self, index):
        entry = self.json_weibo[index]
        image_name_1 = str(entry['image_id']) + '.jpg'
        image_path_1 = entry['image_path']
        if(self.key_split):
            text_for_img = utils_myself.tokenize(entry['text_for_img'])
        else:
            text_for_img = entry['text_for_img']
#         历史博文样本，用于丰富语料
#         text_sample = entry['text_sample']
        lable = entry['happiness']
#         情绪标签
#         sentiment = entry['sentiment']
        if image_name_1 in self.img_fileList:
            img1 = Image.open(image_path_1).convert('RGB')
        else:
            print("%s 不存在"%(image_path_1))
            img1 = Image.new('RGB', (150, 150), (255, 255, 255))
        transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        img1 = transform(img1)
        return text_for_img, img1, lable

    def __len__(self):
        return len(self.json_weibo)
