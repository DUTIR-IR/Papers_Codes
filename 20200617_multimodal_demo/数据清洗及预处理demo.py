#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import oriDataOp


# In[3]:


def read_json(path_json):
    # 读取存储于json文件中的列表
    with open(path_json, 'r', encoding='utf-8') as f_obj:
        json_list = json.load(f_obj)
    return json_list


# In[6]:


path_happy = 'D:/Development/github/weibo-search/结果文件/%23欢乐时光%23/'
path_labled_target = "data/labled_data/"


def data_clean_happy_data(path_happy, path_labled_target):
    path_csv_ori = path_happy + '%23欢乐时光%23.csv'
    df_data_id, df_data_text, df_data_other, df_data_atUser = oriDataOp.funcLoadOriCsv(path_csv_ori)
    print(df_data_id[:5])
    #  save dataframe to csv(not save pic url)
    path_data_text = path_labled_target + "df_data_text.csv"
    path_data_other = path_labled_target + "df_data_other.csv"
    path_data_atUser = path_labled_target + "df_data_atUser.csv"
    df_data_text.to_csv(path_data_text, index= False)
    df_data_other.to_csv(path_data_other, index= False)
    df_data_atUser.to_csv(path_data_atUser, index= False)
    path_pic_ori = path_happy + 'images/'
    path_pic_target = path_labled_target + 'images_final_224/'
    list_uid_have_pic, list_uid_no_pic = oriDataOp.funcLoadOriPic(path_pic_ori, path_pic_target, df_data_id)
    # save the weibo that have img
    list_uid_useful = list_uid_have_pic
    df_data_text, df_data_other, df_data_atUser = oriDataOp.funcSaveUsefulData(list_uid_useful, path_data_text, path_data_other, path_data_atUser)
    path_new_data_text = path_labled_target + "new_data_text.csv"
    df_data_text.to_csv(path_new_data_text, index= False)
    # save final data to json
    path_dir_img = path_pic_target
    json_weibo = oriDataOp.funcDataToJson(path_new_data_text, path_dir_img)
    path_json_weibo_labled = "data/json_weibo_labled.json"
    with open(path_json_weibo_labled, "w") as fp:
        fp.write(json_weibo)
    return json_weibo
    


# In[7]:


json_weibo_labled = data_clean_happy_data(path_happy, path_labled_target)

