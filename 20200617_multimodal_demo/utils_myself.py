import json
import pkuseg


def read_json(path_json):
    # 读取存储于json文件中的列表
    with open(path_json, 'r', encoding='utf-8') as f_obj:
        json_list = json.load(f_obj)
    return json_list



def tokenize(this_text): return pkuseg.pkuseg().cut(this_text) #分词函数,后续操作中会用到



def pad(x, max_len=64):
    return x[:max_len] if len(x) > max_len else x + [0] * (max_len - len(x))