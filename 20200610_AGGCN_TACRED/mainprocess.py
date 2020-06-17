
# coding: utf-8

# In[2]:


def mani_fold(data2):
    import pickle
    import re
    import numpy as np
    import random
    from gensim.models import KeyedVectors
    import gensim
    from gensim.models import Word2Vec
    from sklearn import manifold
    wv_from_bin=gensim.models.KeyedVectors.load("/home/sda/wangbolin/datasets/test840B300.model")#glove
    keys = list(wv_from_bin.vocab.keys())
    weights=[]
    for key in keys[7000:8001]:
        weights.append(wv_from_bin[key])
    data=np.array(weights)
    n_neighbors =1000
    n_components = 300
    lle=manifold.LocallyLinearEmbedding(n_neighbors, n_components,eigen_solver='auto',method='ltsa').fit(data)
    data = lle.transform(data2)
    print('111111111111111111111111111111111111111111')
    return data

