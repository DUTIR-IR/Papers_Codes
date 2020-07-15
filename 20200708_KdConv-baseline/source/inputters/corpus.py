#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/inputters/corpus.py
"""

import os
import torch

from tqdm import tqdm
from source.inputters.dataset import Dataset


class Corpus(object):
    """
    Corpus
    """
    def __init__(self,
                 data_dir,
                 data_prefix,
                 tokenizer=None):
        self.data_dir = data_dir
        self.data_prefix = data_prefix

        prepared_data_file = data_prefix + ".data.pt"
        self.prepared_data_file = os.path.join(data_dir, prepared_data_file)
        self.sort_fn = None
        self.data = None
        self.tokenizer = tokenizer
        self.padding_idx = self.tokenizer.pad_token_id

    def load(self):
        """
        load
        """
        if not os.path.exists(self.prepared_data_file):
            self.build()
        self.load_data(self.prepared_data_file)

    def reload(self, data_type='test'):
        """
        reload
        """
        data_file = os.path.join(self.data_dir, self.data_prefix + "." + data_type)
        data_raw = self.read_data(data_file, data_type="test")
        data_examples = self.build_examples(data_raw)
        self.data[data_type] = Dataset(data_examples)

        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_data(self, prepared_data_file=None):
        """
        load_data
        """
        prepared_data_file = prepared_data_file or self.prepared_data_file
        print("Loading prepared data from {} ...".format(prepared_data_file))
        data = torch.load(prepared_data_file)
        self.data = {"train": Dataset(data['train']),
                     "valid": Dataset(data["valid"]),
                     "test": Dataset(data["test"])}
        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def read_data(self, data_file, data_type=None):
        """
        Returns
        -------
        data: ``List[Dict]``
        """
        raise NotImplementedError

    def build_examples(self, data):
        """
        Args
        ----
        data: ``List[Dict]``
        """
        examples = []
        for raw_data in tqdm(data):
            example = {}
            for name, strings in raw_data.items():
                example[name] = self.tokenizer.encode(strings)
            examples.append(example)
        if self.sort_fn is not None:
            print("Sorting examples ...")
            examples = self.sort_fn(examples)
        return examples

    def build(self):
        """
        build
        """
        print("Start to build corpus!")
        train_file = os.path.join(self.data_dir, self.data_prefix + ".train")
        valid_file = os.path.join(self.data_dir, self.data_prefix + ".valid")
        test_file = os.path.join(self.data_dir, self.data_prefix + ".test")

        print("Reading data ...")
        train_raw = self.read_data(train_file, data_type="train")
        valid_raw = self.read_data(valid_file, data_type="valid")
        test_raw = self.read_data(test_file, data_type="test")

        print("Building TRAIN examples ...")
        train_data = self.build_examples(train_raw)
        print("Building VALID examples ...")
        valid_data = self.build_examples(valid_raw)
        print("Building TEST examples ...")
        test_data = self.build_examples(test_raw)

        data = {"train": train_data,
                "valid": valid_data,
                "test": test_data}

        print("Saving prepared data ...")
        torch.save(data, self.prepared_data_file)
        print("Saved prepared data to '{}'".format(self.prepared_data_file))

    def create_batches(self, batch_size, data_type="train",
                       shuffle=False, device=None):
        """
        create_batches
        """
        try:
            data = self.data[data_type]
            data_loader = data.create_batches(batch_size, shuffle, device)
            return data_loader
        except KeyError:
            raise KeyError("Unsported data type: {}!".format(data_type))

    def transform(self, data_file, batch_size,
                  data_type="test", shuffle=False, device=None):
        """
        Transform raw text from data_file to Dataset and create data loader.
        """
        raw_data = self.read_data(data_file, data_type=data_type)
        examples = self.build_examples(raw_data)
        data = Dataset(examples)
        data_loader = data.create_batches(batch_size, shuffle, device)
        return data_loader


class SrcTgtCorpus(Corpus):
    """
    SrcTgtCorpus
    """
    def __init__(self,
                 data_dir,
                 data_prefix,
                 tokenizer=None):
        super(SrcTgtCorpus, self).__init__(data_dir=data_dir,
                                           data_prefix=data_prefix,
                                           tokenizer=tokenizer)

    def read_data(self, data_file, data_type="train"):
        """
        read_data
        """
        data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                src, tgt = line.strip().split('\t')[:2]
                data.append({'src': src, 'tgt': tgt})

        print("Read {} {} examples ".format(len(data), data_type.upper()))
        return data
