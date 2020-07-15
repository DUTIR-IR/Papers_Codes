#!/usr/bin/env python
# -*- coding: utf-8 -*-
######################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
# @file network.py
#
######################################################################
"""
File: network.py
"""

import os
import json
import random
import logging
import argparse
import torch
import numpy as np
from datetime import datetime

from source.inputters.tokenizers import ModBertTokenizer
from source.inputters.corpus import SrcTgtCorpus
from source.models.seq2seq import Seq2Seq
from source.utils.engine import Trainer
from source.utils.generator import TopKGenerator
from source.utils.engine import evaluate, evaluate_generation
from source.utils.misc import str2bool


def model_config():
    """
    model_config
    """
    parser = argparse.ArgumentParser()

    # Data
    data_arg = parser.add_argument_group("Data")
    data_arg.add_argument("--data_dir", type=str, default="./data/KdConv")
    data_arg.add_argument("--data_prefix", type=str, default="film")
    data_arg.add_argument("--save_dir", type=str, default="./models/")
    data_arg.add_argument("--embed_file", type=str, default=None)

    # Network
    net_arg = parser.add_argument_group("Network")
    net_arg.add_argument("--embed_size", type=int, default=300)
    net_arg.add_argument("--hidden_size", type=int, default=800)
    net_arg.add_argument("--bidirectional", type=str2bool, default=True)
    net_arg.add_argument("--max_vocab_size", type=int, default=30000)
    net_arg.add_argument("--min_len", type=int, default=1)
    net_arg.add_argument("--max_len", type=int, default=500)
    net_arg.add_argument("--num_layers", type=int, default=1)
    net_arg.add_argument("--attn", type=str, default='dot',
                         choices=['none', 'mlp', 'dot', 'general'])
    net_arg.add_argument("--share_vocab", type=str2bool, default=True)
    net_arg.add_argument("--with_bridge", type=str2bool, default=True)
    net_arg.add_argument("--tie_embedding", type=str2bool, default=True)

    # Training / Testing
    train_arg = parser.add_argument_group("Training")
    train_arg.add_argument("--optimizer", type=str, default="Adam")
    train_arg.add_argument("--lr", type=float, default=0.0001)
    train_arg.add_argument("--grad_clip", type=float, default=5.0)
    train_arg.add_argument("--dropout", type=float, default=0.3)
    train_arg.add_argument("--num_epochs", type=int, default=100)
    train_arg.add_argument("--lr_decay", type=float, default=0.5)
    train_arg.add_argument("--use_embed", type=str2bool, default=True)

    # Geneation
    gen_arg = parser.add_argument_group("Generation")
    gen_arg.add_argument("--max_dec_len", type=int, default=50)
    gen_arg.add_argument("--ignore_unk", type=str2bool, default=True)
    gen_arg.add_argument("--length_average", type=str2bool, default=True)
    gen_arg.add_argument("--gen_file", type=str, default="./test.result")
    gen_arg.add_argument("--gold_score_file", type=str, default="./gold.scores")

    # MISC
    misc_arg = parser.add_argument_group("Misc")
    misc_arg.add_argument("--gpu", type=int, default=3)
    misc_arg.add_argument("--log_steps", type=int, default=50)
    misc_arg.add_argument("--valid_steps", type=int, default=100)
    misc_arg.add_argument("--max_patience_num", type=int, default=5)
    misc_arg.add_argument("--seed", type=int, default=42)
    misc_arg.add_argument("--batch_size", type=int, default=128)
    misc_arg.add_argument("--ckpt", type=str)
    # misc_arg.add_argument("--ckpt", type=str, default="models/best.model")
    misc_arg.add_argument("--check", action="store_true")
    misc_arg.add_argument("--test", action="store_true")
    misc_arg.add_argument("--interact", action="store_true")
    # misc_arg.add_argument("--interact", type=str2bool, default=True)

    config = parser.parse_args()

    return config


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def main():
    """
    main
    """
    config = model_config()
    set_seed(config.seed)
    if config.check:
        config.save_dir = "./tmp/"
    config.use_gpu = torch.cuda.is_available() and config.gpu >= 0
    device = config.gpu
    torch.cuda.set_device(device)
    # Tokenizer definition
    tokenizer = ModBertTokenizer('bert-base-chinese', new_tokens=['[Name]', '[AttrName]', '[AttrValue]'])
    config.src_field = tokenizer
    config.tgt_field = tokenizer
    # Data definition
    corpus = SrcTgtCorpus(data_dir=config.data_dir, data_prefix=config.data_prefix,
                          tokenizer=tokenizer)
    corpus.load()
    if config.test and config.ckpt:
        corpus.reload(data_type='test')
    train_iter = corpus.create_batches(
        config.batch_size, "train", shuffle=True, device=device)
    valid_iter = corpus.create_batches(
        config.batch_size, "valid", shuffle=False, device=device)
    test_iter = corpus.create_batches(
        config.batch_size, "test", shuffle=False, device=device)
    # Model definition
    model = Seq2Seq(config)
    model_name = model.__class__.__name__
    # Generator definition
    generator = TopKGenerator(model, config)
    # Interactive generation testing
    if config.interact and config.ckpt:
        model.load(config.ckpt)
        return generator
    # Testing
    elif config.test and config.ckpt:
        print(model)
        model.load(config.ckpt)
        print("Testing ...")
        metrics, scores = evaluate(model, test_iter)
        print(metrics.report_cum())
        print("Generating ...")
        evaluate_generation(generator, test_iter, save_file=config.gen_file, verbos=True)
    else:
        # Load word embeddings
        # if config.use_embed and config.embed_file is not None:
        #     model.encoder.embedder.load_embeddings(
        #         corpus.SRC.embeddings, scale=0.03)
        #     model.decoder.embedder.load_embeddings(
        #         corpus.TGT.embeddings, scale=0.03)
        # Optimizer definition
        optimizer = getattr(torch.optim, config.optimizer)(
            model.parameters(), lr=config.lr)
        # Learning rate scheduler
        if config.lr_decay is not None and 0 < config.lr_decay < 1.0:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                      factor=config.lr_decay, patience=2, verbose=True,
                                                                      min_lr=1e-5)
        else:
            lr_scheduler = None
        # Save directory
        date_str, time_str = datetime.now().strftime("%Y%m%d-%H%M%S").split("-")
        result_str = "{}-{}".format(model_name, time_str)
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)
        # Logger definition
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
        fh = logging.FileHandler(os.path.join(config.save_dir, "train.log"))
        logger.addHandler(fh)
        # Save config
        # params_file = os.path.join(config.save_dir, f"{config.data_prefix}_params.json")
        # with open(params_file, 'w') as fp:
        #     json.dump(config.__dict__, fp, indent=4, sort_keys=True)
        # print("Saved params to '{}'".format(params_file))
        # logger.info(model)
        # Train
        logger.info("Training starts ...")
        trainer = Trainer(model=model, optimizer=optimizer, train_iter=train_iter,
                          valid_iter=valid_iter, logger=logger, data_prefix=config.data_prefix,
                          generator=generator, valid_metric_name="-loss", num_epochs=config.num_epochs,
                          save_dir=config.save_dir, log_steps=config.log_steps,
                          valid_steps=config.valid_steps, grad_clip=config.grad_clip,
                          lr_scheduler=lr_scheduler, save_summary=False,
                          max_patience_num=config.max_patience_num)
        if config.ckpt is not None:
            trainer.load(file_prefix=config.ckpt)
        trainer.train()
        logger.info("Training done!")
        # Test
        logger.info("")
        trainer.load(os.path.join(config.save_dir, f"best_{config.data_prefix}"))
        logger.info("Testing starts ...")
        metrics = evaluate(model, test_iter)
        logger.info(metrics.report_cum())
        logger.info("Generation starts ...")
        test_gen_file = os.path.join(config.save_dir, f"{config.data_prefix}.test.result")
        evaluate_generation(generator, test_iter, save_file=test_gen_file)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
