#!/bin/bash
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################

# set python path according to your actual environment
pythonpath='python'

# put all data set that used and generated for training under this folder: datapath
# for more details, please refer to the following data processing instructions
datapath=./data/KdConv_test

# the prefix of the file name used by the model, must be consistent with the configuration in network.py
prefix=(film music travel)
for ((i=0; i<${#prefix[*]}; i++))
do
    # in train stage, use "train.txt" to train model, and use "dev.txt" to eval model
    # the "train.txt" and "dev.txt" are the original data provided by the organizer and
    # need to be placed in this folder: datapath/resource/
    # the following preprocessing will generate the actual data needed for model training
    # DATA_TYPE = "train" or "dev"
    datatype=(train valid)

    # data preprocessing
    for ((j=0; j<${#datatype[*]}; j++))
    do
        # ensure that each file is in the correct path
        #     1. put the data provided by the organizers under this folder: datapath/resource/
        #            - the data provided consists of three parts: train.txt dev.txt test.txt
        #            - the train.txt and dev.txt are session data, the test.txt is sample data
        #            - in train stage, we just use the train.txt and dev.txt
        #     2. the sample data extracted from session data is in this folder: datapath/resource/
        #     3. the text file required by the model is in this folder: datapath
        #     4. the topic file used to generalize data is in this directory: datapath
        corpus_file=${datapath}/resource/${datatype[$j]}_${prefix[$i]}.json
        sample_file=${datapath}/resource/sample.${datatype[$j]}_${prefix[$i]}.txt
        text_file=${datapath}/${prefix[$i]}.${datatype[$j]}

        # step 1: firstly have to convert session data to sample data
        ${pythonpath} ./tools/convert_session_to_sample.py ${corpus_file} ${sample_file}

        # step 2: convert sample data to text data required by the model
        ${pythonpath} ./tools/convert_conversation_corpus_to_model_text.py ${sample_file} ${text_file}
    done

    # step 3: in train stage, we just use train.txt and dev.txt, so we copy dev.txt to test.txt for model training
    cp ${datapath}/${prefix[$i]}.valid ${datapath}/${prefix[$i]}.test

    # step 4: train model, you can find the model file in ./models/ after training
    ${pythonpath} ./network.py --gpu 1 --data_dir ./data/KdConv_test  --data_prefix ${prefix[$i]} \
      --save_dir ./models --num_epochs 100  # > log.txt

done



