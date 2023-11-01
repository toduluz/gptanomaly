import sys
sys.path.append('../')

import os
import re
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from logparser import Spell, Drain

# get [log key, delta time] as input for deeplog
input_dir  = os.path.expanduser('./')
output_dir = './'  # The output directory of parsing results
log_file   = "HDFS.log"  # The input log file name

log_structured_file = output_dir + log_file + "_structured.csv"
log_templates_file = output_dir + log_file + "_templates.csv"
log_sequence_file = output_dir + "hdfs_sequence.csv"

def mapping():
    log_temp = pd.read_csv(log_templates_file)
    log_temp.sort_values(by = ["Occurrences"], ascending=False, inplace=True)
    log_temp_dict = {event: idx+1 for idx , event in enumerate(list(log_temp["EventId"])) }
    print(log_temp_dict)
    with open (output_dir + "hdfs_log_templates.json", "w") as f:
        json.dump(log_temp_dict, f)


def parser(input_dir, output_dir, log_file, log_format, type='drain'):
    if type == 'spell':
        tau        = 0.5  # Message type threshold (default: 0.5)
        regex      = [
            "(/[-\w]+)+", #replace file path with *
            "(?<=blk_)[-\d]+" #replace block_id with *

        ]  # Regular expression list for optional preprocessing (default: [])

        parser = Spell.LogParser(indir=input_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex, keep_para=False)
        parser.parse(log_file)

    elif type == 'drain':
        regex = [
            r"(?<=blk_)[-\d]+", # block_id
            r'\d+\.\d+\.\d+\.\d+\:\d+',  # IP:Port
            r"(/[-\w]+)+",  # file path
            #r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$',  # Numbers
        ]
        # the hyper parameter is set according to http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf
        st = 0.5  # Similarity threshold
        depth = 3  # Depth of all leaf nodes


        parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex, keep_para=False)
        parser.parse(log_file)


def hdfs_sampling(log_file, window='session'):
    assert window == 'session', "Only window=session is supported for HDFS dataset."
    print("Loading", log_file)
    df = pd.read_csv(log_file, engine='c',
            na_filter=False, memory_map=True, dtype={'Date':object, "Time": object})

    # with open(output_dir + "hdfs_log_templates.json", "r") as f:
    #     event_num = json.load(f)
    # df["EventId"] = df["EventId"].apply(lambda x: event_num.get(x, -1))

    data_dict = defaultdict(list) #preserve insertion order of items
    for idx, row in tqdm(df.iterrows()):
        blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            data_dict[blk_Id].append(row["EventTemplate"])

    data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventTemplate'])
    # Join the elements of each list into a string separated by period and a space
    data_df['EventTemplate'] = data_df['EventTemplate'].apply(lambda lst: '|'.join(map(str, lst)))

    data_df.to_csv(log_sequence_file, index=None)
    print("hdfs sampling done")


def generate_train_test_v1(hdfs_sequence_file):
    blk_label_dict = {}
    blk_label_file = os.path.join(input_dir, "anomaly_label.csv")
    blk_df = pd.read_csv(blk_label_file)
    for _ , row in tqdm(blk_df.iterrows()):
        blk_label_dict[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0

    seq = pd.read_csv(hdfs_sequence_file)
    seq["labels"] = seq["BlockId"].apply(lambda x: blk_label_dict.get(x)) #add label to the sequence of each blockid

    # seq["text"] = [' '.join(map(str, l)) for l in seq["EventSequence"]]
    seq.rename(columns={"EventTemplate": "text"}, inplace=True)
    # print(type(seq["text"][0]))

    normal_seq = seq[seq["labels"] == 0][["text", "labels"]]
    train = normal_seq[:10000]
    train = train.sample(frac=.5, random_state=42) # sample normal data

    abnormal_seq= seq[(seq["labels"] == 1)][["text", "labels"]]

    val_set_normal = normal_seq[10000:14500]
    val_set_abnormal = abnormal_seq[:500]
    test_set_normal = normal_seq[14500:19000]
    test_set_abnormal = abnormal_seq[500:1000]

    validation = pd.concat([val_set_normal, val_set_abnormal]).sample(frac=1, random_state=42)
    test = pd.concat([test_set_normal, test_set_abnormal]).sample(frac=1, random_state=42)

    # train.rename(columns={"EventSequence":"text", "Label":"labels"}, inplace=True)
    # validation.rename(columns={"EventSequence":"text", "Label":"labels"}, inplace=True)
    # test.rename(columns={"EventSequence":"text", "Label":"labels"}, inplace=True)
    # train.sort_index(inplace=True)

    train_len = len(train)
    val_len = len(validation)
    test_len = len(test)
   
    print("train size {0}, validation size {1}, test size {2}".format(train_len, val_len, test_len))

    train.to_csv(output_dir + "train.csv", index=False)
    validation.to_csv(output_dir + "validation.csv", index=False)
    test.to_csv(output_dir + "test.csv", index=False)
    print("generate train validation test data done")

def generate_train_test(hdfs_sequence_file):
    blk_label_dict = {}
    blk_label_file = os.path.join(input_dir, "anomaly_label.csv")
    blk_df = pd.read_csv(blk_label_file)
    for _ , row in tqdm(blk_df.iterrows()):
        blk_label_dict[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0

    seq = pd.read_csv(hdfs_sequence_file)
    seq["labels"] = seq["BlockId"].apply(lambda x: blk_label_dict.get(x)) #add label to the sequence of each blockid

    # seq["text"] = [' '.join(map(str, l)) for l in seq["EventSequence"]]
    seq.rename(columns={"EventTemplate": "text"}, inplace=True)
    seq = seq[["text", "labels"]]
    # print(type(seq["text"][0]))

    train_ratio = 0.8
    val_ratio = 0.1

    seq_len = len(seq)
    train_len = int(seq_len * train_ratio)
    # val_len = int(seq_len * val_ratio)
    train = seq[:train_len]
    train_normal = train[train["labels"] == 0]
    train_normal = train_normal.sample(frac=1, random_state=42)
    train_abnormal = train[train["labels"] == 1]
    train_sample_len = int(len(train_normal) * (1-val_ratio))
    train = train_normal[:train_sample_len]
    # train = train[train["labels"] == 0]
    
    val_anomaly_len = int(0.1/0.9 * len(train_normal[train_sample_len:]))
    validation = pd.concat([train_normal[train_sample_len:], train_abnormal[:val_anomaly_len]], axis=0)
    validation = validation.sample(frac=1, random_state=42)
    # validation = seq[train_len:train_len+val_len]

    test = seq[train_len:]

    # train.rename(columns={"EventSequence":"text", "Label":"labels"}, inplace=True)
    # validation.rename(columns={"EventSequence":"text", "Label":"labels"}, inplace=True)
    # test.rename(columns={"EventSequence":"text", "Label":"labels"}, inplace=True)
    # train.sort_index(inplace=True)

    train_len = len(train)
    val_len = len(validation)
    test_len = len(test)
    val_anomaly = len(validation[validation["labels"] == 1]) / val_len *100
    test_anomaly = len(test[test["labels"] == 1]) / test_len *100
   
    print("train size {0}, validation size {1}, test size {2}".format(train_len, val_len, test_len))
    print("validation anomaly {0} %, test anomaly {1} %".format(val_anomaly, test_anomaly))

    train.to_csv(output_dir + "train.csv", index=False)
    validation.to_csv(output_dir + "validation.csv", index=False)
    test.to_csv(output_dir + "test.csv", index=False)
    print("generate train validation test data done")


def df_to_file(df, file_name):
    with open(file_name, 'w') as f:
        for _, row in df.items():
            f.write(', '.join([str(ele) for ele in eval(row)]))
            f.write('\n')


if __name__ == "__main__":
    # 1. parse HDFS log
    # log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'  # HDFS log format
    # parser(input_dir, output_dir, log_file, log_format, 'drain')
    # mapping()
    # hdfs_sampling(log_structured_file)
    generate_train_test(log_sequence_file)
