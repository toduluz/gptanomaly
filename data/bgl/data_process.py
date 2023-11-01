import sys
sys.path.append('../')

import os
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm
from session import sliding_window, session_window, session_window_bgl, fixed_window
from logparser import Spell, Drain
import shutil
import pickle
from sklearn.utils import shuffle
import json
import random

# tqdm.pandas()
# pd.options.mode.chained_assignment = None  # default='warn'
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)

# In the first column of the log, "-" indicates non-alert messages while others are alert messages.
def _count_anomaly(log_path):
    total_size = 0
    normal_size = 0
    with open(log_path, errors='ignore') as f:
        for line in f:
            total_size += 1
            if line.split(' ')[0] == '-':
                normal_size += 1
    print("total size {}, abnormal size {}, abnormal size % {}%".format(total_size, total_size - normal_size, ((total_size-normal_size)/total_size*100)))


def sample_raw_data(data_file, output_file, sample_window_size, sample_step_size):
    """
    only sample supercomputer dataset such as bgl
    """
    sample_data = []
    labels = []
    idx = 0

    with open(data_file, 'r', errors='ignore') as f:
        for line in f:
            labels.append(line.split()[0] != '-')
            sample_data.append(line)

            if len(labels) == sample_window_size:
                abnormal_rate = sum(np.array(labels)) / len(labels)
                print(f"{idx + 1} lines, abnormal rate {abnormal_rate}")
                break

            idx += 1
            if idx % sample_step_size == 0:
                print(f"Process {round(idx / sample_window_size * 100, 4)} % raw data", end='\r')

    with open(output_file, "w") as f:
        f.writelines(sample_data)
    print("Sampling done")


def _file_generator(filename, df, features):
    with open(filename, 'w') as f:
        for _, row in df.iterrows():
            for val in zip(*row[features]):
                f.write(','.join([str(v) for v in val]) + ' ')
            f.write('\n')


def process_dataset(data_dir, output_dir, log_file, dataset_name, window_type, window_size, step_size, train_size,
                    random_sample=False, session_type="entry"):
    """
    creating log sequences by sliding window
    :param data_dir:
    :param output_dir:
    :param log_file:
    :param window_size:
    :param step_size:
    :param train_size:
    :return:
    """
    ########
    # count anomaly
    ########
    _count_anomaly(data_dir + log_file)

    ##################
    # Transformation #
    ##################
    print("Loading", f'{data_dir}{log_file}_structured.csv')
    df = pd.read_csv(f'{data_dir}{log_file}_structured.csv')

    # build log sequences
    if window_type == "sliding":
        # data preprocess
        if 'bgl' in dataset_name:
            df["datetime"] = pd.to_datetime(df['Time'], format='%Y-%m-%d-%H.%M.%S.%f')
        else:
            df['datetime'] = pd.to_datetime(df["Date"] + " " + df['Time'], format='%Y-%m-%d %H:%M:%S')

        df["Label"] = df["Label"].apply(lambda x: int(x != "-"))
        df['timestamp'] = df["datetime"].values.astype(np.int64) // 10 ** 9
        df['deltaT'] = df['datetime'].diff() / np.timedelta64(1, 's')
        df['deltaT'].fillna(0)
        n_train = int(len(df) * train_size)
        if session_type == "entry":
            sliding = fixed_window
        else:
            sliding = sliding_window
            window_size = float(window_size) * 60
            step_size = float(step_size) * 60
        print(random_sample)
        if random_sample:
            print("???")
            window_df = sliding(df[["timestamp", "Label", "EventId", "deltaT", "EventTemplate", "Content"]],
                                       para={"window_size": window_size,
                                             "step_size": step_size})
            window_df = shuffle(window_df).reset_index(drop=True)
            n_train = int(len(window_df) * train_size)
            train_window = window_df.iloc[:n_train, :].to_dict("records")
            test_window = window_df.iloc[n_train:, :].to_dict("records")
        else:
            train_window = sliding(
                df[["timestamp", "Label", "EventId", "deltaT", "EventTemplate", "Content"]].iloc[:n_train, :],
                para={"window_size": window_size,
                      "step_size": step_size})#.to_dict("records")
            test_window = sliding(
                df[["timestamp", "Label", "EventId", "deltaT", "EventTemplate", "Content"]].iloc[n_train:, :].reset_index(
                    drop=True),
                para={"window_size": window_size, "step_size": step_size})#.to_dict("records")

    elif window_type == "session":
        # only for hdfs
        if dataset_name == "hdfs":
            id_regex = r'(blk_-?\d+)'
            label_dict = {}
            blk_label_file = os.path.join(data_dir, "anomaly_label.csv")
            blk_df = pd.read_csv(blk_label_file)
            for _, row in tqdm(enumerate(blk_df.to_dict("records"))):
                label_dict[row["BlockId"]] = 1 if row["Label"] == "Anomaly" else 0

            window_df = session_window(df, id_regex, label_dict, window_size=int(window_size))
            # window_df = shuffle(window_df).reset_index(drop=True)
            n_train = int(len(window_df) * train_size)
            train_window = window_df[:n_train]
            test_window = window_df[n_train:]
        else:
            window_df = session_window_bgl(df)
            n_train = int(len(window_df) * train_size)
            train_window = window_df[:n_train]
            test_window = window_df[n_train:]
    else:
        raise NotImplementedError(f"{window_type} is not implemented")

    if not os.path.exists(output_dir):
        print(f"creating {output_dir}")
        os.mkdir(output_dir)
    # save pickle file
    # print(train_window_df.head())
    # print(test_window_df.head())
    # train_window = train_window_df.to_dict("records")
    # test_window = test_window_df.to_dict("records")
    # with open(os.path.join(output_dir, "train.pkl"), mode="wb") as f:
    #     pickle.dump(train_window, f)
    # with open(os.path.join(output_dir, "test.pkl"), mode="wb") as f:
    #     pickle.dump(test_window, f)
    train = train_window.loc[:, ['EventTemplate', 'Label']]
    train.rename(columns={"EventTemplate": "text", "Label": "labels"}, inplace=True)
    train['text'] = train['text'].apply(lambda x: '|'.join(x))
    train_normal = train[train["labels"] == 0]
    train_normal.sample(frac=1, random_state=42)
    train_abnormal = train[train["labels"] == 1]
    train_sample_len = int(len(train_normal) * (1-0.1))
    train_set = train_normal[:train_sample_len]

    print("training size {}".format(len(train_set)))
    train_set.to_csv(output_dir + "train.csv", index=False)
    print("abnormal size {}".format(len(train_abnormal)))
    
    val_abnormal_len = int(0.1/0.9 * len(train_normal[train_sample_len:]))
    validation = pd.concat([train_normal[train_sample_len:], train_abnormal[:val_abnormal_len]], axis=0)
    validation = validation.sample(frac=1, random_state=42)

    print("validation size {}".format(len(validation)))
    print("validation anomaly {} %".format(len(validation[validation["labels"] == 1]) / len(validation) *100))
    validation.to_csv(output_dir + "validation.csv", index=False)

    test = test_window.loc[:, ['EventTemplate', 'Label']]
    test.rename(columns={"EventTemplate": "text", "Label": "labels"}, inplace=True)
    test['text'] = test['text'].apply(lambda x: '|'.join(x))

    print("test size {}".format(len(test)))
    test_abnormal_len = len(test[test["labels"] == 1])
    print("test abnormal size {}".format(test_abnormal_len))
    print("test anomaly {} %".format(test_abnormal_len / len(test) *100))
    test.to_csv(output_dir + "test.csv", index=False)


def _file_generator2(filename, df):
    if "train" in filename:
        is_duplicate = {}
        with open(filename, 'w') as f:
            for _, seq in enumerate(df):
                seq = " ".join(seq)
                if seq not in is_duplicate.keys():
                    f.write(seq + "\n")
                    is_duplicate[seq] = 1
    else:
        with open(filename, 'w') as f:
            for _, seq in enumerate(df):
                seq = " ".join(seq)
                f.write(seq + "\n")


def process_instance(data_dir, output_dir, train_file, test_file):
    """
    creating log sequences by sliding window
    :param data_dir:
    :param output_dir:
    :param log_file:
    :param window_size:
    :param step_size:
    :param train_size:
    :return:
    """
    ########
    # count anomaly
    ########
    # _count_anomaly(data_dir + log_file)

    ##################
    # Transformation #
    ##################

    with open(os.path.join(data_dir, train_file), mode='rb') as f:
        train = pickle.load(f)

    with open(os.path.join(data_dir, test_file), mode='rb') as f:
        test = pickle.load(f)

    train = shuffle(train)

    if not os.path.exists(output_dir):
        print(f"creating {output_dir}")
        os.mkdir(output_dir)

    #########
    # Train #
    #########
    train_normal = [x.src_event_ids for x in train if not x.is_anomaly]
    # print(len(train_normal[0]))
    _file_generator2(os.path.join(output_dir, 'train'), train_normal)
    shutil.copyfile(os.path.join(output_dir, "train"), os.path.join(data_dir, "train"))

    train_abnormal = [x.src_event_ids for x in train if x.is_anomaly]
    _file_generator2(os.path.join(output_dir, 'train_abnormal'), train_abnormal)
    shutil.copyfile(os.path.join(output_dir, "train_abnormal"), os.path.join(data_dir, "train_abnormal"))

    train = [(x.src_event_ids, x.is_anomaly) for x in train]
    train_x = [x[0] for x in train]
    train_y = [x[1] for x in train]

    with open("bgl-train.pkl", mode="wb") as f:
        pickle.dump((train_x, train_y), f)

    print("training size {}".format(len(train_normal)))

    ###############
    # Test Normal #
    ###############
    # test_normal = df_normal[train_len:]
    test_normal = [x.src_event_ids for x in test if not x.is_anomaly]
    _file_generator2(os.path.join(output_dir, 'test_normal'), test_normal)
    shutil.copyfile(os.path.join(output_dir, "test_normal"), os.path.join(data_dir, "test_normal"))
    print("test normal size {}".format(len(test_normal)))

    # del df_normal
    # del train
    # del test_normal
    # gc.collect()

    #################
    # Test Abnormal #
    #################
    # df_abnormal = window_df[window_df["Label"] == 1]

    test_abnormal = [x.src_event_ids for x in test if x.is_anomaly]
    _file_generator2(os.path.join(output_dir, 'test_abnormal'), test_abnormal)
    shutil.copyfile(os.path.join(output_dir, "test_abnormal"), os.path.join(data_dir, "test_abnormal"))
    print('test abnormal size {}'.format(len(test_abnormal)))
    test = [(x.src_event_ids, x.is_anomaly) for x in test]
    test_x = [x[0] for x in test]
    test_y = [x[1] for x in test]

    with open("bgl-test.pkl", mode="wb") as f:
        pickle.dump((test_x, test_y), f)

def parse_log(data_dir, output_dir, log_file, parser_type,log_format, regex, keep_para=False, st=0.3, depth=3, max_child=100, tau=0.35):
    if parser_type == "drain":
        # the hyper parameter is set according to http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf
        # Drain is modified
        parser = Drain.LogParser(log_format,
                                 indir=data_dir,
                                 outdir=output_dir,
                                 depth=depth,
                                 st=st,
                                 rex=regex,
                                 keep_para=keep_para, maxChild=max_child)
        parser.parse(log_file)

    elif parser_type == "spell":
        # tau = 0.35
        parser = Spell.LogParser(indir=data_dir,
                                 outdir=output_dir,
                                 log_format=log_format,
                                 tau=tau,
                                 rex=regex,
                                 keep_para=keep_para)
        parser.parse(log_file)

if __name__ == '__main__':
    # seed_everything()
    # log_format = '<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>'
    # regex = [
    #     r'(0x)[0-9a-fA-F]+', #hexadecimal
    #     r'\d+.\d+.\d+.\d+',
    #     # r'/\w+( )$'
    #     r'\d+'
    # ]
    # parse_log("./", "./", "BGL.log", "drain", log_format, regex)
    process_dataset(data_dir="./", output_dir="./", log_file="BGL.log", dataset_name="bgl",
                    window_type="sliding", window_size=50, step_size=50, train_size=0.8, random_sample=False,
                    session_type="entry")