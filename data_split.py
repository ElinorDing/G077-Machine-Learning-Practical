import os
import pandas as pd
import datasets
import random
from shutil import copy2


def data_set_split(src_data_folder1, target_data_folder1, train_scale=0.25, val_scale=0.3, test_scale=0.0):
    print("START THE SPLIT")
    print(os.listdir(src_data_folder1)[4])
    class_name = os.listdir(src_data_folder1)[4]
    split_names = ['train', 'val', 'test']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder1, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.mkdir(split_path)

    current_class_data_path = os.path.join(src_data_folder1, class_name)
    print("current path: ",current_class_data_path)
    current_all_data = pd.read_csv(current_class_data_path)
    current_data_length = len(current_all_data)
    print(current_data_length)
    current_data_index_list = list(range(current_data_length))
    random.shuffle(current_data_index_list)
    train_folder = os.path.join(target_data_folder1, 'train')
    val_folder = os.path.join(target_data_folder1, 'val')
    test_folder = os.path.join(target_data_folder1, 'test')
    train_stop_flag = current_data_length * train_scale
    val_stop_flag = current_data_length * (train_scale + val_scale)
    current_idx = 0
    train_num = 0
    val_num = 0
    test_num = 0
    df_train = pd.DataFrame(columns=['Sentences','Corresponding_words'])
    df_val = pd.DataFrame(columns=['Sentences','Corresponding_words'])
    df_test = pd.DataFrame(columns=['Sentences','Corresponding_words'])
    for i in current_data_index_list:
        print(i)
        # print("type: ",type(current_all_data.iloc[i]))
        src_img_path = current_all_data.iloc[i]
        if current_idx <= train_stop_flag:
            df_train = df_train.append(src_img_path)
            # copy2(src_img_path, train_folder)
            train_num = train_num + 1
        elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
            df_val = df_val.append(src_img_path)
            # copy2(src_img_path, val_folder)
            val_num = val_num + 1
        else:
            df_test = df_test.append(src_img_path)
            # copy2(src_img_path, test_folder)
            test_num = test_num + 1
        current_idx = current_idx + 1

    df_train.to_csv('training_ptb_70.csv')
    # copy2('count_training_clean.csv', train_folder)
    # df_val.to_csv('count_val_clean.csv')
    # copy2('count_val_clean.csv', val_folder)
    # df_test.to_csv('count_test_clean.csv')
    # copy2('count_test_clean.csv',test_folder)

    print("*********************************{}*************************************".format(class_name))
    print(
        "finished split {} according to {}：{}：{}，{} data in total".format(
            class_name, train_scale, val_scale, test_scale, current_data_length))
    print("train{}：{}".format(train_folder, train_num))
    # print("dev{}：{}".format(val_folder, val_num))
    # print("test{}：{}".format(test_folder, test_num))

def merge_dataset(train, val, test):
    df_train = pd.read_csv(train)
    df_train = df_train.rename(columns = {"Sentences":"target_text","Corresponding_words":"source_text"})
    df_train = df_train[['source_text', 'target_text']]
    df_val = pd.read_csv(val)
    df_val = df_val.rename(columns = {"Sentences":"target_text","Corresponding_words":"source_text"})
    df_val = df_val[['source_text', 'target_text']]
    df_test = pd.read_csv(test)
    df_test = df_test.rename(columns = {"Sentences":"target_text","Corresponding_words":"source_text"})
    df_test = df_test[['source_text', 'target_text']]

    train_dataset = datasets.Dataset.from_dict(df_train)
    train_dataset.save_to_disk("train_dataset")
    print("The type of train_dataset: ",train_dataset)
    val_dataset = datasets.Dataset.from_dict(df_val)
    train_dataset.save_to_disk("val_dataset")
    test_dataset = datasets.Dataset.from_dict(df_test)
    train_dataset.save_to_disk("test_dataset")
    my_dataset_dict = datasets.DatasetDict({"train":train_dataset,"val":val_dataset,"test":test_dataset})
    # my_dataset_dict.save_to_disk("merged_data")
    # print(my_dataset_dict)
    # print(type(my_dataset_dict['test']['target_text']))
    return my_dataset_dict



if __name__ == '__main__':

    src_data_folder = r"/Users/vonnet/Master/mlp/G077-Machine-Learning-Practical/Data/Clean_data/train"
    tar_data_folder = r"/Users/vonnet/Master/mlp/G077-Machine-Learning-Practical/Data/split_data"
    data_set_split(src_data_folder, tar_data_folder)

    # Dataset with occurrence

    # src_data_folder = r"/Users/dyt/Documents/WORK/GitHub/seq2seq/raw_data"
    # tar_data_folder = r"/Users/dyt/Documents/WORK/GitHub/seq2seq/wordsCount"
    # data_set_split(src_data_folder, tar_data_folder)

    # train = r"/Users/dyt/Documents/WORK/GitHub/seq2seq/wordsCount/train/count_training_clean.csv"
    # val = r"/Users/dyt/Documents/WORK/GitHub/seq2seq/wordsCount/val/count_val_clean.csv"
    # test = r"/Users/dyt/Documents/WORK/GitHub/seq2seq/wordsCount/test/count_test_clean.csv"
    # merge_dataset(train,val,test)
    # data = datasets.load_from_disk("merged_data")
    # print(type(data['test']['target_text']))


    # train = r"/Users/dyt/Documents/WORK/GitHub/seq2seq/training_ptb_10.csv"
    # val = r"/Users/dyt/Documents/WORK/GitHub/seq2seq/count_val_ptb.csv"
    # test = r"/Users/dyt/Documents/WORK/GitHub/seq2seq/count_test_ptb.csv"
