import os
import shutil
import random



def data_set_split(src_data_folder1, target_data_folder1, train_scale):
    print("START THE SPLIT")
    print(os.listdir(src_data_folder1))
    class_names = os.listdir(src_data_folder1)
    split_name = 'train_'+ str(train_scale)
    split_path = os.path.join(target_data_folder1, split_name)
    print(split_path)
    if os.path.isdir(split_path):
        pass
    else:
        os.mkdir(split_path)

    for one in class_names:
        current_class_data_path = os.path.join(src_data_folder1, one)
        print("current path: ",current_class_data_path)
        current_all_data = os.listdir(current_class_data_path)
        current_data_length = len(current_all_data)
        print('current data length: ',current_data_length)
        current_data_index_list = list(range(current_data_length))
        random.shuffle(current_data_index_list)
        train_folder = os.path.join(split_path, one)
        print('train_folder: ',train_folder)
        if os.path.isdir(train_folder):
            pass
        else:
            os.mkdir(train_folder)
        train_stop_flag = current_data_length * train_scale
        current_idx = 0
        train_num = 0
        for i in current_data_index_list:
            # print(i)
            src_img_path = current_all_data[i]
            # print(os.path.join(train_folder,src_img_path))
            if current_idx <= train_stop_flag:
                old_path = os.path.join(current_class_data_path,src_img_path)
                new_path = os.path.join(train_folder,src_img_path)
                shutil.copy(old_path, new_path)
                train_num = train_num + 1
            else:
                break
            current_idx = current_idx + 1

        print("*********************************{}*************************************".format(one))
        print(
            "finished split {} according to {}:{} data in total".format(
                one, train_scale, current_data_length))
        print("train{}:{}".format(train_folder, train_num))






if __name__ == '__main__':

    src_data_folder = r"/Users/vonnet/Master/mlp/G077-Machine-Learning-Practical/Data/Clean_data/train"
    tar_data_folder = r"/Users/vonnet/Master/mlp/G077-Machine-Learning-Practical/Data/split_data"
    data_set_split(src_data_folder, tar_data_folder,train_scale=0.25)
