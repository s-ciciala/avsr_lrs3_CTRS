import torch
import os
import os.path
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import numpy as np

from config import args
from models.visual_frontend import VisualFrontend
from utils.preprocessing import preprocess_sample
from sklearn.model_selection import train_test_split
import collections


def lrs3_parse(example):
    splt = example.split("{")
    print("Had to split")
    print("Was " + str(example))
    print("Is " + str(splt[0]))
    return splt[0]


def set_device():
    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")
    print("Using Device " + str(device))
    return device


def get_filelist():
    # walking through the data directory and obtaining a list of all files in the dataset
    filesList = list()
    for root, dirs, files in os.walk(args["DATA_DIRECTORY"]):
        #     print(root,dirs,files)
        for file in files:
            if file.endswith(".mp4"):
                filesList.append(os.path.join(root, file[:-4]))
    # print(filesList)
    # Preprocessing each sample
    print("\nNumber of data samples to be processed = %d" % (len(filesList)))
    print("\n\nStarting preprocessing ....\n")
    return filesList


def preprocess_all_samples(filesList, device):
    # declaring the visual frontend module
    vf = VisualFrontend()
    torch.cuda.empty_cache()
    print("Device is " + str(device))
    os.environ["CUDA_AVAILABLE_DEVICES"] = "0,1,2,3"
    print("Forcing device is " + str(device))
    # vf.load_state_dict(torch.load(args["TRAINED_FRONTEND_FILE"], map_location=device))
    device = "cpu"
    vf.load_state_dict(torch.load(args["TRAINED_FRONTEND_FILE"], map_location=device))
    vf.to(device)
    params = {"roiSize": args["ROI_SIZE"], "normMean": args["NORMALIZATION_MEAN"], "normStd": args["NORMALIZATION_STD"],
              "vf": vf}
    for file in tqdm(filesList, leave=True, desc="Preprocess", ncols=75):
        preprocess_sample(file, params)
    print("\nPreprocessing Done.")


# def generate_train_file():
#     # Generating train.txt for splitting the pretrain set into train sets
#     train_dir = args["TRAIN_DIRECTORY"]
#     train_dir_file = args["DATA_DIRECTORY"] + "/train.txt"
#     example_dict = {
#         "ID": [],
#         "TEXT": []
#     }
#     print("\n\nGenerating the train.txt file from the directory" + train_dir)
#     dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
#     print("\nAvaliable folders include: " + str(dirs))
#     print("\nTotal number of folders included: " + str(len(dirs)))
#
#     print("\nFor each folder we will extract the number of txt examples")
#     for folder in tqdm(dirs):
#
#         print("\nParsing Folder:" + folder)
#         example_dir = train_dir + folder + "/"
#         examples = [f for f in listdir(example_dir) if isfile(join(example_dir, f))]
#         ##NOTE assumption that each text file HAS an associated .mp4
#         examples_textonly = [ex for ex in examples if ".txt" in ex]
#         print("Parsing exmaples text files:" + str(examples_textonly))
#         print("NOTE we assume that each text file HAS an associated .mp4")
#
#         ##Current parse is absolute filename -> text
#         print("\nReading Each example")
#         for ex in examples_textonly:
#             examples_textonly_dir = example_dir + ex
#             with open(examples_textonly_dir, "r") as f:
#                 lines = f.readlines()
#                 print(examples_textonly_dir)
#                 examples_npy_dir = examples_textonly_dir.split("txt")[0][:-1]
#                 print(examples_npy_dir)
#
#                 example_dict["ID"].append(examples_npy_dir)
#                 string_to_add = str(lines[0][6: -1])
#                 print(string_to_add)
#                 example_dict["TEXT"].append(string_to_add)
#
#     with open(train_dir_file, "w") as f:
#         for i in range(len(example_dict["ID"])):
#             f.writelines(example_dict["ID"][i])
#             f.writelines(example_dict["TEXT"][i])
#             f.writelines("\n")


# def generate_val_file():
#     # Generating val.txt for splitting the pretrain set into validation sets
#     val_dir = args["VAL_DIRECTORY"]
#     val_dir_file = args["DATA_DIRECTORY"] + "/val.txt"
#     example_dict = {
#         "ID": [],
#         "TEXT": []
#     }
#     print("\n\nGenerating the val.txt file from the directory" + val_dir)
#     dirs = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
#     print("\nAvaliable folders include: " + str(dirs))
#     print("\nTotal number of folders included: " + str(len(dirs)))
#
#     print("\nFor each folder we will extract the number of txt examples")
#     for folder in tqdm(dirs):
#
#         print("\nParsing Folder:" + folder)
#         example_dir = val_dir + folder + "/"
#         examples = [f for f in listdir(example_dir) if isfile(join(example_dir, f))]
#         ##NOTE assumption that each text file HAS an associated .mp4
#         examples_textonly = [ex for ex in examples if ".txt" in ex]
#         print("Parsing exmaples text files:" + str(examples_textonly))
#         print("NOTE we assume that each text file HAS an associated .mp4")
#
#         ##Current parse is absolute filename -> text
#         print("\nReading Each example")
#         for ex in examples_textonly:
#             examples_textonly_dir = example_dir + ex
#             with open(examples_textonly_dir, "r") as f:
#                 lines = f.readlines()
#                 examples_npy_dir = examples_textonly_dir.split("txt")[0][:-1]
#                 example_dict["ID"].append(examples_npy_dir)
#                 string_to_add = str(lines[0][6: -1])
#                 example_dict["TEXT"].append(string_to_add)
#
#     with open(val_dir_file, "w") as f:
#         for i in range(len(example_dict["ID"])):
#             f.writelines(example_dict["ID"][i])
#             f.writelines(example_dict["TEXT"][i])
#             f.writelines("\n")


def split_trainval(fileList):
    trainval_only = [x for x in fileList if (args["VIDEO_TRAINVAL_NAME"] in x)]
    print("We have a total of :" + str(len(trainval_only)))
    print("Now we want a split 80/20")
    train, val = train_test_split(trainval_only, test_size=.20, shuffle=False)
    if collections.Counter(train) == collections.Counter(val):
        print("WARNING: TRAIN AND VAL HAVE SOME OVERLAPPING ELEMENTS")
        exit()
    return train, val


def generate_train_file(train):
    # Generating train.txt for splitting the pretrain set into train sets
    train_dir_file = args["DATA_DIRECTORY"] + "/train.txt"
    example_dict = {
        "ID": [],
        "TEXT": []
    }
    print("Generating train file...")
    for train_dir in tqdm(train):
        text_file = train_dir + ".txt"
        with open(text_file, "r") as f:
            lines = f.readlines()
            # print(text_file)
            examples_npy_dir = text_file.split("txt")[0][:-1]
            # print(examples_npy_dir)
            example_dict["ID"].append(examples_npy_dir)
            string_to_add = str(lines[0][6: -1])
            if "{" in string_to_add:
                string_to_add = lrs3_parse(string_to_add)
            # print(string_to_add)
            example_dict["TEXT"].append(string_to_add)
            # print(example_dict)

    if os.path.isfile(train_dir_file):
        os.remove(train_dir_file)
    with open(train_dir_file, "w") as f:
        for i in range(len(example_dict["ID"])):
            f.writelines(example_dict["ID"][i])
            f.writelines(example_dict["TEXT"][i])
            f.writelines("\n")


def generate_val_file(val):
    # Generating val.txt for splitting the pretrain set into validation sets
    val_dir_file = args["DATA_DIRECTORY"] + "/val.txt"
    example_dict = {
        "ID": [],
        "TEXT": []
    }
    print("Generating val file...")
    for val_dir in tqdm(val):
        text_file = val_dir + ".txt"
        with open(text_file, "r") as f:
            lines = f.readlines()
            # print(text_file)
            examples_npy_dir = text_file.split("txt")[0][:-1]
            # print(examples_npy_dir)
            example_dict["ID"].append(examples_npy_dir)
            string_to_add = str(lines[0][6: -1])
            if "{" in string_to_add:
                string_to_add = lrs3_parse(string_to_add)
            # print(string_to_add)
            example_dict["TEXT"].append(string_to_add)
            # print(example_dict)

    if os.path.isfile(val_dir_file):
        os.remove(val_dir_file)

    with open(val_dir_file, "w") as f:
        for i in range(len(example_dict["ID"])):
            f.writelines(example_dict["ID"][i])
            f.writelines(example_dict["TEXT"][i])
            f.writelines("\n")


def generate_test_file(test):
    # Generating train.txt for splitting the pretrain set into train sets
    test_dir_file = args["DATA_DIRECTORY"] + "/test.txt"
    example_dict = {
        "ID": [],
        "TEXT": []
    }
    print("Generating test file...")
    for test_dir in tqdm(test):
        text_file = test_dir + ".txt"
        with open(text_file, "r") as f:
            lines = f.readlines()
            # print(text_file)
            examples_npy_dir = text_file.split("txt")[0][:-1]
            # print(examples_npy_dir)
            example_dict["ID"].append(examples_npy_dir)
            string_to_add = str(lines[0][6: -1])
            if "{" in string_to_add:
                string_to_add = lrs3_parse(string_to_add)
            # print(string_to_add)
            example_dict["TEXT"].append(string_to_add)
            # print(example_dict)

    if os.path.isfile(test_dir_file):
        os.remove(test_dir_file)
    with open(test_dir_file, "w") as f:
        for i in range(len(example_dict["ID"])):
            f.writelines(example_dict["ID"][i])
            f.writelines(example_dict["TEXT"][i])
            f.writelines("\n")


def check_files_correct_len(train, val, test):
    train_len = len(train)
    val_len = len(val)
    test_len = len(test)
    val_dir_file = args["DATA_DIRECTORY"] + "/val.txt"
    train_dir_file = args["DATA_DIRECTORY"] + "/train.txt"
    test_dir_file = args["DATA_DIRECTORY"] + "/test.txt"

    with open(train_dir_file) as f:
        text = f.readlines()
        train_file_len = len(text)

    with open(val_dir_file) as f:
        text = f.readlines()
        val_file_len = len(text)

    with open(test_dir_file) as f:
        text = f.readlines()
        test_file_len = len(text)

    print("Expected train len: " + str(train_len) + " Got train len: " + str(train_file_len))
    print("Expected val len: " + str(val_len) + " Got val len: " + str(val_file_len))
    print("Expected test len: " + str(test_len) + " Got test len: " + str(test_file_len))


if __name__ == "__main__":
    device = set_device()
    fileList = get_filelist()
    train, val = split_trainval(fileList)
    test = [x for x in fileList if (args["VIDEO_TEST_SET"] in x)]
    # print([x for x in fileList if (args["VIDEO_PREPROC_SET"] in x)])
    print("Size of train set" + str(len(train)))
    print("Size of val set:" + str(len(val)))
    print("Size of test set:" + str(len(test)))
    # preprocess_all_samples(fileList,device)
    generate_train_file(train)
    generate_val_file(val)
    generate_test_file(test)
    check_files_correct_len(train, val, test)
    print("Completed")
