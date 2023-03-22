from os import listdir
from os.path import isfile, join

import torch
import os
from tqdm import tqdm
from scipy.io import wavfile
import numpy as np

from config import args
from utils.preprocessing import preprocess_sample


def set_device():
    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")
    return device


# def remove_lrs3_nuggets(filesList):
#     print("Removing `{LG}` and such")
#     for file in filesList:
#         txt = file + ".txt"
#         print("File")
#         current_example = open(txt, 'r')
#         lines = file1.readlines()
#         for line in lines:
#             if "{" in line:
#                 line = line.split("{")[0]
#         with open("test.txt", "w") as f:
#             f.writelines(lines)


def string_filter(file, folder, val=False):
    if val:
        dir = args["TEST_DIRECTORY"] + folder + "/" + file
    else:
        dir = args["TRAIN_DIRECTORY"] + folder + "/" + file
    with open(dir, "r") as f:
        lines = f.readlines()
        string_to_add = str(lines[0][6: -1])
        if "{" in string_to_add:
            string_to_add = lrs3_parse(string_to_add)
        characters = len([ele for ele in string_to_add if ele.isalpha()])
        print("String :" + str(string_to_add))
        print("characters :" + str(characters) + " and MAX is " + str(args["MAX_CHAR_LEN"]))
            # print(characters,args["MAX_CHAR_LEN"])
        if characters <= args["MAX_CHAR_LEN"]:
            return True
    return False


def check_valid_dirs(fileList):
    valid_dirs = list()
    for file in fileList:
        if args["TRAIN_DIRECTORY"] in file:
            print(file)
            print(file)
            print(os.path.isdir(file[:-5]))
            txt = file + ".txt"
            if (os.path.isdir(file[:-5]) and os.path.isfile(txt)):
                valid_dirs.append(file)
        # os.path.isdir()
    return valid_dirs


def filer_lengths(fileList):
    filesListFiltered = list()
    for file in fileList:
        text = file + ".txt"
        with open(text, "r") as f:
            lines = f.readlines()
            string_to_add = str(lines[0][6: -1])
            if "{" in string_to_add:
                string_to_add = lrs3_parse(string_to_add)
                characters = len([ele for ele in string_to_add if ele.isalpha()])
                if characters <= args["MAX_CHAR_LEN"]:
                    filesListFiltered.append(file)

    print("\nNumber of data samples to after filtering = %d" % (len(filesListFiltered)))
    return filesListFiltered


def get_filelist():
    # walking through the data directory and obtaining a list of all files in the dataset
    filesList = list()
    for root, dirs, files in os.walk(args["DATA_DIRECTORY"]):
        # print(root,dirs,files)
        for file in files:
            if file.endswith(".mp4"):
                # if check_len(file,args["MAX_CHAR_LEN"]):
                filesList.append(os.path.join(root, file[:-4]))
    # print(filesList)
    # Preprocessing each sample
    print("\nNumber of data samples to be processed = %d" % (len(filesList)))
    print("\n\nStarting preprocessing ....\n")
    return filesList


def generate_noise_file(filesList):
    # Generating a 1 hour noise file
    # Fetching audio samples from 20 random files in the dataset and adding them up to generate noise
    # The length of these clips is the shortest audio sample among the 20 samples
    print("\n\nGenerating the noise file ....")

    noise = np.empty((0))
    while len(noise) < 16000 * 3600:
        noisePart = np.zeros(16000 * 60)
        indices = np.random.randint(0, len(filesList), 20)
        for ix in indices:
            sampFreq, audio = wavfile.read(filesList[ix] + ".wav")
            audio = audio / np.max(np.abs(audio))
            pos = np.random.randint(0, abs(len(audio) - len(noisePart)) + 1)
            if len(audio) > len(noisePart):
                noisePart = noisePart + audio[pos:pos + len(noisePart)]
            else:
                noisePart = noisePart[pos:pos + len(audio)] + audio
        noise = np.concatenate([noise, noisePart], axis=0)
    noise = noise[:16000 * 3600]
    noise = (noise / 20) * 32767
    noise = np.floor(noise).astype(np.int16)
    wavfile.write(args["DATA_DIRECTORY"] + "/noise.wav", 16000, noise)

    print("\nNoise file generated.")


def preprocess_all_samples(filesList):
    # declaring the visual frontend module
    # Preprocessing each sample
    print("\nNumber of data samples to be processed = %d" % (len(filesList)))
    print("\n\nStarting preprocessing ....\n")
    for file in tqdm(filesList, leave=True, desc="Preprocess", ncols=75):
        preprocess_sample(file)
    print("\nPreprocessing Done.")


def lrs3_parse(example):
    splt = example.split("{")
    print("Had to split")
    print("Was " + str(example))
    print("Is "+ str(splt))
    return splt[0]


def generate_train_file():
    # Generating train.txt for splitting the pretrain set into train sets
    train_dir = args["TRAIN_DIRECTORY"]
    train_dir_file = args["DATA_DIRECTORY"] + "/train.txt"
    example_dict = {
        "ID": [],
        "TEXT": []
    }
    print("\n\nGenerating the train.txt file from the directory" + train_dir)
    dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    print("\nAvaliable folders include: " + str(dirs))
    print("\nTotal number of folders included: " + str(len(dirs)))

    print("\nFor each folder we will extract the number of txt examples")
    for folder in tqdm(dirs):

        print("\nParsing Folder:" + folder)
        example_dir = train_dir + folder + "/"
        examples = [f for f in listdir(example_dir) if isfile(join(example_dir, f))]
        ##NOTE assumption that each text file HAS an associated .mp4
        examples_textonly = [ex for ex in examples if ".txt" in ex]

        print("Parsing exmaples text files:" + str(examples_textonly))
        print("NOTE we assume that each text file HAS an associated .mp4")

        ##Current parse is absolute filename -> text
        print("\nReading Each example")
        for ex in examples_textonly:
            examples_textonly_dir = example_dir + ex
            with open(examples_textonly_dir, "r") as f:
                lines = f.readlines()
                # print(examples_textonly_dir)

                examples_npy_dir = examples_textonly_dir.split("txt")[0][:-1]
                # print(examples_npy_dir)
                string_to_add = str(lines[0][6: -1])
                if "{" in string_to_add:
                    string_to_add = lrs3_parse(string_to_add)
                # print(string_to_add)
                if string_filter(ex, folder, False):
                    print("adding :" + str(examples_npy_dir))
                    print("adding :" + str(string_to_add))
                    example_dict["ID"].append(examples_npy_dir)
                    example_dict["TEXT"].append(string_to_add)

    with open(train_dir_file, "w") as f:
        for i in range(len(example_dict["ID"])):
            f.writelines(example_dict["ID"][i])
            f.writelines(example_dict["TEXT"][i])
            f.writelines("\n")


def generate_val_file():
    # Generating val.txt for splitting the pretrain set into validation sets
    val_dir = args["VAL_DIRECTORY"]
    val_dir_file = args["DATA_DIRECTORY"] + "/val.txt"
    example_dict = {
        "ID": [],
        "TEXT": []
    }
    print("\n\nGenerating the val.txt file from the directory" + val_dir)
    dirs = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
    print("\nAvaliable folders include: " + str(dirs))
    print("\nTotal number of folders included: " + str(len(dirs)))

    print("\nFor each folder we will extract the number of txt examples")
    for folder in tqdm(dirs):

        print("\nParsing Folder:" + folder)
        example_dir = val_dir + folder + "/"
        examples = [f for f in listdir(example_dir) if isfile(join(example_dir, f))]
        ##NOTE assumption that each text file HAS an associated .mp4
        examples_textonly = [ex for ex in examples if ".txt" in ex]
        print("Parsing exmaples text files:" + str(examples_textonly))
        # print("NOTE we assume that each text file HAS an associated .mp4")

        ##Current parse is absolute filename -> text
        print("\nReading Each example")
        for ex in examples_textonly:
            examples_textonly_dir = example_dir + ex
            with open(examples_textonly_dir, "r") as f:
                lines = f.readlines()
                examples_npy_dir = examples_textonly_dir.split("txt")[0][:-1]
                string_to_add = str(lines[0][6: -1])
                if "{" in string_to_add:
                    string_to_add = lrs3_parse(string_to_add)
                if string_filter(ex, folder, True):
                    print("adding :" + str(examples_npy_dir))
                    print("adding :" + str(string_to_add))
                    example_dict["ID"].append(examples_npy_dir)
                    example_dict["TEXT"].append(string_to_add)

    with open(val_dir_file, "w") as f:
        for i in range(len(example_dict["ID"])):
            f.writelines(example_dict["ID"][i])
            f.writelines(example_dict["TEXT"][i])
            f.writelines("\n")

def generate_test_file():
    # Generating val.txt for splitting the pretrain set into validation sets
    test_dir = args["TEST_DIRECTORY"]
    test_dir_file = args["DATA_DIRECTORY"] + "/test.txt"
    example_dict = {
        "ID": [],
        "TEXT": []
    }
    print("\n\nGenerating the val.txt file from the directory" + test_dir)
    dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    print("\nAvaliable folders include: " + str(dirs))
    print("\nTotal number of folders included: " + str(len(dirs)))

    ##CUT EXAMPLES:
    print("Length of examples before the cull: " + str(len(dirs)))
    dirs = dirs[:args["TEST_SIZE"]]
    print("Length of examples after the cull: " + str(len(dirs)))

    print("\nFor each folder we will extract the number of txt examples")
    for folder in tqdm(dirs):

        print("\nParsing Folder:" + folder)
        example_dir = test_dir + folder + "/"
        examples = [f for f in listdir(example_dir) if isfile(join(example_dir, f))]
        ##NOTE assumption that each text file HAS an associated .mp4
        examples_textonly = [ex for ex in examples if ".txt" in ex]
        print("Parsing exmaples text files:" + str(examples_textonly))
        # print("NOTE we assume that each text file HAS an associated .mp4")

        ##Current parse is absolute filename -> text
        print("\nReading Each example")

        for ex in examples_textonly:
            examples_textonly_dir = example_dir + ex
            with open(examples_textonly_dir, "r") as f:
                lines = f.readlines()
                examples_npy_dir = examples_textonly_dir.split("txt")[0][:-1]
                string_to_add = str(lines[0][6: -1])
                if "{" in string_to_add:
                    string_to_add = lrs3_parse(string_to_add)
                if string_filter(ex, folder, True):
                    print("adding :" + str(examples_npy_dir))
                    print("adding :" + str(string_to_add))
                    example_dict["ID"].append(examples_npy_dir)
                    example_dict["TEXT"].append(string_to_add)

    with open(test_dir_file, "w") as f:
        for i in range(len(example_dict["ID"])):
            f.writelines(example_dict["ID"][i])
            f.writelines(example_dict["TEXT"][i])
            f.writelines("\n")

if __name__ == "__main__":
    device = set_device()
    fileList = get_filelist()
    # fileList = filer_lengths(fileList)
    # fileList = check_valid_dirs(fileList)
    print("File List complete")
    preprocess_all_samples(fileList)
    # generate_noise_file(fileList)
    # generate_train_file()
    # generate_val_file()
    # generate_test_file()
    print("Completed")
