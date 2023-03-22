import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, shutil

from config import args
from models.audio_net import AudioNet
from data.lrs3_dataset import LRS3Main
from data.utils import collate_fn
from utils.general import num_params, train, evaluate
from tqdm import tqdm
from sys import exit


def set_device():
    print("GPU on?:" + str(torch.cuda.is_available()))
    print("Backend on?:" + str(torch.backends.cudnn.enabled))
    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    print("available_gpus: " + str(len(available_gpus)))
    print("device_count: " + str(torch.cuda.device_count()))

    matplotlib.use("Agg")
    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    if not (torch.cuda.is_available()):
        exit()
    device = torch.device(str(args["GPU"]) if len(available_gpus) != 0 else "cpu")
    kwargs = {"num_workers": args["NUM_WORKERS"], "pin_memory": True} if torch.cuda.is_available() else {}
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return device, kwargs


def get_training_data(device, kwargs):
    dataset = "train"
    datadir = args["DATA_DIRECTORY"]
    reqInpLen = args["MAIN_REQ_INPUT_LENGTH"]
    charToIx = args["CHAR_TO_INDEX"]
    stepSize = args["STEP_SIZE"]

    # declaring the train and validation datasets and their corresponding dataloaders
    audioParams = {"stftWindow": args["STFT_WINDOW"], "stftWinLen": args["STFT_WIN_LENGTH"],
                   "stftOverlap": args["STFT_OVERLAP"]}
    noiseParams = {"noiseFile": args["DATA_DIRECTORY"] + "/noise.wav", "noiseProb": args["NOISE_PROBABILITY"],
                   "noiseSNR": args["NOISE_SNR_DB"]}

    trainData = LRS3Main(dataset, datadir, reqInpLen, charToIx, stepSize, audioParams, noiseParams)

    trainLoader = DataLoader(trainData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True, **kwargs)

    ###CHECK BATCH INDEX
    # myiter = iter(trainLoader)
    # myiter.__next__()
    # one, two, three, four, five = next(iter(trainLoader))
    # print(one,two,three,four,five)
    # print("*"*80)
    # print(five)
    # exit()
    ###CHECK BATCH INDEX

    noiseParams = {"noiseFile": args["DATA_DIRECTORY"] + "/noise.wav", "noiseProb": 0, "noiseSNR": args["NOISE_SNR_DB"]}

    valData = LRS3Main("val", args["DATA_DIRECTORY"], args["MAIN_REQ_INPUT_LENGTH"], args["CHAR_TO_INDEX"],
                       args["STEP_SIZE"], audioParams, noiseParams)

    valLoader = DataLoader(valData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=True, **kwargs)

    # declaring the model, optimizer, scheduler and the loss function
    #PREVIOUS DECLIRATION OF AUDIONET
    # model = AudioNet(args["TX_NUM_FEATURES"], args["TX_ATTENTION_HEADS"], args["TX_NUM_LAYERS"], args["PE_MAX_LENGTH"],
    #                  args["AUDIO_FEATURE_SIZE"], args["TX_FEEDFORWARD_DIM"], args["TX_DROPOUT"], args["NUM_CLASSES"])
    #LSTM DECLIRATION
    model = AudioNet(dModel=args["TX_NUM_FEATURES"],
                     numLayers=args["TX_NUM_LAYERS"],
                     inSize=args["AUDIO_FEATURE_SIZE"],
                     fcHiddenSize=args["TX_FEEDFORWARD_DIM"],
                     dropout=args["TX_DROPOUT"],
                     numClasses=args["NUM_CLASSES"])

    ##added multiprocessing
    if args["LIMITGPU"]:
        model = nn.DataParallel(model, device_ids=args["GPUID"])
    else:
        model = nn.DataParallel(model)
    model.to(device)
    return trainData, trainLoader, valData, valLoader, model


def get_optimiser_and_checkpoint_dir(model):
    optimizer = optim.Adam(model.parameters(), lr=args["INIT_LR"], betas=(args["MOMENTUM1"], args["MOMENTUM2"]))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args["LR_SCHEDULER_FACTOR"],
                                                     patience=args["LR_SCHEDULER_WAIT"],
                                                     threshold=args["LR_SCHEDULER_THRESH"],
                                                     threshold_mode="abs", min_lr=args["FINAL_LR"], verbose=True)
    loss_function = nn.CTCLoss(blank=0, zero_infinity=True)
    if args["CHECKPOINTS"]:
        if os.path.exists(args["CODE_DIRECTORY"] + "/checkpoints"):
            while True:
                ch = input("Continue and remove the 'checkpoints' directory? y/n: ")
                if ch == "y":
                    break
                elif ch == "n":
                    exit()
                else:
                    print("Invalid input")
            shutil.rmtree(args["CODE_DIRECTORY"] + "/checkpoints")

    if not os.path.exists(args["CODE_DIRECTORY"] + "audio_only_checkpoints/"):
        os.makedirs(args["CODE_DIRECTORY"] + "audio_only_checkpoints/")
    if not os.path.exists(args["CODE_DIRECTORY"] + "audio_only_checkpoints/models"):
        os.makedirs(args["CODE_DIRECTORY"] + "audio_only_checkpoints/models")
    if not os.path.exists(args["CODE_DIRECTORY"] + "audio_only_checkpoints/plots"):
        os.makedirs(args["CODE_DIRECTORY"] + "audio_only_checkpoints/plots")

    return optimizer, scheduler, loss_function


def train_model(model, trainLoader, valLoader, optimizer, loss_function, device):
    print("\nTraining the model .... \n")

    trainParams = {"spaceIx": args["CHAR_TO_INDEX"][" "], "eosIx": args["CHAR_TO_INDEX"]["<EOS>"]}
    valParams = {"decodeScheme": "greedy", "spaceIx": args["CHAR_TO_INDEX"][" "],
                 "eosIx": args["CHAR_TO_INDEX"]["<EOS>"]}

    trainingLossCurve = list()
    validationLossCurve = list()
    trainingCERCurve = list()
    validationCERCurve = list()
    trainingWERCurve = list()
    validationWERCurve = list()

    for step in range(args["NUM_STEPS"]):
        # train the model for one step
        trainingLoss, trainingCER, trainingWER = train(model, trainLoader, optimizer, loss_function, device,
                                                       trainParams)
        trainingLossCurve.append(trainingLoss)
        trainingCERCurve.append(trainingCER)
        trainingWERCurve.append(trainingWER)

        # evaluate the model on validation set
        validationLoss, validationCER, validationWER = evaluate(model, valLoader, loss_function, device, valParams)
        #def evaluate(model, dataloader, criterion, device):
        # validationLoss, validationCER, validationWER = evaluate(model, valLoader, loss_function, device)
        validationLossCurve.append(validationLoss)
        validationCERCurve.append(validationCER)
        validationWERCurve.append(validationWER)
        # printing the stats after each step
        print(
            "Step: %03d || Tr.Loss: %.6f  Val.Loss: %.6f || Tr.CER: %.3f  Val.CER: %.3f || Tr.WER: %.3f  Val.WER: %.3f"
            % (step, trainingLoss, validationLoss, trainingCER, validationCER, trainingWER, validationWER))

        # make a scheduler step
        scheduler.step(validationWER)
        save_dict = {
            'epoch': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': trainingLoss,
        }
        # saving the model weights and loss/metric curves in the checkpoints directory after every few steps
        if ((step % args["SAVE_FREQUENCY"] == 0) or (step == args["NUM_STEPS"] - 1)) and (step != 0):
            savePath = args["CODE_DIRECTORY"] + "/audio_only_checkpoints/models/train-step_{:04d}-wer_{:.3f}.pt".format(
                step,
                validationWER)
            torch.save(save_dict, savePath)

            plt.figure()
            plt.title("Loss Curves")
            plt.xlabel("Step No.")
            plt.ylabel("Loss value")
            plt.plot(list(range(1, len(trainingLossCurve) + 1)), trainingLossCurve, "blue", label="Train")
            plt.plot(list(range(1, len(validationLossCurve) + 1)), validationLossCurve, "red", label="Validation")
            plt.legend()
            plt.savefig(
                args["CODE_DIRECTORY"] + "/audio_only_checkpoints/plots/train-step_{:04d}-loss.png".format(step))
            plt.close()

            plt.figure()
            plt.title("WER Curves")
            plt.xlabel("Step No.")
            plt.ylabel("WER")
            plt.plot(list(range(1, len(trainingWERCurve) + 1)), trainingWERCurve, "blue", label="Train")
            plt.plot(list(range(1, len(validationWERCurve) + 1)), validationWERCurve, "red", label="Validation")
            plt.legend()
            plt.savefig(args["CODE_DIRECTORY"] + "/audio_only_checkpoints/plots/train-step_{:04d}-wer.png".format(step))
            plt.close()

            plt.figure()
            plt.title("CER Curves")
            plt.xlabel("Step No.")
            plt.ylabel("CER")
            plt.plot(list(range(1, len(trainingCERCurve) + 1)), trainingCERCurve, "blue", label="Train")
            plt.plot(list(range(1, len(validationCERCurve) + 1)), validationCERCurve, "red", label="Validation")
            plt.legend()
            plt.savefig(
                args["CODE_DIRECTORY"] + "/audio_only_checkpoints/plots/train-step_{:04d}-loss.png".format(step))
            plt.close()
    print("\nTraining Done.\n")


if __name__ == "__main__":
    device, kwargs = set_device()
    print("Training using :" + str(device))
    trainData, trainLoader, valData, valLoader, model = get_training_data(device, kwargs)

    optimizer, scheduler, loss_function = get_optimiser_and_checkpoint_dir(model)

    numTotalParams, numTrainableParams = num_params(model)
    print("\nNumber of total parameters in the model = %d" % (numTotalParams))
    print("Number of trainable parameters in the model = %d\n" % (numTrainableParams))
    torch.cuda.empty_cache()
    train_model(model, trainLoader, valLoader, optimizer, loss_function, device)
    print("Completed")
