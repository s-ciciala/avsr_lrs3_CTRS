"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""
import math
import numpy as np

import torch
from tqdm import tqdm

from .metrics import compute_cer, compute_wer
from .decoders import ctc_greedy_decode, ctc_search_decode


def num_params(model):
    """
    Function that outputs the number of total and trainable paramters in the model.
    """
    numTotalParams = sum([params.numel() for params in model.parameters()])
    numTrainableParams = sum([params.numel() for params in model.parameters() if params.requires_grad])
    return numTotalParams, numTrainableParams


def train(model, trainLoader, optimizer, loss_function, device, trainParams):
    """
    Function to train the model for one iteration. (Generally, one iteration = one epoch, but here it is one step).
    It also computes the training loss, CER and WER. The CTC decode scheme is always 'greedy' here.
    """

    trainingLoss = 0
    trainingCER = 0
    trainingWER = 0

    for batch, (inputBatch, targetBatch, inputLenBatch, targetLenBatch, index) in enumerate(
            tqdm(trainLoader, leave=False, desc="Train",
                 ncols=75)):


        inputBatch, targetBatch = (inputBatch.float()).to(device), (targetBatch.float()).to(device)
        inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device), (targetLenBatch.int()).to(device)
        optimizer.zero_grad()
        model.train()
        # if len(inputBatch) >
        outputBatch = model(inputBatch)

        print(inputBatch, len(outputBatch))
        with torch.backends.cudnn.flags(enabled=True):
            print("\n")
            print("inputLenBatch " + str(inputLenBatch))
            print("targetLenBatch " + str(targetLenBatch))
            print("inputBatch " + str(inputBatch))
            print("targetBatch " + str(targetBatch))
            print("outputLenBatch " + str(len(outputBatch)))
            print("targetLenBatch " + str(len(targetBatch)))
            if len(outputBatch) < inputLenBatch:
                # print("CATCH")
                # print(outputBatch)
                # print(inputLenBatch)
                new_inputLenBatch = torch.tensor([len(outputBatch)], dtype=torch.int32, device=device)
                loss = loss_function(outputBatch, targetBatch, new_inputLenBatch, targetLenBatch)
            else:
                loss = loss_function(outputBatch, targetBatch, inputLenBatch, targetLenBatch)
        loss.backward()
        optimizer.step()
        # print("LOSS" * 10)
        # print(index)
        # print(trainLoader.dataset.datalist[index])
        # print(trainLoader.dataset[index][0].sum())
        # print(trainLoader.dataset[index][1].sum())
        # print(loss.item())
        if (loss.item() == math.inf):
            print(trainLoader.dataset.datalist[index])
            print(trainLoader)
            print(trainLoader.dataset[index])
            exit()
        trainingLoss = trainingLoss + loss.item()
        predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch.detach(), inputLenBatch,
                                                                trainParams["eosIx"])
        trainingCER = trainingCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
        trainingWER = trainingWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch,
                                                trainParams["spaceIx"])

    print("LEN OF TRAIN lOADER")
    print(len(trainLoader))
    trainingLoss = trainingLoss / len(trainLoader)
    trainingCER = trainingCER / len(trainLoader)
    trainingWER = trainingWER / len(trainLoader)
    return trainingLoss, trainingCER, trainingWER


def evaluate(model, evalLoader, loss_function, device, evalParams):
    """
    Function to evaluate the model over validation/test set. It computes the loss, CER and WER over the evaluation set.
    The CTC decode scheme can be set to either 'greedy' or 'search'.
    """

    evalLoss = 0
    evalCER = 0
    evalWER = 0

    for batch, (inputBatch, targetBatch, inputLenBatch, targetLenBatch) in enumerate(
            tqdm(evalLoader, leave=False, desc="Eval",
                 ncols=75)):

        inputBatch, targetBatch = (inputBatch.float()).to(device), (targetBatch.int()).to(device)
        inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device), (targetLenBatch.int()).to(device)

        model.eval()
        with torch.no_grad():
            outputBatch = model(inputBatch)
            with torch.backends.cudnn.flags(enabled=False):
                loss = loss_function(outputBatch, targetBatch, inputLenBatch, targetLenBatch)

        evalLoss = evalLoss + loss.item()
        if evalParams["decodeScheme"] == "greedy":
            predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch, inputLenBatch, evalParams["eosIx"])
        elif evalParams["decodeScheme"] == "search":
            predictionBatch, predictionLenBatch = ctc_search_decode(outputBatch, inputLenBatch,
                                                                    evalParams["beamSearchParams"],
                                                                    evalParams["spaceIx"], evalParams["eosIx"],
                                                                    evalParams["lm"])
        else:
            print("Invalid Decode Scheme")
            exit()

        evalCER = evalCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
        evalWER = evalWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch,
                                        evalParams["spaceIx"])

    evalLoss = evalLoss / len(evalLoader)
    evalCER = evalCER / len(evalLoader)
    evalWER = evalWER / len(evalLoader)
    return evalLoss, evalCER, evalWER
