"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""
import math
import numpy as np

import Levenshtein
import torch
from tqdm import tqdm

from .metrics import compute_cer, compute_wer
from .decoders import ctc_greedy_decode, ctc_search_decode
from config import args


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

        with torch.backends.cudnn.flags(enabled=True):
            arry = []
            for btch in inputLenBatch:
                # print("HERE")
                # print(btch)
                if len(outputBatch) < btch:
                    arry.append(len(outputBatch))
                else:
                    arry.append(btch)
            new_inputLenBatch = torch.tensor(arry, dtype=torch.int32, device=device)
            loss = loss_function(outputBatch, targetBatch, new_inputLenBatch, targetLenBatch)
            # if len(outputBatch) < inputLenBatch:
            #     # print("CATCH")
            #     # print(outputBatch)
            #     # print(inputLenBatch)
            #     new_inputLenBatch = torch.tensor([len(outputBatch)], dtype=torch.int32, device=device)
            #     loss = loss_function(outputBatch, targetBatch, new_inputLenBatch, targetLenBatch)
            # else:
            #     loss = loss_function(outputBatch, targetBatch, inputLenBatch, targetLenBatch)
        loss.backward()
        optimizer.step()
        if loss.item() == math.inf:
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

    # print("LEN OF TRAIN lOADER")
    # print(len(trainLoader))
    trainingLoss = trainingLoss / len(trainLoader)
    trainingCER = trainingCER / len(trainLoader)
    trainingWER = trainingWER / len(trainLoader)
    return trainingLoss, trainingCER, trainingWER
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

import torch

class GreedyDecoder:
    def __call__(self, outputs):
        _, max_indices = torch.max(outputs, dim=2)
        return max_indices

def decode_predictions(outputs, targets, idx2char):
    greedy_decoder = GreedyDecoder()

    # Decode the output probabilities
    decoded_preds = greedy_decoder(outputs)
    decoded_preds = [indices_to_text(seq, idx2char) for seq in decoded_preds]

    # Decode the targets
    decoded_targets = [indices_to_text(seq, idx2char) for seq in targets]

    return decoded_preds, decoded_targets

def indices_to_text(indices, idx2char):
    text = ''.join([idx2char[idx] for idx in indices])
    return text


def calculate_metrics(decoded_preds, decoded_targets):
    total_cer, total_wer = 0.0, 0.0
    total_chars, total_words = 0, 0

    for pred, target in zip(decoded_preds, decoded_targets):
        # Calculate CER
        cer = Levenshtein.distance(pred, target)
        total_cer += cer
        total_chars += len(target)

        # Calculate WER
        pred_words = pred.split()
        target_words = target.split()
        wer = Levenshtein.distance(pred_words, target_words)
        total_wer += wer
        total_words += len(target_words)

    avg_cer = total_cer / total_chars
    avg_wer = total_wer / total_words

    return avg_cer, avg_wer


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_cer, total_wer = 0.0, 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets, input_lengths, target_lengths, index = batch
            inputs.float().to(device)
            inputs, targets = inputs.float().to(device), targets.float().to(device)

            # Forward pass
            outputs = model(inputs)
            outputs = outputs.permute(1, 0, 2)  # (B, T, C) -> (T, B, C)

            # Calculate loss
            print("in","tar")
            print(len(input_lengths),len(target_lengths))
            loss = criterion(outputs, targets, input_lengths, target_lengths)
            running_loss += loss.item()

            # Calculate CER and WER
            batch_size = inputs.size(0)

            decoded_preds, decoded_targets = decode_predictions(outputs, targets)
            cer, wer = calculate_metrics(decoded_preds, decoded_targets)
            total_cer += cer * batch_size
            total_wer += wer * batch_size
            total_samples += batch_size

    # Calculate average loss, CER, and WER
    avg_loss = running_loss / len(dataloader)
    avg_cer = total_cer / total_samples
    avg_wer = total_wer / total_samples

    return avg_loss, avg_cer, avg_wer


# def evaluate(model, evalLoader, loss_function, device, evalParams):
#     """
#     Function to evaluate the model over validation/test set. It computes the loss, CER and WER over the evaluation set.
#     The CTC decode scheme can be set to either 'greedy' or 'search'.
#     """
#
#     evalLoss = 0
#     evalCER = 0
#     evalWER = 0
#
#     index_to_char = args["INDEX_TO_CHAR"]
#
#     for batch, (inputBatch, targetBatch, inputLenBatch, targetLenBatch, index) in enumerate(
#             tqdm(evalLoader, leave=False, desc="Eval",
#                  ncols=75)):
#
#         inputBatch, targetBatch = (inputBatch.float()).to(device), (targetBatch.float()).to(device)
#         inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device), (targetLenBatch.int()).to(device)
#
#         model.eval()
#         with torch.no_grad():
#             outputBatch = model(inputBatch)
#             with torch.backends.cudnn.flags(enabled=True):
#                 arry = []
#                 for btch in inputLenBatch:
#                     if len(outputBatch) < btch:
#                         arry.append(len(outputBatch))
#                     else:
#                         arry.append(btch)
#                 new_inputLenBatch = torch.tensor(arry, dtype=torch.int32, device=device)
#                 loss = loss_function(outputBatch, targetBatch, new_inputLenBatch, targetLenBatch)
#
#         evalLoss = evalLoss + loss.item()
#         if evalParams["decodeScheme"] == "greedy":
#             predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch, inputLenBatch, evalParams["eosIx"])
#         elif evalParams["decodeScheme"] == "search":
#             predictionBatch, predictionLenBatch = ctc_search_decode(outputBatch, inputLenBatch,
#                                                                     evalParams["beamSearchParams"],
#                                                                     evalParams["spaceIx"], evalParams["eosIx"],
#                                                                     evalParams["lm"])
#         else:
#             print("Invalid Decode Scheme")
#             exit()
#
#         # Convert prediction and target tensors to strings
#         predictionStrings = []
#         targetStrings = []
#         predictionString = ""
#         for i in range(len(predictionBatch)):
#             item_idx = predictionBatch[i].item()
#             charrr = index_to_char[item_idx]
#             # print(index_to_char[item_idx])
#             predictionString += str(charrr)
#         # print("------------------PREDICTION------------------")
#         # print(predictionString)
#         predictionStrings.append(predictionString)
#         #
#         # for i in range(targetBatch.shape[0]):
#         #     targetString = ""
#         #     for j in range(targetLenBatch[i]):
#         #         targetString += index_to_char[targetBatch[i][j].item()]
#         #     targetStrings.append(targetString)
#
#         targetString = ""
#         for i in range(len(targetBatch)):
#             item_idx = targetBatch[i].item()
#             charrr = index_to_char[item_idx]
#             # print(index_to_char[item_idx])
#             targetString += str(charrr)
#         # print("------------------TARGET------------------")
#         # print(targetString)
#         predictionStrings.append(predictionString)
#
#         targetBatch = targetBatch.cpu()
#         targetLenBatch = targetLenBatch.cpu()
#
#         preds = list(torch.split(predictionBatch, predictionLenBatch.tolist()))
#         trgts = list(torch.split(targetBatch, targetLenBatch.tolist()))
#         # print(preds)
#         # print(len(preds))
#         predictionStrings = []
#         targetStrings = []
#         for prediction in preds:
#             curr_string = ""
#             for char in prediction:
#                 # print(char.item())
#                 item_idx = char.item()
#                 charrr = index_to_char[item_idx]
#                 # print(charrr)
#                 curr_string += charrr
#             predictionStrings.append(curr_string)
#
#         for target in trgts:
#             curr_string = ""
#             for char in target:
#                 # print(char.item())
#                 item_idx = char.item()
#                 charrr = index_to_char[item_idx]
#                 # print(charrr)
#                 curr_string += charrr
#             targetStrings.append(curr_string)
#         if args["DISPLAY_PREDICTIONS"]:
#             for i in range(len(predictionStrings)):
#                 print("------------------PREDICTION------------------")
#                 print("------------------PREDICTION------------------")
#                 print(predictionStrings[i])
#                 print("------------------TARGET------------------")
#                 print("------------------TARGET------------------")
#                 print(targetStrings[i])
#
#     evalCER = evalCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
#     evalWER = evalWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch,
#                                     evalParams["spaceIx"])
#
#     evalLoss = evalLoss / len(evalLoader)
#     evalCER = evalCER / len(evalLoader)
#     evalWER = evalWER / len(evalLoader)
#     return evalLoss, evalCER, evalWER
