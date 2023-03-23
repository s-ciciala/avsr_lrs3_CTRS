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
        model.module.listener.flatten_parameters()
        model.module.speller.flatten_parameters()
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


def evaluate(model, evalLoader, loss_function, device, evalParams):
    """
    Function to evaluate the model on the given dataset. It computes the evaluation loss, CER, and WER.
    The CTC decode scheme is always 'greedy' here.
    """

    evalLoss = 0
    evalCER = 0
    evalWER = 0

    model.eval()
    index_to_char = args["INDEX_TO_CHAR"]
    predictionStrings = []
    targetStrings = []
    with torch.no_grad():
        for batch, (inputBatch, targetBatch, inputLenBatch, targetLenBatch, index) in enumerate(
                tqdm(evalLoader, leave=False, desc="Eval",
                     ncols=75)):

            inputBatch, targetBatch = (inputBatch.float()).to(device), (targetBatch.float()).to(device)
            inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device), (targetLenBatch.int()).to(device)
            model.module.listener.flatten_parameters()
            model.module.speller.flatten_parameters()
            outputBatch = model(inputBatch)

            with torch.backends.cudnn.flags(enabled=True):
                arry = []
                for btch in inputLenBatch:
                    if len(outputBatch) < btch:
                        arry.append(len(outputBatch))
                    else:
                        arry.append(btch)
                new_inputLenBatch = torch.tensor(arry, dtype=torch.int32, device=device)
                loss = loss_function(outputBatch, targetBatch, new_inputLenBatch, targetLenBatch)

            evalLoss = evalLoss + loss.item()
            predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch.detach(), inputLenBatch,
                                                                    evalParams["eosIx"])
            evalCER = evalCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
            evalWER = evalWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch,
                                            evalParams["spaceIx"])
            ##Per batch, predict what it should be , show the target
            # Convert prediction and target tensors to strings
            index_to_char = args["INDEX_TO_CHAR"]
            predictionString = ""
            for i in range(len(predictionBatch)):
                item_idx = predictionBatch[i].item()
                charrr = index_to_char[item_idx]
                predictionString += str(charrr)
            predictionStrings.append(predictionString)
            targetString = ""
            for i in range(len(targetBatch)):
                item_idx = targetBatch[i].item()
                charrr = index_to_char[item_idx]
                targetString += str(charrr)
            targetStrings.append(targetString)

        evalLoss = evalLoss / len(evalLoader)
        evalCER = evalCER / len(evalLoader)
        evalWER = evalWER / len(evalLoader)
        if args["DISPLAY_PREDICTIONS"]:
            for i in range(len(predictionStrings)):
                print("------------------PREDICTION------------------")
                print("------------------PREDICTION------------------")
                print(predictionStrings[i])
                print("------------------TARGET------------------")
                print("------------------TARGET------------------")
                print(targetStrings[i])
            with open('test_results_audio_only.txt', 'w') as f:
                for i in range(len(predictionStrings)):
                    f.write("------------------TARGET------------------\n")
                    f.write("%s\n" % str(targetStrings[i]))
                    f.write("------------------PREDICTION------------------\n")
                    f.write("%s\n" % str(predictionStrings[i]))
                f.write("\n" + "evalLoss: " + str(evalLoss))
                f.write("\n" + "evalCER: " + str(evalCER))
                f.write("\n" + "evalWER: " + str(evalWER))

    return evalLoss, evalCER, evalWER


#
# def evaluate(model, dataloader, criterion, device, eval_params):
#     model.eval()
#     total_loss = 0
#     total_cer = 0
#     total_wer = 0
#     idx2char = args["INDEX_TO_CHAR"]
#
#     with torch.no_grad():
#         for batch_idx, (inputs, targets, input_lengths, target_lengths, index) in enumerate(dataloader):
#             inputs = inputs.float().to(device)
#             targets = targets.float().to(device)
#             input_lengths = input_lengths.int().to(device)
#             target_lengths = target_lengths.int().to(device)
#
#             # outputs = model(inputs, input_lengths)
#             outputs = model(inputs)
#
#             adjusted_input_lengths = torch.min(input_lengths, torch.tensor(outputs.size(1), device=device))
#             loss = criterion(outputs, targets, adjusted_input_lengths, target_lengths)
#             total_loss += loss.item()
#
#             if eval_params["decodeScheme"] == "greedy":
#                 predictionBatch, predictionLenBatch = ctc_greedy_decode(outputs, input_lengths, eval_params["eosIx"])
#             elif eval_params["decodeScheme"] == "search":
#                 predictionBatch, predictionLenBatch = ctc_search_decode(outputs, input_lengths,
#                                                                         eval_params["beamSearchParams"],
#                                                                         eval_params["spaceIx"], eval_params["eosIx"],
#                                                                         eval_params["lm"])
#             # Use the provided decode_predictions function
#     #     evalCER = evalCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
#     #     evalWER = evalWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch,
#     #                                     evalParams["spaceIx"])
#     #
#     #     evalLoss = evalLoss / len(evalLoader)
#     #     evalCER = evalCER / len(evalLoader)
#     #     evalWER = evalWER / len(evalLoader)
#     #     return evalLoss, evalCER, evalWER
#     #         decoded_preds, decoded_targets = decode_predictions(outputs, targets, idx2char)
#     #
#     #         # Use the provided calculate_metrics function
#     #         batch_cer, batch_wer = calculate_metrics(decoded_preds, decoded_targets)
#     #
#     #         total_cer += batch_cer
#     #         total_wer += batch_wer
#     #
#     # avg_loss = total_loss / len(dataloader)
#     # avg_cer = total_cer / len(dataloader)
#     # avg_wer = total_wer / len(dataloader)
#     #
#     # return avg_loss, avg_cer, avg_wer
#

# def indices_to_text(indices, idx2char):
#     text = ''.join([idx2char[idx] for idx in indices])
#     return text
#
#
# class GreedyDecoder:
#     def __call__(self, outputs):
#         _, max_indices = torch.max(outputs, dim=2)
#         return max_indices
#
#
# def decode_predictions(outputs, targets, idx2char):
#     greedy_decoder = GreedyDecoder()
#
#     # Decode the output probabilities
#     decoded_preds = greedy_decoder(outputs)
#     decoded_preds_text = []
#     for seq in decoded_preds:
#         # print("*"*80)
#         # print(seq)
#         text = ""
#         for char1 in seq.data:
#             text += idx2char[char1.item()]
#         decoded_preds_text.append(text)
#     # decoded_preds = [indices_to_text(seq, idx2char) for seq in decoded_preds]
#
#     # Decode the targets
#     # decoded_targets = [indices_to_text(seq, idx2char) for seq in targets]
#     decoded_targets_text = []
#     for seq in targets:
#         # print("*"*80)
#         # print(seq)
#         try:
#             ##Targets can be singletons
#             # print(seq.item())
#             char_index = int(seq.item())
#             text = idx2char[char_index]
#             decoded_targets_text.append(text)
#         except:
#             # print("Not a singleton hence will try a sequence")
#             text = ""
#             for char2 in seq.data:
#                 text += idx2char[char2.item()]
#             decoded_targets_text.append(text)
#
#     return decoded_preds_text, decoded_targets_text
#

#
#
# def evaluate(model, dataloader, loss_function, device):
#     model.eval()
#     running_loss = 0.0
#     total_cer, total_wer = 0.0, 0.0
#     total_samples = 0
#
#     with torch.no_grad():
#         for batch in dataloader:
#             inputs, targets, input_lengths, target_lengths, index = batch
#             inputs, targets = inputs.float().to(device), targets.float().to(device)
#
#             # Forward pass
#             outputs = model(inputs)
#             outputs = outputs.permute(1, 0, 2)  # (B, T, C) -> (T, B, C)
#
#             # Calculate loss
#             print("in","tar")
#             print(len(input_lengths),len(target_lengths))
#             print("in", "tar")
#             print(len(outputs),len(targets))
#             print(len(targets))
#             target_sequences = torch.split(targets, target_lengths.tolist(), dim=0)
#             print( "tar")
#             print(len(target_sequences))
#             loss = loss_function(outputs, target_sequences, input_lengths, target_lengths)
#             running_loss += loss.item()
#
#             # Calculate CER and WER
#             batch_size = inputs.size(0)
#
#             decoded_preds, decoded_targets = decode_predictions(outputs, targets)
#             cer, wer = calculate_metrics(decoded_preds, decoded_targets)
#             total_cer += cer * batch_size
#             total_wer += wer * batch_size
#             total_samples += batch_size
#
#     # Calculate average loss, CER, and WER
#     avg_loss = running_loss / len(dataloader)
#     avg_cer = total_cer / total_samples
#     avg_wer = total_wer / total_samples
#
#     return avg_loss, avg_cer, avg_wer


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
#     predictionStrings = []
#     targetStrings = []
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
#         evalCER = evalCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
#         evalWER = evalWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch,
#                                         evalParams["spaceIx"])
#
#         ##Per batch, predict what it should be , show the target
#         # Convert prediction and target tensors to strings
#         index_to_char = args["INDEX_TO_CHAR"]
#         predictionString = ""
#         for i in range(len(predictionBatch)):
#             item_idx = predictionBatch[i].item()
#             charrr = index_to_char[item_idx]
#             predictionString += str(charrr)
#         predictionStrings.append(predictionString)
#         targetString = ""
#         for i in range(len(targetBatch)):
#             item_idx = targetBatch[i].item()
#             charrr = index_to_char[item_idx]
#             targetString += str(charrr)
#         targetStrings.append(targetString)
#
#     evalLoss = evalLoss / len(evalLoader)
#     evalCER = evalCER / len(evalLoader)
#     evalWER = evalWER / len(evalLoader)
#     if args["DISPLAY_PREDICTIONS"]:
#         for i in range(len(predictionStrings)):
#             print("------------------PREDICTION------------------")
#             print("------------------PREDICTION------------------")
#             print(predictionStrings[i])
#             print("------------------TARGET------------------")
#             print("------------------TARGET------------------")
#             print(targetStrings[i])
#         with open('test_results_audio_only.txt', 'w') as f:
#             for i in range(len(predictionStrings)):
#                 f.write("------------------TARGET------------------\n")
#                 f.write("%s\n" % str(targetStrings[i]))
#                 f.write("------------------PREDICTION------------------\n")
#                 f.write("%s\n" % str(predictionStrings[i]))
#             f.write("\n" + "evalLoss: " + str(evalLoss))
#             f.write("\n" + "evalCER: " + str(evalCER))
#             f.write("\n" + "evalWER: " + str(evalWER))
#
#     return evalLoss, evalCER, evalWER
