import torch
from tqdm import tqdm

from .metrics import compute_cer, compute_wer
from .decoders import ctc_greedy_decode, ctc_search_decode
from sys import exit
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
    
    for batch, (inputBatch, targetBatch, inputLenBatch, targetLenBatch) in enumerate(tqdm(trainLoader, leave=False, desc="Train",
                                                                                          ncols=75)):

        inputBatch, targetBatch = (inputBatch.float()).to(device), (targetBatch.int()).to(device)
        inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device), (targetLenBatch.int()).to(device)

        optimizer.zero_grad()
        model.train()
        outputBatch = model(inputBatch)
        with torch.backends.cudnn.flags(enabled=False):
            loss = loss_function(outputBatch, targetBatch, inputLenBatch, targetLenBatch)
        loss.backward()
        optimizer.step()

        trainingLoss = trainingLoss + loss.item()
        predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch.detach(), inputLenBatch, trainParams["eosIx"])
        trainingCER = trainingCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
        trainingWER = trainingWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, trainParams["spaceIx"])

    trainingLoss = trainingLoss/len(trainLoader)
    trainingCER = trainingCER/len(trainLoader)
    trainingWER = trainingWER/len(trainLoader)
    return trainingLoss, trainingCER, trainingWER



def evaluate(model, evalLoader, loss_function, device, evalParams):

    """
    Function to evaluate the model over validation/test set. It computes the loss, CER and WER over the evaluation set.
    The CTC decode scheme can be set to either 'greedy' or 'search'.
    """

    evalLoss = 0
    evalCER = 0
    evalWER = 0

    for batch, (inputBatch, targetBatch, inputLenBatch, targetLenBatch) in enumerate(tqdm(evalLoader, leave=False, desc="Eval",
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
            predictionBatch, predictionLenBatch = ctc_search_decode(outputBatch, inputLenBatch, evalParams["beamSearchParams"],
                                                                    evalParams["spaceIx"], evalParams["eosIx"], evalParams["lm"])
        else:
            print("Invalid Decode Scheme")
            exit()

        # Convert prediction and target tensors to strings
        index_to_char = args["INDEX_TO_CHAR"]
        predictionStrings = []
        targetStrings = []
        predictionString = ""
        for i in range(len(predictionBatch)):
            item_idx = predictionBatch[i].item()
            charrr = index_to_char[item_idx]
            # print(index_to_char[item_idx])
            predictionString += str(charrr)
        # print("------------------PREDICTION------------------")
        # print(predictionString)
        predictionStrings.append(predictionString)
        #
        # for i in range(targetBatch.shape[0]):
        #     targetString = ""
        #     for j in range(targetLenBatch[i]):
        #         targetString += index_to_char[targetBatch[i][j].item()]
        #     targetStrings.append(targetString)
        targetString = ""
        for i in range(len(targetBatch)):
            item_idx = targetBatch[i].item()
            charrr = index_to_char[item_idx]
            # print(index_to_char[item_idx])
            targetString += str(charrr)
        # print("------------------TARGET------------------")
        # print(targetString)
        predictionStrings.append(predictionString)
        targetBatch = targetBatch.cpu()
        targetLenBatch = targetLenBatch.cpu()
        preds = list(torch.split(predictionBatch, predictionLenBatch.tolist()))
        trgts = list(torch.split(targetBatch, targetLenBatch.tolist()))
        predictionStrings = []
        targetStrings = []
        for prediction in preds:
            curr_string = ""
            for char in prediction:
                item_idx = char.item()
                charrr = index_to_char[item_idx]
                curr_string += charrr
            predictionStrings.append(curr_string)

        for target in trgts:
            curr_string = ""
            for char in target:
                # print(char.item())
                item_idx = char.item()
                charrr = index_to_char[item_idx]
                # print(charrr)
                curr_string += charrr
            targetStrings.append(curr_string)
        if args["DISPLAY_PREDICTIONS"]:
            for i in range(len(predictionStrings)):
                print("------------------PREDICTION------------------")
                print("------------------PREDICTION------------------")
                print(predictionStrings[i])
                print("------------------TARGET------------------")
                print("------------------TARGET------------------")
                print(targetStrings[i])
            with open('test_results.txt', 'w') as f:
                for i in range(len(predictionStrings)):
                    f.write("------------------TARGET------------------")
                    f.write("%s\n" % str(targetStrings[i]))
                    f.write("------------------PREDICTION------------------")
                    f.write("%s\n" % str(predictionStrings[i]))
        evalCER = evalCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
        evalWER = evalWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, evalParams["spaceIx"])

    evalLoss = evalLoss/len(evalLoader)
    evalCER = evalCER/len(evalLoader)
    evalWER = evalWER/len(evalLoader)
    return evalLoss, evalCER, evalWER
