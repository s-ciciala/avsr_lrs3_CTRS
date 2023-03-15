"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

import cv2 as cv
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.parameter as Parameter
import torch.distributed as dist




def preprocess_sample(file, params):

    """
    Function to preprocess each data sample.
    """

    videoFile = file + ".mp4"
    roiFile = file + ".png"
    visualFeaturesFile = file + ".npy"

    roiSize = params["roiSize"]
    normMean = params["normMean"]
    normStd = params["normStd"]
    vf = params["vf"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_devices = torch.cuda.device_count()


    #for each frame, resize to 224x224 and crop the central 112x112 region
    captureObj = cv.VideoCapture(videoFile)
    roiSequence = list()
    while (captureObj.isOpened()):
        ret, frame = captureObj.read()
        if ret == True:
            grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            grayed = grayed/255
            grayed = cv.resize(grayed, (224,224))
            roi = grayed[int(112-(roiSize/2)):int(112+(roiSize/2)), int(112-(roiSize/2)):int(112+(roiSize/2))]
            roiSequence.append(roi)
        else:
            break
    captureObj.release()

    # Concatenate the sequence of ROIs along the time axis to create a batch of data
    roiBatch = np.concatenate(roiSequence, axis=0)

    # Normalize the data using the provided mean and standard deviation
    roiBatch = (roiBatch - normMean) / normStd

    # Convert the numpy array to a PyTorch tensor
    roiBatch = torch.from_numpy(roiBatch).float()
    print("\nnum devices :" + str(torch.cuda.device_count() ))
    if torch.cuda.device_count() > 1:
        roiBatch = nn.DataParallel(roiBatch)
    else:
        # Move the tensor to the appropriate device
        roiBatch = roiBatch.to(device)
    cv.imwrite(roiFile, np.floor(255*np.concatenate(roiSequence, axis=1)).astype(np.int_))


    #normalise the frames and extract features for each frame using the visual frontend
    #save the visual features to a .npy file
    inp = np.stack(roiSequence, axis=0)
    inp = np.expand_dims(inp, axis=[1,2])
    inp = (inp - normMean)/normStd
    # Transpose inputBatch
    inputBatch = np.transpose(inp, (0, 3, 1, 2))
    # inputBatch = torch.from_numpy(inp)
    if num_devices > 1:
        inputBatch = Parameter(torch.from_numpy(inputBatch).float())
        inputBatch = nn.DataParallel(inputBatch)
    else:
        inputBatch = torch.from_numpy(inputBatch).float().to(device)
        # inputBatch = (inputBatch.float()).to(device)
    vf.eval()
    with torch.no_grad():
        outputBatch = vf(inputBatch)
    out = torch.squeeze(outputBatch, dim=1)
    out = out.cpu().numpy()
    np.save(visualFeaturesFile, out)
    return
