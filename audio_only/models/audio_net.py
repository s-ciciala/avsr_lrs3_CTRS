import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioNet(nn.Module):
    def __init__(self, dModel, numLayers, inSize, fcHiddenSize, dropout, numClasses):
        super(AudioNet, self).__init__()
        self.audioConv = nn.Conv1d(inSize, dModel, kernel_size=4, stride=4, padding=0)

        # Listener (encoder) - Bidirectional LSTM
        self.listener = nn.LSTM(input_size=dModel, hidden_size=dModel, num_layers=numLayers, dropout=dropout,
                                bidirectional=True, batch_first=True)

        # Attention Mechanism
        self.attention = nn.MultiheadAttention(embed_dim=dModel * 2, num_heads=1)

        # Speller (decoder) - LSTM
        self.speller = nn.LSTM(input_size=dModel * 2, hidden_size=dModel * 2, num_layers=numLayers, dropout=dropout,
                               batch_first=True)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.outputConv = nn.Conv1d(dModel * 2, numClasses, kernel_size=1, stride=1, padding=0)

    def forward(self, inputBatch):
        inputBatch = inputBatch.transpose(0, 1).transpose(1, 2)
        batch = self.audioConv(inputBatch)
        batch = batch.transpose(1, 2).transpose(0, 1)

        # Listener (encoder) - Bidirectional LSTM
        self.listener.flatten_parameters()  # Add this line
        batch, _ = self.listener(batch)

        # Attention Mechanism
        attn_output, _ = self.attention(batch, batch, batch)

        # Speller (decoder) - LSTM
        self.speller.flatten_parameters()  # Add this line
        speller_output, _ = self.speller(attn_output)

        speller_output = speller_output.transpose(0, 1).transpose(1, 2)
        batch = self.pool(speller_output)
        batch = self.outputConv(batch)
        batch = batch.transpose(1, 2).transpose(0, 1)
        outputBatch = F.log_softmax(batch, dim=2)

        return outputBatch
    # def forward(self, inputBatch):
    #     inputBatch = inputBatch.transpose(0, 1).transpose(1, 2)
    #     batch = self.audioConv(inputBatch)
    #     batch = batch.transpose(1, 2).transpose(0, 1)
    #
    #     # Listener (encoder) - Bidirectional LSTM
    #     batch, _ = self.listener(batch)
    #
    #     # Attention Mechanism
    #     attn_output, _ = self.attention(batch, batch, batch)
    #
    #     # Speller (decoder) - LSTM
    #     speller_output, _ = self.speller(attn_output)
    #
    #     speller_output = speller_output.transpose(0, 1).transpose(1, 2)
    #     batch = self.pool(speller_output)
    #     batch = self.outputConv(batch)
    #     batch = batch.transpose(1, 2).transpose(0, 1)
    #     outputBatch = F.log_softmax(batch, dim=2)
    #
    #     return outputBatch
# ###Previous transformer architecture
# class PositionalEncoding(nn.Module):
#
#     """
#     A layer to add positional encodings to the inputs of a Transformer model.
#     Formula:
#     PE(pos,2i) = sin(pos/10000^(2i/d_model))
#     PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
#     """
#
#     def __init__(self, dModel, maxLen):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(maxLen, dModel)
#         position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(dim=-1)
#         denominator = torch.exp(torch.arange(0, dModel, 2).float()*(math.log(10000.0)/dModel))
#         pe[:, 0::2] = torch.sin(position/denominator)
#         pe[:, 1::2] = torch.cos(position/denominator)
#         pe = pe.unsqueeze(dim=0).transpose(0, 1)
#         self.register_buffer("pe", pe)
#
#
#     def forward(self, inputBatch):
#         outputBatch = inputBatch + self.pe[:inputBatch.shape[0],:,:]
#         return outputBatch
#
#
# class AudioNet(nn.Module):
#     def __init__(self, dModel, nHeads, numLayers, peMaxLen, inSize, fcHiddenSize, dropout, numClasses):
#         super(AudioNet, self).__init__()
#         self.audioConv = nn.Conv1d(inSize, dModel, kernel_size=4, stride=4, padding=0)
#         self.positionalEncoding = PositionalEncoding(dModel=dModel, maxLen=peMaxLen)
#         encoderLayer = nn.TransformerEncoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
#         self.audioEncoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
#         self.audioDecoder = nn.TransformerEncoder(encoderLayer, num_layers=numLayers)
#         self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
#         self.outputConv = nn.Conv1d(dModel, numClasses, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, inputBatch):
#         inputBatch = inputBatch.transpose(0, 1).transpose(1, 2)
#         batch = self.audioConv(inputBatch)
#         batch = batch.transpose(1, 2).transpose(0, 1)
#         batch = self.positionalEncoding(batch)
#         batch = self.audioEncoder(batch)
#         batch = self.audioDecoder(batch)
#         batch = batch.transpose(0, 1).transpose(1, 2)
#         batch = self.pool(batch)
#         batch = self.outputConv(batch)
#         batch = batch.transpose(1, 2).transpose(0, 1)
#         outputBatch = F.log_softmax(batch, dim=2)
#         return outputBatch
