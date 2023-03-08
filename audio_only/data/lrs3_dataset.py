from torch.utils.data import Dataset
from scipy.io import wavfile
import numpy as np

from .utils import prepare_pretrain_input
from .utils import prepare_main_input


class LRS3Main(Dataset):

    """
    A custom dataset class for the LRS3 main (includes train, val, test) dataset
    """

    def __init__(self, dataset, datadir, reqInpLen, charToIx, stepSize, audioParams, noiseParams):
        super(LRS3Main, self).__init__()
        with open(datadir + "/" + dataset + ".txt", "r") as f:
            lines = f.readlines()
        self.datalist = [line.strip().split(" ")[0] for line in lines]
        self.reqInpLen = reqInpLen
        self.charToIx = charToIx
        self.dataset = dataset
        self.stepSize = stepSize
        self.audioParams = audioParams
        _, self.noise = wavfile.read(noiseParams["noiseFile"])
        self.noiseSNR = noiseParams["noiseSNR"]
        self.noiseProb = noiseParams["noiseProb"]
        return


    def __getitem__(self, index):
        #using the same procedure as in pretrain dataset class only for the train dataset
        # if self.dataset == "train":
        base = self.stepSize * np.arange(int(len(self.datalist)/self.stepSize)+1)
        ixs = base + index
        ixs = ixs[ixs < len(self.datalist)]
        index = np.random.choice(ixs)

        #passing the audio file and the target file paths to the prepare function to obtain the input tensors
        audioFile = self.datalist[index] + ".wav"
        targetFile = self.datalist[index] + ".txt"
        if np.random.choice([True, False], p=[self.noiseProb, 1-self.noiseProb]):
            noise = self.noise
        else:
            noise = None
        inp, trgt, inpLen, trgtLen = prepare_main_input(audioFile, targetFile, noise, self.reqInpLen, self.charToIx, self.noiseSNR,
                                                        self.audioParams)
        return  inp, trgt, inpLen, trgtLen,index


    def __len__(self):
        #using step size only for train dataset and not for val and test datasets because
        #the size of val and test datasets is smaller than step size and we generally want to validate and test
        #on the complete dataset
        return len(self.datalist)
#         if self.dataset == "train":
#             return self.stepSize
#         else:
#             return len(self.datalist)
