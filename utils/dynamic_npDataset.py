import logging
import torch
from torch.utils.data import Dataset

from utils.bold import process_dynamic_fc

import os
import glob
import numpy as np

from natsort import natsorted, natsort_keygen
import re

from scipy import io


class npDataset(Dataset):

    def __init__(self,
                 data_file,
                 label_file,
                 task = None):

        # get rois
        self.data = np.load(data_file)


        # get label
        self.label = np.load(label_file)

        self.task = task


    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        # get ROI
        roi = self.data[idx]


        # get label
        label = self.label[idx]
        if self.task == 'dig' and label > 1:
            label = 1

        sample = {'ROI':torch.as_tensor(roi[0, :, :, 0]).float(),
                  'label':torch.as_tensor(label).long()}

        return sample
