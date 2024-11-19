# -*-coding:utf-8 -*-

import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from .utils import preprocess_input
import cv2
import numpy as np
from spfca import SPFCA

class UNetDataset(Dataset):
    def __init__(self, train, dataset_path, auto_label=True, auto_label_params={"n_segments": 1000, "compactness": 0.5}):
        super(UNetDataset, self).__init__()
        self.data = []
        self.label = []
        self.train = train
        self.dataset_path = dataset_path
        if self.train:
            image_directory = os.path.join(self.dataset_path, "train_images")
            label_directory = os.path.join(self.dataset_path, "train_labels")
            if os.path.exists(image_directory):
                for image_file in os.listdir(image_directory):
                    if image_file.endswith('.jpg') or image_file.endswith('.png'):
                        image_path = os.path.join(image_directory, image_file)
                        image = cv2.imread(image_path)
                        if image is None:
                            raise ValueError(f"{image_path} file is None")
                        input_image = np.transpose(preprocess_input(np.array(image, np.float32)), (2, 0, 1))
                        self.data.append(input_image)
                        if auto_label:
                            label,_ = SPFCA(image, auto_label_params["n_segments"], auto_label_params["compactness"])
                            label = label[np.newaxis,...]
                            self.label.append(label)
                        else:
                            label_path = os.path.join(label_directory, image_file.split('.')[0] + '.npy')
                            if os.path.exists(label_path):
                                label = np.array(np.load(label_path),np.float32)
                                label = label[np.newaxis,...]
                                self.label.append(label)
                            else:
                                raise ValueError(f"{label_path} is not exist")
            else:
                raise ValueError(f"{image_directory} is not exist")
        
        else:
            image_directory = os.path.join(self.dataset_path, "valid_images")
            label_directory = os.path.join(self.dataset_path, "valid_labels")
            if os.path.exists(image_directory) and os.path.exists(label_directory):
                for image_file in os.listdir(image_directory):
                    if image_file.endswith('.jpg') or image_file.endswith('.png'):
                        image_path = os.path.join(image_directory, image_file)
                        label_path = os.path.join(label_directory, image_file.split('.')[0] + '.npy')
                        image = cv2.imread(image_path)
                        label = np.array(np.load(label_path),np.float32)
                        if image is None:
                            raise ValueError(f"{image_path} file is None")
                        input_image = np.transpose(preprocess_input(np.array(image, np.float32)), (2, 0, 1))
                        self.data.append(input_image)
                        label = label[np.newaxis,...]
                        self.label.append(label)
            else:
                raise ValueError(f"{image_directory} or {label_directory} is not exist")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index): 
        image = self.data[index]
        label = self.label[index]
        return image, label


