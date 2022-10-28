import os
import json
import random
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import imgaug
import torch
import torch.multiprocessing as mp
import numpy as np
import models
seed = 1234
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main(config_path):
    configs = json.load(open(config_path))
    configs["cwd"] = os.getcwd()
    model = get_model(configs)
    train_set, val_set, test_set = get_dataset(configs)
    from trainers.trainer import FER2013Trainer
    trainer = FER2013Trainer(model, train_set, val_set, test_set, configs)
    trainer.train()

def get_model(configs):
    try:
        return models.__dict__[configs["arch"]]
    except KeyError:
        return segmentation.__dict__[configs["arch"]]

def get_dataset(configs):
    from utils.datasets.fer2013dataset import fer2013
    train_set = fer2013("train", configs)
    val_set = fer2013("val", configs)
    test_set = fer2013("test", configs, tta=True, tta_size=10)
    return train_set, val_set, test_set

if __name__ == "__main__":
    main("/content/drive/MyDrive/Segmentation_VGG/Segmentation_VGG/configs/config.json")