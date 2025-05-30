from time import process_time_ns
import torch
import clip
from PIL import Image
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
 
from sklearn.metrics import confusion_matrix
import itertools
import torch.nn.functional as F

import torch.nn as nn
from torch.utils.data import random_split
from torch import nn
from torchvision import transforms
import sys
import argparse
import time
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve
from transformers import BlipProcessor, BlipForConditionalGeneration

from defakedemonstrator.blipmodels import blip_decoder

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_list, num_classes):
        super(NeuralNet, self).__init__()
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, hidden_size_list[0])
        self.fc2 = nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.fc3 = nn.Linear(hidden_size_list[1], num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out

def preprocess_image2():#, image_size=224):
    #()
    _, preprocess = clip.load("ViT-B/32", device='cuda:0')
    img = img_path #Image.open(img_path)
    #img = img.resize((image_size, image_size))
    return preprocess(img)

def get_requirements():
    return [
            "--index-url https://gitlab.com/api/v4/projects/67921119/packages/pypi/simple",
            "truebees_model_helpers==0.2.0"
        ]

def build_transform():
    transformation = [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
    return transforms.Compose(transformation)






def get_network(model_name: str, task: str) -> nn.Module:
    
    if model_name == "defake":
        
        model = NeuralNet(1024,[512,256],10)
        # Freezing layers (optional, more relevant for pre-trained models)
        if task=='test':
            for param in model.parameters():
                param.requires_grad = False

        return model
    
    else:
        raise ValueError(f"Unsupported model_name: {model_name}. Currently only 'simple_mlp' is supported.")







##################




