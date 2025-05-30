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
        "--index-url https://gitlab.com/api/v4/projects/70323435/packages/pypi/simple",
        "model_helpers==0.1.0",
        "azure-core==1.34.0",
        "azure-storage-blob==12.25.1",
        "blinker==1.9.0",
        "cachetools==5.5.2",
        "certifi==2025.4.26",
        "click==8.2.1",
        "clip @ git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1",
        "databricks-sdk==0.53.0",
        "huggingface-hub==0.31.4",
        "joblib==1.5.0",
        "matplotlib==3.10.3",
        "mlflow==2.22.0",
        "mlflow-skinny==2.22.0",
        "numpy==2.2.6",
        "pandas==2.2.3",
        "pillow==11.2.1",
        "PyYAML==6.0.2",
        "regex==2024.11.6",
        "requests==2.32.3",
        "safetensors==0.5.3",
        "scikit-learn==1.6.1",
        "scipy==1.15.3",
        "setuptools==80.8.0",
        "six==1.17.0",
        "sympy==1.14.0",
        "timm==1.0.15",
        "tokenizers==0.21.1",
        "torch==2.7.0",
        "torchaudio==2.7.0",
        "torchvision==0.22.0",
        "tqdm==4.67.1",
        "transformers==4.52.2",
        "fairscale==0.4.13"
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




