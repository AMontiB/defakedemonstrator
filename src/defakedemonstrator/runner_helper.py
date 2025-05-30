import multiprocessing
import os
from abc import abstractmethod

import clip
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import ImageFile
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration


from tqdm import tqdm

from defakedemonstrator.model_builder import get_network, build_transform


ImageFile.LOAD_TRUNCATED_IMAGES = True
import mlflow


class DeFakeRunner:

    def __init__(self, model_flag,device, load_id, num_threads, task_type, run_directory: str = "runs/"):
        super().__init__()

        self.linear = get_network(
            model_name=model_flag,
            task=task_type,
        )
        #_, preprocess = clip.load("ViT-B/32", device=device)
        self.transform = self.preprocess_image(device)
        #specific of defake
        self.transform_blip = build_transform()
        # 
        self.run_directory = run_directory
        self.device = device
        self.load_id = load_id
        self.num_threads = num_threads
        
    def preprocess_image(self, device):#, image_size=224):
        _, preprocess = clip.load("ViT-B/32", device=device)
        return preprocess
    
    @abstractmethod
    def predict(self, image):
        
        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        #call DEFAKE
        self.linear.load_state_dict(torch.load(os.path.join(self.run_directory, 'checkpoints/train_linear_model.pth'), map_location=device))
        self.linear.to(device)
        self.linear.eval()
        #call CLIP
        model, _ = clip.load("ViT-B/32", device=device)
        model.load_state_dict(torch.load(os.path.join(self.run_directory, 'checkpoints/train_clip_model.pth'), map_location=device))
        model.eval()
        #call BLIP
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        blip.eval()
        blip = blip.to(device)
        
        image_blip = self.transform_blip(image)
        image_blip = image_blip.unsqueeze(0).to(device)
        inputs = blip_processor(images=image_blip, return_tensors="pt").to(device)
        
        
        #Actual Test Image
        image = self.transform(image)
        test_loader = DataLoader(dataset=image[None,], batch_size=1, shuffle=False, num_workers=0)
        #
       
        caption = blip.generate(**inputs)
        prompt = blip_processor.decode(caption[0], skip_special_tokens=True)
        text = clip.tokenize([prompt]).to(device)
        
        
        with torch.no_grad():
             with tqdm(test_loader, unit='batch', mininterval=0.5) as tbatch:
                 tbatch.set_description(f'Test')
                 for data in tbatch:
                    data = data.to(self.device)
                    # extract features
                    image_features = model.encode_image(data)
                    text_features = model.encode_text(text)
                    emb = torch.cat((image_features, text_features),1)
                    #run your model
                    scores = self.linear(emb.float())
                    #print('SCORES: ', torch.softmax(scores, -1))
                    print('FINAL DECISION DETECTOR N: ', torch.softmax(scores, -1).argmax(1).cpu().numpy()[0], 'with probability: ', torch.softmax(scores, -1)[0,torch.softmax(scores, -1).argmax(1).cpu().numpy()[0]].cpu().numpy())
        
        return scores.argmax(1).cpu().numpy()


