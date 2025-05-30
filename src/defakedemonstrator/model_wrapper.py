import logging
from typing import Type

from PIL.Image import Image
from pandas import DataFrame
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import os 


from model_helpers.data_loader.common import AbstractDataLoader
from model_helpers.post_process.common import AbstractPostProcessor
from model_helpers.utils.codecs import dataframe_to_pil
from model_helpers.utils.device import get_device
from model_helpers.wrapper import ModelWrapper as BaseWrapper

from defakedemonstrator.model_builder import get_network, build_transform
from transformers import BlipProcessor, BlipForConditionalGeneration
import clip
import torch



logger = logging.getLogger(__name__)


class ModelWrapper(BaseWrapper):

    def __init__(self,
                 model_class: str,
                 task: str, 
                 loader_class: Type[AbstractDataLoader],
                 post_processors_class: Type[AbstractPostProcessor],
                 **kwargs
                 ):
        super().__init__(
            loader_class=loader_class,
            post_processors_class=post_processors_class,
            **kwargs
        )
        self._model_class = model_class
        self.task = task
        
        self.model = None
        self.device = None
        self.transform = None
        self.transform_blip = None

        # DO NOT initialize the model in __init__ method, the model should be initialized in load_context method (https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PythonModel.load_context)



    def load_context(self, context):
        super().load_context(context)
        logger.info(f"Calling load_context with context: {context}")
        logger.info(f"artifacts: {context.artifacts}")

        artifact_path_linear = context.artifacts["linear"]
        artifact_path_clip = context.artifacts["model"]

        self.load_model(artifact_path_linear, artifact_path_clip)

    def load_model(self, artifact_path_linear, artifact_path_clip):
        self.device = get_device(None)
        logger.info(f"device {self.device}")

        self.linear = get_network(
            model_name=self._model_class,
            task=self.task
        )
        self.transform = self.preprocess_image(self.device)
        self.transform_blip = build_transform()


        logger.info(f"Loading model from: {artifact_path_linear}")
        checkpoint = torch.load(artifact_path_linear, map_location=self.device)

        self.linear.load_state_dict(checkpoint)
        self.linear.to(self.device)
        self.linear.eval()
        
        model, _ = clip.load("ViT-B/32", device=self.device)
        logger.info(f"Loading model from: {artifact_path_clip}")
        self.model = model
        self.model.load_state_dict(torch.load(artifact_path_clip, map_location=self.device))
        self.model.eval()
         

        
    def preprocess_image(self, device):#, image_size=224):
        _, preprocess = clip.load("ViT-B/32", device=device)
        return preprocess
    

    def run_prediction(self, image: Image | DataFrame):

        if not isinstance(image, Image):
             image = dataframe_to_pil(image, "RGB")
             
        device = "cuda" if torch.cuda.is_available() else "cpu"
        

        #call CLIP
        #model, _ = clip.load("ViT-B/32", device=device)
        #model.load_state_dict(torch.load(os.path.join('runs/checkpoints/train_clip_model.pth'), map_location=device))
        #model.eval()  
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
        
        #features encoder
        with torch.no_grad():
             with tqdm(test_loader, unit='batch', mininterval=0.5) as tbatch:
                 tbatch.set_description(f'Test')
                 for data in tbatch:
                    data = data.to(self.device)
                    # extract features
                    image_features = self.model.encode_image(data)
                    text_features = self.model.encode_text(text)
                    emb = torch.cat((image_features, text_features),1)
                    #run your model
                    scores = self.linear(emb.float())

                    print('FINAL DECISION DETECTOR N: ', torch.softmax(scores, -1).argmax(1).cpu().numpy()[0], 'with probability: ', torch.softmax(scores, -1)[0,torch.softmax(scores, -1).argmax(1).cpu().numpy()[0]].cpu().numpy())

        return torch.softmax(scores, -1)

        #return torch.softmax(scores, -1).argmax(1).cpu().numpy()[0], torch.softmax(scores, -1)[0,torch.softmax(scores, -1).argmax(1).cpu().numpy()[0]].cpu().numpy()
