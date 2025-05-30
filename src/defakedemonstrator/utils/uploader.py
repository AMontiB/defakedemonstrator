import logging
from copy import deepcopy

import mlflow

from PIL import Image
import pandas as pd
from mlflow.models import infer_signature
from model_helpers.data_loader.azure.azure_image_loader import ImageAzureLoader
from model_helpers.data_loader.image.data_frame import DataFrameLoader
from model_helpers.post_process.common import BasePostProcessor
from model_helpers.utils.codecs import pil_to_dataframe

from defakedemonstrator.model_builder import get_requirements
from defakedemonstrator.model_wrapper import ModelWrapper

logger = logging.getLogger(__name__)


def upload_artifact(artifact_path1, artifact_path2, model_class: str, model_freeze: bool):

    for loader_class, artifact_name in [(DataFrameLoader, "wrapped_model"), (ImageAzureLoader, "wrapped_model_azure")]:

        logger.info(f"Loading model with loader class: {loader_class}")

        wrapper = ModelWrapper(
            model_class=model_class,
            task='test',
            loader_class=loader_class,
            post_processors_class=BasePostProcessor,
        )

        model = deepcopy(wrapper)
        model.load_model(artifact_path1, artifact_path2)

        # Use the loader's expected input format for signature and input_example
        model_input_signature = loader_class.get_data_example()
        input_example = model_input_signature.head()

        # Use real image only for test prediction, not for logging
        image_path = './img_1.png'
        image = Image.open(image_path).convert("RGB")
        test_input_df = pil_to_dataframe(image)

        prediction = model.run_prediction(test_input_df)

        class_labels = [
            'AdobeFirefly', 'Dall-E3', 'Flux.1', 'Flux.1.1Pro', 'Freepik',
            'LeonardoAI', 'Midjourney', 'StableDiffusion3.5', 'StableDiffusionXL', 'StarryAI'
        ]
        prediction_df = pd.DataFrame(prediction.detach().cpu().numpy(), columns=class_labels)

        model_output_df = model.post_process_elem(test_input_df, prediction_df)
        model_output_df = pd.DataFrame([model_output_df])

        # Signature matches expected input structure, not the image data
        signature = infer_signature(input_example, model_output_df.values[0])

        mlflow.pyfunc.log_model(
            python_model=wrapper,
            input_example=input_example,
            signature=signature,
            artifact_path=artifact_name,
            artifacts={
                "linear": artifact_path1,
                "model": artifact_path2,
            },
            code_path=["defakedemonstrator/"],
            pip_requirements=get_requirements()
        )
