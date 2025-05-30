import logging
import os
from copy import deepcopy
from shutil import rmtree

import pandas as pd

import click
import mlflow
from PIL import Image
from mlflow.models import infer_signature
from model_helpers.data_loader.image.data_frame import DataFrameLoader
from model_helpers.post_process.common import BasePostProcessor
from model_helpers.utils.codecs import pil_to_dataframe

from defakedemonstrator.model_builder import get_requirements
from defakedemonstrator.model_wrapper import ModelWrapper
from defakedemonstrator.utils.uploader import upload_artifact

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

def download_artifact(run_id, artifact_path, local_path):
    client = mlflow.tracking.MlflowClient()
    client.download_artifacts(run_id, artifact_path, local_path)



@click.command()
@click.option("--run_id", help="Run ID")
@click.option("--artifact_path", help="Artifact path", default="wrapped_model")
@click.option("--model_class", help="Model class", default="nodown")
@click.option("--model_freeze", help="Model freeze", default=True)
def re_upload_artifact(run_id: str, artifact_path: str, model_class: str, model_freeze: bool):
    client = mlflow.tracking.MlflowClient()

    if os.path.exists("./tmp_artifact"):
        logger.info("tmp_artifact directory already exists, removing it")
        rmtree("./tmp_artifact")

    logger.info("Creating tmp_artifact directory")
    os.makedirs("./tmp_artifact")

    client.download_artifacts(run_id, artifact_path, "./tmp_artifact")

    with (mlflow.start_run()):

        artifact_path = "./tmp_artifact/wrapped_model/artifacts/best.pt" # TODO update the artifact path

        upload_artifact(
            artifact_path=artifact_path,
            model_class=model_class,
            model_freeze=model_freeze
        )


if __name__ == "__main__":
    re_upload_artifact()