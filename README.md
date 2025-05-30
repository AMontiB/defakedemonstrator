# Model

This repository contains a Cookiecutter template for creating machine learning and deep learning models that can be seamlessly integrated into the designed platform. It provides a structured starting point to accelerate development and ensure compatibility with the platform's requirements.

## Getting started

First, install [Cookiecutter](https://cookiecutter.readthedocs.io/en/stable/installation.html)

Then the template can be launched using the latest stable version with the command:

```shell
cookiecutter https://github.com/AMontiB/template-model
```

It is also possible to switch to another branch or tag by adding the `--checkout` argument followed by the branch or tag in which you want to check out.


### What are the options for configuring it

After the launch of the command the following configuration options should be configured:

* `project_name`: The name of the project.
* `project_slug`: The slug of the project that is used as folder name of the project.


### What to do after the initial creation on the new project

After the initial creation of the new project, the following steps should be performed:

1. Looks for `TODO` comment in the code, some parts of the repository should be personalized. The project is meant to be used as a template, so the code should be personalized to fit the needs of the project.
2. Follow the insrction in the generated README.md file to install the dependencies and run the project.

## BEST USE

To implement your own, I suggest the following steps:

1.  Start with `model_builder.py`. This file should contain your model and all functions necessary to support inference. Your model is required to perform only **evaluation**, not training.

2.  Move to `model_wrapper.py`. Uncomment and, if necessary, adapt `load_context()`. In my case, I had to change a couple of things as DeFake used not one but two models during inference. Then, edit `load_model()`; this function is essential for loading your network, allocating GPU, performing evaluation, and correctly pre-processing the data. Finally, edit `prediction()`.

3. Edit `uploader.py` to run inference and to upload everything on MLflow.

4. Edit `runner_helper.py` to run inference without load artifacts and results on mlflow

5. Finally edit `runner.py`

6. Please place your weights inside 'src/$projectslug/runs/checkpoints'

To run/debug see [https://mlflow.org/docs/latest/tracking/tutorials/local-database](https://mlflow.org/docs/latest/tracking/tutorials/local-database) , then

```
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000/
export MLFLOW_EXPERIMENT_NAME= $YourModelName
mlflow run --env-manager local .

cd src/
python -m $yourcookiecuttername.runner 
```



