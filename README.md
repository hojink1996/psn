# PSN

The code that is provided in this repository has the objective of showing how PSN works in a specific dataset. The dataset that is being shown is the ZINC dataset.


### Requirements

The repository has the following dependencies (with Python >= 3.7):

```
pytorch-lightning==1.1.5
numpy==1.20.3
torch==1.6.0
torch-geometric==1.6.0
torch-cluster==1.5.9
torch-scatter==2.0.6
torch-spline-conv==1.2.1
```

### Model Training

To run the experiments you can use the `training.py` script. This script already contains the hyperparameters used in the paper as the default values. Because of this you can simply use the following command to run the experiments:

```
python training.py
```

To run experiments for a different dataset you have to modify the `training.py` script to specify the dataset you want to use, and change the `PSNLightning` Class to specify the loss and evaluator you want to use. The hyperparameters used for each of the datasets can be found in the paper.

