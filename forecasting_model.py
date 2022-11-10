#!/usr/bin/env python
# coding: utf-8

# This is (to almost its entirety) copied from this excellent article 
# [here](https://coderzcolumn.com/tutorials/machine-learning/simple-guide-to-optuna-for-hyperparameters-optimization-tuning)
# 
# Let's import some basic libraries
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
# from pytorch_lightning.profiler import Profiler, AdvancedProfiler
from pytorch_lightning.callbacks import EarlyStopping

import logging
import click
import mlflow
import tempfile

# Custom made classes and functions used for reading/processing data at input/output of model
from model_utils import ClickParams, Regression
from model_utils import read_csvs, train_test_valid_split, \
                        feauture_target_split, cross_plot_pred_data, calculate_metrics

def train_test_best_model():
    """
    This function is used to train and test our pytorch lightning model
    based on parameters give to it
    Parameters: None (basically global click_params)
    Returns: 
        actuals: 1-D list containing target labels
        predictions:  1-D list containing predictions for said labels
    """
    params = vars(click_params)

    params['output_dims'] = [value for key,value in params.items() 
                                  if key.startswith('n_units_l')]

    trainer = Trainer(max_epochs=params['max_epochs'], auto_scale_batch_size=None, 
                    #   profiler="simple", #add simple profiler
                      callbacks=[EarlyStopping(monitor="val_loss", mode="min", verbose=True)],
                      gpus=0 if torch.cuda.is_available() else None,
                      deterministic=True) 

    pl.seed_everything(click_params.seed, workers=True)    
    model = Regression(params) # double asterisk (dictionary unpacking)

    trainer.fit(model)

    mlflow.pytorch.log_model(model, "model")

    # Either best or path to the checkpoint you wish to test. 
    # If None and the model instance was passed, use the current weights. 
    # Otherwise, the best model from the previous trainer.fit call will be loaded.
    trainer.test()

    trainer.validate()

    preds = trainer.predict(ckpt_path='best')
    actuals = []
    for x,y in model.test_dataloader(): 
        actuals.append(y)

    # torch.vstack: convert list of tensors to (rank 2) tensor
    # .tolist(): convert (rank 2) tensor to list of lists
    # final outcome: list of floats
    preds = [item for sublist in torch.vstack(preds).tolist() for item in sublist]
    actuals = [item for sublist in torch.vstack(actuals).tolist() for item in sublist]

    return preds, actuals

from sklearn.neural_network import MLPRegressor

def sklearn_regress():
    """
    Performs training base on scikit learn MLPRegressor class
    Used for direct comparisson of performance with my pytorch lightning model
    Parametrers: None
    Returns: 
        actuals: list containing target labels
        predictions: list containing predictions for said labels
    """
    train_X, train_Y = feauture_target_split(train_data,click_params.l_window,click_params.f_horizon)  
    test_X, test_Y = feauture_target_split(test_data,click_params.l_window,click_params.f_horizon)

    regr = MLPRegressor(random_state=42,
                        early_stopping=True,
                        verbose=True,
                        shuffle=True, 
                        max_iter=500).fit(train_X.to_numpy(), train_Y.to_numpy())
    preds = regr.predict(test_X.to_numpy())
    
    score = regr.score(test_X.to_numpy(), test_Y.to_numpy())
    print(f'R-squared score: {score}')

    actuals = test_Y.to_numpy()
    preds = [item for sublist in np.vstack(preds).tolist() for item in sublist]
    actuals = [item for sublist in np.vstack(actuals).tolist() for item in sublist]

    return preds, actuals

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Click ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Remove whitespace from your arguments
@click.command(
    help= "Given a folder path for CSV files (see load_raw_data), use it to create a model, find\
            find ideal hyperparameters and train said model to reduce its loss function"
)

@click.option("--dir_in", type=str, default='../preprocessed_data/', help="File containing csv files used by the model")
@click.option("--countries", type=str, default="Portugal", help='csv names from dir_in used by the model')
@click.option("--seed", type=str, default="42", help='seed used to set random state to the model')
# @click.option("--opt_model_path", type=str, default="./optuna_model.pt")

@click.option("--n_trials", type=str, default="20", help='number of trials - different tuning oh hyperparams')
@click.option("--max_epochs", type=str, default="200", help='range of number of epochs used by the model')
@click.option("--n_layers", type=str, default="1", help='range of number of layers used by the model')
@click.option("--layer_size", type=str, default="100", help='range of size of each layer used by the model')
@click.option("--l_window", type=str, default="240", help='range of lookback window (input layer size) used by the model')
@click.option("--f_horizon", type=str, default="24", help='range of forecast horizon (output layer size) used by the model')
@click.option("--l_rate", type=str, default="1e-4", help='range of learning rate used by the model')
@click.option("--activation", type=str, default="ReLU", help='activations function experimented by the model')
@click.option("--optimizer_name", type=str, default="Adam", help='optimizers experimented by the model') # SGD
@click.option("--batch_size", type=str, default="200", help='possible batch sizes used by the model') #16,32,


def forecasting_model(**kwargs):
    """
    This is the main function of the script. 
    1. It loads the data from csvs
    2. splits to train/test/valid
    3. trains model based on hyper params set
    4. computes MAPE and plots graph of best model
    Parameters
        kwargs: dictionary containing click paramters used by the script
    Returns: None 
    """
    # Auto log all MLflow entities
    mlflow.pytorch.autolog()

    # store mlflow metrics/artifacts on temp file
    train_tmpdir = tempfile.mkdtemp()

    global click_params
    click_params = ClickParams(kwargs)

    print("=================== Click Params ===================")
    print(vars(click_params))
    
    #  read csv files
    df = read_csvs(click_params)
    df_backup = df.copy()

    # split data in train/test/validation
    global train_data, test_data, val_data
    train_data, test_data, val_data = train_test_valid_split(df)

    train_data.to_csv(f"{train_tmpdir}/train_data.csv")
    test_data.to_csv(f"{train_tmpdir}/test_data.csv")
    val_data.to_csv(f"{train_tmpdir}/val_data.csv")

    # train model with hparams set to best_params of optuna 
    plot_pred, plot_actual = train_test_best_model()
    # plot_pred, plot_actual = sklearn_regress()

    # calculate metrics
    metrics = calculate_metrics(plot_actual,plot_pred)

    # plot prediction/actual data on common axis system 
    cross_plot_pred_data(df_backup, plot_pred, plot_actual, train_tmpdir)

    print("\nUploading training csvs and metrics to MLflow server...")
    logging.info("\nUploading training csvs and metrics to MLflow server...")
    mlflow.set_tag("stage", "model")
    mlflow.log_artifacts(train_tmpdir, "train_results")
    mlflow.log_metrics(metrics)

if __name__ == '__main__':
    print("\n=========== Forecasing Model =============")
    logging.info("\n=========== Forecasing Model =============")
    forecasting_model()