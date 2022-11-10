#!/usr/bin/env python
# coding: utf-8

# This is (to almost its entirety) copied from this excellent article 
# [here](https://coderzcolumn.com/tutorials/machine-learning/simple-guide-to-optuna-for-hyperparameters-optimization-tuning)

# Let's import some basic libraries
import numpy as np
import matplotlib as mpl
mpl.use("webagg")

import torch
import torch.nn as nn
import pytorch_lightning as pl
# from pytorch_lightning.profiler import Profiler, AdvancedProfiler
from torchmetrics import MeanAbsolutePercentageError

import logging
import pickle
import click
import mlflow
import os

import torchensemble 
from torchensemble.utils import io
import re
import tempfile

# Custom made classes and functions used for reading/processing data at input/output of model
from model_utils import ClickParams, Regression
from model_utils import read_csvs, train_test_valid_split, \
                        cross_plot_pred_data, calculate_metrics

def findNewRows(N,K):
    """
    Function to find the largest number
    smaller than or equal to N
    that is divisible by k
    """
    rem = N % K
    if (rem == 0):
        return N
    else:
        return N - rem

def ensemble(load=True, save_dir='./ensemble_models/', ensemble_filename='BaggingRegressor_Regression_10_ckpt.pth'):
    """
    This function performs the ensembling method
    1. Reads models params from pickle file
    2. Constructs the custom pl model
    3. create ensemble model using torchensemble library
    4. evaluates model using MSE loss
    5. creates 1-D list for data targets and its relative predictions

    Parameters:
        load:   boolean for checking if ensemble model params should be taken from
                ensemble_filename or not
        save_dir:   directory path (relative to current path) for storing fully trained
                    ensemble model
        ensemble_filename: file path (relative to current path) for loading fully trained
                    ensemble model
    Returns:
        actuals: 1-D list containing target labels
        predictions:  1-D list containing predictions for said labels
    """
    a_file = open("optuna_best_trial.pkl", "rb")
    best_params = pickle.load(a_file)

    best_params['output_dims'] = [value for key,value in best_params.items() 
                                if key.startswith('n_units_l')]
    print(best_params)

    pl.seed_everything(click_params.seed, workers=True)    
    regression_model = Regression(best_params) # double asterisk (dictionary unpacking)

    train_loader = regression_model.train_dataloader()
    test_loader = regression_model.test_dataloader()
    val_loader = regression_model.val_dataloader()

    ensemble_model = None
    if(load==True and os.path.exists(save_dir)):
        # parse filename (e.g 'BaggingRegressor_Regression_10_ckpt.pth') for:
        #   numeric value of n_estimators (10)
        #   regressor used (BaggingRegressor)
        n_estimators = re.findall("\d+", ensemble_filename)[0]
        regressor = ensemble_filename.split('_')[0]

        ensemble_model = getattr(torchensemble,regressor)(
            estimator=regression_model,
            n_estimators=n_estimators, 
            cuda=False
        )
        io.load(ensemble_model, save_dir)  # reload
    else:
        ensemble_model = torchensemble.BaggingRegressor(
            estimator=regression_model,
            n_estimators=3, #click_params.max_estimators,
            cuda=False
        )

        ensemble_model.set_optimizer(optimizer_name=best_params['optimizer_name'],
                                    lr=best_params['l_rate'])
        ensemble_model.set_criterion(MeanAbsolutePercentageError())

        ensemble_model.fit(epochs=best_params['max_epochs'],
                        train_loader=train_loader,
                        test_loader=val_loader,
                        save_model=True,
                        save_dir=save_dir)

    mse_loss = ensemble_model.evaluate(test_loader)
    print(f'Testing mean squared error of the fitted ensemble: {mse_loss}')

    actuals_X = []
    actuals_Y = []
    for x,y in test_loader: 
        actuals_X.append(x)
        actuals_Y.append(y)

    # actuals: tensor of tensor-batches
    # actuals[0]: first tensor-batch
    # actuals[0][0]: first feauture/target of first batch

    # torch.vstack: convert list of tensors to (rank 2) tensor
    # .tolist(): convert (rank 2) tensor to list of lists
    # final outcome: list of floats
    actuals_X = [item for sublist in torch.vstack(actuals_X).tolist() for item in sublist]
    actuals_Y = [item for sublist in torch.vstack(actuals_Y).tolist() for item in sublist]

    # find max len of list (rows of later np.array) that is 
    # divisible by lookback_window to be fed as batches to predict
    new_len = findNewRows(len(actuals_X),best_params['l_window'])
    actuals_X = actuals_X[:new_len]
    print(f"new_len: {new_len}")

    # reshape 1-D matrix (created by list train_X) for it to be multipled by input layer matrix
    # matrix shape: [any, lookback_window]
    preds = ensemble_model.predict(np.array(actuals_X).reshape(-1, best_params['l_window']))

    # convert preds to 1-D list
    # keep in actuals_Y, the targets of feautures used by predict()
    preds = [item for sublist in preds.tolist() for item in sublist]
    actuals_Y = actuals_Y[:len(preds)]

    return preds, actuals_Y

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Click ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Remove whitespace from your arguments
@click.command(
    help= "Given a folder path for CSV files (see load_raw_data), use it to create a model, find\
            find ideal hyperparameters and train said model to reduce its loss function"
)

@click.option("--dir_in", type=str, default='../preprocessed_data/', help="File containing csv files used by the model")
@click.option("--countries", type=str, default="Portugal", help='csv names from dir_in used by the model')
@click.option("--seed", type=str, default="42", help='seed used to set random state to the model')
@click.option("--max_estimators", type=str, default="10", help='number of estimators (models) used in ensembling')

def forecasting_model(**kwargs):
    """
    This is the main function of the script. 
    1. It loads the data from csvs
    2. splits to train/test/valid
    3. uses 'torchensemble' library to ensemble model based on those data
    4. computes MAPE and plots graph
    Parameters
        kwargs: dictionary containing click paramters used by the script
    Returns: None 
    """
    # Auto log all MLflow entities
    mlflow.pytorch.autolog()

    # store mlflow metrics/artifacts on temp file
    ensemble_tmpdir = tempfile.mkdtemp()

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

    train_data.to_csv(f"{ensemble_tmpdir}/train_data.csv")
    test_data.to_csv(f"{ensemble_tmpdir}/test_data.csv")
    val_data.to_csv(f"{ensemble_tmpdir}/val_data.csv")

    # train model with hparams set to best_params of optuna 
    plot_pred, plot_actual = ensemble(load=False)
    # plot_pred, plot_actual = sklearn_regress()

    # calculate metrics
    metrics = calculate_metrics(plot_actual,plot_pred)

    # plot prediction/actual data on common axis system 
    cross_plot_pred_data(df_backup, plot_pred, plot_actual, ensemble_tmpdir)

    print("\nUploading training csvs and metrics to MLflow server...")
    logging.info("\nUploading training csvs and metrics to MLflow server...")
    mlflow.set_tag("stage", "ensemble")
    mlflow.log_artifacts(ensemble_tmpdir, "ensemble_results")
    mlflow.log_metrics(metrics)

if __name__ == '__main__':
    print("\n=========== Ensemble Model =============")
    logging.info("\n=========== Ensemble Model =============")
    forecasting_model()