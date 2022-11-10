#!/usr/bin/env python
# coding: utf-8

# This is (to almost its entirety) copied from this excellent article 
# [here](https://coderzcolumn.com/tutorials/machine-learning/simple-guide-to-optuna-for-hyperparameters-optimization-tuning)

# Let's import some basic libraries
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use("webagg")
import warnings
import copy
import tempfile

import optuna
from optuna.integration import PyTorchLightningPruningCallback

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
# from pytorch_lightning.profiler import Profiler, AdvancedProfiler
from pytorch_lightning.callbacks import EarlyStopping

import logging
import pickle
import click
import mlflow

from model_utils import ClickParams, Regression
from model_utils import read_csvs, train_test_valid_split, \
                        cross_plot_pred_data, calculate_metrics

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Globals ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# dataScaler = None 
click_params = None
opt_tmpdir = None

"""
This function is use by the Optuna library: An open source hyperparameter optimization framework
to automate hyperparameter search
In Optuna, there are two major terminologies, namely: 
1) Study: The whole optimization process is based on an objective function 
   i.e the study needs a function which it can optimize. 
2) Trial: A single execution of the optimization function is called a trial. 
Thus the study is a collection of trials

Optuna is a black-box optimizer, which means it needs an objective function, 
which returns a numerical value to evaluate the performance of the hyperparameters, 
and decide where to sample in upcoming trials

In Optuna:
    We create a "study", which corresponds to an optimization task (maximize/minimize the objective function)
    We then try to optimise (maximise/minimize) that objective 
        Since our objective functions returns (and therefore is based on) validation loss, we want
        to minimize that value rather than maximize
    Each "execution" of the objective function is basically a "trial"
        trial (in context): one training of model for specific set of suggestive parameters (check params)
"""

def objective(trial):
    """
    Function used by optuna for hyperparameter tuning
    Each execution of this function is basically a new trial with its params
    changed as suggest by suggest_* commands
    Parameters: 
        trial object (default)
    Returns: 
        validation loss of model used for checking progress of tuning 
    """
    global click_params

    max_epochs = trial.suggest_int('max_epochs', click_params.max_epochs[0], click_params.max_epochs[1])
    n_layers = trial.suggest_int("n_layers", click_params.n_layers[0], click_params.n_layers[1])

    params = {
        'l_window': trial.suggest_int("l_window", click_params.l_window[0], click_params.l_window[1]),
        'f_horizon': trial.suggest_int("f_horizon", click_params.f_horizon[0], click_params.f_horizon[1]),
        'output_dims': [trial.suggest_int("n_units_l{}".format(i), click_params.layer_size[0], click_params.layer_size[1], log=True) for i in range(n_layers)],
        'l_rate':    trial.suggest_float('l_rate', click_params.l_rate[0], click_params.l_rate[1], log=True), # loguniform will become deprecated
        'activation': trial.suggest_categorical("activation", click_params.activation), #SiLU (Swish) performs good
        'optimizer_name': trial.suggest_categorical("optimizer_name", click_params.optimizer_name),
        'batch_size': trial.suggest_categorical('batch_size', click_params.batch_size)
    }
    print(params)
    print(f"max_epochs: {max_epochs}")

    # The default logger in PyTorch Lightning writes to event files to be consumed by
    # TensorBoard. We create a simple logger instead that holds the log in memory so that the
    # final accuracy can be obtained after optimization. When using the default logger, the
    # final accuracy could be stored in an attribute of the `Trainer` instead.
    trainer = Trainer(max_epochs=max_epochs, auto_scale_batch_size=None, logger=True,
                    #   profiler="simple", #add simple profiler
                      gpus=0 if torch.cuda.is_available() else None,
                      callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
                      deterministic=True)

    pl.seed_everything(click_params.seed, workers=True)    
    model = Regression(params) # double asterisk (dictionary unpacking)
    trainer.logger.log_hyperparams(params)
    trainer.fit(model)
    return trainer.callback_metrics["val_loss"].item()

#  Example taken from Optuna github page:
# https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py

# Theory:
# https://coderzcolumn.com/tutorials/machine-learning/simple-guide-to-optuna-for-hyperparameters-optimization-tuning

def print_optuna_report(study):
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

    print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~ Optuna Report ~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # write binary, overwrite if file exists, creates file if not exists
    best_trial_file = open("optuna_best_trial.pkl", "wb") 
    pickle.dump(study.best_params, best_trial_file)
    best_trial_file.close()    

    best_result = copy.deepcopy(study.best_params)
    best_result['value'] = study.best_trial.value

    # appends, pointer at EOF if file exists, creates file if not exists    
    with open('best_trial_diary.txt','a') as trial_diary_file: 
        trial_diary_file.write(str(best_result)+ "\n")
        
# The pruners module defines a BasePruner class characterized by an abstract prune() method, which, 
# for a given trial and its associated study, returns a boolean value 
# representing whether the trial should be pruned (aborted)
# optuna.pruners.MedianPruner() 
# optuna.pruners.NopPruner() (no pruning)
# Hyperband performs best with default sampler for non-deep learning tasks
def optuna_optimize():
    """
    Function used to setup optuna for study
    Parameters: None
    Returns: study object containing info about trials
    """
    # The pruners module defines a BasePruner class characterized by an abstract prune() method, which, 
    # for a given trial and its associated study, returns a boolean value 
    # representing whether the trial should be pruned (aborted)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

    # optuna.pruners.MedianPruner() 
    # optuna.pruners.NopPruner() (no pruning)
    # Hyperband performs best with default sampler for non-deep learning tasks
    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner() # experimentally better performance
        
    # default sampler: TPMESampler    
    study = optuna.create_study(direction="minimize", pruner=pruner)
    """
    timeout (Union[None, float]) – Stop study after the given number of second(s). 
    None represents no limit in terms of elapsed time. 
    The study continues to create trials until: the number of trials reaches n_trials, 
                                                timeout period elapses, stop() is called or, 
                                                a termination signal such as SIGTERM or Ctrl+C is received.
    """

    study.optimize(objective,
                #    timeout=600, # 10 minutes
                   n_trials=click_params.n_trials) # 10 trials
    print_optuna_report(study)
    return study

def optuna_visualize(study):
    if(optuna.visualization.is_available()):
        optuna.visualization.plot_param_importances(study).show()
        optuna.visualization.plot_optimization_history(study).show()
        optuna.visualization.plot_intermediate_values(study).show()
        optuna.visualization.plot_slice(study, study.best_params).show()
        optuna.visualization.plot_contour(study).show()

    plt.close() # close any mpl figures (important, doesn't work otherwise)
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig(f"{opt_tmpdir}/plot_param_importances.png"); plt.close()

    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig(f"{opt_tmpdir}/plot_optimization_history.png"); plt.close()
    
    optuna.visualization.matplotlib.plot_slice(study)
    plt.savefig(f"{opt_tmpdir}/plot_slice.png"); plt.close()

    optuna.visualization.matplotlib.plot_intermediate_values(study)
    plt.savefig(f"{opt_tmpdir}/plot_intermediate_values.png"); plt.close()

def train_test_best_model(study):
    """
    This function is used to train and test our pytorch lightning model
    based on parameters give to it
    Parameters: 
        study object
    Returns: 
        actuals: 1-D list containing target labels
        predictions:  1-D list containing predictions for said labels
    """

    best_params = study.best_params.copy()

    best_params['output_dims'] = [value for key,value in best_params.items() 
                                  if key.startswith('n_units_l')]

    trainer = Trainer(max_epochs=10, auto_scale_batch_size=None, 
                      callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
                      gpus=0 if torch.cuda.is_available() else None,
                      deterministic=True) 

    model = Regression(best_params) # double asterisk (dictionary unpacking)

    trainer.fit(model)

    # Either best or path to the checkpoint you wish to test. 
    # If None and the model instance was passed, use the current weights. 
    # Otherwise, the best model from the previous trainer.fit call will be loaded.
    trainer.test(ckpt_path='best')

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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Click ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Remove whitespace from your arguments
@click.command(
    help= "Given a folder path for CSV files (see load_raw_data), use it to create a model, find\
            find ideal hyperparameters and train said model to reduce its loss function"
)

@click.option("--dir_in", type=str, default='../preprocessed_data/', help="File containing csv files used by the model")
@click.option("--countries", type=str, default="Portugal", help='csv names from dir_in used by the model')
@click.option("--seed", type=str, default="42", help='seed used to set random state to the model')
@click.option("--opt_model_path", type=str, default="./optuna_model.pt")

@click.option("--n_trials", type=str, default="20", help='number of trials - different tuning oh hyperparams')
@click.option("--max_epochs", type=str, default="5,10", help='range of number of epochs used by the model')
@click.option("--n_layers", type=str, default="1,2", help='range of number of layers used by the model')
@click.option("--layer_size", type=str, default="90,150", help='range of size of each layer used by the model')
@click.option("--l_window", type=str, default="240,250", help='range of lookback window (input layer size) used by the model')
@click.option("--f_horizon", type=str, default="24,25", help='range of forecast horizon (output layer size) used by the model')
@click.option("--l_rate", type=str, default="1e-5, 1e-4", help='range of learning rate used by the model')
@click.option("--activation", type=str, default="ReLU,SiLU", help='activations function experimented by the model')
@click.option("--optimizer_name", type=str, default="Adam,RMSprop", help='optimizers experimented by the model') # SGD
@click.option("--batch_size", type=str, default="64,128,256,512", help='possible batch sizes used by the model') #16,32,

def forecasting_model(**kwargs):
    """
    This is the main function of the script. 
    1. It loads the data from csvs
    2. splits to train/test/valid
    3. uses 'optuna' library to tune hyperparameters in range based on click params
    4. computes MAPE and plots graph of best model
    Parameters
        kwargs: dictionary containing click paramters used by the script
    Returns: None 
    """
    # Auto log all MLflow entities
    mlflow.pytorch.autolog()
    
    # store mlflow metrics/artifacts on temp file
    global opt_tmpdir
    opt_tmpdir = tempfile.mkdtemp()

    global click_params
    click_params = ClickParams(kwargs)

    print("=================== Click Params ===================")
    print(vars(click_params))
    
    #  read csv files
    df = read_csvs(click_params)
    df_backup = df.copy()

    # global dataScaler 
    # dataScaler = DataScaler(min=0,max=1)

    # use minmax to scale data between feauture range set by dataScaler (default min_max) 
    # df = dataScaler.scale_transform(df)

    # split data in train/test/validation
    global train_data, test_data, val_data
    train_data, test_data, val_data = train_test_valid_split(df)

    train_data.to_csv(f"{opt_tmpdir}/train_data.csv")
    test_data.to_csv(f"{opt_tmpdir}/test_data.csv")
    val_data.to_csv(f"{opt_tmpdir}/val_data.csv")

    # use Optuna to make a study for best hyperparamter values for maxiumum reduction of loss function
    study = optuna_optimize()

    study.trials_dataframe().to_csv(f"{opt_tmpdir}/trials_dataframe.csv")

    # visualize results of study
    optuna_visualize(study)

    # train model with hparams set to best_params of optuna 
    plot_pred, plot_actual = train_test_best_model(study)

    # rescale values based on mean and std of 'Load' column
    # plot_actual_r = dataScaler.inverse_scale_transform(pd.DataFrame(plot_actual,columns=['Load']))
    # plot_pred_r = dataScaler.inverse_scale_transform(pd.DataFrame(plot_pred,columns=['Load']))

    # calculate metrics
    metrics = calculate_metrics(plot_actual,plot_pred)

    # plot prediction/actual data on common axis system 
    cross_plot_pred_data(df_backup, plot_pred, plot_actual, opt_tmpdir)

    print("\nUploading training csvs and metrics to MLflow server...")
    logging.info("\nUploading training csvs and metrics to MLflow server...")
    mlflow.set_tag("stage", "optuna")
    mlflow.log_artifacts(opt_tmpdir, "optuna_results")
    mlflow.log_metrics(metrics)

if __name__ == '__main__':
    print("\n=========== Optuna Hyperparameter Tuning =============")
    logging.info("\n=========== Optuna Hyperparameter Tuning =============")
    forecasting_model()