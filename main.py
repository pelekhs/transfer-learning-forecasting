#!/usr/bin/env python
# coding: utf-8

import os
import click
import tempfile
import matplotlib.pyplot as plt
import tempfile
import shutil
import pickle
import logging 
import copy

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint
from mlflow.tracking.fluent import _get_experiment_id

from model_utils import ClickParams

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
S3_ENDPOINT_URL = os.environ.get('MLFLOW_S3_ENDPOINT_URL')
EXPERIMENT_NAME = "mlflow-pipeline"
# EXP = mlflow.set_experiment(EXPERIMENT_NAME)

def print_run_info(runs):
    for r in runs:
        print("run_id: {}".format(r.info.run_id))
        print("lifecycle_stage: {}".format(r.info.lifecycle_stage))
        print("metrics: {}".format(r.data.metrics))
        print("Status: {}".format(r.info.to_proto().status))
        # Exclude mlflow system tags
        tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
        print("tags: {}".format(tags))

def _already_ran(entry_point_name, parameters, experiment_id=None):
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    client = MlflowClient()
    all_run_infos = reversed(client.search_runs(experiment_id))

    for run_info in all_run_infos:
        full_run = client.get_run(run_info.info.run_id)
        tags = full_run.data.tags
        if tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None) != entry_point_name:
            continue
        match_failed = False
        for param_key, param_value in parameters.items():
            run_value = full_run.data.params.get(param_key)
            if run_value != param_value:
                match_failed = True
                break
        if match_failed:
            continue
        
        # if failed runs exist delete them and their folder
        if run_info.info.to_proto().status == RunStatus.FAILED:
            mlflow.delete_run(run_info.info.run_id) 
            print(f"Delete file: {run_info.info.run_id}")
            shutil.rmtree(f"./mlruns/0/{run_info.info.run_id}", ignore_errors=True)

        if run_info.info.to_proto().status != RunStatus.FINISHED:
            eprint(
                ("Run matched, but is not FINISHED, so skipping (run_id=%s, status=%s)")
                % (run_info.info.run_id, run_info.status)
            )
            continue

        return client.get_run(run_info.info.run_id)
    eprint("No matching run has been found.")
    return None

# TODO(aaron): This is not great because it doesn't account for:
# - changes in code
# - changes in dependant steps
def _get_or_run(entrypoint, parameters, ignore_previous_run=False, use_cache=True):
# TODO: this was removed to always run the pipeline from the beginning.
    if not ignore_previous_run:
        existing_run = _already_ran(entrypoint, parameters)
        if use_cache and existing_run:
            # print(f'Found existing run (with ID=%s) for entrypoint=%s and parameters=%s')
            logging.warning(f"Found existing run (with ID={existing_run.info.run_id}) for entrypoint={entrypoint} and parameters={parameters}")
            return existing_run
    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    print(f"Parameters: {parameters}")
    submitted_run = mlflow.run(".", entrypoint,
                                parameters=parameters,
                                env_manager="local")
    return MlflowClient().get_run(submitted_run.run_id)

def find_idx(cur_stage):
    client = MlflowClient()
    all_run_infos = reversed(client.search_runs(_get_experiment_id()))
    idx = 0
    # find how many "entry_points" stages are
    if(all_run_infos):
        for runs in all_run_infos:
            tags = {k: v for k, v in runs.data.tags.items() if not k.startswith("mlflow.")}
            # print(tags)
            if 'stage' in tags and tags['stage'] == cur_stage: 
                print('Found one!')
                idx += 1
    return idx

# Remove whitespace from your arguments
@click.command()
@click.option("--stages", type=str, default='all', help='comma seperated entry point names to execute from pipeline')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ETL Params ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@click.option("--dir_in", type=str, default='../original_data/', help="File containing csv files used by the model")
@click.option("--dir_out", type=str, default="../preprocessed_data/", help="Local directory where preprocessed timeseries csvs will be stored")
@click.option("--tmp_folder", type=str, default="../tmp_data/", help="Temporary directory to store merged-countries data with no outliers (not yet imputed)")
@click.option("--local_tz", type=bool, default=False, help="flag if you want local (True) or UTC (False) timezone")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optuna (exclusive) Params ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@click.option("--n_trials", type=str, default="20", help='number of trials - different tuning oh hyperparams')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Train Params ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@click.option("--countries", type=str, default="Portugal", help='csv names from dir_in used by the model')
@click.option("--seed", type=str, default="42", help='seed used to set random state to the model')
@click.option("--max_epochs", type=str, default="200", help='range of number of epochs used by the model')
@click.option("--n_layers", type=str, default="1", help='range of number of layers used by the model')
@click.option("--layer_sizes", type=str, default="100", help='range of size of each layer used by the model')
@click.option("--l_window", type=str, default="240", help='range of lookback window (input layer size) used by the model')
@click.option("--f_horizon", type=str, default="24", help='range of forecast horizon (output layer size) used by the model')
@click.option("--l_rate", type=str, default="0.0001", help='range of learning rate used by the model')
@click.option("--activation", type=str, default="ReLU", help='activations function experimented by the model')
@click.option("--optimizer_name", type=str, default="Adam", help='optimizers experimented by the model') # SGD
@click.option("--batch_size", type=str, default="200", help='possible batch sizes used by the model') #16,32,
@click.option("--transfer_mode", type=str, default="0", help='indicator to use transfer learning techniques')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Ensemble (exclusive) Params ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@click.option("--n_estimators", type=str, default="3", help='number of estimators (models) used in ensembling')

def workflow(**kwargs):
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.

    click_params = ClickParams(kwargs)

    # idx = find_idx("main")
    # print(f"Creating main run with index: {idx}")
    # RUN_NAME = f"run_{idx}"

    with mlflow.start_run(run_name="main") as main_run:
        optuna_run = None; train_run = None
        train_params = None; optuna_params = None

        mlflow.set_tag("stage", "main")

        print(f"### Stages: {click_params.stages}")
        
        if 'etl' in click_params.stages or 'all' in click_params.stages:
            print("\n=========== ETL Stage =============")
            logging.info("\n=========== ETL Stage =============")
            etl_params = {
                'dir_in':  click_params.dir_in,
                'dir_out': click_params.dir_out,
                'tmp_folder': click_params.tmp_folder,
                'local_tz': click_params.local_tz
            }
            etl_run = _get_or_run("etl", parameters=etl_params)   

        if 'optuna' in click_params.stages or 'all' in click_params.stages:
            print("\n=========== Optuna Stage =============")
            logging.info("\n=========== Optuna Stage =============")            
            # if max_epochs int and not list (range), then main's click params (default or manually set)
            #    are meant for 'model' entrypoint and therefore SOME be ignored by optuna  
            # if not, it means that main's click params should not be passed to optuna run
            if(isinstance(click_params.max_epochs, int)):
                optuna_params = {
                    'dir_in': click_params.dir_in,
                    'countries': click_params.countries,
                    'seed': click_params.seed,
                    'n_trials': click_params.n_trials
                }
                optuna_run = _get_or_run("optuna", parameters=optuna_params)   
            else:
                exclusion_list = {'stages','dir_out','tmp_folder','local_tz',
                                'transfer_mode','n_estimators'}
                optuna_params = {x: kwargs[x] for x in kwargs if x not in exclusion_list}
                optuna_run = _get_or_run("optuna", parameters=optuna_params)   

        if 'model' in click_params.stages or 'all' in click_params.stages:
            print("\n=========== Model Stage =============")
            logging.info("\n=========== Model Stage =============")

            if('optuna' in click_params.stages):
                print("\n########### Optuna used for model ###########")
                logging.info("\n########### Optuna used for model ###########")
                best_trial_path = f'runs:/{optuna_run.info.run_id}/optuna_results/optuna_best_trial.pkl'
                train_params = mlflow.artifacts.download_artifacts(best_trial_path)
                train_params = pickle.load( open(train_params, 'rb') )

                print(f'"params: {optuna_run.data.params}')

                train_params['transfer_mode'] = kwargs['transfer_mode']
                train_params['countries'] = kwargs['countries'] ############################

                train_run = _get_or_run("model", parameters=train_params) 
            else:  
                print("\n########### Custom hyperparams used for model ###########")
                logging.info("\n########### Custom hyperparams used for model ###########")
                exclusion_list = {'stages','dir_out','tmp_folder',
                                'local_tz','n_estimators','n_trials'}
                train_params = {x: kwargs[x] for x in kwargs if x not in exclusion_list}
                train_run = _get_or_run("model", parameters=train_params) 

        if 'ensemble' in click_params.stages or 'all' in click_params.stages:
            print("\n=========== Ensemble Stage =============")
            logging.info("\n=========== Ensemble Stage =============")
            ensemble_params = {'n_estimators': click_params.n_estimators}
            if(not optuna_run): # if no optuna (train only)
                ensemble_params.update(train_params)
            else: #if optuna (with or without train - doesn't matter)
                ensemble_params.update(optuna_params) 
                del ensemble_params['n_trials'] 
                ensemble_params['transfer_mode'] = click_params.transfer_mode
            ensemble_run = _get_or_run("ensemble", parameters=ensemble_params) 

if __name__ == "__main__":
    workflow()