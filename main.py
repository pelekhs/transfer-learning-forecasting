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

from model_utils import ClickParams, Transfer, TestCase

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
S3_ENDPOINT_URL = os.environ.get('MLFLOW_S3_ENDPOINT_URL')

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
    submitted_run = mlflow.run(".", entrypoint,
                                parameters=parameters,
                                env_manager="local")
    return MlflowClient().get_run(submitted_run.run_id)

def pipeline(kwargs):
    pipeline_name = 'SOURCE'
    if(kwargs['transfer_mode'] != str(Transfer.NO_TRANSFER.value)):
        pipeline_name = 'TARGET'

    with mlflow.start_run(run_name=pipeline_name,nested=True) as pipeline_run:
        mlflow.set_tag("stage", pipeline_name)
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optuna entrypoint ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        optuna_run = None; optuna_params = None
        if 'optuna' in kwargs['stages'] or 'all' in kwargs['stages']:
            print("\n=========== Optuna Stage =============")
            logging.info("\n=========== Optuna Stage =============")            
            # if max_epochs int and not list (range), then main's click params (default or manually set)
            #    are meant for 'model' entrypoint and therefore should be ignored by optuna  
            # if not, it means that main's click params should be passed to optuna run
            if(not isinstance(kwargs['max_epochs'], list)):
                optuna_params = {
                    'dir_in': kwargs['dir_in'],
                    'countries': kwargs['countries'],
                    'seed': kwargs['seed'],
                    'n_trials': kwargs['n_trials']
                }
                optuna_run = _get_or_run("optuna", parameters=optuna_params)   
            else:
                exclusion_list = {'stages','local_tz','transfer_mode','n_estimators'}
                optuna_params = {x: kwargs[x] for x in kwargs if x not in exclusion_list}
                optuna_run = _get_or_run("optuna", parameters=optuna_params)   

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Train entrypoint ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        train_run = None; train_params = None 
        if 'model' in kwargs['stages'] or 'all' in kwargs['stages']:
            print("\n=========== Model Stage =============")
            logging.info("\n=========== Model Stage =============")

            if('optuna' in kwargs['stages']):
                print("\n########### Optuna used for model ###########")
                logging.info("\n########### Optuna used for model ###########")
                best_trial_path = f'runs:/{optuna_run.info.run_id}/optuna_results/optuna_best_trial.pkl'
                train_params = mlflow.artifacts.download_artifacts(best_trial_path)
                train_params = pickle.load( open(train_params, 'rb') )

                train_params['transfer_mode'] = kwargs['transfer_mode']
                train_params['countries'] = kwargs['countries'] ############################

                train_run = _get_or_run("model", parameters=train_params) 
            else:  
                print("\n########### Custom hyperparams used for model ###########")
                logging.info("\n########### Custom hyperparams used for model ###########")
                exclusion_list = {'stages','local_tz','n_estimators','n_trials'}
                train_params = {x: kwargs[x] for x in kwargs if x not in exclusion_list}
                train_run = _get_or_run("model", parameters=train_params) 

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Ensemble entrypoint ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ensemble_run = None; ensemble_params = None
        if 'ensemble' in kwargs['stages'] or 'all' in kwargs['stages']:
            print("\n=========== Ensemble Stage =============")
            logging.info("\n=========== Ensemble Stage =============")
            ensemble_params = {'n_estimators': kwargs['n_estimators']}
            if(not optuna_run): # if no optuna (train only)
                ensemble_params.update(train_params)
            else: #if optuna (with or without train - doesn't matter)
                ensemble_params.update(optuna_params) 
                del ensemble_params['n_trials'] 
                ensemble_params['transfer_mode'] = kwargs['transfer_mode']
            ensemble_run = _get_or_run("ensemble", parameters=ensemble_params) 

# ################################# Click Parameters ########################################
# Remove whitespace from your arguments
@click.command()
@click.option("--stages", type=str, default='all', help='comma seperated entry point names to execute from pipeline')
@click.option("--src_countries", type=str, default="Portugal", help='csv names from dir_in used by the source model')
@click.option("--tgt_countries", type=str, default="Portugal", help='csv names from dir_in used by the target model')
@click.option("--test_case", type=str, default="1", help='case number -check TestCase enum in model_utils and test_script.py-')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ETL Params ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@click.option("--dir_in", type=str, default='../original_data/', help="File containing csv files used by the model")
@click.option("--local_tz", type=bool, default=False, help="flag if you want local (True) or UTC (False) timezone")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optuna (exclusive) Params ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@click.option("--n_trials", type=str, default="20", help='number of trials - different tuning oh hyperparams')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Train Params ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
@click.option("--num_workers", type=str, default="3", help='accelerator (cpu/gpu) processesors and threads used')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Ensemble (exclusive) Params ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@click.option("--n_estimators", type=str, default="3", help='number of estimators (models) used in ensembling')

def workflow(**kwargs):
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run(run_name="main") as main_run:
        mlflow.set_tag("stage", "main")
        mlflow.set_tag("test_case", TestCase(int(kwargs['test_case'])).name)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Load Data entrypoint ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        load_run = None; load_params = None; load_data_path = None
        print("\n=========== Load Data Stage =============")
        logging.info("\n=========== Load Data Stage =============")    
        load_params = {
            'dir_in':  kwargs['dir_in'],
            'countries': kwargs['src_countries'] if kwargs['transfer_mode'] == str(Transfer.NO_TRANSFER.value) \
                                                    else f"{kwargs['src_countries']},{kwargs['tgt_countries']}",
        }
        load_run = _get_or_run("load", parameters=load_params)
        # fetch output folder containing  loaded datasets from load_data, to be used as input for the etl entrypoint            
        load_data_path = mlflow.artifacts.download_artifacts(f'runs:/{load_run.info.run_id}/load_data')
        print(f'load_data_path : {load_data_path}')

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ETL entrypoint ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        etl_run = None; etl_params = None; preprocessed_data_path = None
        if('etl' in kwargs['stages'] or 'all' in kwargs['stages']):
            print("\n=========== ETL Stage =============")
            logging.info("\n=========== ETL Stage =============")
            etl_params = {
                'dir_in':  load_data_path if load_run else kwargs['dir_in'],
                'countries': kwargs['src_countries'] if kwargs['transfer_mode'] == str(Transfer.NO_TRANSFER.value) \
                                                     else f"{kwargs['src_countries']},{kwargs['tgt_countries']}",
                'local_tz': kwargs['local_tz']
            }
            print(etl_params)
            etl_run = _get_or_run("etl", parameters=etl_params)

            # fetch output folder containing preprocessed datasets from etl, to be used as input for the following entrypoints
            preprocessed_data_path = mlflow.artifacts.download_artifacts(f'runs:/{etl_run.info.run_id}/preprocessed_data')
            print(f'preprocessed_data_path : {preprocessed_data_path}')
            # 'dir_in' in the context of etl is original data folder, while 
            # in the context of everything  else, is the output of dir_in
            # kwargs['dir_in'] = preprocessed_data_path
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Source Domain ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # copy click parameters, update 'countries' element with:
        #   - 'src_countries' element's value when in source domain's pipeline
        #       > NO_TRANSFER 
        #   - 'tgt_countries' element's value when in target domain's pipeline
        #       > transfer technique (if set for transfer learning)
        exclusion_list = ['test_case','src_countries','tgt_countries']
        train_kwargs = {x: kwargs[x] for x in kwargs if x not in exclusion_list}
        train_kwargs['countries'] = kwargs['src_countries']
        train_kwargs['transfer_mode'] = str(Transfer.NO_TRANSFER.value)
        train_kwargs['dir_in'] = preprocessed_data_path

        pipeline(train_kwargs)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Target Domain ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # if there is transfer learning
        if(kwargs['transfer_mode'] != str(Transfer.NO_TRANSFER.value)):
            train_kwargs['countries'] = kwargs['tgt_countries']
            train_kwargs['transfer_mode'] = kwargs['transfer_mode']
            pipeline(train_kwargs)

if __name__ == "__main__":
    print("\n=========== Experimentation Pipeline =============")
    logging.info("\n=========== Experimentation Pipeline =============")    
    workflow()