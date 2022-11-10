#!/usr/bin/env python
# coding: utf-8

import os
import click
import tempfile
import matplotlib.pyplot as plt
import tempfile
import shutil

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint
from mlflow.tracking.fluent import _get_experiment_id

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
    print_run_info(all_run_infos)

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
            shutil.rmtree("./mlruns/0/"+run_info.info.run_id, ignore_errors=True)

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
def _get_or_run(entrypoint, parameters,ignore_previous_run=False, use_cache=True):
# TODO: this was removed to always run the pipeline from the beginning.
    if not ignore_previous_run:
        existing_run = _already_ran(entrypoint, parameters)
        if use_cache and existing_run:
            print("Found existing run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
            return existing_run
    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters, env_manager="local")

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
@click.option("--stages",
              type=str,
              default='all',
              help='string containing stages (entry points) to execute from pipeline seperated by commas' 
)

def workflow(stages):
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.

    idx = find_idx("main")
    print(f"Creating main run with index: {idx}")
    RUN_NAME = f"run_{idx}"

    # with mlflow.start_run() as active_run:
    with mlflow.start_run(run_name="main") as active_run:
        mlflow.set_tag("stage", "main")

        pipeline = stages.split(',')

        return
        # print(stages)
        
        if 'etl' in pipeline or 'all' in pipeline:
            etl_run = _get_or_run("etl", {})   
            # etl_series_uri =  etl_run.data.tags["series_uri"].replace("s3:/", S3_ENDPOINT_URL)
            # etl_time_covariates_uri =  etl_run.data.tags["time_covariates_uri"].replace("s3:/", S3_ENDPOINT_URL)
            # ratings_parquet_uri = os.path.join(etl_data_run.info.artifact_uri, "etl-dir")

        if 'optuna' in pipeline or 'all' in pipeline:
            optuna_run = _get_or_run("optuna", {})   
            # optuna_uri = os.path.join(optuna_run.info.artifact_uri, "optuna-dir")

        if 'model' in pipeline or 'all' in pipeline:
            train_run = _get_or_run("model", {})   
            # ensemble_uri = os.path.join(ensemble_run.info.artifact_uri, "model-dir")

            # Log train params (mainly for logging hyperparams to father run)
            for param_name, param_value in train_run.data.params.items():
                try:
                    mlflow.log_param(param_name, param_value)
                except mlflow.exceptions.RestException:
                    pass
                except mlflow.exceptions.MlflowException:
                    pass

        if 'ensemble' in pipeline or 'all' in pipeline:
            ensemble_run = _get_or_run("ensemble", {})   
            # ensemble_uri = os.path.join(model_data_run.info.artifact_uri, "ensemble-dir")

            # Log eval metrics to father run for consistency and clear results
            mlflow.log_metrics(ensemble_run.data.metrics)

    # print(f"Status: {active_run.info.to_proto().status}")
    # if(active_run.info.to_proto().status != RunStatus.FINISHED):
    #     mlflow.delete_run(active_run.info.run_id) 
    #     shutil.rmtree("./mlruns/0/"+active_run.info.run_id, ignore_errors=True)

if __name__ == "__main__":
    workflow()