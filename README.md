# transfer-learning-forecasting

The repository for load forecasting through Transfer Learning techniques.

## Installation

This project is implemented in [MLFlow] (https://mlflow.org/docs/latest/index.html) to handle the different stages in the pipeline. Each stage can be run independently as an entry point, and its inputs and outputs are stored in its respected MLflow run file -- see **MLProject** file for details regarding the inputs of each entry point.

| Entrypoint |            Filename           |
|:----------:|:-----------------------------:|
| main       | main.py                       |
| load       | load_raw_data.py              |
| etl        | etl.py                        |
| optuna     | forecasting_model_optuna.py   |
| model      | forecasting_model.py          |
| ensemble   | forecasting_model_ensemble.py |
| eval       | forecasting_model_eval.py     |
| snaive*     | forecasting_model_naive.py   |

***snaive** is an independent entrypoint used for comparative evaluation of our scenarios with naive forecasts

**model_utils.py** is a python file containing general-purpose functions used in more than one pipeline stages

Use the package manager [pip] (https://pip.pypa.io/en/stable/) to install MLFlow.
```bash
pip install mlflow
```
By executing an entrypoint, **MLProject** checks package dependencies (see **python_env.yaml**) and proceeds to install them.

## Data format

Our pipeline is capable of processing multiple files from different or same countries. 
Data must: 
* all be in a single directory provided at user input (dir_in) 
* be in csv format 
* contain one datetime column (named "Start") and one column containing energy load data (named "Load")
* be named after the country name (e.g Greece) or code (e.g GR) they represent

## Usage

As mentioned, **MLProject** offers a wide range of parameters that should be tuned, with regards to each stage in pipeline.

**Example:** locally train a model in the Greek-Spanish dataset, apply AbO warm-start transfer learning on Italy and store in an experiment with name "full_pipeline":

```bash
mlflow run . --env-manager=local -P stages='all'
             -P src_countries='Greece,Spain' -P tgt_countries='Italy' 
             -P test_case=2 -P transfer_mode=1
             --experiment-name=full_pipeline
```

**transfer_mode** and **test_case** are integers determined by the following Enums:

```python
class TestCase(IntEnum):
    NAIVE = 0
    BENCHMARK = 1
    ALL_FOR_ONE = 2 
    CLUSTER_FOR_ONE = 3

class Transfer(IntEnum):
    NO_TRANSFER = 0
    WARM_START = 1 

```


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

