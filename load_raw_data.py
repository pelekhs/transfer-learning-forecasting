import sys
import tempfile
import requests
import mlflow
import os
import click
import logging
import shutil
import pycountry

def download_online_file(url, dst_filename=None, dst_dir=None):

    if dst_dir is None:
        dst_dir = tempfile.mkdtemp()
    else:
        os.makedirs(dst_dir, exist_ok=True)
    req = requests.get(url)
    if req.status_code != 200:
        raise Exception(f"\nResponse is not 200\nProblem downloading: {url}")
        sys.exit()
    url_content = req.content
    if dst_filename is None:
        dst_filename = url.split('/')[-1]
    filepath = os.path.join(dst_dir, dst_filename)
    file = open(filepath, 'wb')
    file.write(url_content)
    file.close()

    return filepath

def download_mlflow_file(url, dst_dir=None):
    S3_ENDPOINT_URL = os.environ.get('MLFLOW_S3_ENDPOINT_URL')

    if dst_dir is None:
        dst_dir = tempfile.mkdtemp()
    else:
        os.makedirs(dst_dir, exist_ok=True)
    if url.startswith('s3://mlflow-bucket/'):
        url = url.replace("s3:/", S3_ENDPOINT_URL)
        local_path = download_online_file(
            url, dst_dir=dst_dir)
    elif url.startswith('runs:/'):
        # client = mlflow.tracking.MlflowClient()
        run_id = url.split('/')[1]
        mlflow_path = '/'.join(url.split('/')[3:])
        # local_path = client.download_artifacts(run_id, mlflow_path, dst_dir)
        local_path = mlflow.artifacts.download_artifacts(run_id, mlflow_path, dst_dir)
    elif url.startswith('http://'):
        local_path = download_online_file(url, dst_dir=dst_dir)
    elif url.startswith('file:///'):
        mlflow_path = '/'.join(url.split('/')[3:])
        local_path = mlflow.artifacts.download_artifacts(mlflow_path, dst_path=dst_dir)
    else:
        return url

    return local_path

def find_country(filename):
    # Get lists of country names and codes based on pycountry lib
    country_names = list(map(lambda country : country.name, pycountry.countries))
    country_codes = list(map(lambda country : country.alpha_2, pycountry.countries))

    cand_countries = [country for country in country_names + ["Czech Republic"] + country_codes 
                    if country in filename and not country in ["BZ", "BA"]]
    if len(cand_countries) >= 1:
        #Always choose the first candidate country
        country_name = pycountry.countries.search_fuzzy(cand_countries[0])[0].name
        country_code = pycountry.countries.search_fuzzy(cand_countries[0])[0].alpha_2
        print(f"File \"{filename}\" belongs to \"{country_name}\" with code \"{country_code}\"")
        return country_name, country_code
    else:
        return None, None

@click.command()
@click.option("--dir_in", 
              type=str, 
              default="file:///C:/Users/User/Desktop/thesis/code/pipeline/mlruns/0/f163c5e155b04a779a2615ece69da35e/artifacts/preprocessed_data",
              help='path to input, can be filepath or remote filepath: http address or artifact URI ')
@click.option("--countries", type=str, default="Portugal", help='csv names from dir_in used by the model')

def load_raw_data(dir_in, countries):
    """
    series_uri: folder or file used as input for pipeline
    """
    with mlflow.start_run(run_name="load_data",nested=True) as load_start:

        if(not os.path.exists(dir_in)):
            input_path = download_mlflow_file(dir_in, dst_dir=None)
        else:
            input_path = dir_in

        input_path = input_path.replace('/', os.path.sep)

        with tempfile.TemporaryDirectory() as load_tmpdir: 
            for root, dirs, files in os.walk(input_path):
                for name in files:
                    country_name, country_code = find_country(name)
                    if(not (country_name in countries.split(',') or country_code in countries.split(','))):
                        print(f'Country \"{country_name}\" not in list for preprocessing. Ignoring...')
                        continue                    
                    print(f'Found file with name \"{name}\"')
                    shutil.copyfile(f'{input_path}/{name}', f'{load_tmpdir}/{name}')

            print("\nUploading training csvs and metrics to MLflow server...")
            logging.info("\nUploading training csvs and metrics to MLflow server...")
            mlflow.log_param("dir_in", dir_in)
            mlflow.log_artifacts(load_tmpdir, "load_data")
            mlflow.set_tag("run_id", load_start.info.run_id)        
            mlflow.set_tag('load_data_uri', f'{load_start.info.artifact_uri}/load_data')
            mlflow.set_tag("stage", "load")

if __name__ == '__main__':
    print("\n=========== Load Data =============")
    logging.info("\n=========== Load Data =============")
    load_raw_data()