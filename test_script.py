import os
import itertools
# from subprocess import run, Popen, PIPE
import click
from enum import IntEnum
from model_utils import ClickParams

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Globals ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

click_params = None

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def all_for_one(countries,combinations):
    # prepare combinations to be used as command argument
    source_domain = []
    for comb in combinations:
        source_domain.append(','.join(str(country) for country in comb))

    # print(source_domain)

    # get list of missing value of combination (target domain of combination)
    target_domain = []
    for target_countries in source_domain:
        missing_country = set(countries).difference(target_countries.split(','))
        target_domain.append(list(missing_country)[0].strip("'"))
    
    return source_domain, target_domain

def case_1(params,domain):
    """
    Case 1: Source and target domain is the same one country
    """
    print("=============== Case 1 ===============")        
    print(f"Source/Target Domain: {domain}")
    for country in domain:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")               
        print(f"Source/Target Domain: {country}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")       
        command = (
            f'mlflow run . --env-manager=local -P stages={params.stages} '
            f'-P src_countries={country} -P transfer_mode=0 -P test_case=1 ' ## add/remove max_epochs
            f'-P max_epochs=2 -P n_trials=2 '
            f'--experiment-name={params.experiment_name}'
        )     
        print(command)
        os.system(command)
    
def case_2(params,source_domain,target_domain):
    """
    Case 2: 
        Source domain: N-1 countries
        Target domain: 1 (remaining) country
    """
    print("=============== Case 2 ===============")        
    counter = 0
    for src, tgt in zip(source_domain,target_domain):
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"Source Domain: {src}")
        print(f'Target Domain: {tgt}')
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")   
        command = (
            f'mlflow run . --env-manager=local -P stages={params.stages} '
            f'-P src_countries={src} -P tgt_countries={tgt} -P test_case=2 '
            f'-P transfer_mode={params.transfer_mode} ' ## add/remove max_epochs
            f'-P max_epochs=2 -P n_trials=2 '
            f'--experiment-name={params.experiment_name}'
        )     
        print(command)
        os.system(command)
        counter+= 1
        if(counter > 0): break;

#### Enum used to define test case used in run of test_script  
class TestCase(IntEnum):
    BENCHMARK = 1
    ALL_FOR_ONE = 2 
    CLUSTER_FOR_ONE = 3

# Remove whitespace from your arguments
@click.command()
@click.option("--case", type=str, default="1", help='test cases to use')
@click.option("--stages", type=str, default='model', help='comma seperated entry point names to execute from pipeline')
@click.option("--transfer_mode", type=str, default="1", help='indicator to use transfer learning techniques')
@click.option("--experiment_name", type=str, default="alex_trash", help='indicator to use transfer learning techniques')

def test_script(**kwargs):
    countries = os.listdir('../preprocessed_data/') #get list of countries
    countries = [x.rstrip('.csv') for x in countries] # remove '.csv' suffix in each one

    # get list of lists for all possible combinations with N-1 countries
    combinations = list(itertools.combinations(countries, len(countries)-1)) 

    # create source/target list of countries
    source_domain,target_domain = all_for_one(countries,combinations)

    click_params = ClickParams(kwargs)

    if(TestCase(click_params.case) == TestCase.BENCHMARK):
        case_1(click_params,target_domain)
    if(TestCase(click_params.case) == TestCase.ALL_FOR_ONE):
        case_2(click_params,source_domain,target_domain)
    if(TestCase(click_params.case) == TestCase.CLUSTER_FOR_ONE):
        pass
    
if __name__ == "__main__":
    test_script()