#!/usr/bin/env python
# coding: utf-8

# This is (to almost its entirety) copied from this excellent article 
# [here](https://coderzcolumn.com/tutorials/machine-learning/simple-guide-to-optuna-for-hyperparameters-optimization-tuning)
# 
# Let's import some basic libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
# mpl.use("webagg")
from matplotlib import pyplot as plt
import pathlib
from datetime import datetime
from sklearn import preprocessing
import warnings
import copy

import optuna
from optuna.integration import PyTorchLightningPruningCallback

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.profiler import Profiler, AdvancedProfiler
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import MeanAbsolutePercentageError

import logging
import pickle
import click
import mlflow
import sys

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Classes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
class ClickParams:
    def __init__(self, click_args):
        super(ClickParams, self).__init__()
        
        for key, value in click_args.items():
            split_value = value.split(',') # split string into a list delimited by comma

            if not split_value: # if empty list
                sys.exit(f"Empty value for paramter {key}")
            
            if(len(split_value) == 1): # if single value, check if int/float/str
                if(value.isdigit()): setattr(self, key, int(value)); continue;
                if(self.is_floats(value)): setattr(self, key, float(value)); continue;
                else: setattr(self, key, value); continue; # categorical shoul be kept as a list  

            # since value is not a list
            if all(ele.isdigit() for ele in split_value): # if all characters in each value in list are digits (int)
                setattr(self, key, [int(x) for x in split_value]); continue;
            elif self.is_floats(split_value): # check if all values in list can be converted to floats (decimals or scientific notation)
                setattr(self, key, [float(x) for x in split_value]); continue;
            else: # then it's a list of strings
                setattr(self, key, split_value)
    
    def is_floats(self,x):
        """ True if x is convertible to a float, or is a list or tuple (or nested list or 
            tuple) of values that are all convertible to floats """    
        if isinstance(x, (list, tuple)):
            return all(self.is_floats(item) for item in x)
        try:
            float(x)
            return True
        except ValueError:
            return False

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Globals ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
click_params = None
train_data = None
test_data = None 
validation_data = None

## Loading Data
def read_csvs():
    """
    Reads all csv files in folder path given and stores them in a single dataframe df 
    Parameters: list of countries to read their csv
    ----------
    Returns
    -------
    pandas.DataFrame
        Dataframe where rows are individual data and columns for indicative
        information on the data like datetime, month, week, country etc
    """

    df = pd.DataFrame()
    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

    # For every csv file in folder 
    for csv in pathlib.Path(click_params.dir_in).glob('*.csv'):
        if(click_params.countries is None or csv.stem in click_params.countries):            
            print("Loading dataset: "+csv.stem)
            temp_df = pd.read_csv(csv,
                                  parse_dates=['Date'],
                                  dayfirst=True,
                                  date_parser=dateparse)
            temp_df['country'] = csv.stem #set cokumn with the name of the dataset this datum came from
            df = pd.concat([df,temp_df]) 
    print('Data loading complete!')
    # sort values based on Start column (Datetime)
    df = df[df['year'] != 2022] # remove rows of year 2022 since its an incomplete year (not currently ended)
    if(click_params.countries is None):
        df.sort_values(by=['Date'], ascending=True, inplace=True)
    df['Unnamed: 0'] = range(1, len(df) + 1)     # Reset "Unnamed: 0" based on concated df 
    return df

# ### Preparing the data in general

# #### Categorical variables: one-hot encoding
# 
# A column like weekday actually contains 7 possible values, 0 through 6, each of which represents a different day of the week. Instead of representing this data as 1 feature with 7 values, it works better if we represent if as 7 binary features. This process is known as one-hot encoding and is nicely explained by Chris Albon [here](https://chrisalbon.com/machine_learning/preprocessing_structured_data/one-hot_encode_nominal_categorical_features/). 
# Note that 'holiday' is effectively already one-hot encoded as it either has a value 0 for 'not a holiday' or 1 for 'is a holiday'.

# #### Continuous variables: scaling
# It is common practice to scale the continuous variables, i.e. to standardize them on a zero mean with a standard deviation of 1. That way, coefficients of variables with large variance are small and thus less penalized by the model. 
# The ultimate goal to perform standardization is to bring down all the features to a common scale without distorting the differences in the range of the values. 
# We do this even for 'Load' which is our target variable, and therefore we need to be able to get back again to get an actual number of bikes predicted so we'll save away the various means and standard deviations for converting back later.

# #### Data we won't use

# ### Preparing the data for training
# 
# #### Train / test / validation split
# This is a timeseries problem, so it would be typical to train on the earlier data, and test / validate on the later data.

def train_test_valid_split(df):
    """
    we choose to split data with validation/test data to be at the end of time series
    Since it's a time series, it is intuitive to predict future values, rather than values in between
    validation/test should hold at least an entire year, since:
        - most seasonality exists between years
        - there are fluctuation inside year (e.g heatwaves at summer)
    Parameters:
        pandas.dataframe containing dataframe to split
    Returns:
        pandas.dataframe containing train/test/valiation data
        pandas.dataframe containing valiation data
        pandas.dataframe containing test data
    """
    # train_data: data from year [2015,2019]
    # validation_data: data from year 2020
    # test_data: data from year 2021
    # drop all columns except 'Load' (use 'backup_df' to restore them)
    global train_data, test_data, validation_data

    train_data = df[~df['year'].isin(['2020','2021'])][['Load']] 
    test_data = df[df['year'] == 2021][['Load']]
    validation_data = df[df['year'] == 2020][['Load']]

    # print("dataframe shape: {}".format(df.shape))
    # print("train shape: {}".format(train_data.shape))
    # print("test shape: {}".format(test_data.shape))
    # print("validation shape: {}".format(validation_data.shape))

def feauture_target_split(df, lookback_window=168, forecast_horizon=36):# lookback_window: 168 = 7 days(* 24 hours)
    """
    This function gets a column of a dataframe and splits it to input and target
    
    **lookback_window**
    In a for-loop of 'lookback_window' max iterations, starting from 0 
    At N-th iteration (iter): 
        1. create a shifted version of 'Load' column by N rows (vertically) and 
        2. stores it in a column* (feauture_'N')

    Same pseudo-code for 'forecast_horizon' loop
    
    *At first iterations of both loops, the new columns (feauture/target) are going to be firstly created
    but for each iteration, the same columns are going to be used
    
    We store each new column created in a dictionary which, at the end, convert it to dataframe
    The reason behind this choice is that if it was initially a dataframe, for large amount of loops,
    fast insertion of new columns would cause a performance issue (slower) even though the end result
    would propably not be altered
    
    Parameters: 
        df: pandas.dataframe containing column to parse
        lookback_window: lookback_window - # feature columns - # inputs in model
        forecast_horizon: forecast_horizon - # target columns - # outputs in model
    ----------
    Returns
        'subset'_X: pandas.dataframe containing feautures of df after preprocess for model
        'subset'_Y: pandas.dataframe containing targets of df after preprocess for model
    -------
    """

    # Reset "Unnamed: 0" based on concated df 
    df['Unnamed: 0'] = range(1, len(df) + 1)

    df_copy = df.copy()    

    df_new = {}
        
    for inc in range(0,int(lookback_window)):
        df_new['feauture_' + str(inc)] = df_copy['Load'].shift(-inc)

    # shift 'load' column permanently for as many shifts happened at 'lookback_window' loops  
    df_copy['Load'] = df_copy['Load'].shift(-int(lookback_window))
                    
    for inc in range(0,int(forecast_horizon)):
        df_new['target_' + str(inc)] = df_copy['Load'].shift(-inc)    
    
    df_new = pd.DataFrame(df_new, index=df_copy.index)
    df_new = df_new.dropna().reset_index(drop=True)    
                        
    # store new dataset to csv (if needed) 
    df_new.to_csv("preprocess_for_model.csv")
    
    return df_new.iloc[:,:lookback_window] , df_new.iloc[:,-forecast_horizon:]

# ### Defining the model and hyperparameters
# Before proceeding with this section, I'd recommend taking a look at the PyTorch Lightning [INTRODUCTION GUIDE](https://pytorch-lightning.readthedocs.io/en/latest/introduction_guide.html) which describes how each section of the LightningModule template contributes and fits together. You'll recognize the same components that you're used to in PyTorch, but they are organized into functions with specific naming conventions that need to be adhered to.

# From here on out we start preparing the data for PyTorch - let's first get the libraries we'll need

class Regression(pl.LightningModule):
    """
    Regression  Techniques are used when the output is real-valued based on continuous variables. 
                For example, any time series data. This technique involves fitting a line
    Feature: Features are individual independent variables that act as the input in your system. 
             Prediction models use features to make predictions. 
             New features can also be obtained from old features using a method known as ‘feature engineering’. 
             More simply, you can consider one column of your data set to be one feature. 
             Sometimes these are also called attributes. T
             The number of features are called dimensions
    Target: The target is whatever the output of the input variables. 
            In our case, it is the output value range of load. 
            If the training set is considered then the target is the training output values that will be considered.
    Labels: Label: Labels are the final output. You can also consider the output classes to be the labels. 
            When data scientists speak of labeled data, they mean groups of samples that have been tagged to one or more labels.

    ### The Model ### 
    Initialize the layers
    Here we have:
        one input layer (size 'lookback_window'), 
        one output layer (size 36 as we are predicting next 36 hours)
        hidden layers define by 'params' argument of init
    """
    def __init__(self,params):
        super(Regression, self).__init__()
        self.l_rate = params['l_rate']
        self.batch_size = params['batch_size']
        self.l_window = params['l_window']
        self.f_horizon = params['f_horizon']
        # self.loss = nn.MSELoss() # MSE
        # self.loss = nn.L1Loss() # MAE 
        self.loss = MeanAbsolutePercentageError() #MAPE
        self.optimizer_name = params['optimizer_name']
        self.output_dims = params['output_dims']
        self.activation = params['activation']
                
        layers = [] #list of layer to add at nn

        input_dim = self.l_window #input dim set to lookback_window
        output_dim = self.f_horizon #output dim set to f_horizon
        
        # create datasets used by dataloaders
        # 'subset'_X: dataset containing features of subset (train/test/validation) dataframe
        # 'subset'_Y: dataset containing targets of subset (train/test/validation) dataframe
        self.train_X, self.train_Y = feauture_target_split(train_data,input_dim,output_dim)  
        self.validation_X, self.validation_Y = feauture_target_split(validation_data,input_dim,output_dim)  
        self.test_X, self.test_Y = feauture_target_split(test_data,input_dim,output_dim)
        
        """
        Each loop is the setup of a new layer
        At each iteration:
            1. add previous layer to the next (with parameters gotten from output_dims)
                    at first iteration previous layer is input layer
            2. add activation function
            3. set current_layer as next layer
        connect last layer with cur_layer
        """
        for cur_layer in self.output_dims: 
            layers.append(nn.Linear(input_dim, cur_layer))
            layers.append(getattr(nn, self.activation)()) # nn.activation_function (as suggested by Optuna)
            input_dim = cur_layer #connect cur_layer with previous layer (at first iter, input layer)
        
        #connect last layer (stored at input_din) with target layer (output_dim)            
        layers.append(nn.Linear(input_dim, output_dim)) 
        self.layers = nn.Sequential(*layers) #list of nn layers

    # Perform the forward pass
    def forward(self, x):
        return self.layers(x)

### The Data Loaders ###     
    # Define functions for data loading: train / validate / test
    def train_dataloader(self):
        feature = torch.tensor(self.train_X.values).float() #feauture tensor train_X
        target = torch.tensor(self.train_Y.values).float() #target tensor train_Y
        train_dataset = TensorDataset(feature, target)  # dataset bassed on feature/target
        train_loader = DataLoader(dataset = train_dataset, 
                                  shuffle=True, 
                                  batch_size = self.batch_size)
        return train_loader
            
    def test_dataloader(self):
        feature = torch.tensor(self.test_X.values).float()
        target = torch.tensor(self.test_Y.values).float()
        test_dataset = TensorDataset(feature, target)
        test_loader = DataLoader(dataset = test_dataset, 
                                 batch_size = self.batch_size)
        return test_loader

    def val_dataloader(self):
        feature = torch.tensor(self.validation_X.values).float()
        target = torch.tensor(self.validation_Y.values).float()
        validation_dataset = TensorDataset(feature, target)
        validation_loader = DataLoader(dataset = validation_dataset,
                                       batch_size = self.batch_size)
        return validation_loader

    def predict_dataloader(self):
        return self.test_dataloader()
    
### The Optimizer ### 
    # Define optimizer function: here we are using ADAM
    def configure_optimizers(self):
        return getattr(optim, self.optimizer_name)( self.parameters(),
                                                    # momentum=0.9, 
                                                    # weight_decay=1e-4,                   
                                                    lr=self.l_rate)

### Training ### 
    # Define training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        # Add logging
        logs = {'loss': loss}
        self.log("loss", loss, on_epoch=True) # computes train_loss mean at end of epoch        
        return {'loss': loss, 'log': logs}

### Validation ###  
    # Define validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log("val_loss", loss)
        self.log("avg_val_loss", loss, on_epoch=True)  # computes avg_loss mean at end of epoch
        return {'val_loss': loss}

### Testing ###     
    # Define test step
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        correct = torch.sum(logits == y.data)
        # I want to visualize my predictions vs my actuals so here I'm going to 
        # add these lines to extract the data for plotting later on
        self.log('test_loss', loss, on_epoch=True)        
        return {'test_loss': loss, 'test_correct': correct, 'logits': logits}

### Prediction ###
    # Define prediction step
        # This method takes as input a single batch of data and makes predictions on it. 
        # It then returns predictions
    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self.forward(x)

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

    mlflow.pytorch.autolog()

    # Save state of model to file with path 'opt_model_path'
    # A state_dict is simply a Python dictionary object that maps each layer to its parameter tensor 
    # torch.save(model.state_dict(), click_params.opt_model_path)
    # print(f"State_dict: {str(model.state_dict())}")

    trainer.fit(model)

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

def mape(actual, pred): 
    """
    Function that computes MAPE from two 1-D lists containing target labels and their 
    relevant predictions from the model
    Parameters:
        actuals: list containing target labels
        predictions: list containing predictions for said labels
    Returns:
        float containing MAPE rounded to 2 decimals 
    """
    actual, pred = np.array(actual), np.array(pred)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    return round(mape,2) # round to 2 decimals

def cross_plot_pred_data(df_backup, plot_pred_r, plot_actual_r):
    """
    Function used to plot lists of target labels and their predictions from the model
    in a shared axis system
    Parameters:
        df_backup: backup dataframe taken when reading csvs
        plot_actual: list containing target labels
        plot_pred: list containing predictions for said labels
    Returns: None
    """
    warnings.filterwarnings("ignore")

    # Get dates for plotting (test_data)
    datesx = list(df_backup[df_backup['year'] == 2021]['Date'])

    # And finally we can see that our network has done a decent job of estimating!
    fig, ax = plt.subplots(figsize=(16,4))

    # convert list of lists to list (np.hstack)
    # get every N-th value of predicion and data liiiiiist and display them in common plot
    N = 1000
    ax.plot(plot_pred_r[::N], label='Prediction Data')
    ax.plot(plot_actual_r[::N], label='Actual Data')
    ax.set_xticklabels(datesx[::N], rotation=45)
    ax.legend();
    plt.show()

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
    # with mlflow.start_run(run_name='model', nested=True) as mlrun:

        global click_params
        click_params = ClickParams(kwargs)

        print("=================== Click Params ===================")
        print(vars(click_params))
        
        #  read csv files
        df = read_csvs()
        
        df_backup = df.copy()

        # split data in train/test/validation
        train_test_valid_split(df)

        # train model with hparams set to best_params of optuna 
        plot_pred, plot_actual = train_test_best_model()
        # plot_pred, plot_actual = sklearn_regress()

        # calculate MAPE
        mape_value = mape(plot_actual,plot_pred)
        print(f"Mean Absolute Percentage Error (MAPE): {mape_value} %")

        # plot prediction/actual data on common axis system 
        cross_plot_pred_data(df_backup, plot_pred, plot_actual)


if __name__ == '__main__':
    print("\n=========== Forecasing Model =============")
    logging.info("\n=========== Forecasing Model =============")
    forecasting_model()