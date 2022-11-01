#!/usr/bin/env python
# coding: utf-8

# This is (to almost its entirety) copied from this excellent article 
# [here](https://coderzcolumn.com/tutorials/machine-learning/simple-guide-to-optuna-for-hyperparameters-optimization-tuning)

# Let's import some basic libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("webagg")
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
# from pytorch_lightning.profiler import Profiler, AdvancedProfiler
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import MeanAbsolutePercentageError

import logging
import pickle
import click
import mlflow
import sys
import os

import torchensemble 
from torchensemble.utils import io
import re

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
                else: setattr(self, key, value); continue;

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

    print("dataframe shape: {}".format(df.shape))
    print("train shape: {}".format(train_data.shape))
    print("test shape: {}".format(test_data.shape))
    print("validation shape: {}".format(validation_data.shape))

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
        # self.loss = nn.L1Loss() # MAE 
        self.loss = MeanAbsolutePercentageError() #MAPE
        # self.loss = nn.MSELoss() # MSE
        self.optimizer_name = params['optimizer_name']
        # self.dropout = params['dropout']
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
            3. add dropout 
            4. set current_layer as next layer
        connect last layer with cur_layer
        """
        for cur_layer in self.output_dims: 
            layers.append(nn.Linear(input_dim, cur_layer))
            layers.append(getattr(nn, self.activation)()) # nn.activation_function (as suggested by Optuna)
            # layers.append(nn.Dropout(self.dropout))
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
        return getattr(optim, self.optimizer_name)(self.parameters(), lr=self.l_rate)

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

def cross_plot_pred_data(df_backup, plot_pred, plot_actual):
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
    ax.plot(plot_pred[::N], label='Rescaled Prediction')
    ax.plot(plot_actual[::N], label='Rescaled Actual Data')
    ax.set_xticklabels(datesx[::N], rotation=45)
    ax.legend();
    plt.show()

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

        # use ensemble algorithms to find averaged prediction 
        plot_pred, plot_actual = ensemble(load=False)

        # calculate MAPE
        mape_value = mape(plot_actual,plot_pred)
        print(f"Mean Absolute Percentage Error (MAPE): {mape_value} %")

        # plot prediction/actual data on common axis system 
        cross_plot_pred_data(df_backup, plot_pred, plot_actual)

if __name__ == '__main__':
    print("\n=========== Forecasing Model =============")
    logging.info("\n=========== Forecasing Model =============")
    forecasting_model()