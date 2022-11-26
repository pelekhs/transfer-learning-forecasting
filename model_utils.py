import pandas as pd
from enum import IntEnum
from datetime import datetime
import pathlib
from matplotlib import pyplot as plt
# import matplotlib as mpl
# mpl.use("webagg")
import warnings
import sys
import copy

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.profiler import Profiler, AdvancedProfiler
from torchmetrics import MeanAbsolutePercentageError

import pytorch_lightning as pl

import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Globals ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
click_params = None
train_data = None
test_data = None 
val_data = None

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Classes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
class DataScaler:
    def __init__(self, min=0,max=1):
        super(DataScaler, self).__init__()
        self.min = min
        self.max = max
        self.scaler = preprocessing.MinMaxScaler(feature_range=(min, max))

    def scale_transform(self, dframe, columns=['Load']):
        temp_df = copy.deepcopy(dframe)
        temp_df['Load'] = self.scaler.fit_transform(dframe[columns])
        return temp_df

    def inverse_scale_transform(self, dframe, columns=['Load']):
        temp_df = copy.deepcopy(dframe)
        temp_df[columns] = self.scaler.inverse_transform(dframe[columns])
        return temp_df

#### Enum used to define transfer learning techniques used  
class Transfer(IntEnum):
    NO_TRANSFER = 0
    WARM_START = 1 
    BOUNDED_EPOCHS = 2
    FREEZING = 3
    HEAD_REPLACEMENT = 4

#### Class used to store and utilize click arguments passed to the script
class ClickParams:
    def __init__(self, click_args):
        super(ClickParams, self).__init__()
        
        for key, value in click_args.items():
            if(isinstance(value, bool)):
               setattr(self, key, value); continue 

            split_value = value.split(',') # split string into a list delimited by comma

            if not split_value: # if empty list
                sys.exit(f"Empty value for parameter {key}")
            
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

#### Define the model and hyperparameters

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
    def __init__(self, **params):
        super(Regression, self).__init__()

        self.loss = MeanAbsolutePercentageError() #MAPE

        # by default, no transfer learning, init in case of optuna/ensemble        
        self.transfer_mode = Transfer.NO_TRANSFER 
        if('transfer_mode' in params): 
            self.transfer_mode = Transfer(params['transfer_mode'])

        # enable Lightning to store all the provided arguments 
        # under the self.hparams attribute. 
        # These hyperparameters will also be stored within the model checkpoint
        self.save_hyperparameters()

        input_dim = self.hparams.l_window #input dim set to lookback_window
        output_dim = self.hparams.f_horizon #output dim set to f_horizon
        
        # create datasets used by dataloaders
        # 'subset'_X: dataset containing features of subset (train/test/validation) dataframe
        # 'subset'_Y: dataset containing targets of subset (train/test/validation) dataframe
        global train_data, test_data, val_data
        self.train_X, self.train_Y = feature_target_split(train_data,input_dim,output_dim)  
        self.test_X, self.test_Y = feature_target_split(test_data,input_dim,output_dim)
        self.validation_X, self.validation_Y = feature_target_split(val_data,input_dim,output_dim)          

        """
        feature_extractor: all layers before classifier
        classifier: last layer connecting output with rest of network (not always directly)
        We load proper pretrained model, and use its feauture_extractor for the new untrained one
        (Also check forward pass commentary)
        """
        self.feature_extractor = None        
        self.classifier = None

        if(self.transfer_mode == Transfer.NO_TRANSFER):
            print(f'input_dim: {input_dim}')
            feature_layers, last_dim = self.make_hidden_layers()
            self.feature_extractor = nn.Sequential(*feature_layers) #list of nn layers
            print(f'last_dim: {last_dim}')
            self.classifier = nn.Linear(last_dim, output_dim)
        else:
            # get best pretrained model and load its feature extractor
            best_run = self.search_proper_run()
            model = mlflow.pytorch.load_model(f"runs:/{best_run.info.run_id}/model") 

            # get dimension of last hidden layer of model (one before output)
            last_dim = model.hparams.layer_sizes[-1]
            self.feature_extractor = model.feature_extractor # "old" feature extractor
            self.classifier = nn.Linear(last_dim, output_dim) # new classifier

            if(self.transfer_mode == Transfer.WARM_START):
                print('Using \"WARM START\" technique...')
                self.classifier = model.classifier # "old" classifier

            if(self.transfer_mode == Transfer.FREEZING):
                print('Using \"FREEZING\" technique...') # freeze feature layers
                # self.feature_extractor.eval() # disable dropout/BatchNorm layers using eval()
                self.feature_extractor.requires_grad_(False) # freeze params
                # self.feature_extractor.freeze() 

    def search_proper_run(self):
        # find best pretrained model by searching runs with best performance (MAPE)
        # (NOT CORRECT, CHECK RUNS BASED ON WHAT CSVS ARE USED + PERFORMANCE) 
        filter_string = f'tags.stage = "model"' # search only runs in 'model' stage
        best_run = MlflowClient().search_runs(experiment_ids="0",
                                              filter_string=filter_string,
                                              run_view_type=ViewType.ACTIVE_ONLY,
                                              max_results=1,
                                              order_by=["metrics.MAPE DESC"]
                                            #   order_by=['end_time DESC']
                                              )[0]
        print(f"Found best model run with ID: {best_run.info.run_id}")
        return best_run

    def make_hidden_layers(self):
        """
        Each loop is the setup of a new layer
        At each iteration:
            1. add previous layer to the next (with parameters gotten from layer_sizes)
                    at first iteration previous layer is input layer
            2. add activation function
            3. set current_layer as next layer
        connect last layer with cur_layer
        Parameters: None
        Returns: 
            layers: list containing input layer through last hidden one
            cur_layer: size (dimension) of the last hidden layer      
        """
        layers = [] # list of layer to add at nn
        cur_layer = self.hparams.l_window

        for next_layer in self.hparams.layer_sizes: 
            layers.append(nn.Linear(cur_layer, next_layer))
            layers.append(getattr(nn, self.hparams.activation)()) # nn.activation_function (as suggested by Optuna)
            cur_layer = next_layer #connect cur_layer with previous layer (at first iter, input layer)
        return layers, cur_layer

    # Perform the forward pass
    def forward(self, x):
        """
        In forward pass, we pass input through (freezed or not) feauture extractor
        and then its output through the classifier 
        """
        representations = self.feature_extractor(x)
        return self.classifier(representations)

### The Data Loaders ###     
    # Define functions for data loading: train / validate / test
    def train_dataloader(self):
        feature = torch.tensor(self.train_X.values).float() #feature tensor train_X
        target = torch.tensor(self.train_Y.values).float() #target tensor train_Y
        train_dataset = TensorDataset(feature, target)  # dataset bassed on feature/target
        train_loader = DataLoader(dataset = train_dataset, 
                                  shuffle=True, 
                                  batch_size = self.hparams.batch_size)
        return train_loader
            
    def test_dataloader(self):
        feature = torch.tensor(self.test_X.values).float()
        target = torch.tensor(self.test_Y.values).float()
        test_dataset = TensorDataset(feature, target)
        test_loader = DataLoader(dataset = test_dataset, 
                                 batch_size = self.hparams.batch_size)
        return test_loader

    def val_dataloader(self):
        feature = torch.tensor(self.validation_X.values).float()
        target = torch.tensor(self.validation_Y.values).float()
        val_dataset = TensorDataset(feature, target)
        validation_loader = DataLoader(dataset = val_dataset,
                                       batch_size = self.hparams.batch_size)
        return validation_loader

    def predict_dataloader(self):
        return self.test_dataloader()
    
### The Optimizer ### 
    # Define optimizer function: here we are using ADAM
    def configure_optimizers(self):
        return getattr(optim, self.hparams.optimizer_name)( self.parameters(),
                                                            # momentum=0.9, 
                                                            # weight_decay=1e-4,                   
                                                            lr=self.hparams.l_rate)

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

## Loading Data
def read_csvs(click_params):
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
    global train_data, test_data, val_data

    train_data = df[~df['year'].isin(['2020','2021'])][['Load']] 
    test_data = df[df['year'] == 2021][['Load']]
    val_data = df[df['year'] == 2020][['Load']]

    # print("dataframe shape: {}".format(df.shape))
    # print("train shape: {}".format(train_data.shape))
    # print("test shape: {}".format(test_data.shape))
    # print("validation shape: {}".format(val_data.shape))
    return train_data, test_data, val_data

def feature_target_split(df, lookback_window=168, forecast_horizon=36):# lookback_window: 168 = 7 days(* 24 hours)
    """
    This function gets a column of a dataframe and splits it to input and target
    
    **lookback_window**
    In a for-loop of 'lookback_window' max iterations, starting from 0 
    At N-th iteration (iter): 
        1. create a shifted version of 'Load' column by N rows (vertically) and 
        2. stores it in a column* (feature_'N')

    Same pseudo-code for 'forecast_horizon' loop
    
    *At first iterations of both loops, the new columns (feature/target) are going to be firstly created
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
        'subset'_X: pandas.dataframe containing features of df after preprocess for model
        'subset'_Y: pandas.dataframe containing targets of df after preprocess for model
    -------
    """

    # Reset "Unnamed: 0" based on concated df 
    df['Unnamed: 0'] = range(1, len(df) + 1)

    df_copy = df.copy()    

    df_new = {}
        
    for inc in range(0,int(lookback_window)):
        df_new['feature_' + str(inc)] = df_copy['Load'].shift(-inc)

    # shift 'load' column permanently for as many shifts happened at 'lookback_window' loops  
    df_copy['Load'] = df_copy['Load'].shift(-int(lookback_window))
                    
    for inc in range(0,int(forecast_horizon)):
        df_new['target_' + str(inc)] = df_copy['Load'].shift(-inc)    
    
    df_new = pd.DataFrame(df_new, index=df_copy.index)
    df_new = df_new.dropna().reset_index(drop=True)    
                        
    # store new dataset to csv (if needed) 
    df_new.to_csv("preprocess_for_model.csv")
    
    return df_new.iloc[:,:lookback_window] , df_new.iloc[:,-forecast_horizon:]

def cross_plot_pred_data(df_backup, plot_pred_r, plot_actual_r, tmpdir):
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

    # display it on window/browser tab
    # plt.show()

    plt.savefig(f"{tmpdir}/cross_plot.png")
    plt.close()

def calculate_metrics(plot_actual,plot_pred):

    # Evaluate the model prediction
    metrics = {
        "MSE" : mean_squared_error(plot_actual, plot_pred, squared=True),
        "RMSE" : mean_squared_error(plot_actual, plot_pred, squared=False),
        "MAE" : mean_absolute_error(plot_actual, plot_pred),
        "MAPE" : mean_absolute_percentage_error(plot_actual, plot_pred),
        "R_squared": r2_score(plot_actual, plot_pred)
    }

    print("  Metrics: ")
    for key, value in metrics.items():
        print("    {}: {}".format(key, value))
    print(f"Mean Absolute Percentage Error (MAPE): {metrics['MAPE']*100} %")

    return metrics
