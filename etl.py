#!/usr/bin/env python
# coding: utf-8

# File Preprocessing 

import pytz
from datetime import datetime
from datetime import timezone
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import holidays
import logging
import warnings

from darts import TimeSeries
from darts.timeseries import TimeSeries
import os
import pycountry
import dateutil

import pathlib
import click
import shutil
import mlflow

"""
Αρχικά ορίζεται η συνάρτηση read_and_validate_input που διαβάζει τα αρχεία και ελέγχει αν είναι σωστά. 
Πιο συγκεκριμένα, προκύπτει Exception αν το αρχείο δεν έχει ημερομηνίες σε αύξουσα σειρά 
(που ενδεχομένως σημαίνει ότι έχει διαβαστεί με λάθος τρόπο ή λάθος τύπο ημερομηνίας) 
ή αν δεν έχει τον σωστό αριθμό ή όνομα στήλων (2 στήλες με ονόματα Date και Load στο csv)
"""

class DatesNotInOrder(Exception):
    """
    Exception raised if dates in series_csv are not sorted.
    """
    def __init__(self):
        super().__init__("Dates in series_csv are not sorted. Check date format in input csv.")

class WrongColumnNames(Exception):
    """
    Exception raised if series_csv has wrong column names.
    """
    def __init__(self, columns):
        self.message = "series_csv must have 2 columns named Date and Load."
        self.columns = columns
        super().__init__(self.message)
    def __str__(self):
        return f'Column names provided: {self.columns}. {self.message}'

def read_and_validate_input(series_csv: str = "../../RDN/Load_Data/2009-2019-global-load.csv",
                            day_first: str = "true"):
    """
    Validates the input after read_csv is called and
    throws apropriate exception if it detects an error

    Parameters
    ----------
    series_csv
        The path to the local file of the series to be validated
    day_first
        Whether to read the csv assuming day comes before the month

    Returns
    -------
    pandas.DataFrame
        The resulting dataframe from series_csv
    """
    ts = pd.read_csv(series_csv,
                        delimiter=',',
                        header=0,
                        index_col=0,
                        parse_dates=True,
                        dayfirst=day_first)
    print(ts.index)
    ts.sort_index(ascending=True, inplace=True)
    print(ts.index)
    print(ts.index.sort_values())
    if not ts.index.sort_values().equals(ts.index):
        raise DatesNotInOrder()
    elif not (len(ts.columns) == 1 and 
              ts.columns[0] == 'Load' and ts.index.name == 'Date'):
        raise WrongColumnNames(list(ts.columns) + [ts.index.name])
    return ts


"""
Έπειτα καλούμε την συνάστηση read_csvs για να διαβάσουμε τα αρχεία και να επεξεργαστούμε με κατάλληλο τρόπο. 
Πιο συγκεκριμένα, κάποια αρχεία (DE-LU MBA.csv, Ukraine BEI CTA.csv, Ukraine IPS CTA.csv) δεν είχαν τα δεδομένα τους με χρονολογική σειρά. 
Έτσι, καλείται η συνάρτηση sort_csvs με σκοπό την ταξινόμηση των ημερομηνιών σε αύξουσα σειρά.

Τα αρχεία δεν έχουν σταθερό όνομα ανάλογα με την χώρα στην οποία ανήκει το καθένα. 
Για αυτό, γίνεται χρήση της βιβλιοθήκης pycountry για να γίνει αυτόματη αναγνώριση της χώρας του κάθε αρχείου.
 Έτσι, αποθηκεύεται στο λεξικό country_ts το όνομα κάθε χώρας και η λίστα από τα αρχεία της.

Καποιες χώρες (Ιταλία, Ουκρανία, Δανία, Νορβηγία, Σουηδία) είναι χωρισμένες σε πολλαπλές BZ (Betting Zones). 
Αυτό έχει ως αποτέλεσμα μεγαλύτερες διακυμάνσεις στα δεδομένα και προβλήματα σχετικά με το ότι 
μια χώρα με πολλές χρονοσειρές θα επηρέαζε τα αποτελέσματα του υπερμοντέλου περισσότερο από ότι θα έπρεπε 
(πχ η Ιταλία με 6 χρονοσειρές θα επηρέαζε περισσότερο το αποτέλεσμα από ότι η Γερμανία με μία χρονοσειρά). 
Για αυτό, στην συνάρτηση save_results, γίνεται άθροιση των csvs που αναφέρονται στην ίδια χώρα. 
Τελικά, δηλαδή, έχουμε 1 csv για κάθε χώρα, το οποίο αποθηκεύεται στον tmp_folder.
"""

#Reads each file and sorts its dates in ascending order
def sort_csvs(series_csv: str = "Load_Data/2009-2019-global-load.csv",
             day_first: str = "true"):
    date_parser = lambda x: dateutil.parser.parse(x, ignoretz=True)
    ts = pd.read_csv(series_csv,
                    delimiter=',',
                    header=0,
                    index_col=0,
                    names=['Date', 'Load'],
                    usecols=[1,2],
                    parse_dates=True,
                    date_parser=date_parser,
                    dayfirst=day_first)
    csvname = os.path.basename(series_csv)
    return ts.sort_values(by=['Date'], ascending=True)

def merge_country_csvs(): # previously know 'read_csvs'
    dir_in = click.get_current_context().params["dir_in"] # make click argument accessible by function
    country_names = list(map(lambda country : country.name, pycountry.countries))
    country_codes = list(map(lambda country : country.alpha_2, pycountry.countries))
    country_ts = {}
    country_file_to_name = {}
    
    for root, dirs, files in os.walk(dir_in):
        for name in files:
            if os.path.splitext(name)[1] == ".csv":
                print(f"~~~Now processing file: {name}~~~")
                logging.info(f"~~~Now processing file: {name}~~~")
                #Sorting the ts first
                ts = sort_csvs(os.path.join(root, name))
                ts = ts[~ts.index.duplicated(keep='first')]
                ts = ts.asfreq('1H')
                #Candidate countries that this file might belong to. Checking all countries and country codes.
                #"BZ" and "BA" is never a country code in this context
                #Because full country names are checked first they have priority
                cand_countries = [country for country in country_names + ["Czech Republic"] + country_codes if country in name and not country in ["BZ", "BA"]]
                if len(cand_countries) >= 1:
                    #Always choose the first candidate country
                    first = pycountry.countries.search_fuzzy(cand_countries[0])[0].name
                    print(f"File {name} belongs to {first}")
                    logging.info(f"File {name} belongs to {first}")
                    #Append the ts to the chosen country's list
                    if first in country_ts:              
                        country_ts[first].append(ts)
                    else:
                        country_ts[first] = [ts]
                    country_file_to_name[name] = pycountry.countries.search_fuzzy(cand_countries[0])[0].name
                else:
                    print(f"No match for {name}")
                    logging.info(f"No match for {name}")

    return country_ts, country_file_to_name

def save_results(country_ts, country_file_to_name):
    multiple_ts = []
    names = []
    tmp_folder = click.get_current_context().params["tmp_folder"] # make click argument accessible by function

    for country in country_ts:
        ts = country_ts[country][0]
        #Add all ts's of the same country together
        for df in country_ts[country][1:]:
            ts = ts + df
        ts.to_csv(f"{tmp_folder}{country}.csv")
        multiple_ts.append(ts)
        names.append(f"{tmp_folder}{country}.csv")
    return multiple_ts, names

# country_ts, country_file_to_name = merge_country_csvs()
# multiple_ts, names = save_results(country_ts, country_file_to_name)

"""
Σε πολλές χρονοσειρές υπάρχουν μηδενικές τιμές, και επίσης πολλά άλλα outliers (δηλαδή τιμές πολύ μεγαλύτερες ή πολύ μικρότερες από τις γειτονικές τους)
 τα οποία πιθανόν οφείλονται σε λάθος και θα προκαλούσαν προβλήματα στην εκπαίδευση των μοντέλων, και στο MinMaxScaling. 
 Για αυτό γίνεται χρήση της συνάρτησης remove_outliers η οποία αντικαθιστά τις προαναφερθείσες τιμές με NaNs. 

Η συνάρτηση αυτή, αφού υπολογίσει τον μέσο όρο και την τυπική απόκλιση κάθε μήνα στο dataset, 
αφαιρεί όσες τιμές βρίσκονται περισσότερο από std_dev τυπικές αποκλίσεις μακριά από τον μέσο όρο αυτόν. 
Ως default τιμή του std_dev επιλέχθηκε το 4.5 έτσι ώστε να αφαιρούνται μόνο οι τιμές που βρίσκονται πολλές τυπικές αποκλίσεις μακριά από τον μέσο όρο,
οι οποίες είναι το πιθανότερο να είναι λανθασμένες, και να κρατούνται στο dataset οι τιμές που βρίσκονται λιγότερες τυπικές αποκλίσεις μακριά από τον μέσο όρο,
οι οποίες είναι πιθανότερο να είναι μακριά από τον μέσο όρο απλά λόγω κάποιου εξωτερικού γεγονότος (πχ καιρού). 

Επιπλέον, αφού οι σειρές αυτές δεν εμφανίζουν τάση, δεν χρειάζεται να πραγματοποιηθεί detrend.
"""

def remove_outliers(ts: pd.DataFrame, 
                    name: str = "Portugal", 
                    std_dev: float = 4.5, 
                    print_removed: bool = True):
    """
    Reads the input dataframe and replaces its outliers with NaNs by removing 
    values that are more than std_dev standard deviations away from their 1 month 
    mean or both. This function works with datasets that have NaN values.

    Parameters
    ----------
    ts
        The pandas.DataFrame to be processed
    name
        The name of the country to be displayed on the plots
    std_dev
        The number to be multiplied with the standard deviation of
        each 1 month  period of the dataframe. The result is then used as
        a cut-off value as described above
    print_removed
        If true it will print the removed values
        
    Returns
    -------
    pandas.DataFrame
        The original dataframe with its outliers values replaced with NaNs
    """
    #Dates with NaN values are removed from the dataframe
    ts = ts.dropna()
    #Removing all non postive values
    a = ts.loc[ts["Load"] <= 0]
    #Calculating monthly mean and standard deviation and removing values 
    #that are more than std_dev standard deviations away from the mean
    mean_per_month = ts.groupby(lambda x: x.month).mean().to_numpy()
    mean = ts.index.to_series().apply(lambda x: mean_per_month[x.month - 1])
    std_per_month = ts.groupby(lambda x: x.month).std().to_numpy()
    std = ts.index.to_series().apply(lambda x: std_per_month[x.month - 1])
    a = pd.concat([a, ts.loc[-std_dev * std + mean > ts['Load']]])
    a = pd.concat([a, ts.loc[ts['Load'] > std_dev * std + mean]])

    #Plotting Removed values and new series
    a = a.sort_values(by='Date')
    a = a[~a.index.duplicated(keep='first')]
    if print_removed: print(f"Removed outliers from {name}: {a.shape[0]} rows")

#     fig, ax = plt.subplots(figsize=(8,5))
#     ax.plot(ts.index, ts['Load'], color='black', label = f'Load of {name}')
#     ax.scatter(a.index, a['Load'], color='blue', label = 'Removed Outliers')
#     plt.legend()
#     plt.show()
    res = ts.drop(index=a.index)
#     fig, ax = plt.subplots(figsize=(8,5))
#     ax.plot(res.index, res['Load'], color='black', label = f'Load of {name}')
#     plt.legend()
#     plt.show()
    return res.asfreq('1H')

def read_csvs():
    tmp_folder = click.get_current_context().params["tmp_folder"] # make click argument accessible by function
    multiple_ts = []
    names = []
    result= {}
    raw_result = {}
    for root, dirs, files in os.walk(tmp_folder):
        for name in files:
            ts = read_and_validate_input(os.path.join(root, name))
            ts = ts[~ts.index.duplicated(keep='first')]
            ts = ts.asfreq('1H')
            # print("Removing outliers...")
            result[name] = remove_outliers(ts, name)
            raw_result[name] = ts
    return result, raw_result

# result, raw_result = read_csvs()


"""
Όπως φαίνεται παραπάνω, ως outliers θεωρούνται μόνο ακραία υψηλές και χαμηλές τιμές που είναι το πιο πιθανό να οφείλονται σε λάθος των δεδομένων.
Έτσι, για παράδειγμα, οι υψηλές τιμές που παρατηρήθηκαν στην Ελλάδα το καλοκαίρι του 2021 λόγω των υψηλών θερμοκρασιών για μεγάλο χρονικό διάστημα διατηρούνται,
παρόλο που είναι πολύ υψηλές για την εποχή τους. Ταυτόχρονα όμως η πολύ υψηλή τιμή τον Ιανουάριο του 2022 στην Γαλλία θεωρείται outlier, 
αφού είναι ακραία υψηλότερη από τους γείτονές της, και έτσι είναι πολύ πιθανό να οφείλεται σε λάθος 
"""


import holidays
from pytz import timezone
import pytz
from calendar import isleap
def isholiday(x, holiday_list):
    if x in holiday_list:
        return True
    return False


def isweekend(x):
    if x == 6 or x == 0:
        return True
    return False


def create_calendar(timeseries, timestep_minutes, holiday_list, local_timezone):
    local_tz = click.get_current_context().params["local_tz"] # make click argument accessible by function

    calendar = pd.DataFrame(
        timeseries.index.tolist(),
        columns=['datetime']
    )
        
    calendar['year'] = calendar['datetime'].apply(lambda x: x.year)
    calendar['month'] = calendar['datetime'].apply(lambda x: x.month)
    calendar['yearweek'] = calendar['datetime'].apply(
        lambda x: int(x.strftime("%V")) - 1)
    calendar['day'] = calendar['datetime'].apply(lambda x: x.day)
    calendar['hour'] = calendar['datetime'].apply(lambda x: x.hour)
    calendar['minute'] = calendar['datetime'].apply(lambda x: x.minute)
    calendar['second'] = calendar['datetime'].apply(lambda x: x.second)
    calendar['weekday'] = calendar['datetime'].apply(lambda x: x.weekday())
    calendar['monthday'] = calendar['datetime'].apply(
        lambda x: int(x.strftime("%d")) - 1)
    calendar['weekend'] = calendar['weekday'].apply(lambda x: isweekend(x))
    calendar['yearday'] = calendar['datetime'].apply(
        lambda x: int(x.strftime("%j")) - 1)

    if(local_tz):
        calendar['timestamp'] = calendar['datetime'].apply(
            lambda x: x.timestamp()).astype(int)
    else:
        # first convert to utc and then to timestamp
        calendar['timestamp'] = calendar['datetime'].apply(
            lambda x: local_timezone.localize(x).replace(tzinfo=pytz.utc).timestamp()).astype(int)

    # national_holidays = Province(name="valladolid").national_holidays()
    # regional_holidays = Province(name="valladolid").regional_holidays()
    # local_holidays = Province(name="valladolid").local_holidays()
    # holiday_list = national_holidays + regional_holidays + local_holidays

    calendar['holiday'] = calendar['datetime'].apply(
        lambda x: isholiday(x.date(), holiday_list))
    WNweekday = calendar['datetime'].apply(
        lambda x: x.weekday() if not isholiday(x.date(), holiday_list) else 5 if x.weekday() == 4 else 6) 
    calendar['WN'] = WNweekday + calendar['hour']/24 + calendar['minute']/(24*60)
    return calendar

"""
Τέλος ορίζεται η συνάρτηση imputation για αντικατάσταση των NaNs με αριθμητικές τιμές, η οποία αποτελεί υλοποίηση του: 
https://ieeexplore.ieee.org/abstract/document/7781213. Αυτό γίνεται χρησιμοποιώντας τον σταθμισμένο μέσο όρο ιστορικών δεδομένων και γραμμικής παρεμβολής. 
Τα βάρη της κάθε μεθόδου εξαρτώνται εκθετικά από την απόσταση της τρέχουσας NaN τιμής από την κοντινότερη ημερομηνία που έχει τιμή σύμφωνα με τους τύπους:
                               w = e^(a * di), result = w * L + (1 - w) * Η
όπου L η γραμμική παρεμβολή και H τα ιστορικά δεδομένα. Επομένως, όσο αυξάνεται η απόσταση αυτή, 
μειώνεται η συνεισφορά της γραμμικής παρεμβολής και αυξάνεται η συνεισφορά των ιστορικών δεδομένων. 

Τα ιστορικά δεδομένα για μια τιμή NaN (εδώ τροποποιείται λίγο ο ορισμός του paper) ορίζονται ως 
ο μέσος όρος των τιμών της ίδιας μέρας της εβδομάδας που επίσης έχουν λιγότερο από WNcutoff (default τιμή 1 λεπτό) απόσταση από την ώρα της NaN,
 λιγότερο από Ycutoff (default τιμή 3 χρόνια) απόσταση από τον χρόνο της NaN και λιγότερο από YDcutoff (default τιμή 30 μέρες) απόσταση από την ημέρα του χρόνου της NaN.

Τα holidays θεωρούνται είτε Σάββατα(αν η πραγματική μέρα ειναι Παρασκευή) ή Κυριακές(σε κάθε άλλη περίπτωση)
"""

def impute(ts: pd.DataFrame, 
           holidays, 
           max_thhr: int = 48, 
           a: float = 0.3, 
           WNcutoff: float = 1/(24 * 60), 
           Ycutoff: int = 3, 
           YDcutoff: int = 30, 
           debug: bool = False): 
    """
    Reads the input dataframe and imputes the timeseries using a weighted average of historical data
    and simple interpolation. The weights of each method are exponentially dependent on the distance
    to the nearest non NaN value. More specifficaly, with increasing distance, the weight of
    simple interpolation decreases, and the weight of the historical data increases. If there is
    a consecutive subseries of NaNs longer than max_thhr, then it is not imputed and returned with NaN
    values.

    Parameters
    ----------
    ts
        The pandas.DataFrame to be processed
    holidays
        The holidays of the country this timeseries belongs to
    max_thhr
        If there is a consecutive subseries of NaNs longer than max_thhr, 
        then it is not imputed and returned with NaN values
    a
        The weight that shows how quickly simple interpolation's weight decreases as 
        the distacne to the nearest non NaN value increases 
    WNcutoff
        Historical data will only take into account dates that have at most WNcutoff distance 
        from the current null value's WN(Week Number)
    Ycutoff
        Historical data will only take into account dates that have at most Ycutoff distance 
        from the current null value's year
    YDcutoff
        Historical data will only take into account dates that have at most YDcutoff distance 
        from the current null value's yearday
    debug
        If true it will print helpfull intermediate results
        
    Returns
    -------
    pandas.DataFrame
        The imputed dataframe
    """
    
    #Returning calendar of the country ts belongs to
    calendar = create_calendar(ts, 60, holidays, timezone("UTC"))
    calendar.index = calendar["datetime"]
    
    #null_dates: Series with all null dates to be imputed
    null_dates = ts[ts["Load"].isnull()].index
    
    if debug:
        for date in null_dates:
            print(date)
    
    #isnull: An array which stores whether each value is null or not
    isnull = ts["Load"].isnull().values
    
    #d: List with distances to the nearest non null value
    d = [len(null_dates) for _ in range(len(null_dates))]
    
    #leave_nan: List with all the values to be left NaN because there are
    #more that max_thhr consecutive ones
    leave_nan = [False for _ in range(len(null_dates))]
    
    #Calculating the distances to the nearest non null value that is earlier in the series
    count = 1
    for i in range(len(null_dates)):
        d[i] = min(d[i], count)
        if i < len(null_dates) - 1:
            if null_dates[i+1] == null_dates[i] + pd.offsets.DateOffset(hours=1):
                count += 1
            else:
                count = 1
                
    #Calculating the distances to the nearest non null value that is later in the series
    count = 1
    for i in range(len(null_dates)-1, -1, -1):
        d[i] = min(d[i], count)
        if i > 0:
            if null_dates[i-1] == null_dates[i] - pd.offsets.DateOffset(hours=1):
                count += 1
            else:
                count = 1

    # max_consecutive_nans = 0

    #We mark this subseries so that it does not get imputed
    for i in range(len(null_dates)):
        if d[i] == max_thhr // 2:
            for ii in range(max(0, i - max_thhr // 2 + 1), min(i + max_thhr // 2, len(null_dates))):
                leave_nan[ii] = True
        elif d[i] > max_thhr // 2:
            if(d[i] > max_consecutive_nans):
                max_consecutive_nans = d[i]
            leave_nan[i] = True
                
    #This is the interpolated version of the time series
    ts_interpolatied = ts.interpolate(inplace=False)
    
    #We copy the time series so that we don't change it while iterating
    res = ts.copy()
    
    for i, null_date in enumerate(null_dates):
        if leave_nan[i]: continue
        
        #WN: Day of the week + hour/24 + minute/(24*60). Holidays are handled as
        #either Saturdays(if the real day is a Friday) or Sundays(in every other case)
        currWN = calendar.loc[null_date]['WN']
        currYN = calendar.loc[null_date]['yearday']
        currY = calendar.loc[null_date]['year']
        
        #weight of interpolated series, decreases as distance to nearest known value increases
        w = np.e ** (-a * d[i])
        
        #Historical value is calculated as the mean of values that have at most WNcutoff distance to the current null value's
        #WN, Ycutoff distance to its year, and YDcutoff distance to its yearday
        historical = ts[(~isnull) & ((calendar['WN'] - currWN < WNcutoff) & (calendar['WN'] - currWN > -WNcutoff) &\
                                    (calendar['year'] - currY < Ycutoff) & (calendar['year'] - currY > -Ycutoff) &\
                                    (((calendar['yearday'] - currYN) % (365 + calendar['year'].apply(lambda x: isleap(x))) < YDcutoff) |\
                                    ((-calendar['yearday'] + currYN) % (365 + calendar['year'].apply(lambda x: isleap(x))) < YDcutoff)))]["Load"]
        
        if debug: print("~~~~~~Date~~~~~~~",null_date, "~~~~~~~Dates summed~~~~~~~~~~",historical,sep="\n")
        
        historical = historical.mean()
        
        #imputed value is calculated as a wheighted average of the histrorical value and the value from intrepolation
        res.loc[null_date] = w * ts_interpolatied.loc[null_date] + (1 - w) * historical
        
        # if debug:
        #     print(res.loc[null_date])

    # print("----------------------------- NaN values in dataframe: "+str(res.isnull().sum().sum()))

    # merge callendar with res
    res_with_calendar = res.merge(calendar,left_index=True, right_index=True, how = 'inner')
  
    return res_with_calendar, null_dates

def test_impute_function(country, result, max_thhr, a):
    code = compile(f"holidays.{country}()", "<string>", "eval")
    holidays_ = eval(code)
    res, null_dates = impute(result[f"{country}.csv"], holidays_, max_thhr, a, debug=False)
    
    res_ts = TimeSeries.from_dataframe(res)
    res_ts_with_na = TimeSeries.from_dataframe(result[f"{country}.csv"])
    prev = null_dates[0]
    for curr in null_dates:
        if curr > prev + pd.offsets.DateOffset(hours=1):
            _, example_imp = res_ts.split_before(max(pd.Timestamp(prev - pd.offsets.DateOffset(days=4)), res.index[0] + pd.offsets.DateOffset(hours=1)))
            example_imp, _ = example_imp.split_before(min(pd.Timestamp(prev + pd.offsets.DateOffset(days=4)), res.index[-1] - pd.offsets.DateOffset(hours=1)))
            _, example_no_imp = res_ts_with_na.split_before(max(pd.Timestamp(prev - pd.offsets.DateOffset(days=4)), res.index[0] + pd.offsets.DateOffset(hours=1)))
            example_no_imp, _ = example_no_imp.split_before(min(pd.Timestamp(prev + pd.offsets.DateOffset(days=4)), res.index[-1] - pd.offsets.DateOffset(hours=1)))
            example_imp.plot(label="Interpolation", new_plot=True)
            example_no_imp.plot(label="Original")
        prev = curr
    _, example_imp = res_ts.split_before(max(pd.Timestamp(prev - pd.offsets.DateOffset(days=4)), res.index[0] + pd.offsets.DateOffset(hours=1)))
    example_imp, _ = example_imp.split_before(min(pd.Timestamp(prev + pd.offsets.DateOffset(days=4)), res.index[-1] - pd.offsets.DateOffset(hours=1)))
    _, example_no_imp = res_ts_with_na.split_before(max(pd.Timestamp(prev - pd.offsets.DateOffset(days=4)), res.index[0] + pd.offsets.DateOffset(hours=1)))
    example_no_imp, _ = example_no_imp.split_before(min(pd.Timestamp(prev + pd.offsets.DateOffset(days=4)), res.index[-1] - pd.offsets.DateOffset(hours=1)))
    example_imp.plot(label="Interpolation", new_plot=True)
    example_no_imp.plot(label="Original")

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

from pytz import country_timezones

def utc_to_local(temp_df, country_code):
    # Get dictionary of countries and their timezones
    timezone_countries = {country: timezone 
                            for country, timezones in country_timezones.items()
                            for timezone in timezones}
    local_timezone = timezone_countries[country_code]

    # reset timezone info
    # temp_df['Start'] = temp_df['Start'].dt.tz_convert(None)

    # convert dates to given timezone, get timezone info
    temp_df['Start'] = temp_df['Start'].dt.tz_convert(local_timezone)

    # remove timezone information-naive, because next localize() recquires it 
    # but keep dates to local timezone
    temp_df['Start'] = temp_df['Start'].dt.tz_localize(None)

    # localize based on timezone, ignore daylight saving time, shift forward if ambiguous datetimes
    temp_df['Start'] = temp_df['Start'].dt.tz_localize(local_timezone,
                                                        ambiguous=np.array([False] * temp_df.shape[0]),
                                                        nonexistent='shift_forward')
    # crop datetime format (remove '+timezone' suffix)
    # becomes problemaric when there are multiple timezones on same country
    # for setting frequency at imputation
    temp_df['Start'] = temp_df['Start'].dt.strftime('%Y-%m-%d %H:%M:%S')

def store_with_local_time():
    dir_in = click.get_current_context().params["dir_in"] # make click argument accessible by function
    tmp_folder = click.get_current_context().params["tmp_folder"] # make click argument accessible by function

    country_ts = {}

    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S%z')
    
    # For every csv file in folder 
    for root, dirs, files in os.walk(dir_in):
        for name in files:
            print("Loading dataset: "+name)
            temp_df = pd.read_csv(os.path.join(root, name),
                                parse_dates=['Start'],
                                dayfirst=True,
                                date_parser=dateparse)
            country_name, country_code = find_country(name)
            # temp_df['country'], temp_df['code']  = country_name, country_code                                        
            utc_to_local(temp_df, country_code) # convert 'Start' column from utc to local datetime 
            temp_df.rename(columns={"Start": "Date"}, errors="raise", inplace=True) #rename column 'Start' to 'Date'
            temp_df = temp_df[['Date','Load']] # keep only 'Date' and 'Load' columns
            temp_df.set_index('Date', inplace=True) #set 'Date' as index (neeed for 'read_and_validate...') 
            temp_df.sort_index(ascending=True, inplace=True)

            # dictionary country_ts has a key for every country and value
            # the appended dataframes of all csv files related to that country
            if country_name in country_ts: 
                country_df = country_ts[country_name]
                country_df = pd.concat([country_df,temp_df]) 
                # country_ts[country_name].append(temp_df)
            else:
                country_ts[country_name] = temp_df.copy()

    for country in country_ts:
        country_ts[country].to_csv(f"{tmp_folder}{country}.csv")

            # if country_name in name:
            #     temp_df.to_csv(f"{tmp_folder}{name}")
            #     print(f'Store to file \"{name}\"')
            # else:
            #     new_name = name.replace(country_code,country_name)
            #     temp_df.to_csv(f"{tmp_folder}{new_name}")
            #     print(f'Store to file \"{new_name}\"')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Click ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Remove whitespace from your arguments
@click.command(
    help="Given a CSV file (see load_raw_data), transforms it into Parquet "
    "in an mlflow artifact called 'ratings-parquet-dir'"
)
@click.option("--dir_in",
    type=str,
    default="../original_data/",
    help="Local directory where original timeseries csvs are stored"
)
@click.option("--dir_out",
    type=str,
    default="../preprocessed_data/",
    help="Local directory where preprocessed timeseries csvs will be stored"
)
@click.option("--tmp_folder",
    type=str,
    default="../tmp_data/",    
    help="Temporary directory to store merged-countries data with no outliers (not yet imputed)"
)
@click.option("--local_tz",
    type=bool,
    default=True,    
    help="flag if you want local (True) or UTC (False) timezone"
)
# @click.option("--csv_names", '-csv', 
#     default='csv_all',   
#     multiple=True,
#     help="csv files I want to use for preprocesssing" 
# )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Main ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def etl(dir_in, dir_out, tmp_folder, local_tz):

    with mlflow.start_run(run_name='etl', nested=True) as etl_start:
        mlflow.set_tag("stage", "etl")

        # countries = csv_names.split(',')
        print(dir_in, dir_out)
        # print(countries)

        if not os.path.exists(dir_out): os.makedirs(dir_out) #create output directory if not exists

        # set as global because they are used by most function
        global country_ts, country_file_to_name, multiple_ts, names

        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder) #create tmp directory if not exists

            """
            if we want our data to be based on local timezone of 
            their country of origin
            """   
            if(local_tz):
                store_with_local_time()
            else: 
                # these two lines where outside of function in vanilla script
                country_ts, country_file_to_name = merge_country_csvs()
                multiple_ts, names = save_results(country_ts, country_file_to_name)

        result, raw_result = read_csvs() # read csv files and remove outliers

        for root, dirs, files in os.walk(tmp_folder):
            for name in files:
                # if any(string in countries for string in ('csv_all', csv.stem)):
                print(f'Inputing data of csv: "{name}"')
                logging.info(f'\nInputing data of csv: "{name}"')
                
                country = None
                if(local_tz):
                    country, country_code = find_country(name)
                else:
                    country = pathlib.Path(name).stem #find country of csv (its name given the circumstances)
                code = compile(f"holidays.{country}()", "<string>", "eval") #find country's holidays
                holidays_ = eval(code)
                res, null_dates = impute(result[f"{name}"], holidays_, 5000, 0.2, debug=False) # impute datasets and

                print(f'Stored to "{dir_out}{name}"')
                logging.info(f'Stored to "{dir_out}{name}"')
                res.to_csv(f"{dir_out}{name}") #store them on seperate folder

        # delete temp folder and all its contents
        try:
            shutil.rmtree(tmp_folder)
        except OSError as e:
            print ("Error: %s - %s." % (e.filename, e.strerror))

        # mlflow tags
        mlflow.set_tag("run_id", etl_start.info.run_id)
        mlflow.set_tag("stage", "etl")
        mlflow.set_tag('series_uri', f'{etl_start.info.artifact_uri}/features/series.csv')
        mlflow.set_tag("dir_out", dir_out)
        mlflow.set_tag("dir_in", dir_in)        

if __name__ == '__main__':
    print("\n=========== ETL =============")
    logging.info("\n=========== ETL =============")
    etl()