# -*- coding: utf-8 -*-
"""
Created on Sat Febr 26 17h 2022
@author: VWHU0EL
"""

"""
Description of purpose of the script
Based from the Comparison of Multiclass vs. Binary class:
    - Most interessting classes are_
        - Vibration sensor data (most data av)
        - Power Sensor data (worst performance from Multiclass optimizer)
"""

"""
1. Overview available TSC algos that are of relevance:
    1. Features of time series:
        - Vibration: High frequent oszilating time series
        - Power: Very different data structures

Chosen Algortihms based on Table 8 from The great time series classification 
bake off and Training time analysis from MPM Thesis

1. HIVE-COTE
2. Shapelet Transform
3. Elastic Ensemble
4. Time CNN based on https://arxiv.org/pdf/1809.04356.pdf section 5.4
5. RISE

"""
#%%Import packages
import pandas as pd
import numpy as np
import time
from pickle import load
import glob
import matplotlib.pyplot as plt
import copy
import scipy
from sklearn.metrics import plot_confusion_matrix
import seaborn as sn
import tensorflow
import winsound

#Import Sklearn Packages
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Import Sktime
#from sktime.classification.compose import TimeSeriesForestClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier, ElasticEnsemble, ProximityForest
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.hybrid import HIVECOTEV2
from sktime.transformations.panel.shapelet_transform import (RandomShapeletTransform)
from sktime.transformations.panel.summarize import RandomIntervalFeatureExtractor
from sktime.contrib.vector_classifiers._rotation_forest import RotationForest
from sktime.utils.slope_and_trend import _slope

#Import own Packages
from Classification.custom_classifiers.classifiers import RISErejectOption_entropy, custom_fbeta
from Classification.custom_classifiers.utils import build_sktime_data, calc_accuracy, data_stats
from Classification.data_handling.basics import read_in_data, handling_data, map_to_plaintext_labels



#%% Prepare data bases for binary class predictions

#Required manual import of sensor dic
try:
    sensor_dic
except NameError:
    print("Please import OS Data")
    
#%% GVL
series_length = 1000

#%% Data preprocessing
#build Dataframe in correct format for sktime. Sensor length set to 1000 in build_sktime_data
keys_sorted, all_data = build_sktime_data(sensor_dic, series_length)
X, y = all_data['X'], all_data['y']


#%%Restructure database for individual binary class predictions

#Initiate dict with [sensorname]{[sensorname], [other]} as tag structure
database = {}

selected_keys = ['vibration_sensor', 'power_sensor']

#Iterate over all the available sensor types
for el in selected_keys:
    sensor_type = el
    #copy complete dict once again under tag for the sensor type
    database[sensor_type] = copy.deepcopy(sensor_dic)
    #Declare a list of sensor types that are stored under new tag other and can be removed
    del_list = []
    data = pd.DataFrame()

    
    for key in keys_sorted:    
        if (key != sensor_type):
            if 'other' not in database[sensor_type]:
                #case 1: add new key 'other' to dict
                database[sensor_type]['other'] = copy.deepcopy(database[sensor_type][key])
                #print('Sensortype initialisierung: '+sensor_type)
                #print('Key initialisierung: '+key)
                #for dataframes in database[sensor_type][key]:
                #    print('Df initialisierung: '+dataframes)

            else:
                #case 2: other is initialised and the dataframe needs to be joined to the existing df under other 
                for df in database[sensor_type][key]:
                    
                    if df in database[sensor_type]['other']:
                        
                        #print('Key if: '+key)
                        #print('Df if: '+df)
                            
                        data = database[sensor_type][key][df]
                        data = pd.DataFrame(data)
                        database[sensor_type]['other'][df] = pd.DataFrame(database[sensor_type]['other'][df]).join(data)
                        data = pd.DataFrame()
                    
                    #case 3: other initialized and dfs to be added not existing
                    
                    else:
                        database[sensor_type]['other'][df] = pd.DataFrame(database[sensor_type][key][df])
                        #print('Key else: '+key)
                        #print('Df else: '+df)
                        data_neu = database[sensor_type][key][df]
                        

            #fill del_list    
            del_list.append(key)

        
    #del keys that are now stores in other
    for e in del_list:
        database[sensor_type].pop(e, None)
        
#%%HIVE-COTE BASE VIBRATION

#Implemennt HIVE COTE BASE for Vibration data
keys_sorted, all_data = build_sktime_data(database['vibration_sensor'], series_length)
X, y = all_data['X'], all_data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.6, test_size= 0.4, stratify=y, random_state=123)


X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

clf = HIVECOTEV2(
    stc_params={
        "estimator": RotationForest(n_estimators=3),
        "n_shapelet_samples": 500, 
        "max_shapelets": 20,
        "batch_size": 100,
    },
    drcif_params={"n_estimators": 10},
    arsenal_params={"num_kernels": 100, "n_estimators": 5},
    tde_params={
        "n_parameter_samples": 25,
        "max_ensemble_size": 5,
        "randomly_selected_params": 10,
    },
    random_state=123
)

#Train model
clf.fit(X_train, y_train)

#Predict
y_pred = clf.predict(X_test)

cm_vibration = confusion_matrix(y_test, y_pred)

#%%HIVE-COTE BASE VIBRATION: ANALYSIS OF RESULTS

#Merge y_test (real class) and y_pred (predicted class)
y_test_test = copy.deepcopy(y_test)
y_pred_test = pd.DataFrame(y_pred)


y_test_test= pd.DataFrame(y_test_test)
y_test_test= y_test_test.assign(pred=y_pred_test[0].values)

#Res is the df storing all cases, where algorithm predicted 1 but true class is 0
res = y_test_test[(y_test_test['y'] == 0) & (y_test_test['pred'] == 1)]


#%% Analysis results
"""
Missed (for current state)

Acutal 1 but predicted 0:
    cwr_default0hp1797rpm:x262_fe_time
    femto_vib_c1:horizontal_acceleration
    sensiml:accelerometerz
    
Critical case: Actual 0 but predicted 1
    sensiml:gyroscopey
    

"""



#%%% HIVE-COTE OPTIMIZATION (ROBUSTNESS CHECK) FOR LATER
#Based on Thesis from MP Mathieu
"""
#Implemennt HIVE COTE OPTIMZED for Vibration data
keys_sorted, all_data = build_sktime_data(database['vibration_sensor'], series_length)
X, y = all_data['X'], all_data['y']


#generate random integers to fix random_seed for each model in the inner loop of the nested CV
beta = 0.5
n_outer_splits = 3
n_inner_splits = 3
rng = np.random.default_rng(seed=231984)
rise_random_seeds = rng.integers(low=0, high=1000000, size = n_outer_splits)

#------------------------Nested-CV RISE-----------------------------------------------
print("Results for NCV with entropy threshold")
#reset iteration variable i
i = 0
#create individual scorer
fpoint5_scorer = make_scorer(custom_fbeta, beta=beta)
# configure the cross-validation procedure
cv_outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
# enumerate splits
outer_results = list()
for train_ix, test_ix in cv_outer.split(X,y):
    # split data
    X_train, X_test = all_data['X'][train_ix], all_data['X'][test_ix]
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train, y_test = all_data['y'][train_ix], all_data['y'][test_ix]
    #verify data structure
    #tt_data_stats = data_stats(database['vibration_sensor'].keys(),y_train,y_test)
    #print(tt_data_stats)
    # configure the cross-validation procedure
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
    # define the model
    model = HIVECOTEV2(random_state=rise_random_seeds[i], n_jobs=-1)
    # define search space
    space = dict()
    space['stc_params'] = [{"estimator": RotationForest(n_estimators=3), "n_shapelet_samples": 500, "max_shapelets": 20, "batch_size": 100},
                           {"estimator": RotationForest(n_estimators=3), "n_shapelet_samples": 100, "max_shapelets": 4, "batch_size": 20},
                           {"estimator": RotationForest(n_estimators=3), "n_shapelet_samples": 750, "max_shapelets": 30, "batch_size": 150}]
    space['drcif_params'] = [{"n_estimators": 10},
                             {"n_estimators": 2},
                             {"n_estimators": 15}]    
    space['arsenal_params'] = [{"num_kernels": 100, "n_estimators": 5},
                               {"num_kernels": 20, "n_estimators": 2},
                               {"num_kernels": 150, "n_estimators": 8}]
    space['tde_params'] = [{"n_parameter_samples": 25, "max_ensemble_size": 5, "randomly_selected_params": 10},
                            {"n_parameter_samples": 5, "max_ensemble_size": 2, "randomly_selected_params": 2},
                            {"n_parameter_samples": 38, "max_ensemble_size": 8, "randomly_selected_params": 15}]
    # define search
    search = GridSearchCV(model, space, cv=cv_inner, refit=True)
    # execute search
    result = search.fit(X_train, y_train)
    # get the best performing model fit on the whole training set
    best_model = result.best_estimator_
    # evaluate model on the hold out dataset
    yhat = best_model.predict(X_test)
    # evaluate the model
    fbeta = custom_fbeta(y_test, yhat, beta)
    # store the result
    outer_results.append(fbeta)
    #iterate
    i += 1
    # report progress
    print(f'>Score on test set: fbeta_micro={fbeta:.3f}, Mean score best_estimator validation set ={result.best_score_:.3f}, cfg={result.best_params_}')
# summarize the estimated performance of the model
print(f'Fbeta-micro: {np.mean(outer_results):.3f} Std: {np.std(outer_results):.3f}')
"""