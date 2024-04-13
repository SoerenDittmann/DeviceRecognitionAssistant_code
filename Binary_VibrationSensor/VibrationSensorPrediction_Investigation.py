#%%Import packages
import pandas as pd
import numpy as np
import time
from pickle import load
import glob
#import matplotlib.pyplot as plt
import copy
import scipy
#from sklearn.metrics import plot_confusion_matrix
#import seaborn as sn
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
from sktime.classification.sklearn._rotation_forest import RotationForest
from sktime.utils.slope_and_trend import _slope

#Import own Packages
from Classification.custom_classifiers.classifiers import RISErejectOption_entropy, custom_fbeta
from Classification.custom_classifiers.utils import build_sktime_data, calc_accuracy, data_stats
#from Classification.data_handling.basics import read_in_data, handling_data, map_to_plaintext_labels



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

selected_keys = ['vibration_sensor']

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

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.6, test_size= 0.4, stratify=y, random_state=122)


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

