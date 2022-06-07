#%% Importing Packages
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
from sktime.classification.hybrid import HIVECOTEV1
from sktime.classification.hybrid import HIVECOTEV2
from sktime.contrib.vector_classifiers._rotation_forest import RotationForest
from sktime.transformations.panel.summarize import RandomIntervalFeatureExtractor
from sktime.utils.slope_and_trend import _slope

#Import own Packages
from Classification.custom_classifiers.classifiers import RISErejectOption_entropy, custom_fbeta
from Classification.custom_classifiers.utils import build_sktime_data, calc_accuracy, data_stats
from Classification.data_handling.basics import read_in_data, handling_data, map_to_plaintext_labels
#from Classification.custom_classifiers.train_model import rise_training

#%% General
# Orga: Define ring
duration = 2000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)

# Technical: Define max. time series length
series_length = 1000 #Time series length for predictions

#%% 1. Load OS Data
##############################################################################

#%% 1.1GVL and Load OS Data

#First load spydata df into workspace: 20210530_Diss_Data_DataFile.spydata
#Required manual import
try:
    sensor_dic
except NameError:
    print("Please import OS Data")


#%% Multi-class predictions - base
#-----------------------------------------------------------

#%% Implement Multiclass prediction for OS-Data

#build Dataframe in correct format for sktime. Sensor length set to 1000 in build_sktime_data
keys_sorted, all_data = build_sktime_data(sensor_dic, series_length)
X, y = all_data['X'], all_data['y']

#Test train split with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, stratify=y, random_state=123)

#Convert to DataFrame
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

#Print overall and test/train stats
overall_stats = data_stats(keys_sorted, all_data['y'])
tt_data_stats = data_stats(keys_sorted, y_train, y_test)



#%% Multiclass RISE - base
#-----------------Random Interval Spectral Ensemble-----------------
"""
ATTENTION: THE NEW VERSION OF RISE needed in l.95 the definition of the 
max_interval parameter (even though not used here). Manually added to the
classifiers file
"""

#algorithm parameter definition
n_estimators=200
min_interval=16
#max_interval = 22
acf_lag=100
acf_min_values=4
threshold = 1

rise = RISErejectOption_entropy(random_state=123,
                                n_estimators=n_estimators,
                                min_interval=min_interval,
                                #max_interval=max_interval,
                                acf_lag=acf_lag,
                                acf_min_values=acf_min_values,
                                threshold = threshold)

# Fit model
rise.fit(X_train, y_train)
# Predict labels for test-set
y_pred = rise.predict(X_test) 
# Store posterior probabilities for classes
y_probas = rise.predict_proba(X_test)

#Calculate Confusion Matrix: 57% of the predicted classes are correct
cm = confusion_matrix(y_test, y_pred)


#check which datasets have been classified incorrectly and have not been rejected
cond = ((y_pred != y_test) & (y_pred != len(np.unique(y))))
misclassified = X_test[cond]
misclassified_probas = y_probas[cond]


#calc precision of baseline model
true_pos = np.diag(cm)
false_pos = np.sum(cm, axis=0) - true_pos

#Exclude zero divisors from precision calc
divisor_precision = true_pos + false_pos

#Calc average class precision
precision = np.average(true_pos[divisor_precision != 0]/divisor_precision[divisor_precision != 0])
