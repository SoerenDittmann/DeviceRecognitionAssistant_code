#import keras
import numpy as np
#import tensorflow as tf
from scipy.stats import entropy


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, fbeta_score, confusion_matrix




def custom_f1(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average = 'micro', labels = np.unique(y_true))
    return f1

def custom_fbeta(y_true, y_pred, beta):
    f1 = fbeta_score(y_true, y_pred, average = 'micro', beta=beta, labels = np.unique(y_true))
    return f1

