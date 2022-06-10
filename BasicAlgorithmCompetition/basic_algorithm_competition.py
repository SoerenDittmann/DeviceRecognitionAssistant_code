#%%
if __name__ == "__main__":   #for tsfresh to work on windows everything has to be indented and inside this if-loop. It does matter whether the file is saved or not
    import pandas as pd
    import numpy as np
    import os
    
    import time
    from pickle import load
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from sklearn.tree import DecisionTreeClassifier

    from keras.utils import to_categorical
    from tensorflow.random import set_seed

    from sktime.classification.compose import TimeSeriesForestClassifier
    from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier, ElasticEnsemble, ProximityForest
    from sktime.classification.dictionary_based import BOSSEnsemble
    from sktime.classification.hybrid import HIVECOTEV1
    from sktime.classification.shapelet_based import ShapeletTransformClassifier
    from sktime.classification.interval_based import RandomIntervalSpectralForest
    
    from tsfresh import extract_features, select_features
    from tsfresh.utilities.dataframe_functions import impute
    
    from custom_classifiers.classifiers import Classifier_RESNET
    from data_preprocessing.utils_data_container import build_sktime_data, sktime_to_tsfresh_converter, build_nn_data
    from data_preprocessing.utils_data_handling import data_stats
    
    #%% Load data, determine random seed & train-test-split
    
    #load working directory
    fileDirectory = os.path.dirname(__file__)
    #create relative path to sensor dictionary
    sensor_dic_path = os.path.join(fileDirectory, r'data_preprocessing\sensor_dic.pkl')   
    #load .pkl file
    with open(sensor_dic_path, 'rb') as filehandle:
        # read the data as binary data stream
        sensor_dic = load(filehandle)
    
    ts_length = 1000
    
    #build nested DataFrame in appropriate format
    keys_sorted, all_data = build_sktime_data(sensor_dic, ts_length) 
    
    random_seed = 6345    #set random seed to use for all train-test-splits. Ensures comparability of TSFresh, sktime and NN approaches.
    
    #-----------------Train-Test-Split-----------------
    # With stratified sampling!!!
    X_train, X_test, y_train, y_test = train_test_split(all_data['X'], all_data['y'], 
                                                                        train_size=0.65,
                                                                        test_size=0.35,
                                                                        stratify = all_data['y'],
                                                                        random_state = random_seed)
    
    #Classifier requires DataFrame as X-Input
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    
    #check whether train-test-split worked properly 
    print ("X_train: ", X_train.shape, type(X_train))
    print ("y_train: ", y_train.shape, type(y_train))
    print("X_test: ", X_test.shape, type(X_test))
    print ("y_test: ", y_test.shape, type(y_test))
    
    overall_stats = data_stats(keys_sorted,all_data['y'])
    tt_data_stats = data_stats(keys_sorted,y_train,y_test)
    
    print(overall_stats)
    print(tt_data_stats)
    
    #build table to save results in
    header = np.array(['training time','inference time','accuracy'])
    results_table = pd.DataFrame(columns=header)

#%% 
    
    def measure_times(classifier, X_train, y_train, X_test, y_test):
            
        start_train = time.time()
        classifier.fit(X_train, y_train)
        stop_train = time.time()
        duration_train = (stop_train - start_train)/60    
        
        start_inference = time.time()
        preds = classifier.predict(X_test)
        stop_inference = time.time()
        duration_inference = (stop_inference - start_inference)/60    
        
        cm = confusion_matrix(y_test, preds)
        acc = cm.diagonal().sum()/len(y_test)
        
        model_name = str(classifier)
        
        return duration_train, duration_inference, acc, model_name
    #%% KNN with dtw
    #-----------------knn for time-series classification-----------------
    #Series length is m, number of series is n and number of classes is c
    
    knn = KNeighborsTimeSeriesClassifier() #was n_neighbors=2, metric="dtw"
    
    duration_train, duration_inference, acc, model_name = measure_times(knn, X_train, y_train, X_test, y_test)
    
    knn_results = pd.DataFrame(data=[[duration_train, duration_inference, acc]], columns = header, index=[model_name])
    results_table = results_table.append(knn_results)
    
    #%% HIVECOTE
    #-----------------HIVE COTE-----------------
    # time complexity O(n²m^4)
    
    cote = HIVECOTEV1(random_state = 1398563, n_jobs=-1)
    
    duration_train, duration_inference, acc, model_name = measure_times(cote, X_train, y_train, X_test, y_test)
    
    cote_results = pd.DataFrame(data=[[duration_train, duration_inference, acc]], columns = header, index=[model_name])
    results_table = results_table.append(cote_results)
    
    #%% ElasticEnsemble
    #-----------------ElasticEnsemble Classifier-----------------
    # time complexity O(n²m²), 3h and dtw_distance finished, 7h no ddtw_distance finished ffs
    
    ee = ElasticEnsemble(random_state = 7223364, n_jobs=-1)
    
    duration_train, duration_inference, acc, model_name = measure_times(ee, X_train, y_train, X_test, y_test)
    
    ee_results = pd.DataFrame(data=[[duration_train, duration_inference, acc]], columns = header, index=[model_name])
    results_table = results_table.append(ee_results)
    
    #%% Shapelet Transform
    #-----------------Shapelet-Tranform-Classifier---------------
    # time complexity O(n²m^4), time = 1.13min, accuracy 41.2%
    
    stc = ShapeletTransformClassifier(time_contract_in_mins = 15, random_state = 93844) #threshold=0, time_contract_in_mins=10, n_estimators=500
    
    duration_train, duration_inference, acc, model_name = measure_times(stc, X_train, y_train, X_test, y_test)
    
    stc_results = pd.DataFrame(data=[[duration_train, duration_inference, acc]], columns = header, index=[model_name])
    results_table = results_table.append(stc_results)
    
    #%% RISE
    #-----------------Random Interval Spectral Ensemble-----------------
    #time complexity O(nm²), 0.97min 52.9% accuracy
    rise = RandomIntervalSpectralForest(random_state = 103, n_jobs=-1)

    duration_train, duration_inference, acc, model_name = measure_times(rise, X_train, y_train, X_test, y_test)
    
    rise_results = pd.DataFrame(data=[[duration_train, duration_inference, acc]], columns = header, index=[model_name])
    results_table = results_table.append(rise_results)   
    
    #%% TS Forest
    #-----------------Time-series Forest built-in-----------------
    #time complexity O(rmn log n) - r: number of trees, 0.05min 47.1% accuracy
    tsf = TimeSeriesForestClassifier(random_state = 1932948, n_jobs=-1)

    duration_train, duration_inference, acc, model_name = measure_times(tsf, X_train, y_train, X_test, y_test)
    
    tsf_results = pd.DataFrame(data=[[duration_train, duration_inference, acc]], columns = header, index=[model_name])
    results_table = results_table.append(tsf_results)
    
    #%% BOSS Ensemble
    #-----------------BOSS Ensemble-----------------
    # time complexity O(nm(n-w)) - w: window length, 22.6min 38.2% accuracy on m = 1000
    boss = BOSSEnsemble(random_state = 39485, n_jobs=-1)

    duration_train, duration_inference, acc, model_name = measure_times(boss, X_train, y_train, X_test, y_test)
    
    boss_results = pd.DataFrame(data=[[duration_train, duration_inference, acc]], columns = header, index=[model_name])
    results_table = results_table.append(boss_results)
    
    #%% Proximity Forest
    #----------------Proximity Forest-----------------
    # time complexity O(k*nlog(n)*r*c*l²) - k: number of trees in the forest, r: candidate splits, c: ..., l: ...
    # n_estimators = 100 (default), >1h for 2 trees
    pf = ProximityForest(random_state = 12453, n_jobs=-1) #n_estimators=1, n_jobs=2, verbosity = 1, n_stump_evaluations = 1
    
    # indices of the rows need to be integers instead of str
    X_train_pf, X_test_pf = X_train.reset_index(drop=True), X_test.reset_index(drop=True)
    duration_train, duration_inference, acc, model_name = measure_times(pf, X_train_pf, y_train, X_test, y_test)
    
    pf_results = pd.DataFrame(data=[[duration_train, duration_inference, acc]], columns = header, index=[model_name])
    results_table = results_table.append(pf_results)

    #%% DecisionTree based on features
    
    #use own converter from nested to long, automatic from_nested_to_long seems buggy
    X_long = sktime_to_tsfresh_converter(all_data)
    
    #------------Extract features from sensor data

    X = extract_features(X_long, column_id="id", column_value="value", impute_function = impute)
    X.head()
    
    y = all_data['y'].to_numpy()
    
    # y_train / y_test from this train-test-split are equivalent to the ones from the other train/test split
    # y_train from the non-featurized tt-split contains the data about source and sensor for transferability calculations
    X_train_features, X_test_features, y_train_features, _ = train_test_split(X, y, train_size=0.65, test_size=0.35,
                                                                              stratify = y, random_state = random_seed)
    

    #check whether train-test-split worked properly    
    print ("X_train_features: ", X_train_features.shape, type(X_train_features))
    print ("y_train_features: ", y_train.shape, type(y_train))
    print("X_test_features: ", X_test_features.shape, type(X_test_features))
    print ("y_test_features: ", y_test.shape, type(y_test))
    
    overall_stats_features = data_stats(sensor_dic.keys(),all_data['y'])
    tt_data_stats_features = data_stats(sensor_dic.keys(),y_train,y_test)
    
    print(overall_stats_features)
    print(tt_data_stats_features)
    
    
    #Build classifier from features
    classifier_selected_multi = DecisionTreeClassifier(random_state = 9953765)

    #Start training time
    start_train = time.time()
    
    #Build and select features that are considered significant
    X_train_filtered_multi = select_features(X_train_features, y_train_features, multiclass=True, n_significant=1)
    
    #Fit model & measure time
    classifier_selected_multi.fit(X_train_filtered_multi, y_train)
    stop_train = time.time()
    duration_train = (stop_train - start_train)/60    
    
    #Build X_test
    X_test_filtered_multi = X_test_features[X_train_filtered_multi.columns]
    
    #Measure time and predict
    start_inference = time.time()
    preds = classifier_selected_multi.predict(X_test_filtered_multi)
    stop_inference = time.time()
    duration_inference = (stop_inference - start_inference)/60    
    
    #calculate accuracy
    cm = confusion_matrix(y_test, preds)
    acc = cm.diagonal().sum()/len(y_test)
    
    #save model name
    model_name = str(classifier_selected_multi)
    
    #Add results to results_table
    TSFresh_results = pd.DataFrame(data=[[duration_train, duration_inference, acc]], columns = header, index=[model_name])
    results_table = results_table.append(TSFresh_results)

    #%% Residual Networks

    #set random seed 
    set_seed(2)
    
    keys_sorted, X, y = build_nn_data(sensor_dic, ts_length)
 
    #z-normalization of input data
    #mean = np.mean(X_prenorm,axis=1)
    #std = np.std(X_prenorm,axis=1)
    #X = (X_prenorm - mean[:,None])/std[:,None]
    
    #train-test-split on nn-data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.65,test_size=0.35,
                                                        random_state=random_seed, stratify = y)
    
    #Check whether tt-split worked
    print ("X_train: ", X_train.shape, type(X_train))
    print ("y_train: ", y_train.shape, type(y_train))
    print("X_test: ", X_test.shape, type(X_test))
    print ("y_test: ", y_test.shape, type(y_test))
    
    tt_data_stats = data_stats(keys_sorted,y_train,y_test)
    print(tt_data_stats)
    
    
    #define output directory for model    
    output_dir = fileDirectory
    
    #nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    nb_classes = len(np.unique(y, axis=0))

    # transform the labels from integers to one hot vectors
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)
    
    if len(X_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    #Determine input shape to be able to build network
    input_shape = X_train.shape[1:]
        
    resnet = Classifier_RESNET(output_dir, input_shape, nb_classes, custom_metric = False, threshold = 1, verbose=2)

    epochs = 200 #5s each epoch --> 200 more realistic
    
    #fit and measure time
    start_train = time.time()
    resnet.fit(X_train, y_train, X_test, y_test, y_true, epochs)
    stop_train = time.time()
    duration_train = (stop_train - start_train)/60    
    
    #predict with measuring time
    start_inference = time.time()
    acc = resnet.predict(X_test, y_true, X_train, y_train, y_test, return_df_metrics=True)    
    stop_inference = time.time()
    duration_inference = (stop_inference - start_inference)/60    
        
    #save model name
    model_name = str(resnet)

    #Add results to results_table    
    ResNet_results = pd.DataFrame(data=[[duration_train, duration_inference, acc]], columns = header, index=[model_name])
    results_table = results_table.append(ResNet_results)
