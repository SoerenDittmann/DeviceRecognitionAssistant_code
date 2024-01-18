#%%
if __name__ == "__main__":   #for tsfresh to work on windows everything has to be indented and inside this if-loop. It does matter whether the file is saved or not
    import pandas as pd
    import numpy as np
    import os
    
    from pickle import load
    
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.tree import DecisionTreeClassifier

    from keras.utils import to_categorical
    from tensorflow.random import set_seed

    from sktime.classification.compose import TimeSeriesForestClassifier
    from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
    from sktime.classification.dictionary_based import BOSSEnsemble
    from sktime.classification.shapelet_based import ShapeletTransformClassifier
    from sktime.classification.interval_based import RandomIntervalSpectralForest

    
    from tsfresh import extract_features, select_features
    from tsfresh.utilities.dataframe_functions import impute
    
    from custom_classifiers.classifiers import Classifier_RESNET
    from data_preprocessing.utils_data_container import build_sktime_data, sktime_to_tsfresh_converter, build_nn_data
    from data_preprocessing.utils_data_handling import data_stats, getSource_buildKeys
    
    
    #%% Load data, determine random seed & train-test-split
    
    
    #load working directory
    fileDirectory = os.path.dirname(__file__)
    #create relative path to sensor dictionary
    sensor_dic_path = os.path.join(fileDirectory, r'data_preprocessing\sensor_dic.pkl')   
    sensor_dic_normalized_path = os.path.join(fileDirectory, r'data_preprocessing\sensor_dic_normalized.pkl')   

    #load .pkl file
    with open(sensor_dic_path, 'rb') as filehandle:
        # read the data as binary data stream
        sensor_dic = load(filehandle)
        
    #load .pkl file
    with open(sensor_dic_normalized_path, 'rb') as filehandle:
        # read the data as binary data stream
        sensor_dic_normalized = load(filehandle)
    
    ts_length = 1000
    
    #----Build data for Sktime
    #unnormalized
    keys_sorted, all_data = build_sktime_data(sensor_dic, ts_length)
    X, y = all_data['X'], all_data['y']

    #normalized
    _, all_data_normalized = build_sktime_data(sensor_dic_normalized, ts_length)
    X_norm, _ = all_data_normalized['X'], all_data_normalized['y']
    
    #----Build data for TSFresh
    X_long = sktime_to_tsfresh_converter(all_data)
    #extract features
    X_features = extract_features(X_long, column_id="id", column_value="value", impute_function = impute)
    y_features = all_data['y'].to_numpy()
    
    #----Build data for ResNet
    keys_sorted_nn, X_nn, y_nn = build_nn_data(sensor_dic_normalized, ts_length)    
   
    #List to store all results_table in
    results = []



#%% 
    from sklearn.metrics import confusion_matrix
    
    def measure_KPIs(classifier, X_train, y_train, X_test, y_test):
        
        #train & predict
        classifier.fit(X_train, y_train)
        preds = classifier.predict(X_test)
        
        #calculate accuracy
        cm = confusion_matrix(y_test, preds)
        acc = cm.diagonal().sum()/len(y_test)

        #split indices of y in sources and sensors. Add source and label to key for check
        sources_y_test, sensors_y_test, y_test_keys = getSource_buildKeys(y_test)
        _, _, y_train_keys = getSource_buildKeys(y_train)
        
        #check whether keys of test_dataset have already been in train_dataset
        present_in_training = np.isin(y_test_keys,y_train_keys).flatten()
        
        correct_pred = y_test.to_numpy() == preds #calculate correct predictions
    
        #check all possible combinations of present_in_training and correct_pred
        cm = confusion_matrix(present_in_training, correct_pred)    
               
        #calculate score = correctly transfered / all transfer tasks 
        transferability = (cm[0,1])/(cm[0,:].sum())
                
        
        model_name = str(classifier)
        
        return acc, transferability, model_name
        
#%%
    #determine the number of splits and repeats
    n_splits = 3
    n_repeats = 3
    number_of_algorithms = 7

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state = 2857347)
        
    
    #produce series of random but reproducible seeds to use during loops
    rng = np.random.default_rng(seed=231984)
    loop_random_seeds = rng.integers(low=0, high=1000000, size = n_splits*n_repeats*number_of_algorithms)  

    i = 0
    
    for train_index, test_index in rskf.split(X, y):
        
        #build table to save results of each iteration results
        header = np.array(['accuracy','transferability'])
        results_table = pd.DataFrame(columns=header)        


        #------------Use Train-Test-Split-----------------
        #unnormalized
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        #Sktime classifier require DataFrame as X-Input
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        
        
        #normalized
        X_norm_train, X_norm_test = X_norm[train_index], X_norm[test_index]
        
        #Sktime classifier require DataFrame as X-Input
        X_norm_train = pd.DataFrame(X_norm_train)
        X_norm_test = pd.DataFrame(X_norm_test)
        
        
        #Check for proper tt-split
        print ("X_train: ", X_train.shape, type(X_train))
        print ("y_train: ", y_train.shape, type(y_train))
        print("X_test: ", X_test.shape, type(X_test))
        print ("y_test: ", y_test.shape, type(y_test))
        
        overall_stats = data_stats(keys_sorted,all_data['y'])
        tt_data_stats = data_stats(keys_sorted,y_train,y_test)

        print(overall_stats)
        print(tt_data_stats)
              
        
        
        # KNN with dtw
        #-----------------knn for time-series classification-----------------
        #Series length is m, number of series is n and number of classes is c
        
        knn = KNeighborsTimeSeriesClassifier() #was n_neighbors=2, metric="dtw"
        
        acc, transferability, model_name = measure_KPIs(knn, X_train, y_train, X_test, y_test)
        
        knn_results = pd.DataFrame(data=[[acc, transferability]], columns = header, index=[model_name])
        results_table = results_table.append(knn_results)

        # Shapelet Transform
        #-----------------Shapelet-Tranform-Classifier---------------
        # time complexity O(n²m^4), time = 1.13min, accuracy 41.2%
        stc = ShapeletTransformClassifier(time_contract_in_mins = 15, random_state = loop_random_seeds[i+1]) #threshold=0, time_contract_in_mins=10, n_estimators=500
        
        acc, transferability, model_name = measure_KPIs(stc, X_norm_train, y_train, X_norm_test, y_test)
        
        stc_results = pd.DataFrame(data=[[acc, transferability]], columns = header, index=[model_name])
        results_table = results_table.append(stc_results)
        
        # RISE
        #-----------------Random Interval Spectral Ensemble-----------------
        #time complexity O(nm²), 0.97min 52.9% accuracy
        rise = RandomIntervalSpectralForest(random_state = loop_random_seeds[i+2], n_jobs=-1)
    
        acc, transferability, model_name = measure_KPIs(rise, X_train, y_train, X_test, y_test)
        
        rise_results = pd.DataFrame(data=[[acc, transferability]], columns = header, index=[model_name])
        results_table = results_table.append(rise_results)   
        
        # TS Forest
        #-----------------Time-series Forest built-in-----------------
        #time complexity O(rmn log n) - r: number of trees, 0.05min 47.1% accuracy
        tsf = TimeSeriesForestClassifier(random_state = loop_random_seeds[i+3], n_jobs=-1)
    
        acc, transferability, model_name = measure_KPIs(tsf, X_train, y_train, X_test, y_test)
        
        tsf_results = pd.DataFrame(data=[[acc, transferability]], columns = header, index=[model_name])
        results_table = results_table.append(tsf_results)
    
        # BOSS Ensemble
        #-----------------BOSS Ensemble-----------------
        # time complexity O(nm(n-w)) - w: window length, 22.6min 38.2% accuracy on m = 1000
        boss = BOSSEnsemble(random_state = loop_random_seeds[i+4], n_jobs=-1)
    
        acc, transferability, model_name = measure_KPIs(boss, X_train, y_train, X_test, y_test)
        
        boss_results = pd.DataFrame(data=[[acc, transferability]], columns = header, index=[model_name])
        results_table = results_table.append(boss_results)
    
        # DecisionTree based on features
        #-----------------TSFresh-----------------
        
        # y_train / y_test from this train-test-split are equivalent to the ones from the other train/test split
        # y_train from the non-featurized tt-split contains the data about source and sensor for transferability calculations
        X_train_features, X_test_features = X_features.iloc[train_index], X_features.iloc[test_index]
        y_train_features, y_test_features = y_features[train_index], y_features[test_index]
        
    
        #Build classifier from features
        classifier_selected_multi = DecisionTreeClassifier(random_state = loop_random_seeds[i+5])

        #Build and select features that are considered significant
        X_train_filtered_multi = select_features(X_train_features, y_train_features, multiclass=True, n_significant=1)
                
        #Build X_test
        X_test_filtered_multi = X_test_features[X_train_filtered_multi.columns]
        
        #Measure KPIs (incl. fit & predict)
        acc, transferability, model_name = measure_KPIs(classifier_selected_multi, X_train_filtered_multi, y_train, X_test_filtered_multi, y_test)
        
        #Add results to results_table
        TSFresh_results = pd.DataFrame(data=[[acc, transferability]], columns = header, index=[model_name])
        results_table = results_table.append(TSFresh_results)
    
        
        #% Artificial Neural Networks
        #------------Residual Network------------
        
        #from numpy.random import seed
        set_seed(loop_random_seeds[i+6])

        X_train_nn, X_test_nn = X_nn[train_index,:], X_nn[test_index,:]
        y_train_nn, y_test_nn = y_nn[train_index], y_nn[test_index]
        
        
        
        #Check whether tt-split worked
        print ("X_train_nn: ", X_train_nn.shape, type(X_train_nn))
        print ("y_train_nn: ", y_train_nn.shape, type(y_train_nn))
        print("X_test_nn: ", X_test_nn.shape, type(X_test_nn))
        print ("y_test_nn: ", y_test_nn.shape, type(y_test_nn))
        
        tt_data_stats = data_stats(keys_sorted,y_train_nn,y_test_nn)
        print(tt_data_stats)
        
        
        #define output directory for model    
        output_dir = fileDirectory
        
        #nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
        nb_classes = len(np.unique(y, axis=0))
        
     
        
        # transform the labels from integers to one hot vectors
        y_train_nn_1hot = to_categorical(y_train_nn)
        y_test_nn_1hot = to_categorical(y_test_nn)
        
        # save orignal y because later we will use binary
        #y_true_nn = np.argmax(y_test_nn, axis=1)
        
        if len(X_train_nn.shape) == 2:  # if univariate
            # add a dimension to make it multivariate with one dimension 
            X_train_nn = X_train_nn.reshape((X_train_nn.shape[0], X_train_nn.shape[1], 1))
            X_test_nn = X_test_nn.reshape((X_test_nn.shape[0], X_test_nn.shape[1], 1))
        
        #Determine input shape to be able to build network
        input_shape = X_train_nn.shape[1:]
            
        resnet = Classifier_RESNET(output_dir, input_shape, nb_classes, custom_metric = False, threshold = 1, verbose=2)
    
        epochs = 200
        
        #fit
        resnet.fit(X_train_nn, y_train_nn_1hot, X_test_nn, y_test_nn_1hot, y_test_nn, epochs)

        #----predict labels
        probs = resnet.predict(X_test_nn,y_test_nn,X_train_nn,y_train_nn_1hot,y_test_nn_1hot, return_df_metrics=False)    
        preds = np.argmax(probs, axis=1)
        
        #----calculate accuracy
        acc = resnet.predict(X_test_nn,y_test_nn,X_train_nn,y_train_nn_1hot,y_test_nn_1hot, return_df_metrics=True)
          
        #----calculate transferability
        
        #split indices of y in sources and sensors. Add source and label to key for check
        sources_y_test, sensors_y_test, y_test_keys = getSource_buildKeys(y_test)
        _, _, y_train_keys = getSource_buildKeys(y_train)
        
        #check whether keys of test_dataset have already been in train_dataset
        present_in_training = np.isin(y_test_keys,y_train_keys).flatten()
        
        correct_pred = y_test_nn == preds #calculate correct predictions
    
        #check all possible combinations of present_in_training and correct_pred
        cm = confusion_matrix(present_in_training, correct_pred)    
               
        #calculate score = correctly transfered / all transfer tasks 
        transferability = (cm[0,1])/(cm[0,:].sum())
        
        
        #save model name
        model_name = str(resnet)
        
        #Add results to results_table    
        ResNet_results = pd.DataFrame(data=[[acc, transferability]], columns = header, index=[model_name])
        results_table = results_table.append(ResNet_results)
        
        
        #-------------------
        #Add overall table to the list of tables
        results.append(results_table)
        
        #Use new random seed for next iteration
        i += 1

        
    #------End of for-loop--------
        
    #%% calculate overall table
    
    # build new list to evaluate all runs
    evaluation_acc = []
    evaluation_transf = []
    for result_table in results:
        acc = result_table.loc[:,'accuracy'].to_numpy()
        evaluation_acc.append(acc)

        transf = result_table.loc[:,'transferability'].to_numpy()
        evaluation_transf.append(transf)
       
        
    #create numpy array from nested list to simplify computation
    evaluation_acc = np.array(evaluation_acc)
    evaluation_transf = np.array(evaluation_transf)

    
    #calculate accuracy values
    mean_acc = np.mean(evaluation_acc, axis=0)
    std_acc = np.std(evaluation_acc, axis=0)
    
    #calculate accuracy values
    mean_transf = np.nanmean(evaluation_transf, axis=0)
    std_transf = np.nanstd(evaluation_transf, axis=0)
       
    #use row indices from one of the results_tables
    final_results = results[0]
    
    #overwrite with averaged results
    final_results.loc[:,'accuracy'] = mean_acc
    final_results.loc[:,'transferability'] = mean_transf
    