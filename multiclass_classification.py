import copy
import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, classification_report

from helpers import load_data, clean_data, split_data



# trains the model based on the user's specified model type - 4types allowed
def direct_multiclass_train(model_name, X_train, y_train):
    if model_name == 'dt':
        model = DecisionTreeClassifier()
        model = model.fit(X_train, y_train)
        return model
    if model_name == 'knn':
        model = KNeighborsClassifier()
        model = model.fit(X_train, y_train)
        return model
    if model_name == 'mlp':
        model = MLPClassifier(hidden_layer_sizes=(40,), random_state=1, max_iter=300).fit(X_train, y_train)
        return model
    if model_name == 'rf':
        model = RandomForestClassifier().fit(X_train, y_train)
        return model
    
    # error statement
    print("Unable to train model. Try inputting dt, knn, mlp, or rf for model_name.")


# check the number correct predictions out of the total to find the accuracy
def direct_multiclass_test(model, X_test, y_test):
    correct = 0
    total = 0
    predictions = model.predict(X_test)
    for i in range(len(predictions)):
        tot += 1
        if predictions[i] == y_test.iloc[i]:
            correct += 1
            cor +=1
    return correct/total


# resamples from the original using undersampling strategies
def data_resampling(df, sampling_strategy):
    X = df[df.columns[:-1]]
    y = df[df.columns[-1]]
    if sampling_strategy == 'majority':
        resampleDF = RandomUnderSampler(sampling_strategy= 'majority', random_state=2)
    if sampling_strategy == 'not majority':
        resampleDF = RandomUnderSampler(sampling_strategy= 'not majority', random_state=2)
    if sampling_strategy == 'not minority':
        resampleDF = RandomUnderSampler(sampling_strategy= 'not minority', random_state=2)
    if sampling_strategy == 'all':
        resampleDF = RandomUnderSampler(sampling_strategy= 'all', random_state=2)
    if sampling_strategy == 'binary':
        resampleDF = RandomUnderSampler(sampling_strategy={'BENIGN': 50000, 'MALICIOUS': 50000}, random_state=1)

    # takes specific model and finalizes it
    X_resampled, y_resampled = resampleDF.fit_resample(X, y)
    resampleDF = pd.DataFrame(columns=df.columns)
    resampleDF[resampleDF.columns[:-1]] = X_resampled
    resampleDF[resampleDF.columns[-1]] = y_resampled

    return resampleDF


# split the dataFrame directly into training (80%) and testing (20%)sets
def improved_data_split(df):
    # use a random mask to split data
    mask = np.random.rand(len(df)) < 0.8
    df_train = df[mask]
    df_test = df[~mask]

    return df_train, df_test


# use the improved split function to create a binary dataset - BENIGN and MALICIOUS labels
def get_binary_dataset(df):
    # use the improved split function to split the data into a training and testing set
    df_train, df_test = improved_data_split(df)

    # will return both of these as finished products
    df_binary = df_train.copy()
    df_test_binary = df_test.copy()

    # convert labels
    df_binary.loc[df_binary[' Label'] != 'BENIGN', ' Label'] = 'MALICIOUS'
    df_test_binary.loc[df_test_binary[' Label'] != 'BENIGN', ' Label'] = 'MALICIOUS'

    df_binary = data_resampling(df_binary, 'binary')

    return df_binary, df_test_binary



def main():
    # extracted the .csv files and just kept them in the same folder as the .py files for ease of use (no extra folder_path)
    fname1 = 'Monday-WorkingHours.pcap_ISCX.csv'

    # individual dataFrame creation
    dfRaw = load_data(fname1)

    # copy the data so that we don't need to recreate the original
    dfs = copy.deepcopy(dfRaw)

    # clean the data - will use this dataFrame as the starting point for multiple functions
    dfs = clean_data(dfs)

    # split the data
    X_train1, y_train1, X_test1, y_test1 = split_data(dfs)

    print('succesfully cleaned and split data')

    # train the data using the specified model types
    dtModel = direct_multiclass_train('dt', X_train1, y_train1)
    knnModel = direct_multiclass_train('knn', X_train1, y_train1)
    mlpModel = direct_multiclass_train('mlp', X_train1, y_train1)
    rfModel = direct_multiclass_train('rf', X_train1, y_train1)

    print('succesfully trained the models')

    # test the models' accuracy (correct/total)
    dtAccuracy = direct_multiclass_test(dtModel, X_test1, y_test1)
    print('accuracy of DT: {:.4f}'.format(dtAccuracy))

    knnAccuracy = direct_multiclass_test(knnModel, X_test1, y_test1)
    print('accuracy of KNN: {:.4f}'.format(knnAccuracy))

    mlpAccuracy = direct_multiclass_test(mlpModel, X_test1, y_test1)
    print('accuracy of MLP: {:.4f}'.format(mlpAccuracy))

    rfAccuracy = direct_multiclass_test(rfModel, X_test1, y_test1)
    print('accuracy of RF: {:.4f}'.format(rfAccuracy))

    print('succesfully tested the models')

    # resample the data to help mitigate the unbalanced nature of the data
    dfsResampled = data_resampling(dfs, 'all')

    X_train_resample, y_train_resample, X_test_resample, y_test_resample = split_data(dfsResampled)

    # create new models for the resampled dataFrame
    mlpModelResampled = direct_multiclass_train('mlp', X_train_resample, y_train_resample)
    rfModelResampled = direct_multiclass_train('rf', X_train_resample, y_train_resample)

    # test accuracy of resampled models
    mlpAccuracyResampled = direct_multiclass_test(mlpModelResampled, X_test_resample, y_test_resample)
    print('accuracy of MLP resampled: {:.4f}'.format(mlpAccuracyResampled))

    rfAccuracyResampled = direct_multiclass_test(rfModelResampled, X_test_resample, y_test_resample)
    print('accuracy of MLP resampled: {:.4f}'.format(rfAccuracyResampled))

    print('succesfully resampled and tested the models')

    # convert to a binary dataset
    binary_dataset, binary_test = get_binary_dataset(dfs)

    # compute X and y train and testing sets
    mask = np.random.rand(len(binary_dataset)) < 0.9

    train_set = binary_dataset[mask]
    val_set = binary_dataset[~mask]

    X_train = train_set[train_set.columns[:-1]]
    y_train = train_set[train_set.columns[-1]]

    X_val = val_set[val_set.columns[:-1]]
    y_val = val_set[val_set.columns[-1]]

    X_test = binary_test[binary_test.columns[:-1]]
    y_test = binary_test[binary_test.columns[-1]]

    mlp = direct_multiclass_train('mlp', X_train, y_train)

    # calculate prediction accuracy using a library
    pred = mlp.predict(X_test)
    acc = accuracy_score(pred, y_test)
    print('Validation Accuracy : {:.5f}'.format(accuracy_score(mlp.predict(X_val), y_val)))
    print('Test Accuracy : {:.5f}'.format(acc))
    print('Classification_report:')
    print(classification_report(y_test, pred))

    print('succesfully converted and trained a binary dataset')



if __name__=="__main__":
    main()