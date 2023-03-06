
import time
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


from dataloader import Dataloader
from preprocessor import Preprocessor
from models import random_forest, xgboost
from feature_selector import FeatureSelector

def conf_matrix(y_true, y_pred):
    
    target_labels = np.array(y_true)
    predictions = np.array(y_pred)
    matrix = confusion_matrix(target_labels, predictions)
    
    return matrix

def get_stats(metric):
    '''
    Get mean and std of metrics
    Parameters
    ----------
    metric : list
        containing the metrics of different folds.
    Returns
    -------
    mean_metric : float
        average value of the metric.
    std_metric : float
        standard deviation of the metric.
    '''
    
    metric = np.array(metric)
    mean_metric = np.mean(metric)
    std_metric = np.std(metric)
    
    return mean_metric, std_metric


def cross_val_stats(fold_stats):
    '''
    mean/std of metrics for cross validation experiments
    Parameters
    ----------
    fold_stats : dict
        evaluation metrics for each folds are kept in the dict.
    Returns
    -------
    four average values and four standard deviation values for four metrics 
    including accuracy, sensitivity, specifity, and auroc.
    '''

    acc = []
    sen = []
    spc = []
    auc = []
    for keys, vals in fold_stats.items():
        if type(vals) == dict:
            acc.append(vals['val_accuracy'])
            sen.append(vals['val_sensitivity'])
            spc.append(vals['val_specificity'])
            auc.append(vals['val_auc'])
    acc = np.array(acc)
    sen = np.array(sen)
    spc = np.array(spc)
    auc = np.array(auc)

    mean_acc, std_acc = get_stats(acc)
    mean_sen, std_sen = get_stats(sen)
    mean_spc, std_spc = get_stats(spc)
    mean_auc, std_auc = get_stats(auc)
    
    return mean_acc, mean_sen, mean_spc, mean_auc, std_acc, std_sen, std_spc, std_auc 


def metrics(conf_matrix):
    
    tp = conf_matrix[1][1]
    tn = conf_matrix[0][0]
    fp = conf_matrix[0][1]
    fn = conf_matrix[1][0]
    
    accuracy = (float (tp+tn) / float(tp + tn + fp + fn))
    sensitivity = (tp / float(tp + fn))
    specificity = (tn / float(tn + fp))

    return accuracy, sensitivity, specificity


def smote_balancing(feature_set, label_set):
    '''
    balancing the imabalanced dataset by synthesizing new data to the 
    minority class labels.
    Parameters
    ----------
    feature_set : array
        feature matrix of all data.
    label_set : array
        label vector representing all data.
    Returns
    -------
    feature_set_balanced : array
        augmented feature set.
    label_set_balanced : array
        augmented label vector.
    '''
    
    sm = SMOTE(sampling_strategy='auto',  random_state=42)
    feature_set_balanced, label_set_balanced = sm.fit_resample(feature_set, label_set)  
    
    return feature_set_balanced, label_set_balanced

def learning(clf, x_train, y_train, x_val, y_val, preprocessor, selector, x_test=None):
    '''
    Parameters
    ----------
    clf : class
        A compiled learning algorithms based on sklearn format.
    x_train : array
        A matrix of the training features.
    y_train : array
        A vector representing the class labels of the training set.
    x_val : array
        A matrix of the validation features.
    y_val : array
        A vector representing the class labels of the validation set.
    x_test : array, optional
        A matrix of the validation features. The default is None.
    Returns
    -------
    summary : dict
        metrics as well as predicted values of validation
        and test (optional) sets.
    '''
    
    summary = {}

    x_train, x_val = preprocessor.preprocess_data_cross_val(x_train, x_val)

    x_train, x_val = selector.select_data_cross_val(x_train, y_train, x_val, keyword = 'logistic_regression')

    x_train, y_train = smote_balancing(x_train, y_train)


    clf.fit(x_train, y_train)
    y_val_pred = clf.predict(x_val)
    y_val_pred_prob = clf.predict_proba(x_val)[:,1]
    
    try:
        roc_auc_val = roc_auc_score(y_val, y_val_pred_prob)
    except ValueError:
        roc_auc_val = -999
    y_val_pred_prob = y_val_pred_prob.tolist()
    confusion_matrix = conf_matrix(y_val, y_val_pred)
    accuracy, sensitivity, specificity = metrics(confusion_matrix)
    
    if x_test is not None:        
        y_test_pred_prob = clf.predict_proba(x_test)[:,1]
        y_test_pred_prob = y_test_pred_prob.tolist()
    else:
        y_test_pred_prob = None

    
    summary['val_accuracy'] = accuracy
    summary['val_sensitivity'] = sensitivity
    summary['val_specificity'] = specificity
    summary['val_auc'] = roc_auc_val
    summary['val_pred_prob'] = y_val_pred_prob
    summary['test_pred_prob'] = y_test_pred_prob
    
    return summary, clf
    
def test_data(x_test, y_test, preprocessor, selector, clf):
    x_test = preprocessor.preprocess_val(x_test)
    x_test = selector.select_val(x_test)
    y_test_pred = clf.predict(x_test)
    y_test_pred_prob = clf.predict_proba(x_test)[:,1]
    
    try:
        roc_auc_test = roc_auc_score(y_test, y_test_pred_prob)
    except ValueError:
        roc_auc_test = -999
    y_test_pred_prob = y_test_pred_prob.tolist()
    confusion_matrix = conf_matrix(y_test, y_test_pred)
    accuracy, sensitivity, specificity = metrics(confusion_matrix)

    print('the average AUC, Acc, Sens, Spec value: {}, {}, {}, {}'.format(roc_auc_test, accuracy, sensitivity, specificity))

def main():

    # for learning_rate in np.arange(1, 10, 1):
    #     print(learning_rate)

    train_dataloader = Dataloader(feature_path='/home/smriti/Downloads/csv_files/duke_mri_with_pcr_train_white.csv')

    # data loading
    subject_ids_train ,features_names_train, x_train, y_train = train_dataloader.load_dataset()

    # feature preprocessing
    preprocessor = Preprocessor()


    # feature selection 
    selector = FeatureSelector()

    # and training
    fold_num = 0
    kf = KFold(n_splits = 5, shuffle = False) 
    fold_stats = {}

    for train_index, val_index in kf.split(y_train):
        fold_num += 1
        fold_name = 'fold_'+str(fold_num)
        # print('Working on fold: {}'.format(fold_num))
        
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        
        # feature_selector = lasso(x_train, y_train, x_val, x_test=None, n_fold=5, max_iters=50, thr=0.5)

        # x_train = feature_selector['train']
        # x_val = feature_selector['val']
        # feature_indices = feature_selector['feature_indices']
        # fold_name_features = fold_name+'_selected_features'
        # fold_stats[fold_name_features] = feature_indices
        
        clf = xgboost(n_estimators=8, learning_rate=0.14, max_depth=1)
        # clf = random_forest(n_estimators=5, criterion='gini', max_depth=5, class_weight=None)
        clf_summary, clf = learning(clf, x_train, y_train, x_val, y_val, preprocessor, selector, x_test=None)
        fold_stats[fold_name] = clf_summary
            
    mean_acc, mean_sen, mean_spc, mean_auc, _, _, _, _ = cross_val_stats(fold_stats)
    print('the average AUC, Acc, Sens, Spec value of {} fold cross validation: {}, {}, {}, {}'.format(5, mean_auc, mean_acc, mean_sen, mean_spc))

    print('White')
    # testing on white patients
    test_dataloader = Dataloader(feature_path='/home/smriti/Downloads/csv_files/duke_mri_with_pcr_validation_white.csv')
    subject_ids_test, features_names_test, x_test, y_test= test_dataloader.load_dataset()
    test_data(x_test, y_test, preprocessor, selector, clf)

    # testing on black patients
    # test_dataloader_black = Dataloader(feature_path='/home/smriti/Downloads/csv_files/duke_mri_with_pcr_validation_white.csv')
    # subject_ids_test_black, features_names_test_black, x_test_black, y_test_black= test_dataloader_black.load_dataset()
    # test_data(x_test_black, y_test_black, preprocessor, clf)

    print('Black')
    # testing on other patients
    test_dataloader_mixed = Dataloader(feature_path='/home/smriti/Downloads/csv_files/duke_mri_with_pcr_validation_black.csv')
    subject_ids_test_mixed, features_names_test_mixed, x_test_mixed, y_test_mixed= test_dataloader_mixed.load_dataset()
    test_data(x_test_mixed, y_test_mixed, preprocessor, selector, clf)

main()

# # feature selection - lasso
    # t1 = time.perf_counter()
    
    # select_features(x_train, (y_train > 1).astype(int))

    # t2 = time.perf_counter()

    # print(f'Finished in {t2-t1} seconds')