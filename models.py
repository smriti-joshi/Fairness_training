class Model:
     
     def __init__(self, keyword):
          pass

def random_forest(n_estimators=10, criterion='gini', max_depth=5, class_weight=None):
    '''
    Parameters
    ----------
    n_estimators : int
        set the number of trees in the forest, default is 100
    criterion : string
        selected from either 'gini' or 'entropy'.
    max_depth : int
        the maximum depth of the trees..
    class_weight : dict or list
         Assigining weights to  class labels e.g., (0:1, 1:2).
    Returns
    -------
    clf : class
        A compiled random forest model
    '''
    clf = RandomForestClassifier(criterion = criterion,
                                 n_estimators = n_estimators,
                                 max_depth = max_depth,
                                 class_weight = class_weight,
                                 n_jobs = -1,
                                 random_state = 42)
    return clf

def xgboost(n_estimators=10, learning_rate=1.0, max_depth=1):
     
     clf = GradientBoostingClassifier(n_estimators=n_estimators, 
                                      learning_rate=learning_rate,
                                      max_depth=max_depth,                                     
                                      random_state=42)
     return clf