from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
import pandas as pd

class FeatureSelector:
    def __init__(self) -> None:
       self.model = None
       self.features = None
       self.labels = None

    def set_features_and_labels(self, features, labels):
        self.features = features
        self.labels = labels

    def logistic_regression(self):
        scaler = StandardScaler().fit(self.features)
        clf = LogisticRegression().fit(scaler.transform(self.features), self.labels)
        return clf
  
    def extra_trees(self):
        clf = ExtraTreesClassifier(n_estimators=50).fit(self.features, self.labels)
        return clf
      
    def svc(self):
        clf = LinearSVC(C=0.01, penalty="l1", dual=False).fit(self.features, self.labels)
        return clf

    def select_train(self, keyword):
        # self.scaler = StandardScaler()

        if keyword == 'logistic_regression':
            clf = self.logistic_regression()
        elif keyword == 'extra_trees':
             clf= self.extra_trees()
        elif keyword == 'svm':
            clf = self.svc()
        else:
            print(keyword, ' feature selection method does not exist!')

        self.model = SelectFromModel(clf, prefit=True)
        x_new = self.model.transform(self.features)
        return x_new
    
    def select_val(self, features_val):
        df = pd.DataFrame(features_val)
        return df.iloc[:, self.model.get_support(indices=True)].to_numpy()
    
    def select_data_cross_val(self, features_val, keyword):

        x_train = self.select_train(keyword)

        if self.model is not None:
            x_val = self.select_val(features_val)

        return x_train, x_val
    






# def lasso(x_train, y_train, x_val, x_test=None, n_fold=5, max_iters=50, thr=0.5):
#     from sklearn.linear_model import LassoCV
#     from sklearn.feature_selection import SelectFromModel
#     from sklearn.model_selection import GridSearchCV
#     from sklearn.pipeline import Pipeline
#     import numpy as np

# #     if grid_search:
# #         pipeline = Pipeline([
# #                      ('scaler',StandardScaler()),
# #                      ('model',Lasso())
# # ])
# #         search = GridSearchCV(pipeline,
# #                         {'model__alpha':np.arange(0.1,10,0.1)},
# #                         cv = 2, scoring="accuracy",verbose=3
# #                         )
# #         search.fit(features,labels)
# #         print(search.best_params_)

#     lasso = {}
#     cv_lasso = LassoCV(cv = n_fold, max_iter = max_iters, n_jobs = 1)
#     cv_lasso_model = SelectFromModel(cv_lasso, threshold = thr)
#     cv_lasso_model.fit(x_train, y_train)
#     #n_remained_lasso = cv_lasso_model.transform(x_train).shape[1]
#     remained_lasso_idx = cv_lasso_model.get_support(indices = True)
#     x_train_lasso = x_train[:,remained_lasso_idx] 
#     x_val_lasso = x_val[:,remained_lasso_idx]
#     lasso['train'] = x_train_lasso
#     lasso['val'] = x_val_lasso
    
#     # if x_test is not None:
#     #     x_test_lasso = x_test[:,remained_lasso_idx]
#     #     lasso['test'] = x_test_lasso
#     lasso['feature_indices'] = remained_lasso_idx.tolist()
#     return lasso