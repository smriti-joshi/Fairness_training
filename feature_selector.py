from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import pandas as pd

class FeatureSelector:
    def __init__(self) -> None:
       self.model = None

    def logistic_regression_train(self, x_train, y_train):
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x_train, y_train)
        self.model = SelectFromModel(lsvc, prefit=True)
        x_new = self.model.transform(x_train)
        return x_new
    
    def select_val(self, x_val):
        df = pd.DataFrame(x_val)
        return df.iloc[:, self.model.get_support(indices=True)].to_numpy()
    
    def select_data_cross_val(self, x_train, y_train,  x_val, keyword):

        if keyword == 'logistic_regression':
           x_train = self.logistic_regression_train(x_train, y_train)

        if self.model is not None:
            x_val = self.select_val(x_val)

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