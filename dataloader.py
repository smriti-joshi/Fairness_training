import pandas as pd
import numpy as np

class Dataloader:

    def __init__(self, extract_feature = False, feature_path = None):
        self.extract_feature = extract_feature
        self.feature_path = feature_path

    def load_dataset(self):

        if self.extract_feature:
            pass
        else:
            return self.load_radiomic_set_from_excel()

    def load_radiomic_set_from_excel(self):
        '''
        loading radiomics features saved in a .csv file
        Parameters
        ----------
        feature_path : str
            full path to the .csv file.
        Returns
        -------
        subject_ids : list
        feature_values : array
        label_values : array
        '''
        
        
        features_df = pd.read_csv(self.feature_path) 
        
        # print(features_df.isnull().values.any())
        
        features_df.replace(np.nan, 0)
        features_names = list(features_df.keys()) # == list(features_df.column)
        features_names = features_names[1:-2]       # getting the feature names
        subject_ids =  list(features_df[features_df.columns[0]]) # get the subject ids
        label_values = np.asarray(list(features_df[features_df.columns[-1]])) # get the subject labels
        feature_values = features_df.values  # get the feature values
        feature_values = feature_values[:,1:-2] # the first 3 columns contain order, id, labels 
        feature_values = feature_values.astype(np.float32)

        # labels values - group PCR, DCIS and LCIS

        label_values_updated = []

        for el in label_values:
            if el == 3 or el == 4:
                el = 1
            label_values_updated.append(el)
            
        label_values_updated = np.array(label_values_updated)
                
        return subject_ids ,features_names, feature_values, label_values_updated