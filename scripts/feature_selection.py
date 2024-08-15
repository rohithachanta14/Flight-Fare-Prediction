from sklearn.feature_selection import mutual_info_regression
import pandas as pd

class FeatureSelection:

    def __init__(self,flight_data):
        self.flight_data=flight_data

    def select_features(self,target_feature):
        X=self.flight_data.drop([target_feature],axis=1)
        y=self.flight_data[target_feature]
        imp = mutual_info_regression(X, y)
        imp_df = pd.DataFrame(imp, index=X.columns)
        imp_df.columns=['Importance of Feature']
        print(imp_df.sort_values(by='Importance of Feature' , ascending=False))
        return X,y

