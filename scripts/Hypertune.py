from sklearn.model_selection import RandomizedSearchCV
from scripts.model_building import BuildMLModels

class HyperTuneMLModel(BuildMLModels):
    def __init__(self, ml_model, features, target):
        super().__init__(ml_model, features, target)

    def hypertune(self):
        reg_rf = self.ml_model

        n_estimators = [100, 300, 500, 700, 900]
        max_features = ['sqrt', 'log2']
        max_depth = [10, 20, 30, 40]
        min_samples_split = [5, 10, 15, 100]
        min_samples_leaf = [1, 2, 4, 6]
        bootstrap = [True, False]

        # Creating the random grid or hyper-parameter space

        random_grid = {
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap
        }

        ## Define  searching

        rf_random = RandomizedSearchCV(estimator=reg_rf, param_distributions=random_grid,
                                       n_iter=10, cv=5, n_jobs=-1, verbose=2,
                                       error_score='raise')
        rf_random.fit(self.training_features, self.training_target)
        print('The Best Estimator for your Model')
        print(rf_random.best_estimator_)
        print(rf_random.best_score_)
        return rf_random.best_estimator_
