from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

class BuildMLModels:
    def __init__(self,ml_model,training_features,target_variable):
        self.ml_model=ml_model
        self.training_features, self.testing_features, self.training_target, self.testing_target = train_test_split(
            training_features, target_variable, test_size=0.25, random_state=42)

    def mean_absolute_percentage_error(self, true_values, predicted_values):
        true_values , predicted_values = np.array(true_values) , np.array(predicted_values)
        return np.mean(np.abs((true_values - predicted_values) / true_values)) * 100

    def build_model(self):
        self.ml_model.fit(self.training_features, self.training_target)

    def predict_on_test_data(self):
        self.predicted_values = self.ml_model.predict(self.testing_features)
        print('Predictions:', self.predicted_values)

    def evaluate_model_performance(self):
        print('Training score:', self.ml_model.score(self.training_features, self.training_target))
        r2_score = metrics.r2_score(self.testing_target, self.predicted_values)
        print('R-squared score:', r2_score)
        print('MAE:', metrics.mean_absolute_error(self.testing_target, self.predicted_values))
        print('MSE:', metrics.mean_squared_error(self.testing_target, self.predicted_values))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(self.testing_target, self.predicted_values)))
        print('MAPE:', self.mean_absolute_percentage_error(self.testing_target, self.predicted_values))

    def build_and_evaluate(self):
        self.build_model()
        self.predict_on_test_data()
        self.evaluate_model_performance()





