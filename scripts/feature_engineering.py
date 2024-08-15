import numpy as np
import pandas as pd

class FeatureEngineering:
    """
    Class for performing feature engineering on flight data.
    """

    def __init__(self, flight_data):
        """
        Initializes the class with the flight data.

        Args:
            flight_data (pandas.DataFrame): The flight data to be processed.
        """
        self.flight_data = flight_data.copy()  # Create a copy to avoid modifying original data

    def preprocess_duration(self, duration_string):
        """
        Preprocesses a duration string to ensure a consistent format (e.g., '0h 30m').

        Args:
            duration_string (str): The duration string to be preprocessed.

        Returns:
            str: The preprocessed duration string.
        """
        if 'h' not in duration_string:
            duration_string = '0h' + ' ' + duration_string
        elif 'm' not in duration_string:
            duration_string = duration_string + ' ' + '0m'
        return duration_string

    def create_date_time_features(self, datetime_columns):
        """
        Creates new features from existing datetime columns (day, month, year).

        Args:
            datetime_columns (list): List of datetime column names.
        """
        for feature in datetime_columns:
            self.flight_data[feature + "_day"] = self.flight_data[feature].dt.day
            self.flight_data[feature + "_month"] = self.flight_data[feature].dt.month
            self.flight_data[feature + "_year"] = self.flight_data[feature].dt.year

    def convert_to_datetime(self, column):
        """
        Converts a column to datetime format, assuming the specified format.

        Args:
            column (str): Name of the column to convert.
            format (str, optional): The format string for datetime conversion. Defaults to '%d/%m/%Y'.
        """
        self.flight_data[column] = pd.to_datetime(self.flight_data[column], dayfirst=True)

    def extract_and_create_hour_minute_features(self, time_columns):
        """
        Extracts hour and minute features from existing time columns.

        Args:
            time_columns (list): List of time column names.
        """
        for time_type in time_columns:
            self.flight_data[time_type + "_hours"] = self.flight_data[time_type].dt.hour
            self.flight_data[time_type + "_minutes"] = self.flight_data[time_type].dt.minute

    def extract_and_create_total_duration_features(self, duration_columns):
        """
        Extracts hours, minutes, and total minutes from duration columns.

        Args:
            duration_columns (list): List of duration column names.
        """
        for duration in duration_columns:
            self.flight_data[duration] = self.flight_data[duration].apply(self.preprocess_duration)
            self.flight_data[duration + "_hours"] = self.flight_data[duration].apply(lambda x: int(x.split(' ')[0][0:-1]))
            self.flight_data[duration + "_minutes"] = self.flight_data[duration].apply(lambda x: int(x.split(' ')[1][0:-1]))
            self.flight_data[duration + '_total_mins'] = self.flight_data[duration].str.replace('h', "*60").str.replace(' ', '+').str.replace(
                'm', "*1").apply(eval)

    def encode_categorical_nominal_data(self, data_columns):
        """
        Encodes nominal categorical data using target-guided encoding.

        Args:
            data_columns (list): List of nominal categorical column names.
        """
        for column in data_columns:
            # Calculate mean target value for each category
            category_target_means = self.flight_data.groupby(column)['Price'].mean()

            # Create a mapping dictionary
            encoding_dict = category_target_means.to_dict()

            # Map categories to their corresponding mean target values
            self.flight_data[column] = self.flight_data[column].map(encoding_dict)

    def encode_categorical_ordinal_data(self, data_columns='Total_Stops'):
        stop = {'non-stop': 0, '2 stops': 2, '1 stop': 1, '3 stops': 3, '4 stops': 4}
        self.flight_data[data_columns] = self.flight_data['Total_Stops'].map(stop)

    def handle_outlier_data(self, feature):
        """
        Handles outliers using IQR and capping.

        Args:
          feature (str): The name of the feature to handle outliers for.
        """
        q1 = self.flight_data[feature].quantile(0.25)
        q3 = self.flight_data[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Cap outliers to the lower and upper bounds
        self.flight_data[feature] = np.clip(self.flight_data[feature], lower_bound, upper_bound)