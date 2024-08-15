from scripts.data_collection import load_flight_data
from scripts.feature_engineering import FeatureEngineering
from scripts.data_cleaning import FlightDataCleaner
from sklearn.ensemble import RandomForestRegressor
import pickle
from scripts.feature_selection import FeatureSelection
from scripts.model_building import BuildMLModels
from scripts.Hypertune import HyperTuneMLModel
from pathlib import Path
from scripts.plots import FlightDataPlotter


def main():
    print("Starting Flight Prediction Project")
    filepath=r"data\Data_Train.xlsx"
    # Collect Data
    raw_flight_data = load_flight_data(filepath)

    ## Preprocess the Data for ML Model Training
    data_preprocessor = FeatureEngineering(raw_flight_data)
    data_cleaner = FlightDataCleaner()
    data_preprocessor.flight_data = data_cleaner.drop_missing_data(raw_flight_data)
    print('Null Values Dropped successfully')
    date_time_columns = ['Dep_Time', 'Arrival_Time', 'Date_of_Journey']
    time_columns = ['Dep_Time', 'Arrival_Time']
    new_date_time_columns = ['Date_of_Journey']
    duration_columns = ['Duration']
    for column in date_time_columns:
        data_preprocessor.convert_to_datetime(column)
    data_preprocessor.create_date_time_features(new_date_time_columns)
    data_preprocessor.extract_and_create_hour_minute_features(time_columns)
    data_preprocessor.extract_and_create_total_duration_features(duration_columns)
    print("Converted Object Type features to Date Time")
    nominal_data_columns = ['Source', 'Destination', 'Airline']
    data_preprocessor.encode_categorical_nominal_data(nominal_data_columns)
    data_preprocessor.encode_categorical_ordinal_data()

    ##Plots in Jupyter NoteBook##


    redundant_columns = ['Date_of_Journey', 'Additional_Info', 'Duration_total_mins', 'Source', 'Route', 'Duration',
                         'Dep_Time', 'Arrival_Time']
    data_preprocessor.flight_data = data_cleaner.remove_unnecessary_columns(data_preprocessor.flight_data,
                                                                            redundant_columns)
    data_preprocessor.handle_outlier_data('Price')
    cleaned_data = data_preprocessor.flight_data
    print("Cleaning Data Successful")

    # ## Select the feature using feature Selection
    feature_selector = FeatureSelection(cleaned_data)
    selected_features, target_variable = feature_selector.select_features('Price')

    # # ## Build the ML Model
    model = RandomForestRegressor()
    model_builder = BuildMLModels(model, selected_features, target_variable)
    model_builder.build_and_evaluate()

    # ## HyperTune the Model and evaluate##
    model_tuner = HyperTuneMLModel(model, selected_features, target_variable)
    optimized_model = model_tuner.hypertune()
    hypertuned_model_builder = BuildMLModels(optimized_model, selected_features, target_variable)
    hypertuned_model_builder.build_and_evaluate()

    # ## Save the Model##
    # path=r"D:/Personal/flightpredictionmodel.pkl"
    # save(hypertuned_flightprediction,path)


def save(ml_model, path):
    mlmodelfile = open(path, 'wb');
    pickle.dump(ml_model, mlmodelfile)


if __name__ == "__main__":
    main()
