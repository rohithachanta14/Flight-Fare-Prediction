
class FlightDataCleaner:
    """
    Class for cleaning flight data.
    """

    def __init__(self):
        pass

    def drop_missing_data(self, flight_data):
        """
        Removes rows with missing values from the flight data.

        Args:
            flight_data (pandas.DataFrame): The raw flight data.

        Returns:
            pandas.DataFrame: Flight data with missing values removed.
        """
        flight_data.dropna(inplace=True)
        return flight_data

    def remove_unnecessary_columns(self, flight_data, columns_to_drop):
        """
        Removes specified columns from the flight data.

        Args:
            flight_data (pandas.DataFrame): The flight data.
            columns_to_drop (list): List of column names to remove.

        Returns:
            pandas.DataFrame: Flight data with specified columns removed.
        """
        flight_data.drop(columns_to_drop, axis=1, inplace=True)
        return flight_data
