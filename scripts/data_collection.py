import pandas as pd

def load_flight_data(file_path):
  """
  Loads flight data from an Excel file into a pandas DataFrame.

  Args:
    file_path (str): The path to the Excel file.

  Returns:
    pandas.DataFrame: The loaded flight data.
  """

  flight_data = pd.read_excel(file_path)
  return flight_data