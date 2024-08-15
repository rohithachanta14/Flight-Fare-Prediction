import matplotlib.pyplot as plt
import seaborn as sns


class FlightDataPlotter:

    def __init__(self, flight_data):
        self.flight_data = flight_data  # More descriptive name

    def plot_price_vs_duration_scatter(self):
        sns.scatterplot(x="Duration_total_mins", y="Price", data=self.flight_data)

    def plot_price_vs_total_stops_scatter(self):
        sns.scatterplot(x="Duration_total_mins", y="Price", hue="Total_Stops", data=self.flight_data)

    def plot_airline_vs_price_boxplot(self):
        sns.boxplot(y="Price", x="Airline", data=self.flight_data.sort_values(by="Price", ascending=False))
        plt.xticks(rotation="vertical")
        plt.show()

    def find_most_frequent_route_per_airline(self, airline):
        """
        Finds the most frequent route for a specific airline.

        Args:
            airline (str): The name of the airline to analyze.
        """
        most_frequent_routes = (
            self.flight_data[self.flight_data["Airline"] == airline]
            .groupby("Route")
            .size()
            .sort_values(ascending=False)
        )
        print(most_frequent_routes)

    def plot_departures_by_time_of_day(self, param=None):
        """
        Plots the distribution of flight departures by time of day.
        """
        self.flight_data['Dep_Time_hours'].apply(self.flight_departure_time_category).value_counts().plot(kind="bar",
                                                                                                   color="g")

    def flight_departure_time_category(self, hour):
        """
        Categorizes a departure time based on the hour of the day.

        Args:
            hour (int): The hour of the day (0-23).

        Returns:
            str: The category of the departure time (e.g., "Early Morning", "Morning").
        """
        if 4 <= hour <= 8:
            return "Early Morning"
        elif 8 < hour <= 12:
            return "Morning"
        elif 12 < hour <= 16:
            return "Noon"
        elif 16 < hour <= 20:
            return "Evening"
        elif 20 < hour <= 24:
            return "Night"
        else:
            return "Late Night"
