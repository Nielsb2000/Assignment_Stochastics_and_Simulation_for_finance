import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from shapely.geometry import Point
from statsmodels.distributions.empirical_distribution import ECDF


def data_exploration(data_path, aggregate_period, aggregate_label, show_plots):
    """
    Subquestion 1 of Part 1 (Data analysis)
    Imports and filters the data, plots the amount of earthquakes over time (aggregates per month),
    plots detailed counts at peak times and calculates summary statistics
    :param data_path: relative path to full dataset
    :param aggregate_period: time period to aggregate counts on; so day, week, month or quartile
    :param aggregate_label: label for title in plots to indicate which time period was chosen
    :param show_plots: TRUE/FALSE variable to indicate whether the plots should be shown or not
    :return: filtered dataframes for all, low and high magnitude data
    """

    def filter_data(data_relative_path):
        """
        Selects relevant data from the full dataset, and splits it into low and high magnitude data
        :param data_relative_path: relative path to full dataset
        :return: dataframes with all earthquakes, low magnitude earthquakes and high magnitude earthquakes
        """
        # Import the data to a dataframe
        data_eq = pd.read_csv(data_relative_path)

        # Transform 'time' column values to DateTime objects
        data_eq['DateTime'] = pd.to_datetime(data_eq['time'])
        data_eq = data_eq.sort_values(by='DateTime')   # sort from earliest to newest

        # Select only earthquakes within the Philippines' borders
        mask_Philippines = data_eq['place'].str.contains('Philippines', case=True)
        data_eq = data_eq[mask_Philippines]

        # Only keep important column and set the new DateTime column as index
        all_data = data_eq[['DateTime', 'latitude', 'longitude', 'mag']].set_index('DateTime')

        # Split DataFrame into 2 DataFrames for earthquakes with magnitudes lower and higher than 5
        high_mag_data = all_data[all_data['mag'] >= 5]
        low_mag_data = all_data[all_data['mag'] < 5]

        return all_data, low_mag_data, high_mag_data

    all_eq_data, low_mag_eq_data, high_mag_eq_data = filter_data(data_relative_path=data_path)

    # Count amount of earthquakes per day/week/month
    full_counts = all_eq_data.resample(aggregate_period).size()
    print(f'Full amount of earthquakes: {sum(full_counts)}')

    low_counts = low_mag_eq_data.resample(aggregate_period).size()
    print(f'Full amount of low magnitude earthquakes: {sum(low_counts)}')

    high_counts = high_mag_eq_data.resample(aggregate_period).size()
    print(f'Full amount of high magnitude earthquakes: {sum(high_counts)}')

    # Plot counts of earthquake occurrences
    if show_plots:
        # Make plots to show amount of earthquakes per month
        fig, axs = plt.subplots(2, 1, figsize=(10, 15))

        # Plot for full data
        axs[0].plot(full_counts, label="Full Counts")
        axs[0].set_xlabel('Time Period', size=10)
        axs[0].set_ylabel('Counts', size=10)
        axs[0].set_title("Full Counts", size=15)

        # Plot for low/high magnitude data
        axs[1].plot(low_counts, label="Low Magnitude (< 5)")
        axs[1].plot(high_counts, label="High Magnitude (>= 5)")
        axs[1].set_xlabel('Time Period', size=10)
        axs[1].set_ylabel('Counts', size=10)
        axs[1].set_title("High and Low Magnitudes Counts", size=15)
        axs[1].legend()

        fig.suptitle(f'Amount of reported earthquakes in the Philippines aggregated per {aggregate_label}', size=20)
        plt.show()

    # Reset index so the DataTime column in not the index column anymore
    all_eq_data = all_eq_data.reset_index()
    high_mag_eq_data = high_mag_eq_data.reset_index()
    low_mag_eq_data = low_mag_eq_data.reset_index()

    # Plot distribution of magnitudes during peak period
    if show_plots:
        peak_start = '2023-12-01'  # starting day of peak
        peak_end = '2023-12-08'  # ending day of peak

        # Filter data for peak periods
        peak_df = all_eq_data[(all_eq_data['DateTime'] >= peak_start) & (all_eq_data['DateTime'] <= peak_end)]

        # Plot histogram of magnitudes in peak period
        plt.hist(peak_df['mag'], bins=30, edgecolor='black', density=False)
        plt.title('Earthquake Magnitudes During Peak Period', size=18)
        plt.xlabel('Magnitude', size=11)
        plt.ylabel('Count', size=11)
        plt.show()

    # Data Summary Statistics for magnitudes
    print(all_eq_data['mag'].describe())
    print(high_mag_eq_data['mag'].describe())
    print(low_mag_eq_data['mag'].describe())

    # Largest magnitude earthquakes:
    mag_sorted = all_eq_data.sort_values(by='mag', ascending=False)
    print(mag_sorted.head())

    return all_eq_data, high_mag_eq_data, low_mag_eq_data

def plot_geospatial(data_eq, magnitude_label):
    """
    Plots earthquakes occurrences as points on a map of the Philippines
    :param data_eq: a dataframe with column 'longitude' and 'latitude' to indicate earthquake locations
    :param magnitude_label: label to say if data_eq contains high, low, or all magnitude data; used for plot title
    :return: plot of earthquakes occurrences as points on a map of the Philippines
    """
    # Import map of the earth from geopandas
    worldmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    # Get map of Philippines
    philippines_map = worldmap[worldmap['name'] == 'Philippines']

    # Plot earthquake data points on the Philippines map
    fig, ax = plt.subplots(figsize=(10, 10))

    # Append point locations to dataframe
    data_eq['geometry'] = data_eq.apply(lambda x: Point(x['longitude'], x['latitude']), axis=1)
    data_eq = gpd.GeoDataFrame(data_eq, geometry='geometry')

    # Plot Philippines map and points
    philippines_map.plot(ax=ax, color='lightgray')
    data_eq.plot(ax=ax, color='red', markersize=data_eq['mag'], alpha=0.05 * data_eq['mag'], label='Earthquakes')

    fig.suptitle(f'Locations of {magnitude_label} earthquakes in the Philippines', size=20)
    plt.xlabel('Longitude', size=12)
    plt.ylabel('Latitude', size=12)
    plt.show()

def transform_time_diff_data(data):
    """
    Transforms a dataframe with a 'DateTime' column, with time values, into a dataframe with a column 'time_diff_seconds',
    indicating the interarrival times in seconds
    :param data: a dataframe with a 'DateTime' column
    :return: a dataframe with a 'time_diff' column
    """
    # Calculate differences in time between values in the 'DateTime' column
    data['time_diff'] = data['DateTime'].diff()

    # Drop the first row, as it will be a NaN value (first time has no time to subtract from)
    data = data[['time_diff']].drop(index=0)
    # Convert the time differences to seconds
    data['time_diff_seconds'] = pd.to_timedelta(data['time_diff']).dt.total_seconds()

    # Take only relevant column
    time_diff_data = data['time_diff_seconds'].copy()

    return time_diff_data

def fit_distribution(time_diff_data, show_plots):
    """
    Fit distribution for earthquake interarrival data using the method of methods.
    Gives histogram and eCDF plots and performs KS-tests for various distributions
    :param time_diff_data: a dataframe with a 'time_diff' column
    :param show_plots: TRUE/FALSE variable to indicate whether the plots should be shown or not
    :return: fitted distributions, these being the exponential, normal, gamma and uniform distributions
    """
    # Fitting distributions
    # First and second moment
    M1 = np.mean(time_diff_data)  # first moment
    M2 = np.mean(time_diff_data ** 2)  # second moment
    print(f'First moment: {M1}, Second moment: {M2}')

    # Normal Distribution: Estimates for mu and sigma^2
    muEst = M1
    sigma2Est = M2 - M1 ** 2
    print(f'Normal distribution: {muEst}, {sigma2Est}')

    # Exponential Distribution: Estimates for lambda
    lamEst = 1 / M1
    print(f'Exponential distribution: {lamEst}')

    # Gamma Distribution: Estimates for alpha and béta
    alphaEst = M1 ** 2 / (M2 - M1 ** 2)
    betaEst = M1 / (M2 - M1 ** 2)
    print(f'Gamma distribution: {alphaEst}, {betaEst}')

    # Uniform Distribution: Estimates for a and b
    aEst = np.min(time_diff_data)
    bEst = np.max(time_diff_data)
    print(f'Uniform distribution: a={aEst}, b={bEst}')

    # The estimated distributions
    estExpDist = stats.expon(scale=1 / lamEst)
    estNormDist = stats.norm(muEst, np.sqrt(sigma2Est))
    estGammaDist = stats.gamma(alphaEst, scale=1 / betaEst)
    estUniformDist = stats.uniform(loc=aEst, scale=bEst - aEst)

    # Add theoretical density
    xs = np.arange(np.min(time_diff_data), np.max(time_diff_data))

    if show_plots:
        # Show a plot with the histogram of the time difference data, and the estimated distributions
        plt.figure(figsize=(8, 8))
        plt.hist(time_diff_data, bins=80, edgecolor='black', density=True)
        plt.plot(xs, estExpDist.pdf(xs), 'r', label='Exponential Distribution')
        plt.plot(xs, estNormDist.pdf(xs), 'b', label='Normal Distribution')
        # plt.plot(xs, estGammaDist.pdf(xs), 'y', label='Gamma Distribution')
        plt.plot(xs, estUniformDist.pdf(xs), 'm', label='Uniform Distribution')
        plt.title('Histogram of interarrival times with fitted distributions', size=17)
        plt.xlabel('Interarrival Time (s)', size=11)
        plt.ylabel('Density', size=11)
        plt.legend()
        plt.show()

        # Empirical distribution function (Method 2: using Python function ECDF)
        ecdf = ECDF(time_diff_data)
        plt.figure(figsize=(9, 9))
        plt.step(ecdf.x, ecdf.y, color='black', where='post', label='Interarrival Times eCDF')
        plt.plot(xs, estExpDist.cdf(xs), 'r', label='Exponential Distribution')
        plt.plot(xs, estNormDist.cdf(xs), color='b', label='Normal Distribution')
        plt.plot(xs, estGammaDist.cdf(xs), 'y', label='Gamma Distribution')
        plt.plot(xs, estUniformDist.cdf(xs), 'm', label='Uniform Distribution')
        plt.title('Empirical Distribution Function with fitted cumulative distributions', size=18)
        plt.xlabel('Interarrival Time (s)', size=12)
        plt.ylabel('Cumulative Probability', size=12)
        plt.legend()
        plt.show()

    # Kolmogorov-Smirnov test
    test_exponential = stats.kstest(time_diff_data, estExpDist.cdf)
    test_normal = stats.kstest(time_diff_data, estNormDist.cdf)
    test_gamma = stats.kstest(time_diff_data, estGammaDist.cdf)
    test_uniform = stats.kstest(time_diff_data, estUniformDist.cdf)
    print('KS Test Exponential distribution: ' + str(test_exponential))
    print('KS Test Normal distribution: ' + str(test_normal))
    print('KS Test Gamma distribution: ' + str(test_gamma))
    print('KS Test Uniform distribution: ' + str(test_uniform))

    return estExpDist, estNormDist, estGammaDist, estUniformDist

def calculate_probabilities(data_eq, show_plots):
    """
    Calculates the probability that an arbitrary earthquake has a magnitude of less than 5
    Plots a cumulative probability function to show this probability
    :param data_eq: dataframe with all magnitude earthquake data
    :param show_plots: TRUE/FALSE variable to indicate whether the plots should be shown or not
    :return: probability that an arbitrary earthquake has a magnitude of less than 5
    """

    if show_plots:
        # Cumulative distribution function for magnitudes
        ecdf = ECDF(data_eq['mag'])

        plt.figure(figsize=(9, 9))
        plt.step(ecdf.x, ecdf.y, color='black', where='post', label='Interarrival Times eCDF')

        # Add vertical line at magnitude of 5
        x_value = 5 # mag = 5
        plt.axvline(x=x_value, color='red', linestyle='--')     # vertical line at mag = 5

        # Calculate y-value where the vertical line meets the ECDF
        y_value = ecdf.y[ecdf.x < x_value][-1]    # y-value just before the jump at mag = 5
        plt.axhline(y=y_value, color='blue', linestyle='--')    # horizontal line at y_value

        # Add text to show y-value
        plt.text(x_value - 1.92, y_value, f'{y_value:.4f}', color='blue', va='center')  # 4 decimals

        plt.title('Cumulative distribution function of earthquakes by magnitude', size=18)
        plt.xlabel('Cumulative probability', size=11)
        plt.ylabel('Density', size=11)

        plt.grid()
        plt.show()

    prob = len(data_eq[data_eq['mag'] < 5]) / len(data_eq)
    print(f'Probability that random earthquake has magnitude less than 5 is equal to = {prob}')

    return prob

def simulate_from_distribution(num_of_years, fitted_distribution):
    """
    Simulates T years of earthquakes, by sampling from a previously fitted distribution
    :param num_of_years: times (amount of years) to repeat simulation
    :param fitted_distribution: the fitted distribution that will be used to sample interarrival times
    :return: lists of interarrival times for T years and a numbers with the total count of earthquakes for T years
    """
    def simulate_year_from_fit_dist(fitted_distribution):
        """
        Simulates one year of earthquakes, by sampling from a previously fitted distribution
        :param fitted_distribution: the fitted distribution that will be used to sample interarrival times
        :return: list of interarrival times in the year and a number with the total count of earthquakes in the year
        """
        seconds_in_year = 365.25 * 24 * 3600  # Approx. 31,557,600 seconds
        current_time = 0
        year_of_quakes = []
        # Simulate earthquakes until we exceed one year
        while current_time < seconds_in_year:
            # Sample an interarrival time using a fitted distribution
            sampled_time = fitted_distribution.rvs(1)  # samples random earthquake
            current_time += sampled_time  # calc total time

            if current_time < seconds_in_year:  # if total time is less than a year
                # Add earthquake (time in seconds between the last one and this one) to that year list of earthquakes
                year_of_quakes.append(float(sampled_time[0]))
                quake_count = len(year_of_quakes)
        return year_of_quakes, quake_count

    def sim_num_years(T: int, fitted_distribution):
        """
        Runs simulate_year_from_fit_dist T times, so a distribution of earthquake count per year can be found
        :param T: times (amount of years) to repeat simulation
        :param fitted_distribution: the fitted distribution that will be used to sample interarrival times
        :return: lists of interarrival times for T years and numbers with the total count of earthquakes for T years
        """
        sim_data = []
        num_of_quakes_each_year = []
        for n in range(T): # do simulation for T years
            year_of_quakes, quake_count = simulate_year_from_fit_dist(fitted_distribution)
            sim_data.append(year_of_quakes)
            num_of_quakes_each_year.append(quake_count)
        return sim_data, np.array(num_of_quakes_each_year)

    sim_data, sim_num_quakes = sim_num_years(num_of_years, fitted_distribution)

    return sim_data, sim_num_quakes

def simulation_method_1(num_of_years, fit_dist_low, fit_dist_high):
    """
    Simulates T years of earthquakes, by sampling for type-1 and type-2 earthquakes first, and adding them together
    to get earthquake counts for all magnitudes.
    Samples using a previously fitted distribution for interarrival times.
    :param num_of_years: times (amount of years) to repeat simulation; Also the T variable in N(T)
    :param fit_dist_low: fitted distribution for low magnitude data (type 2)
    :param fit_dist_high: fitted distribution for high magnitude data (type 1)
    :return: lists for amount of earthquakes per year for low, high and all magnitude earthquakes
    """
    # Sample for type 1 earthquakes (Get N_1(T))
    sim_data_high, sim_num_quakes_high = simulate_from_distribution(num_of_years, fit_dist_high)

    # Sample for type 2 earthquakes (Get N_2(T))
    sim_data_high, sim_num_quakes_low = simulate_from_distribution(num_of_years, fit_dist_low)

    # Combine to get number of earthquakes for all magnitude data (Get N_(T))
    sim_num_quakes_total = sim_num_quakes_high + sim_num_quakes_low

    return sim_num_quakes_low, sim_num_quakes_high, sim_num_quakes_total

def simulation_method_2(num_of_years, fit_dist_total, prob_lower_5):
    """
    Simulates T years of earthquakes, by sampling for all magnitude data first and randomly determining how many are
    type 1 and how many are type 2, using a previously calculated probability.
    Samples using a previously fitted distribution for interarrival times.
    :param num_of_years: times (amount of years) to repeat simulation; Also the T variable in N(T)
    :param fit_dist_total: fitted distribution for all magnitude data
    :param prob_lower_5: probability that an arbitrary earthquakes has a magnitude of less than 5
    :return: lists for amount of earthquakes per year for low, high and all magnitude earthquakes
    """
    # Sample for all earthquakes (Get N(T))
    sim_data_total, sim_num_quakes_total = simulate_from_distribution(num_of_years, fit_dist_total)

    # Randomly how many earthquakes are type 1 and type 2
    prob_low = prob_lower_5 # previously calculated probability for earthquakes with magnitude < 5
    prob_high = 1 - prob_lower_5 # probability magnitude >= 5

    # The 2 types of earthquakes to randomly split between
    eq_types = [1, 2]

    # Initialize lists to hold the counts of each type per year
    sim_num_quakes_high = [] # initiate list for earthquake counts per year for type 1 earthquakes
    sim_num_quakes_low = [] # initiate list for earthquake counts per year for type 2 earthquakes

    # Loop through each year and apply the split
    for count in sim_num_quakes_total:
        # Randomly gives values of 1 or 2 (indicate type 1 or 2 earthquakes), according to the probability
        earthquake_types_random = np.random.choice(eq_types, count, p=[prob_high, prob_low])
        count_type_1 = np.sum(earthquake_types_random == 1) # count number of 1's = high magnitude earthquakes
        count_type_2 = np.sum(earthquake_types_random == 2) # count number of 2's = low magnitude earthquakes

        sim_num_quakes_high.append(count_type_1)
        sim_num_quakes_low.append(count_type_2)

    # Convert to correct format (Numpy array with integers (not np.int64)
    sim_num_quakes_high = np.array(sim_num_quakes_high, dtype=int)
    sim_num_quakes_low = np.array(sim_num_quakes_low, dtype=int)

    return sim_num_quakes_low, sim_num_quakes_high, sim_num_quakes_total

def simulate_empirical(num_of_years, empirical_data):
    """
    Simulates T years of earthquakes, by sampling from the empirical interarrival times.
    :param num_of_years: times (amount of years) to repeat simulation
    :param empirical_data: the empirical data, containing the actual interarrival times between earthquakes
    :return: lists of interarrival times for T years and numbers with the total count of earthquakes for T years
    """
    rng = np.random.default_rng()

    def simulate_year_from_emp_data(empirical_data):
        """
        Simulates one year of earthquakes, by sampling from the empirical interarrival times.
        :param empirical_data: the empirical data, containing the actual interarrival times between earthquakes
        :return: list of interarrival times in the year and a number with the total count of earthquakes in the year
        """
        seconds_in_year = 365.25 * 24 * 3600  # Approx. 31,557,600 seconds
        current_time = 0
        year_of_quakes = []
        # Simulate earthquakes until we exceed one year
        while current_time < seconds_in_year:
            # Sample an interarrival time from the empirical data
            sampled_time = rng.choice(empirical_data)  # pick random earthquake
            current_time += sampled_time  # calc total time

            if current_time < seconds_in_year:  # if total time is less than a year
                # Add earthquake (time in seconds between the last one and this one) to that year list of earthquakes
                year_of_quakes.append(sampled_time)
                quake_count = len(year_of_quakes)
        return year_of_quakes, quake_count

    def sim_num_years(T: int, empirical_data):
        """
        Runs simulate_year_from_emp_data T times, so a distribution of earthquake count per year can be found
        :param T: times (amount of years) to repeat simulation
        :param empirical_data: the empirical data, containing the actual interarrival times between earthquakes
        :return: lists of interarrival times for T years and a numbers with the total count of earthquakes for T years
        """
        sim_data = []
        num_of_quakes_each_year = []
        for n in range(T): # do simulation for T years
            year_of_quakes, quake_count = simulate_year_from_emp_data(empirical_data)
            sim_data.append(year_of_quakes)
            num_of_quakes_each_year.append(quake_count)
        return sim_data, np.array(num_of_quakes_each_year)

    sim_data, sim_num_quakes = sim_num_years(num_of_years, empirical_data)

    return sim_data, sim_num_quakes

def fit_distribution_simulation(sim_num_quakes, sim_type, show_plots):
    """
    Fit distribution for earthquake per year count data using the method of methods.
    Gives histogram and eCDF plots and performs KS-tests for various distributions. Also plots confidence interval
    :param sim_num_quakes: list with amount of earthquakes per year, for multiple years
    :param sim_type: low, high or all earthquakes; used for plotting title
    :param show_plots: TRUE/FALSE variable to indicate whether the plots should be shown or not
    """
    # Fitting distributions
    # First and second moment
    M1 = np.mean(sim_num_quakes)  # first moment
    M2 = np.mean(sim_num_quakes ** 2)  # second moment
    print(f'First moment: {M1}, Second moment: {M2}')

    # Normal Distribution: Estimates for mu and sigma^2
    muEst = M1
    sigma2Est = M2 - M1 ** 2
    print(f'Normal distribution: {muEst}, {sigma2Est}')

    # Exponential Distribution: Estimates for lambda
    lamEst = 1 / M1
    print(f'Exponential distribution: {lamEst}')

    # Gamma Distribution: Estimates for alpha and béta
    alphaEst = M1 ** 2 / (M2 - M1 ** 2)
    betaEst = M1 / (M2 - M1 ** 2)
    print(f'Gamma distribution: {alphaEst}, {betaEst}')

    # Uniform Distribution: Estimates for a and b
    aEst = np.min(sim_num_quakes)
    bEst = np.max(sim_num_quakes)
    print(f'Uniform distribution: a={aEst}, b={bEst}')

    # The estimated distributions
    sim_estExpDist = stats.expon(scale=1 / lamEst)
    sim_estNormDist = stats.norm(muEst, np.sqrt(sigma2Est))
    sim_estGammaDist = stats.gamma(alphaEst, scale=1 / betaEst)
    sim_estUniformDist = stats.uniform(loc=aEst, scale=bEst - aEst)

    # Add theoretical density
    xs = np.arange(np.min(sim_num_quakes), np.max(sim_num_quakes))

    if show_plots:
        # Show a plot with the histogram of the simulated earthquake counts, and the estimated distributions
        plt.figure()
        plt.hist(sim_num_quakes, bins=10, edgecolor='black', density=True)
        plt.plot(xs, sim_estExpDist.pdf(xs), 'r', label='Exponential Distribution')
        plt.plot(xs, sim_estNormDist.pdf(xs), 'b', label='Normal Distribution')
        plt.plot(xs, sim_estGammaDist.pdf(xs), 'y', label='Gamma Distribution')
        plt.plot(xs, sim_estUniformDist.pdf(xs), 'm', label='Uniform Distribution')
        plt.title('Distributions of Simulated Earthquakes Count')
        plt.xlabel('Simulated number of earthquakes')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

        # Empirical distribution function (Method 2: using Python function ECDF)
        ecdf = ECDF(sim_num_quakes)
        plt.figure()
        plt.step(ecdf.x, ecdf.y, color='black', where='post', label='Interarrival Times eCDF')
        plt.plot(xs, sim_estExpDist.cdf(xs), 'r', label='Exponential Distribution')
        plt.plot(xs, sim_estNormDist.cdf(xs), color='b', label='Normal Distribution')
        plt.plot(xs, sim_estGammaDist.cdf(xs), 'y', label='Gamma Distribution')
        plt.plot(xs, sim_estUniformDist.cdf(xs), 'm', label='Uniform Distribution')
        plt.legend()
        plt.show()

        # Kolmogorov-Smirnov test
        test_sim_exponential = stats.kstest(sim_num_quakes, sim_estExpDist.cdf)
        test_sim_normal = stats.kstest(sim_num_quakes, sim_estNormDist.cdf)
        test_sim_gamma = stats.kstest(sim_num_quakes, sim_estGammaDist.cdf)
        test_sim_uniform = stats.kstest(sim_num_quakes, sim_estUniformDist.cdf)
        print('KS Test Exponential distribution: ' + str(test_sim_exponential))
        print('KS Test Normal distribution: ' + str(test_sim_normal))
        print('KS Test Gamma distribution: ' + str(test_sim_gamma))
        print('KS Test Uniform distribution: ' + str(test_sim_uniform))

    # Confidence intervals for the mean(Using Slutsky's theorem: estimate the sample variance)
    s2 = np.var(sim_num_quakes)
    m = np.mean(sim_num_quakes)
    z = 1.96 # 5% confidence interval
    half_width = z * np.sqrt(s2 / len(sim_num_quakes))

    interval = (m - half_width, m + half_width) # the confidence interval

    # Values for mean and standard deviation for plotting
    mean = M1
    std_dev = np.sqrt(sigma2Est) # std dev approx from the interval

    print(f'Mean: {mean}')
    print(f'Standard deviation: {std_dev}')
    print(f'95% Confidence Interval: {interval}')

    # Generate x values around the mean
    x_values = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 1000)
    y_values = norm.pdf(x_values, mean, std_dev)

    # Plot for confidence interval
    if show_plots:
        # Create plot to show confidence interval around mean with a normal distribution
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, label="Normal Distribution", color="blue")

        # Confidence interval lines
        plt.axvline(mean, color='red', linestyle='--', label=f'mean of {mean}')
        plt.axvline(interval[0], color='orange', linestyle='--', label=f'left-sided confidence interval of {interval[0]}')
        plt.axvline(interval[1], color='orange', linestyle='--', label=f'right-sided confidence interval of {interval[1]}')

        # Add labels and title
        plt.title(f"Confidence interval of mean of {sim_type} earthquakes")
        plt.xlim((mean - 3 * std_dev, mean + 3 * std_dev))  # Set limits to match x_values range
        plt.xlabel("Value")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # Data analysis - Descriptive Statistics
    all_eq_data, high_mag_eq_data, low_mag_eq_data = data_exploration(data_path='data/query_Philippines.csv',
                                                                      aggregate_period='M',
                                                                      aggregate_label='month',
                                                                      show_plots=False)

    # # Data analysis - Descriptive Statistics (Plot earthquake locations)
    # plot_geospatial(all_eq_data, magnitude_label='all')
    # plot_geospatial(high_mag_eq_data, magnitude_label='high magnitude')
    # plot_geospatial(low_mag_eq_data, magnitude_label='low magnitude')

    # Data Analysis - Distribution Fitting (Transform dataframes with earthquakes time to interarrival times)
    diff_data_eq = transform_time_diff_data(all_eq_data)
    diff_data_low_mag_eq = transform_time_diff_data(low_mag_eq_data)
    diff_data_high_mag_eq = transform_time_diff_data(high_mag_eq_data)

    # Data Analysis - Distribution Fitting (For interarrival times)
    estExpDist, estNormDist, estGammaDist, estUniformDist = (
        fit_distribution(diff_data_eq, show_plots=False))
    estExpDist_low_mag, estNormDist_low_mag, estGammaDist_low_mag, estUniformDist_low_mag = (
        fit_distribution(diff_data_low_mag_eq, show_plots=False))
    estExpDist_high_mag, estNormDist_high_mag, estGammaDist_high_mag, estUniformDist_high_mag = (
        fit_distribution(diff_data_high_mag_eq, show_plots=False))

    # Data Analysis - Distribution Fitting (Probability of < 5 magnitude earthquake)
    prob_mag_less_5 = calculate_probabilities(all_eq_data, show_plots=False)

    # Simulation (Method 1)
    m1_sim_num_quakes_low_mag, m1_sim_num_quakes_high_mag, m1_sim_num_quakes_total = (
        simulation_method_1(num_of_years=100, fit_dist_low=estGammaDist_low_mag, fit_dist_high=estGammaDist_high_mag))

    # # Simulation (Method 1: Find Distribution, mean and sd and plot some figures)
    # fit_distribution_simulation(m1_sim_num_quakes_total, sim_type='all', show_plots=False)
    fit_distribution_simulation(m1_sim_num_quakes_low_mag, sim_type='type-2', show_plots=True)
    # fit_distribution_simulation(m1_sim_num_quakes_high_mag, sim_type='type-1', show_plots=False)


    # Simulation (Method 2)
    m2_sim_num_quakes_low_mag, m2_sim_num_quakes_high_mag, m2_sim_num_quakes_total = (
        simulation_method_2(num_of_years=100, fit_dist_total=estGammaDist, prob_lower_5=prob_mag_less_5))

    # # Simulation (Method 2: Find Distribution, mean and sd and plot some figures)
    # fit_distribution_simulation(m2_sim_num_quakes_total, sim_type='all', show_plots=False)
    fit_distribution_simulation(m2_sim_num_quakes_low_mag, sim_type='type-2', show_plots=True)
    # fit_distribution_simulation(m2_sim_num_quakes_high_mag, sim_type='type-1', show_plots=False)


    # # Sensitivity analysis (Sample from the empirical data)
    # emp_sim_data_total, emp_sim_num_quakes_total = (
    #     simulate_empirical(num_of_years=100, empirical_data=diff_data_eq))
    emp_sim_data_low_mag, emp_sim_num_quakes_low_mag = (
        simulate_empirical(num_of_years=100, empirical_data=diff_data_low_mag_eq))
    # emp_sim_data_high_mag, emp_sim_num_quakes_high_mag = (
    #     simulate_empirical(num_of_years=100, empirical_data=diff_data_high_mag_eq))

    # # Sensitivity analysis (Find Distribution, mean and sd and plot some figures)
    # fit_distribution_simulation(emp_sim_num_quakes_total, sim_type='all', show_plots=False)
    fit_distribution_simulation(emp_sim_num_quakes_low_mag, sim_type='type-2', show_plots=True)
    # fit_distribution_simulation(emp_sim_num_quakes_high_mag, sim_type='type-1', show_plots=False)





