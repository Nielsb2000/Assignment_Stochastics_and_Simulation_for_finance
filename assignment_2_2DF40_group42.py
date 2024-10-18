import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import LineString
import plotly.express as px
from shapely.geometry import Point
from tabulate import tabulate
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm

def plot_counts(region, aggregate_period, aggregate_label):
    # Import the data to a dataframe
    data_eq = pd.read_csv(f"data/query_{region}.csv")

    # Transform 'time' column values to DateTime objects
    data_eq['DateTime'] = pd.to_datetime(data_eq['time'])
    data_eq = data_eq.sort_values(by='DateTime')   # sort from earliest to newest

    # Select only earthquakes within Philippines' borders
    selection2 = data_eq['place'].str.contains('Philippines', case=True)
    data_eq = data_eq[selection2]

    # Only keep important column and set the new DateTime column as index
    data_eq_dropped = data_eq[['DateTime','latitude','longitude','mag']].set_index('DateTime')
    # print(tabulate(data_eq_dropped.head(), headers='keys', tablefmt="github"))

    # Split DataFrame into 2 DataFrames for earthquakes with magnitudes lower and higher than 5
    high_mag_data = data_eq_dropped[data_eq_dropped['mag']>=5]
    low_mag_data = data_eq_dropped[data_eq_dropped['mag']<5]

    # Count amount of earthquakes per day/week/month
    full_counts = data_eq_dropped.resample(aggregate_period).size()
    print(f'Full amount of earthquakes: {sum(full_counts)}')

    low_counts = low_mag_data.resample(aggregate_period).size()
    print(f'Full amount of low magnitude earthquakes: {sum(low_counts)}')

    high_counts = high_mag_data.resample(aggregate_period).size()
    print(f'Full amount of high magnitude earthquakes: {sum(high_counts)}')

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

    fig.suptitle(f'Amount of reported earthquakes in {region} aggregated per {aggregate_label}', size=20)
    plt.show()

    # Returned dataframes
    data_eq = data_eq_dropped.reset_index()
    high_mag_data = high_mag_data.reset_index()
    low_mag_data = low_mag_data.reset_index()

    # Extra: Plot a distribution of magnitudes for peak day
    peak_start = '2023-12-01'
    peak_end = '2023-12-08'

    # Filter data for peak periods
    peak_df = data_eq[(data_eq['DateTime'] >= peak_start) & (data_eq['DateTime'] <= peak_end)]

    # Plot histogram of magnitudes in peak periods
    plt.hist(peak_df['mag'], bins=30, edgecolor='black', density=False)
    plt.title('Earthquake Magnitudes During Peak Period', size=18)
    plt.xlabel('Magnitude', size=11)
    plt.ylabel('Count', size=11)
    plt.show()

    # Data Summary Statistics for magnitudes
    print(data_eq['mag'].describe())
    print(high_mag_data['mag'].describe())
    print(low_mag_data['mag'].describe())

    # Largest magnitude earthquakes:
    mag_sorted = data_eq.sort_values(by='mag', ascending=False)
    print(mag_sorted.head())

    return data_eq, high_mag_data, low_mag_data

def plot_geospatial(data_eq, magnitude_label):
    # Import map of the earth from geopandas
    worldmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

    # Get map of Philippines
    philippines_map = worldmap[worldmap['name'] == 'Philippines']

    # Plot earthquake data points on Philippines map
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
    # Calculate differences in time between values in the 'DateTime' column
    data['time_diff'] = data['DateTime'].diff()

    # Drop the first row, as it will be a NaN value (first time has no time to subtract from)
    data = data[['time_diff']].drop(index=0)
    # Convert the time differences to seconds
    data['time_diff_seconds'] = pd.to_timedelta(data['time_diff']).dt.total_seconds()

    # Take only relevant column
    time_diff_data = data['time_diff_seconds'].copy()

    return time_diff_data

def fit_distribution(time_diff_data, region, magnitude, show_plots):
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

    if show_plots is True:
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

    with open('data/test_results.txt', 'a') as f:
        f.write(f'\n\n{magnitude} magnitudes for region {region}')
        f.write('\nKS Test Exponential distribution: ' + str(test_exponential[1]))
        f.write('\nKS Test Normal distribution: ' + str(test_normal[1]))
        f.write('\nKS Test Gamma distribution: ' + str(test_gamma[1]))
        f.write('\nKS Test Uniform distribution: ' + str(test_uniform[1]))

    return (estExpDist, estNormDist, estGammaDist, estUniformDist)

def calculate_probabilities(data_eq):
    # Cumulative distribution function for magnitudes
    ecdf = ECDF(data_eq['mag'])
    plt.figure(figsize=(9, 9))
    plt.step(ecdf.x, ecdf.y, color='black', where='post', label='Interarrival Times eCDF')

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

def simulation(fitted_distribution):
    def simulate_year_from_empdata(fitted_distribution):
        """Empirical data has to be in seconds and returns a year of earthquakes based on that data"""
        seconds_in_year = 365.25 * 24 * 3600  # Approx. 31,557,600 seconds
        current_time = 0
        year_of_quakes = []
        # Simulate earthquakes until we exceed one year
        while current_time < seconds_in_year:
            # Sample an interarrival time from the empirical data
            sampled_time = fitted_distribution.rvs(1)  # samples random earthquake
            current_time += sampled_time  # calc total time

            if current_time < seconds_in_year:  # if total time is less than a year
                # add earthquake (time in seconds between the last one and this one) to that year list of earthquakes
                year_of_quakes.append(float(sampled_time[0]))
                quake_count = len(year_of_quakes)
        return year_of_quakes, quake_count

    def sim_num_years(T: int, fitted_distribution):
        sim_data = []
        num_of_quakes_each_year = []
        for n in range(T):
            year_of_quakes, quake_count = simulate_year_from_empdata(fitted_distribution)
            sim_data.append(year_of_quakes)
            num_of_quakes_each_year.append(quake_count)
        return sim_data, np.array(num_of_quakes_each_year)

    num_of_years = 100  # Also the T variable in N(T)

    sim_data, sim_num_quakes = sim_num_years(num_of_years, fitted_distribution)

    return sim_data, sim_num_quakes

def fit_distribution_simulation(sim_num_quakes, sim_type):
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

    # Confidence intervals (# 2. Using Slutsky's theorem: estimate the sample variance)
    s2 = np.var(sim_num_quakes)
    # s2 = alphaEst / (betaEst ** 2)
    m = np.mean(sim_num_quakes)
    z = 1.96
    half_width = z * np.sqrt(s2 / len(sim_num_quakes))

    # lower_bound = max(0, m - half_width)
    # upper_bound = m + half_width
    # interval = (lower_bound, upper_bound)

    interval = (m - half_width, m + half_width)

    mean = M1
    # std_dev = np.sqrt(M2) # std dev approx from the interval
    std_dev = np.sqrt(sigma2Est)

    print(f'Mean: {mean}')
    print(f'Standard deviation: {std_dev}')
    print(f'95% Confidence Interval: {interval}')

    # Generate x values around the mean
    x_values = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 1000)
    y_values = norm.pdf(x_values, mean, std_dev)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label="Normal Distribution", color="blue")

    # Plot the vertical line at the given point
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
    # Geographical area of query
    region = 'Philippines'

    # Data analysis - Descriptive Statistics
    data_eq, data_high_mag_eq, data_low_mag_eq = plot_counts(region=region,
                                                             aggregate_period='M',
                                                             aggregate_label='month')

    # # Data analysis - Descriptive Statistics (Plot earthquake locations)
    # plot_geospatial(data_eq, magnitude_label='all')
    # plot_geospatial(data_high_mag_eq, magnitude_label='high magnitude')
    # plot_geospatial(data_low_mag_eq, magnitude_label='low magnitude')

    # Data Analysis - Distribution Fitting (Transform dataframes with earthquakes time to interarrival times)
    diff_data_eq = transform_time_diff_data(data_eq)
    diff_data_low_mag_eq = transform_time_diff_data(data_low_mag_eq)
    diff_data_high_mag_eq = transform_time_diff_data(data_high_mag_eq)

    # # Data Analysis - Distribution Fitting (For interarrival times)
    # estExpDist, estNormDist, estGammaDist, estUniformDist = (
    #     fit_distribution(diff_data_eq, region=region, magnitude='All', show_plots=False))
    estExpDist_low_mag, estNormDist_low_mag, estGammaDist_low_mag, estUniformDist_low_mag = (
        fit_distribution(diff_data_low_mag_eq, region=region, magnitude='Low', show_plots=False))
    estExpDist_high_mag, estNormDist_high_mag, estGammaDist_high_mag, estUniformDist_high_mag = (
        fit_distribution(diff_data_high_mag_eq, region=region, magnitude='High', show_plots=False))

    # # Data Analysis - Distribution Fitting (Probabilities of < 5 magnitude earthquake)
    # calculate_probabilities(data_eq)

    # Simulation (Method 1: Simulate low and high magnitude earthquakes)
    sim_quakes_low_mag, sim_num_quakes_low_mag = simulation(estGammaDist_low_mag)
    sim_quakes_high_mag, sim_num_quakes_high_mag = simulation(estGammaDist_high_mag)

    # Simulation (Method 1: Find Distribution, mean and sd for low and high magnitude earthquakes)
    fit_distribution_simulation(sim_num_quakes_low_mag, 'type-1')
    fit_distribution_simulation(sim_num_quakes_high_mag, 'type-2')

    # Simulation (Method 1: Find Distribution, mean and sd for all earthquakes combined)
    sim_num_quakes_total = sim_num_quakes_low_mag + sim_num_quakes_high_mag
    fit_distribution_simulation(sim_num_quakes_total, 'all')


