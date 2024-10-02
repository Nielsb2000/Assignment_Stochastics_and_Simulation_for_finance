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

def plot_counts(region):
    # Import the data to a dataframe
    data_eq = pd.read_csv(f"data/query_{region}.csv")
    print(tabulate(data_eq.head(), headers='keys', tablefmt="github"))

    # Transform 'time' column values to DateTime objects
    data_eq['DateTime'] = pd.to_datetime(data_eq['time'])
    data_eq = data_eq.sort_values(by='DateTime')   # sort from earliest to newest

    # Select only earthquakes within Philippines' borders
    # selection2 = data['place'].str.contains('Philippines', case=True)
    # data = data[selection2]

    # Only keep important column and set the new DateTime column as index
    data_eq_dropped = data_eq[['DateTime','latitude','longitude','mag']].set_index('DateTime')
    print(tabulate(data_eq_dropped.head(), headers='keys', tablefmt="github"))

    # Split DataFrame into 2 DataFrames for earthquakes with magnitudes lower and higher than 5
    high_mag_data = data_eq_dropped[data_eq_dropped['mag']>=5]
    low_mag_data = data_eq_dropped[data_eq_dropped['mag']<5]

    # Count amount of earthquakes per day/week
    full_daily_counts = data_eq_dropped.resample('D').size()
    print(sum(full_daily_counts))
    full_weekly_counts = low_mag_data.resample('W').size()

    low_daily_counts = low_mag_data.resample('D').size()
    print(sum(low_daily_counts))
    low_weekly_counts = low_mag_data.resample('W').size()

    high_daily_counts = high_mag_data.resample('D').size()
    print(sum(high_daily_counts))
    high_weekly_counts = high_mag_data.resample('W').size()

    # Make plots to show amount of earthquakes per day
    fig, axs = plt.subplots(2, 1, figsize=(10, 15))

    axs[0].plot(full_daily_counts, label="Full Daily Counts")
    axs[0].set_title("Full Counts")

    axs[1].plot(low_daily_counts, label="Low Mag Daily Counts")
    axs[1].set_title("High vs Low Counts")

    axs[1].plot(high_daily_counts, label="High Mag Daily Counts")
    axs[1].legend()
    fig.suptitle('Daily and Weekly Counts Comparison')
    plt.xlabel('Time Period')
    plt.ylabel('Counts')
    plt.show()

    data_eq = data_eq_dropped.reset_index()
    high_mag_data = high_mag_data.reset_index()
    low_mag_data = low_mag_data.reset_index()
    return data_eq, high_mag_data, low_mag_data

def plot_geospatial(data_rest):
    worldmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    print(tabulate(worldmap.head(), headers='keys', tablefmt="github"))

    philippines_map = worldmap[worldmap['name'] == 'Philippines']
    # philippines_map = worldmap[worldmap['name'] == 'Fiji']
    # philippines_map = worldmap[worldmap['name'] == 'Indonesia']


    philippines_map.plot()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 10))

    data_rest['geometry'] = data_rest.apply(lambda x: Point(x['longitude'], x['latitude']), axis=1)
    data_rest = gpd.GeoDataFrame(data_rest, geometry='geometry')


    philippines_map.plot(ax=ax, color='lightgray')
    data_rest.plot(ax=ax, color='red', markersize=data_rest['mag'], alpha=0.05*data_rest['mag'], label='Earthquakes')
    plt.show()

def fit_distribution(data, region, magnitude):
    # Calculate differences in time between values in the 'DateTime' column
    data['time_diff'] = data['DateTime'].diff()

    # Drop the first row, as it will be a NaN value (first time how no time to subtract from)
    data = data[['time_diff']].drop(index=0)
    # Convert the time differences to seconds
    data['time_diff_seconds'] = pd.to_timedelta(data['time_diff']).dt.total_seconds()

    # Take only relevant column
    time_diff_data = data['time_diff_seconds']

    # Plot a histogram to show density of difference in seconds between earthquakes
    plt.hist(time_diff_data, bins=200, rwidth=0.8, edgecolor='black', density=True)
    plt.xlabel('Time Difference (seconds)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Time Differences')
    plt.show()

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

    # The estimated distributions
    estExpDist = stats.expon(scale=1 / lamEst)
    estNormDist = stats.norm(muEst, np.sqrt(sigma2Est))
    estGammaDist = stats.gamma(alphaEst, scale=1 / betaEst)

    # Add theoretical density
    xs = np.arange(np.min(time_diff_data), np.max(time_diff_data))
    print(xs)

    # Show a plot with the histogram of the time difference data, and the estimated distributions
    plt.figure()
    plt.hist(time_diff_data, bins=200, rwidth=0.8, edgecolor='black', density=True)
    plt.plot(xs, estExpDist.pdf(xs), 'r', label='Exponential Distribution')
    plt.plot(xs, estNormDist.pdf(xs), 'b', label='Normal Distribution')
    # plt.plot(xs, estGammaDist.pdf(xs), 'y', label='Gamma Distribution')
    plt.legend()
    plt.show()

    # Empirical distribution function (Method 2: using Python function ECDF)
    ecdf = ECDF(time_diff_data)
    plt.figure()
    plt.step(ecdf.x, ecdf.y, color='black', where='post', label='Time Difference ECDF')
    plt.plot(xs, estExpDist.cdf(xs), 'r', label='Exponential Distribution')
    plt.plot(xs, estNormDist.cdf(xs), color='b', label='Normal Distribution')
    plt.plot(xs, estGammaDist.cdf(xs), 'y', label='Gamma Distribution')
    plt.legend()
    plt.show()

    # Kolmogorov-Smirnov test
    test_exponential = stats.kstest(time_diff_data, estExpDist.cdf)
    test_normal = stats.kstest(time_diff_data, estNormDist.cdf)
    test_gamma = stats.kstest(time_diff_data, estGammaDist.cdf)
    print('KS Test Exponential distribution: ' + str(test_exponential))
    print('KS Test Normal distribution: ' + str(test_normal))
    print('KS Test Gamma distribution: ' + str(test_gamma))

    with open('data/test_results.txt', 'a') as f:
        f.write(f'\n\n{magnitude} magnitudes for region {region}')
        f.write('\nKS Test Exponential distribution: ' + str(test_exponential[1]))
        f.write('\nKS Test Normal distribution: ' + str(test_normal[1]))
        f.write('\nKS Test Gamma distribution: ' + str(test_gamma[1]))


if __name__ == "__main__":
    # region = 'Philippines'
    # region = 'Fiji'
    region = 'Indonesia'
    # region = 'Alaska'
    # region = 'Japan'

    data_eq, data_high_mag_eq, data_low_mag_eq = plot_counts(region=region)

    fit_distribution(data_eq, region=region, magnitude='All')
    fit_distribution(data_low_mag_eq, region=region, magnitude='Low')
    fit_distribution(data_high_mag_eq, region=region, magnitude='High')

    # plot_geospatial(data_eq)
