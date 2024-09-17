import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm

# Load and preprocess the data
@st.cache_data
def load_data():
    file = 'insight7.xlsx'
    data = pd.read_excel(file)
    
    # Calculate total screenings for each weekend
    data['Total First Weekend'] = data[['Friday Screenings', 'Saturday Screenings', 'Sunday Screenings']].sum(axis=1)
    data['Total Second Weekend'] = data[['Second Friday Screenings', 'Second Saturday Screenings', 'Second Sunday Screenings']].sum(axis=1)
    
    # Remove rows where either weekend total is 0
    data = data[(data['Total First Weekend'] != 0) & (data['Total Second Weekend'] != 0)]
    
    # Calculate percentage drop
    data['Percentage Drop'] = ((data['Total First Weekend'] - data['Total Second Weekend']) / data['Total First Weekend']) * 100
    
    return data

# Function to plot daily screenings for a specific movie
def plot_screenings_for_movie(data, selected_movie):
    st.subheader(f"Daily Screenings for '{selected_movie}'")
    
    movie_data = data[data['Film'] == selected_movie]
    movie_data = movie_data[['Friday Screenings', 'Saturday Screenings', 'Sunday Screenings', 
                             'Second Friday Screenings', 'Second Saturday Screenings', 'Second Sunday Screenings']]

    days = ['Friday', 'Saturday', 'Sunday']
    weekend_1 = movie_data[['Friday Screenings', 'Saturday Screenings', 'Sunday Screenings']].values.flatten()
    weekend_2 = movie_data[['Second Friday Screenings', 'Second Saturday Screenings', 'Second Sunday Screenings']].values.flatten()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(days, weekend_1, alpha=0.6, label='First Weekend')
    ax.bar(days, weekend_2, alpha=0.6, label='Second Weekend')
    ax.set_title(f'Daily Screenings for {selected_movie}')
    ax.set_xlabel('Day')
    ax.set_ylabel('Number of Screenings')
    ax.legend()
    
    st.pyplot(fig)
    
    st.write(f"This graph shows the number of screenings allocated for each day of the first and second weekends of '{selected_movie}'. The bars represent the number of screenings on Friday, Saturday, and Sunday during the opening weekend and the second weekend. This helps to understand how the distribution of screenings changes over these two key periods.")

# Function to analyze daily variation during the first weekend
def analyze_daily_variation(data, selected_movie):
    st.subheader(f"Daily Variation Analysis During the First Weekend for '{selected_movie}'")
    
    movie_data = data[data['Film'] == selected_movie]
    daily_screenings = movie_data[['Friday Screenings', 'Saturday Screenings', 'Sunday Screenings']].values.flatten()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(daily_screenings, kde=True)
    ax.set_title(f'Distribution of Daily Screenings During the First Weekend of {selected_movie}')
    ax.set_xlabel('Number of Screenings')
    ax.set_ylabel('Frequency')
    
    st.pyplot(fig)
    
    st.write(f"This histogram shows the distribution of daily screenings for '{selected_movie}' during the first weekend. The Kernel Density Estimate (KDE) line helps visualize the distribution of the number of screenings per day. This analysis helps understand if there are significant fluctuations in the number of screenings on different days during the opening weekend.")
    
    return daily_screenings

# Function to plot Gaussian distribution of daily screenings
def plot_gaussian_distribution(screenings):
    st.subheader("Gaussian Distribution of Daily Screenings")
    
    mu, std = norm.fit(screenings)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(screenings, kde=False, stat='density', bins=20, color='blue')
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2)
    ax.set_title('Gaussian Distribution of Daily Screenings')
    ax.set_xlabel('Number of Screenings')
    ax.set_ylabel('Density')
    
    st.pyplot(fig)
    
    st.write(f"This plot shows the Gaussian distribution of daily screenings for the selected movie. The histogram represents the density of daily screenings, and the overlaid curve shows the fitted normal distribution. This analysis helps to understand the typical number of screenings and whether the distribution follows a normal pattern.")

# Function to analyze percentage drop for all movies
def analyze_percentage_drop(data):
    st.write("**Top Films by Screening Drop**")
    
    sorted_data = data.sort_values(by='Percentage Drop', ascending=True)
    display_data = sorted_data[['Film', 'Total First Weekend', 'Total Second Weekend', 'Percentage Drop']]
    
    def color_change(val):
        try:
            value = float(val)
            if value < 0:
                return 'color: green'
            elif value > 0:
                return 'color: red'
            else:
                return ''
        except:
            return ''
    
    st.write(display_data.style.format({
        'Total First Weekend': '{:.0f}',
        'Total Second Weekend': '{:.0f}',
        'Percentage Drop': '{:.2f}'
    }).applymap(color_change, subset=['Percentage Drop']))

# Function to plot Gaussian distribution of average daily screenings over the opening weekend for all movies (in %)
def plot_gaussian_average_screenings_percentage(data):
    st.subheader("Gaussian Distribution of Average Daily Screenings Over Opening Weekend for All Movies (in %)")
    
    # Calculate the average daily screenings for the first weekend (Friday, Saturday, Sunday)
    data['Average Daily Screenings'] = data[['Friday Screenings', 'Saturday Screenings', 'Sunday Screenings']].mean(axis=1)
    
    # Fit a Gaussian distribution to the average daily screenings
    avg_screenings = data['Average Daily Screenings'].values
    mu, std = norm.fit(avg_screenings)
    
    # Create the histogram with percentage
    fig, ax = plt.subplots(figsize=(10, 6))
    count, bins, _ = ax.hist(avg_screenings, bins=20, density=False, color='blue', alpha=0.6)
    
    # Convert to percentage
    count_percentage = (count / count.sum()) * 100
    
    # Plot histogram as percentages
    ax.cla()  # Clear previous plot
    ax.bar(bins[:-1], count_percentage, width=np.diff(bins), edgecolor='black', align='edge', color='blue')
    
    # Plot the fitted normal distribution as a line
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std) * 100  # Convert to percentage
    ax.plot(x, p, 'k', linewidth=2)
    
    # Labels and titles
    ax.set_title('Gaussian Distribution of Average Daily Screenings Over Opening Weekend (in %)')
    ax.set_xlabel('Average Number of Screenings')
    ax.set_ylabel('Percentage of Movies (%)')
    
    st.pyplot(fig)
    
    st.write(f"This plot shows the Gaussian distribution of the average number of daily screenings over the opening weekend for all movies, expressed as a percentage. "
             f"The bars represent the percentage of movies that fall within each range of daily screenings, and the curve shows the fitted normal distribution.")

# Streamlit app
def insight7():
    st.title("Movie Screening Analysis")

    data = load_data()

    # Dropdown to select a movie, sorted alphabetically
    selected_movie = st.selectbox("Select a movie to analyze", sorted(data['Film'].unique()))

    # Display plots for the selected movie
    plot_screenings_for_movie(data, selected_movie)
    screenings = analyze_daily_variation(data, selected_movie)

    # Plot Gaussian distribution if daily variation is low
    if np.std(screenings) < 100:
        plot_gaussian_distribution(screenings)
    else:
        st.write("The daily variation is significant. Gaussian distribution will not be plotted.")
    
    # Plot Gaussian distribution of average daily screenings over the opening weekend for all movies (in %)
    plot_gaussian_average_screenings_percentage(data)

    # Analyze percentage drop for all movies
    analyze_percentage_drop(data)

if __name__ == '__main__':
    insight7()

