import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path, release_dates_path):
    try:
        # Load the movie revenue data
        df = pd.read_excel(file_path)
        
        # Load the release dates with proper error handling
        release_dates = pd.read_csv(release_dates_path, dayfirst=True)
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

    return df, release_dates

def preprocess_data(df, release_dates):
    # Transpose the DataFrame so dates are rows and movies are columns
    df_transposed = df.set_index('Movie Name').T
    df_transposed.columns.name = None
    df_transposed.reset_index(inplace=True)
    df_transposed.rename(columns={'index': 'Date'}, inplace=True)

    # Convert 'Date' to datetime format
    df_transposed['Date'] = pd.to_datetime(df_transposed['Date'], format='%Y-%m-%d')

    # Transform the DataFrame to long format for analysis
    df_long = df_transposed.melt(id_vars=['Date'], var_name='Movie Name', value_name='Revenue')

    # Convert 'Revenue' to numeric, handling errors by coercing invalid values to NaN
    df_long['Revenue'] = pd.to_numeric(df_long['Revenue'], errors='coerce')

    # Merge with release dates data
    df_long = df_long.merge(release_dates, on='Movie Name', how='left')
    df_long['Release Date'] = pd.to_datetime(df_long['Release Date'], dayfirst=True)

    return df_long

def plot_revenue_trends(df_filtered, selected_movie):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df_filtered, x=df_filtered.index, y='Revenue', ax=ax, marker='o')
    ax.set_title(f'Revenue of {selected_movie} from 7 days before to 21 days after release date')
    ax.set_xlabel('Date')
    ax.set_ylabel('Revenue (Billion VND)')
    st.pyplot(fig)

def calculate_weekend_revenue(df_movie):
    premiere_date = df_movie['Release Date'].iloc[0]
    df_movie = df_movie[df_movie.index >= premiere_date]
    first_weekend_end = premiere_date + pd.Timedelta(days=6)
    second_weekend_end = premiere_date + pd.Timedelta(days=13)
    
    first_weekend_revenue = df_movie.loc[df_movie.index <= first_weekend_end, 'Revenue'].sum()
    second_weekend_revenue = df_movie.loc[(df_movie.index > first_weekend_end) & (df_movie.index <= second_weekend_end), 'Revenue'].sum()

    return first_weekend_revenue, second_weekend_revenue

def insight2(file_path='insight2.xlsx', release_dates_path='release_dates.csv'):
    # Load and preprocess data
    df, release_dates = load_data(file_path, release_dates_path)
    if df is None or release_dates is None:
        return
    
    df_long = preprocess_data(df, release_dates)

    # Calculate total revenue for each movie
    movie_revenue = df_long.groupby('Movie Name')['Revenue'].sum()

    # Initialize button states in session state
    if 'filter_option' not in st.session_state:
        st.session_state.filter_option = 'all'
    
    # Buttons to filter movies
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button('Show Movies with Total Revenue < 45 Billion VND'):
            st.session_state.filter_option = 'less_than_45b'
    
    with col2:
        if st.button('Show Movies with Total Revenue >= 45 Billion VND'):
            st.session_state.filter_option = 'greater_than_45b'

    with col3:
        if st.button('Show All Movies'):
            st.session_state.filter_option = 'all'

    # Filter movies based on the selected option
    if st.session_state.filter_option == 'less_than_45b':
        movies = movie_revenue[movie_revenue < 45e9].index
    elif st.session_state.filter_option == 'greater_than_45b':
        movies = movie_revenue[movie_revenue >= 45e9].index
    else:
        movies = df_long['Movie Name'].unique()

    # Initialize the selected movie index in session state
    if 'selected_index' not in st.session_state:
        st.session_state.selected_index = 0

    # Navigation buttons to move through movies
    col1, col2, col3 = st.columns([1, 5, 1])

    with col1:
        if st.button('Previous Movie'):
            st.session_state.selected_index = (st.session_state.selected_index - 1) % len(movies)

    with col3:
        if st.button('Next Movie'):
            st.session_state.selected_index = (st.session_state.selected_index + 1) % len(movies)

    # Dropdown to select a movie
    selected_movie = st.selectbox('Select a movie to analyze:', movies, index=st.session_state.selected_index)

    # Ensure the selected movie matches the index
    if selected_movie != movies[st.session_state.selected_index]:
        st.session_state.selected_index = list(movies).index(selected_movie)

    # Filter data for the selected movie
    df_movie = df_long[df_long['Movie Name'] == selected_movie]
    df_movie = df_movie[['Date', 'Revenue', 'Release Date']]
    df_movie.set_index('Date', inplace=True)
    df_movie.sort_index(inplace=True)

    # Get the release date for the selected movie
    release_date = df_movie['Release Date'].iloc[0]
    st.write(f"Release date for {selected_movie}: {release_date.strftime('%d/%m/%Y')}")

    # Define the date range centered around the release date
    start_date = release_date - pd.Timedelta(days=7)
    end_date = release_date + pd.Timedelta(days=21)

    # Filter data within the date range
    df_filtered = df_movie[(df_movie.index >= start_date) & (df_movie.index <= end_date)]

    # Plot the revenue trends
    plot_revenue_trends(df_filtered, selected_movie)

    # Calculate and display weekend revenue
    first_weekend_revenue, second_weekend_revenue = calculate_weekend_revenue(df_movie)
    st.write(f"Total revenue in the first weekend: {first_weekend_revenue / 1e9:,.2f} billion VND")
    st.write(f"Total revenue in the second weekend: {second_weekend_revenue / 1e9:,.2f} billion VND")

# Run the app
if __name__ == "__main__":
    insight2()
