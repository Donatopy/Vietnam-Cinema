import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Function to load key dates from an Excel file
def load_key_dates_from_excel():
    key_dates_df = pd.read_excel('key_dates.xlsx')
    key_dates = {}

    # Iterate over each row to process the key dates
    for _, row in key_dates_df.iterrows():
        year = int(row['Year'])
        key_dates[year] = {
            "New Year's Day": datetime.strptime(str(row["New Year's Day"]), '%Y-%m-%d'),
            'Lunar New Year Start': datetime.strptime(str(row['Lunar New Year Start']), '%Y-%m-%d'),
            'Lunar New Year End': datetime.strptime(str(row['Lunar New Year End']), '%Y-%m-%d'),
            "Hung Kings' Temple Festival": datetime.strptime(str(row["Hung Kings' Temple Festival"]), '%Y-%m-%d'),
            'Reunification Day': datetime.strptime(str(row['Reunification Day']), '%Y-%m-%d'),
            'International Labor Day': datetime.strptime(str(row['International Labor Day']), '%Y-%m-%d'),
            'National Day': datetime.strptime(str(row['National Day']), '%Y-%m-%d')
        }

    return key_dates

# Function to check if a date falls on a key event (+- n days margin)
def is_key_date(date, key_dates, margin_days):
    year = date.year
    if year in key_dates:
        for event, event_date in key_dates[year].items():
            if event_date - timedelta(days=margin_days) <= date <= event_date + timedelta(days=margin_days):
                return event
    return 'Other'

# Function to load and process revenue data
def load_and_process_revenue_data(file_path):
    df = pd.read_excel(file_path, engine='openpyxl')
    df['Ca chiếu'] = pd.to_datetime(df['Ca chiếu'], format='%H:%M:%S').dt.time
    df['Hour'] = pd.to_datetime(df['Ca chiếu'], format='%H:%M:%S').dt.hour
    df['Revenue (Billions VND)'] = df['Doanh thu'] / 1_000_000_000
    return df

    
# Function to load and process revenue data from insight2.xlsx
def load_revenue_from_insight2(file_path):
    # Load the Excel file
    revenue_df = pd.read_excel(file_path, engine='openpyxl')

    # Calculate total revenue for each movie by summing the values across date columns
    revenue_df['Total Revenue (Billions VND)'] = revenue_df.iloc[:, 1:].sum(axis=1) / 1_000_000_000
    
    return revenue_df[['Movie Name', 'Total Revenue (Billions VND)']]

# Main function to generate the Streamlit app
def insight5():
    # Read data from the CSV file
    df = pd.read_csv('release_dates.csv')
    df['Release Date'] = pd.to_datetime(df['Release Date'], format='%d/%m/%Y')
    
    # Load key dates from the Excel file
    key_dates = load_key_dates_from_excel()

    # Load revenue data from insight2.xlsx
    revenue_file_path = 'insight2.xlsx'
    revenue_df = load_revenue_from_insight2(revenue_file_path)

    # Merge the revenue data with the main df
    df = df.merge(revenue_df, left_on='Movie Name', right_on='Movie Name', how='left')

    # Margin days setting
    st.title('Movie Release Insights')

    # Filter by year
    st.header('Filter by Year')
    years = ["All Years"] + sorted(df['Release Date'].dt.year.unique().tolist())
    selected_year = st.selectbox("Select Year", options=years, key="year_filter")

    if selected_year != "All Years":
        df = df[df['Release Date'].dt.year == int(selected_year)]

    margin_days = st.slider('Select margin days for key dates', min_value=0, max_value=10, value=3, key="margin_days_slider")
    exclude_other = st.checkbox('Exclude "Other" from Key Date Event chart', key="exclude_other_checkbox")

    df['Key Date Event'] = df['Release Date'].apply(lambda x: is_key_date(x, key_dates, margin_days))
    df['Month'] = df['Release Date'].dt.month

    # Section 1: Optimal Release Periods
    st.header('Optimal Release Periods')
    st.write(f"We also consider a margin of ±{margin_days} days around key dates to account for movies "
             f"released in close proximity to these events.")

    event_counts = df['Key Date Event'].value_counts()
    if exclude_other:
        event_counts = event_counts[event_counts.index != 'Other']
    st.bar_chart(event_counts)

    # Section 2: Releases by Month
    st.header('Releases by Month')

    # Count movies by month
    month_counts = df['Month'].value_counts().sort_index()

    # Count movies by month with revenue > 50 billion VND
    high_revenue_df = df[df['Total Revenue (Billions VND)'] > 50]
    high_revenue_month_counts = high_revenue_df['Month'].value_counts().sort_index()

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot total releases
    ax.bar(month_counts.index, month_counts.values, color='blue', label='Total Releases')

    # Overlay movies with revenue > 50 billion VND
    ax.bar(high_revenue_month_counts.index, high_revenue_month_counts.values, color='orange', label='Releases with Revenue > 50B VND')

    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Releases')
    ax.set_title('Releases by Month with High Revenue Highlight')
    ax.legend()

    st.pyplot(fig)

    # Section 3: Revenue Analysis by Time
    st.header('Revenue Analysis by Time')

    # Load and process revenue data from the original revenue file
    revenue_time_file_path = 'revenue_by_time.xlsx'
    revenue_time_df = load_and_process_revenue_data(revenue_time_file_path)

    total_revenue = revenue_time_df['Revenue (Billions VND)'].sum()
    st.write(f"Total Revenue: {total_revenue:.2f} billion VND")

    revenue_by_hour = revenue_time_df.groupby('Hour')['Revenue (Billions VND)'].sum()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(revenue_by_hour.index, revenue_by_hour.values, marker='o', linestyle='-')

    x_ticks = [0, 10, 13] + list(range(13, 24))
    ax.set_xticks(x_ticks)

    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Revenue (Billions VND)')
    ax.set_title('Revenue by Hour of the Day')
    ax.grid(True)

    st.pyplot(fig)

    st.header('Revenue Breakdown by Hour')
    revenue_table = revenue_by_hour.reset_index()
    revenue_table.columns = ['Hour of the Day', 'Revenue (Billions VND)']
    st.table(revenue_table)

# Call the main function
if __name__ == "__main__":
    insight5()