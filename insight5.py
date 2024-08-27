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
    df['Minutes'] = [t.hour * 60 + t.minute for t in df['Ca chiếu']]
    df['Revenue (Billions VND)'] = df['Doanh thu'] / 1_000_000_000
    return df

# Main function to generate the Streamlit app
def insight5():
    # Read data from the CSV file
    df = pd.read_csv('release_dates.csv')
    df['Release Date'] = pd.to_datetime(df['Release Date'], format='%d/%m/%Y')
    
    # Load key dates from the Excel file
    key_dates = load_key_dates_from_excel()

    # Margin days setting
    st.title('Movie Release Insights')

    # Place the margin days slider and "Exclude Other" checkbox at the top
    margin_days = st.slider('Select margin days for key dates', min_value=0, max_value=10, value=3)
    exclude_other = st.checkbox('Exclude "Other" from Key Date Event chart')

    # Add a column to the DataFrame with the key event using the defined margin
    df['Key Date Event'] = df['Release Date'].apply(lambda x: is_key_date(x, key_dates, margin_days))
    df['Month'] = df['Release Date'].dt.month
    df['Day of Week'] = df['Release Date'].dt.day_name()

    # Section 1: Optimal Release Periods
    st.header('Optimal Release Periods')
    st.write(f"We also consider a margin of ±{margin_days} days around key dates to account for movies "
             f"released in close proximity to these events.")

    # Count movies by key event (including ±n days margin)
    event_counts = df['Key Date Event'].value_counts()

    # Exclude "Other" if selected
    if exclude_other:
        event_counts = event_counts[event_counts.index != 'Other']

    st.bar_chart(event_counts)

    # Section 2: Releases by Month
    st.header('Releases by Month')
    month_counts = df['Month'].value_counts().sort_index()
    st.bar_chart(month_counts)

    # Section 3: Releases by Day of the Week
    st.header('Releases by Day of the Week')
    weekday_counts = df['Day of Week'].value_counts().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    )
    st.bar_chart(weekday_counts)

    # Section 4: Revenue Analysis by Time
    st.header('Revenue Analysis by Time')

    # Fixed file path for revenue data
    revenue_file_path = 'revenue_by_time.xlsx'

    # Load and process revenue data
    revenue_df = load_and_process_revenue_data(revenue_file_path)

    # Display total revenue
    total_revenue = revenue_df['Revenue (Billions VND)'].sum()
    st.write(f"Total Revenue: {total_revenue:.2f} billion VND")

    # Plot revenue over time
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(revenue_df['Minutes'], revenue_df['Revenue (Billions VND)'], marker='o', linestyle='-')

    # Configure x-axis ticks
    ticks = range(0, 1440, 60)  # Every 60 minutes
    labels = [f'{h:02d}:{m:02d}' for h in range(24) for m in [0, 30] if h * 60 + m in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45)

    ax.set_xlabel('Minutes Since Midnight')
    ax.set_ylabel('Revenue (Billions VND)')
    ax.set_title('Revenue by Hour of the Day')
    ax.grid(True)

    st.pyplot(fig)

# Call the main function
if __name__ == "__main__":
    insight5()
