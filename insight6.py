import streamlit as st
from streamlit.components.v1 import html
import pandas as pd

def insight6():
    # Title of the application
    st.title("City Revenue Map and Theaters Revenue")

    # Path to the HTML file generated with Folium
    map_file = 'city_revenue_map.html'

    # Read and display the HTML file in the Streamlit application
    with open(map_file, 'r', encoding='utf-8') as file:
        map_html = file.read()

    # Display the map in Streamlit
    html(map_html, height=700, width=900, scrolling=True)

    # Load the data from Excel
    df = pd.read_excel('normalized_theatres_combined_revenue_with_city.xlsx')

    # Aggregate revenue by city
    city_revenue = df.groupby('Thành phố')['Total Revenue'].sum().reset_index()

    # Calculate the total revenue
    total_revenue = city_revenue['Total Revenue'].sum()

    # Convert total revenue to billions VND for display
    total_revenue_billion_vnd = total_revenue / 1e9

    # Add a selectbox for the user to choose the number of top cities to display or show all cities
    num_top_cities = st.selectbox(
        'Select number of top cities to display (or select "All Cities"):',
        options=['All Cities', 5, 10, 15, 20],
        index=1  # Default to 5 cities
    )

    # Check if the user selected "All Cities" or a specific number of top cities
    if num_top_cities == 'All Cities':
        top_cities = city_revenue.sort_values(by='Total Revenue', ascending=False)
    else:
        top_cities = city_revenue.sort_values(by='Total Revenue', ascending=False).head(num_top_cities)

    # Calculate revenue percentage for each city
    top_cities['Revenue Percentage'] = (top_cities['Total Revenue'] / total_revenue) * 100

    # Convert revenue to billions VND
    top_cities['Total Revenue (Billion VND)'] = top_cities['Total Revenue'] / 1e9

    # Format the data for display
    top_cities = top_cities[['Thành phố', 'Total Revenue (Billion VND)', 'Revenue Percentage']]
    top_cities['Total Revenue (Billion VND)'] = top_cities['Total Revenue (Billion VND)'].apply(lambda x: f"{x:,.2f} Billion VND")
    top_cities['Revenue Percentage'] = top_cities['Revenue Percentage'].apply(lambda x: f"{x:.2f}%")

    # Display the total revenue above the table
    st.subheader(f"Total Revenue: {total_revenue_billion_vnd:,.2f} Billion VND")

    # Display the selected number of top cities or all cities with the highest revenue in a table
    if num_top_cities == 'All Cities':
        st.subheader("All Cities with Revenue Data")
    else:
        st.subheader(f"Top {num_top_cities} Cities with Highest Revenue")
    
    st.dataframe(top_cities)

    # Add a selectbox for the user to choose the number of top theaters to display or show all theaters
    num_top_theaters = st.selectbox(
        'Select number of top theaters to display (or select "All Theaters"):',
        options=['All Theaters', 5, 10, 15, 20],
        index=1  # Default to 5 theaters
    )

    # Aggregate revenue by theater
    theater_revenue = df.groupby(['Tên rạp', 'Thành phố'])['Total Revenue'].sum().reset_index()

    # Check if the user selected "All Theaters" or a specific number of top theaters
    if num_top_theaters == 'All Theaters':
        top_theaters = theater_revenue.sort_values(by='Total Revenue', ascending=False)
    else:
        top_theaters = theater_revenue.sort_values(by='Total Revenue', ascending=False).head(num_top_theaters)

    # Calculate revenue percentage for each theater
    top_theaters['Revenue Percentage'] = (top_theaters['Total Revenue'] / total_revenue) * 100

    # Convert revenue to billions VND
    top_theaters['Total Revenue (Billion VND)'] = top_theaters['Total Revenue'] / 1e9

    # Format the data for display
    top_theaters = top_theaters[['Tên rạp', 'Thành phố', 'Total Revenue (Billion VND)', 'Revenue Percentage']]
    top_theaters['Total Revenue (Billion VND)'] = top_theaters['Total Revenue (Billion VND)'].apply(lambda x: f"{x:,.2f} Billion VND")
    top_theaters['Revenue Percentage'] = top_theaters['Revenue Percentage'].apply(lambda x: f"{x:.2f}%")

    # Display the selected number of top theaters or all theaters with the highest revenue in a table
    if num_top_theaters == 'All Theaters':
        st.subheader("All Theaters with Revenue Data")
    else:
        st.subheader(f"Top {num_top_theaters} Theaters with Highest Revenue")
    
    st.dataframe(top_theaters)

if __name__ == "__main__":
    insight6()
