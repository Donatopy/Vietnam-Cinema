import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def load_and_process_data_insight1(file_name):
    df = pd.read_excel(file_name)
    
    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
    df.dropna(subset=['Revenue'], inplace=True)
    df['Revenue (Billion VND)'] = df['Revenue'] / 1e9
    df['Name'] = df['Name'].astype(str)
    
    return df

def plot_data_insight1(df, release_dates):
    st.header('Filter by Revenue and Year')

    col1, col2 = st.columns([2, 1])

    with col1:
        filter_option = st.radio(
            "Select Revenue Filter",
            ["All Movies", "0-5 Billion", "10-20 Billion", "20-100 Billion", "More than 100 Billion", "Custom Range"]
        )
    
    with col2:
        year_filter = st.selectbox(
            "Select Year",
            options=["All Years"] + sorted(release_dates['Year'].unique().tolist())
        )

    if filter_option == "Custom Range":
        min_revenue = st.number_input("Enter Minimum Revenue (in Billion VND)", min_value=0.0, format="%.2f")
        max_revenue = st.number_input("Enter Maximum Revenue (in Billion VND)", min_value=0.0, format="%.2f")
    else:
        if filter_option == "0-5 Billion":
            min_revenue, max_revenue = 0, 5
        elif filter_option == "10-20 Billion":
            min_revenue, max_revenue = 10, 20
        elif filter_option == "20-100 Billion":
            min_revenue, max_revenue = 20, 100
        elif filter_option == "More than 100 Billion":
            min_revenue, max_revenue = 100, df['Revenue (Billion VND)'].max()
        else:  # "All Movies"
            min_revenue, max_revenue = df['Revenue (Billion VND)'].min(), df['Revenue (Billion VND)'].max()

    filtered_df = df[(df['Revenue (Billion VND)'] >= min_revenue) & (df['Revenue (Billion VND)'] <= max_revenue)]
    
    if year_filter != "All Years":
        filtered_movies = release_dates[release_dates['Year'] == year_filter]['Movie Name']
        filtered_df = filtered_df[filtered_df['Name'].isin(filtered_movies)]

    filtered_df = filtered_df.sort_values(by='Revenue (Billion VND)', ascending=True)
    filtered_df = filtered_df.tail(50)

    fig, ax = plt.subplots(figsize=(12, 12))

    bars = ax.barh(filtered_df['Name'], filtered_df['Revenue (Billion VND)'], color='skyblue')

    # Valores de las líneas según el año seleccionado
    if year_filter == 2019:
        line_value = 18
    elif year_filter == 2020:
        line_value = 18.9
    elif year_filter == 2021:
        line_value = 19.85
    elif year_filter == 2022:
        line_value = 20.84
    elif year_filter == 2023:
        line_value = 21.88
    elif year_filter == 2024:
        line_value = 22.97
    else:
        line_value = 18  # Valor por defecto si no se selecciona un año específico

    # Añadir líneas ajustadas
    ax.axvline(x=line_value, color='red', linestyle='--', label=f'{line_value:.2f} Billion VND (Budget)')
    ax.axvline(x=line_value * 2.5, color='green', linestyle='--', label=f'{(line_value * 2.5):.2f} Billion VND (Profitable)')

    ax.set_xlabel('Revenue (Billion VND)')
    ax.set_ylabel('Movie Name')
    ax.set_title('Revenue of Movies in Billion VND')
    ax.legend()

    for bar in bars:
        width = bar.get_width()
        label = f'{width:.2f} Billion VND'
        ax.text(width, bar.get_y() + bar.get_height()/2, label, va='center', ha='left', fontsize=9, color='black')

    plt.yticks(rotation=0)
    st.pyplot(fig)

    # Add explanation below the chart
    st.markdown(
        """
        - **Red line:** Estimated budget based on 2019 values adjusted for a 5% annual inflation.
        - **Green line:** Revenue needed to be profitable, calculated as 2.5 times the budget.
        """
    )

    # Create the budget and revenue needed table
    budget_data = {
        'Year': [2019, 2020, 2021, 2022, 2023, 2024],
        'Budget (Billion VND)': [f"{18.00:.2f}", f"{18.90:.2f}", f"{19.85:.2f}", f"{20.84:.2f}", f"{21.88:.2f}", f"{22.97:.2f}"],
        'Revenue Needed to be Profitable (Billion VND)': [f"{45.00:.2f}", f"{47.25:.2f}", f"{49.63:.2f}", f"{52.10:.2f}", f"{54.70:.2f}", f"{57.43:.2f}"]
    }
    budget_df = pd.DataFrame(budget_data)
    st.table(budget_df)
