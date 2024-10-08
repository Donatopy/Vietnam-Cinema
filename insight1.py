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
            ["All Movies", "0-20 Billion", "20-50 Billion", "50-100 Billion", "More than 100 Billion", "Custom Range"]
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
        if filter_option == "0-20 Billion":
            min_revenue, max_revenue = 0, 20
        elif filter_option == "20-50 Billion":
            min_revenue, max_revenue = 20, 50
        elif filter_option == "50-100 Billion":
            min_revenue, max_revenue = 50, 100
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
    if year_filter == 2019 or year_filter == "All Years":
        line_value = 18
        year_display = 2019  # Mostrar 2019 si es "All Years"
    elif year_filter == 2020:
        line_value = 18.9
        year_display = 2020
    elif year_filter == 2021:
        line_value = 19.85
        year_display = 2021
    elif year_filter == 2022:
        line_value = 20.84
        year_display = 2022
    elif year_filter == 2023:
        line_value = 21.88
        year_display = 2023
    elif year_filter == 2024:
        line_value = 22.97
        year_display = 2024
    else:
        line_value = 18  # Valor por defecto si no se selecciona un año específico
        year_display = 2019

    # Añadir líneas ajustadas
    ax.axvline(x=line_value, color='red', linestyle='--', label=f'{line_value:.2f} Billion VND (Average Budget)')
    ax.axvline(x=line_value * 2.5, color='green', linestyle='--', label=f'{(line_value * 2.5):.2f} Billion VND (Estimated Break-Even)')

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
        f"""
        - **Red line:** Average budget based on {year_display} values adjusted for a 5% annual inflation.
        - **Green line:** Estimated revenue needed to “break even” using a 2.5x standard formula.
        """
    )
    
    # Add disclaimer
    st.markdown(
        """
        **Disclaimer:** Does not account for Promotion & Advertising (P&A), taxes, or interest rates. For general reference only.
        """
    )

    # Create the budget and revenue needed table
    budget_data = {
        'Year': [2019, 2020, 2021, 2022, 2023, 2024],
        'Budget (Billion VND)': [f"{18.00:.2f}", f"{18.90:.2f}", f"{19.85:.2f}", f"{20.84:.2f}", f"{21.88:.2f}", f"{22.97:.2f}"],
        'Estimated Break-Even Point (Billion VND)': [f"{45.00:.2f}", f"{47.25:.2f}", f"{49.63:.2f}", f"{52.10:.2f}", f"{54.70:.2f}", f"{57.43:.2f}"]
    }
    budget_df = pd.DataFrame(budget_data)
    st.table(budget_df)
    
    # Create a scatterplot for revenue ranges
    st.header('Scatterplot of Films by Revenue Range')

    # Add year filter for scatterplot
    year_filter_scatter = st.selectbox(
        "Select Year for Scatterplot",
        options=["All Years"] + sorted(release_dates['Year'].unique().tolist())
    )
    
    # Filtrar por año si no es "All Years"
    df_scatter = df.copy()
    if year_filter_scatter != "All Years":
        df_scatter = df_scatter[df_scatter['Name'].isin(release_dates[release_dates['Year'] == year_filter_scatter]['Movie Name'])]

    # Verificar el rango de ingresos
    if not df_scatter.empty:
        max_revenue = df_scatter['Revenue (Billion VND)'].max()
        bins = [0, 20, 50, 100, max_revenue + 1]  # Aseguramos que el último bin sea mayor que max_revenue
        labels = ["0-20 Billion", "20-50 Billion", "50-100 Billion", "100+ Billion"]
        
        # Verificar que los bins estén en orden ascendente
        bins = sorted(set(bins))
        
        # Asignar los rangos
        df_scatter['Revenue Range'] = pd.cut(df_scatter['Revenue (Billion VND)'], bins=bins, labels=labels, include_lowest=True)
        
        # Contar el número de películas en cada rango de ingresos
        revenue_counts = df_scatter['Revenue Range'].value_counts().sort_index()

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        scatter = ax2.scatter(revenue_counts.index, revenue_counts.values, color='orange', s=100)

        for i, (x, y) in enumerate(zip(revenue_counts.index, revenue_counts.values)):
            ax2.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9, color='black')

        ax2.set_xlabel('Revenue Range (Billion VND)')
        ax2.set_ylabel('Number of Films')
        ax2.set_title(f'Number of Films by Revenue Range ({year_filter_scatter})')
        
        st.pyplot(fig2)
    else:
        st.write("No data available for the selected year.")
