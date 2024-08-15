import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def load_and_process_data_insight1(file_name):
    df = pd.read_excel(file_name)
    
    # Convertir la columna 'Revenue' a numérico, manejando errores
    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
    
    # Eliminar filas con valores NaN en 'Revenue'
    df.dropna(subset=['Revenue'], inplace=True)
    
    # Convertir Revenue a billions VND
    df['Revenue (Billion VND)'] = df['Revenue'] / 1e9
    
    # Asegurarse de que 'Name' es una cadena de texto
    df['Name'] = df['Name'].astype(str)
    
    return df

def load_release_dates(file_name):
    # Cargar el archivo CSV
    release_dates = pd.read_csv(file_name)
    
    # Convertir 'Release Date' a formato de fecha
    release_dates['Release Date'] = pd.to_datetime(release_dates['Release Date'], format='%d/%m/%Y')
    
    # Extraer el año de 'Release Date'
    release_dates['Year'] = release_dates['Release Date'].dt.year
    
    return release_dates

def plot_data_insight1(df, release_dates):
    st.header('Filter by Revenue and Year')

    # Crear un diseño en columnas para los filtros
    col1, col2 = st.columns([2, 1])  # Ajustar proporciones según necesidad

    with col1:
        # Botón de selección de filtro por ingresos
        filter_option = st.radio(
            "Select Revenue Filter",
            ["All Movies", "0-5 Billion", "10-20 Billion", "20-100 Billion", "More than 100 Billion", "Custom Range"]
        )
    
    with col2:
        # Filtro por año
        year_filter = st.selectbox(
            "Select Year",
            options=["All Years"] + sorted(release_dates['Year'].unique().tolist())
        )

    # Establecer el rango de ingresos basado en la opción seleccionada
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

    # Filtrar el DataFrame según el rango seleccionado
    filtered_df = df[(df['Revenue (Billion VND)'] >= min_revenue) & (df['Revenue (Billion VND)'] <= max_revenue)]
    
    # Filtrar por año
    if year_filter != "All Years":
        filtered_movies = release_dates[release_dates['Year'] == year_filter]['Movie Name']
        filtered_df = filtered_df[filtered_df['Name'].isin(filtered_movies)]

    # Ordenar por Revenue (de mayor a menor)
    filtered_df = filtered_df.sort_values(by='Revenue (Billion VND)', ascending=True)

    # Limitar la cantidad de datos para mejorar la legibilidad
    filtered_df = filtered_df.tail(50)  # Mostrar los 50 elementos con mayor revenue

    # Crear la gráfica
    fig, ax = plt.subplots(figsize=(12, 12))  # Ajustar el tamaño de la figura

    # Crear el gráfico de barras horizontales
    bars = ax.barh(filtered_df['Name'], filtered_df['Revenue (Billion VND)'], color='skyblue')

    # Agregar líneas en 15 mil millones VND y 37 mil millones VND
    ax.axvline(x=15, color='red', linestyle='--', label='15 Billion VND')
    ax.axvline(x=37, color='green', linestyle='--', label='37 Billion VND')

    # Configurar etiquetas y título
    ax.set_xlabel('Revenue (Billion VND)')
    ax.set_ylabel('Movie Name')
    ax.set_title('Revenue of Movies in Billion VND')
    ax.legend()

    # Añadir anotaciones en cada barra
    for bar in bars:
        width = bar.get_width()
        label = f'{width:.2f} Billion VND'
        ax.text(width, bar.get_y() + bar.get_height()/2, label,
                va='center', ha='left', fontsize=9, color='black')

    # Rotar las etiquetas del eje Y para mejorar la legibilidad
    plt.yticks(rotation=0)  # Ajustar según la necesidad, por ejemplo: rotation=45

    # Mostrar la gráfica en Streamlit
    st.pyplot(fig)
