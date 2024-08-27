import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_data(file_path, release_dates_path):
    try:
        # Cargar los datos de ingresos de películas
        df = pd.read_excel(file_path)
        
        # Cargar las fechas de estreno con el manejo de errores adecuado
        release_dates = pd.read_csv(release_dates_path, dayfirst=True)
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

    return df, release_dates

def preprocess_data(df, release_dates):
    # Transponer el DataFrame para que las fechas sean filas y las películas columnas
    df_transposed = df.set_index('Movie Name').T
    df_transposed.columns.name = None
    df_transposed.reset_index(inplace=True)
    df_transposed.rename(columns={'index': 'Date'}, inplace=True)

    # Convertir 'Date' a formato datetime
    df_transposed['Date'] = pd.to_datetime(df_transposed['Date'], format='%Y-%m-%d')

    # Transformar el DataFrame a formato largo para el análisis
    df_long = df_transposed.melt(id_vars=['Date'], var_name='Movie Name', value_name='Revenue')

    # Convertir 'Revenue' a numérico, manejando errores convirtiendo valores inválidos a NaN
    df_long['Revenue'] = pd.to_numeric(df_long['Revenue'], errors='coerce')

    # Fusionar con los datos de fechas de estreno
    df_long = df_long.merge(release_dates, on='Movie Name', how='left')
    df_long['Release Date'] = pd.to_datetime(df_long['Release Date'], dayfirst=True)

    return df_long

def plot_revenue_trends(df_filtered, selected_movie, release_date, highlight_weekends=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Asegúrate de que los datos estén ordenados por fecha
    df_filtered.sort_index(inplace=True)
    
    sns.lineplot(data=df_filtered, x=df_filtered.index, y='Revenue', ax=ax, marker='o')
    
    # Sombrear los fines de semana si está activado
    if highlight_weekends:
        weekends = df_filtered[df_filtered.index.weekday >= 5].index
        for weekend in weekends:
            ax.axvspan(weekend, weekend + pd.Timedelta(days=1), color='grey', alpha=0.3)
    
    # Añadir la línea roja de puntos en la fecha de estreno
    ax.axvline(release_date, color='red', linestyle='--', linewidth=1.5, label='Release Date')
    
    # Formato del eje x
    ax.set_title(f'Revenue of {selected_movie} from 7 days before to 21 days after release date')
    ax.set_xlabel('Date')
    ax.set_ylabel('Revenue (Billion VND)')
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%d/%m/%y'))
    
    ax.legend()
    st.pyplot(fig)


def calculate_revenue_periods(df_movie):
    premiere_date = df_movie['Release Date'].iloc[0]

    # Primeros 7 días
    first_week_revenue = df_movie.loc[(df_movie.index >= premiere_date) & 
                                      (df_movie.index < premiere_date + pd.Timedelta(days=7)), 'Revenue'].sum()
    
    # Siguientes 7 días (Día 7 al Día 13)
    second_week_revenue = df_movie.loc[(df_movie.index >= premiere_date + pd.Timedelta(days=7)) & 
                                       (df_movie.index < premiere_date + pd.Timedelta(days=14)), 'Revenue'].sum()

    # Primer fin de semana
    first_weekend = df_movie.loc[(df_movie.index.weekday >= 5) & 
                                 (df_movie.index < premiere_date + pd.Timedelta(days=7)), 'Revenue'].sum()

    # Segundo fin de semana
    second_weekend = df_movie.loc[(df_movie.index.weekday >= 5) & 
                                  (df_movie.index >= premiere_date + pd.Timedelta(days=7)) & 
                                  (df_movie.index < premiere_date + pd.Timedelta(days=14)), 'Revenue'].sum()

    return first_week_revenue, second_week_revenue, first_weekend, second_weekend

def analyze_profitable_movies(df_long, threshold=45e9):
    profitable_movies = df_long.groupby('Movie Name')['Revenue'].sum()
    profitable_movies = profitable_movies[profitable_movies >= threshold]

    week_results = []
    weekend_results = []

    for movie in profitable_movies.index:
        df_movie = df_long[df_long['Movie Name'] == movie]
        df_movie.set_index('Date', inplace=True)
        df_movie.sort_index(inplace=True)

        first_week_revenue, second_week_revenue, first_weekend, second_weekend = calculate_revenue_periods(df_movie)

        total_revenue = profitable_movies[movie] / 1e9  # Convertir a miles de millones

        # Calcular porcentajes de caída
        week_drop_percentage = ((second_week_revenue - first_week_revenue) / first_week_revenue * 100) if first_week_revenue > 0 else None
        weekend_drop_percentage = ((second_weekend - first_weekend) / first_weekend * 100) if first_weekend > 0 else None

        # Tabla 1: Ingresos semanales
        week_results.append({
            'Film': movie,
            'Week 1 Revenue (Billion VND)': first_week_revenue / 1e9,  # Mantener como número
            'Week 2 Revenue (Billion VND)': second_week_revenue / 1e9,  # Mantener como número
            'Change (%)': week_drop_percentage,
            'Total Revenue (Billion VND)': total_revenue
        })

        # Tabla 2: Ingresos de fines de semana
        weekend_results.append({
            'Film': movie,
            'Weekend 1 Revenue (Billion VND)': first_weekend / 1e9,  # Mantener como número
            'Weekend 2 Revenue (Billion VND)': second_weekend / 1e9,  # Mantener como número
            'Change (%)': weekend_drop_percentage,
            'Total Revenue (Billion VND)': total_revenue
        })

    # Crear DataFrames para las dos tablas
    week_results_df = pd.DataFrame(week_results).sort_values(by='Change (%)', ascending=True).reset_index(drop=True)
    weekend_results_df = pd.DataFrame(weekend_results).sort_values(by='Change (%)', ascending=True).reset_index(drop=True)

    # Función para aplicar colores
    def color_change(val):
        try:
            # Convertir a float para la comparación
            value = float(val)
            if value < 0:
                return 'color: red'
            elif value > 0:
                return 'color: green'
            else:
                return ''
        except:
            return ''

    # Mostrar las tablas en Streamlit
    st.write("**Top Films by Weekly Drop**")
    st.write(week_results_df.style.format({
        'Week 1 Revenue (Billion VND)': '{:.2f}',
        'Week 2 Revenue (Billion VND)': '{:.2f}',
        'Change (%)': '{:.2f}',
        'Total Revenue (Billion VND)': '{:.2f}'
    }).applymap(color_change, subset=['Change (%)']))

    st.write("**Top Films by Weekend Drop**")
    st.write(weekend_results_df.style.format({
        'Weekend 1 Revenue (Billion VND)': '{:.2f}',
        'Weekend 2 Revenue (Billion VND)': '{:.2f}',
        'Change (%)': '{:.2f}',
        'Total Revenue (Billion VND)': '{:.2f}'
    }).applymap(color_change, subset=['Change (%)']))

def analyze_movie_run_length(df_long, revenue_threshold_failed=10e9, revenue_threshold_successful=45e9):
    # Agrupar por película y calcular el ingreso total
    movie_total_revenue = df_long.groupby('Movie Name')['Revenue'].sum()

    # Clasificar películas
    failed_movies = movie_total_revenue[movie_total_revenue < revenue_threshold_failed].index
    breakeven_movies = movie_total_revenue[(movie_total_revenue >= revenue_threshold_failed) & (movie_total_revenue < revenue_threshold_successful)].index
    successful_movies = movie_total_revenue[movie_total_revenue >= revenue_threshold_successful].index

    def calculate_run_length(movies):
        run_lengths = []
        revenue_progressions = []
        for movie in movies:
            movie_data = df_long[df_long['Movie Name'] == movie].sort_values('Date')
            release_date = movie_data['Release Date'].iloc[0]
            last_revenue_date = movie_data[movie_data['Revenue'] > 0]['Date'].max()
            run_length = (last_revenue_date - release_date).days + 1
            run_lengths.append(run_length)

            # Calcular la progresión de ingresos
            movie_data['Days Since Release'] = (movie_data['Date'] - release_date).dt.days
            cumulative_revenue = movie_data.set_index('Days Since Release')['Revenue'].cumsum()
            normalized_revenue = cumulative_revenue / cumulative_revenue.max()
            revenue_progressions.append(normalized_revenue)

        return run_lengths, revenue_progressions

    failed_run_lengths, failed_progressions = calculate_run_length(failed_movies)
    breakeven_run_lengths, breakeven_progressions = calculate_run_length(breakeven_movies)
    successful_run_lengths, successful_progressions = calculate_run_length(successful_movies)

    # Visualizar resultados
    st.write("## Movie Run Length Analysis")
    
    st.write("### Average Run Lengths")
    st.write(f"Failed Movies: {np.mean(failed_run_lengths):.1f} days")
    st.write(f"Break-even Movies: {np.mean(breakeven_run_lengths):.1f} days")
    st.write(f"Successful Movies: {np.mean(successful_run_lengths):.1f} days")

    # Graficar distribución de duración de exhibición
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=[failed_run_lengths, breakeven_run_lengths, successful_run_lengths], ax=ax)
    ax.set_xticklabels(['Failed', 'Break-even', 'Successful'])
    ax.set_ylabel('Run Length (days)')
    ax.set_title('Distribution of Movie Run Lengths')
    st.pyplot(fig)

    # Graficar progresión de ingresos promedio
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_average_progression(failed_progressions, 'Failed', ax)
    plot_average_progression(breakeven_progressions, 'Break-even', ax)
    plot_average_progression(successful_progressions, 'Successful', ax)
    ax.set_xlabel('Days since release')
    ax.set_ylabel('Normalized Cumulative Revenue')
    ax.set_title('Average Revenue Progression')
    ax.legend()
    st.pyplot(fig)

def plot_average_progression(progressions, label, ax):
    max_length = max(len(p) for p in progressions)
    aligned_progressions = [p.reindex(range(max_length), method='pad') for p in progressions]
    average_progression = pd.concat(aligned_progressions, axis=1).mean(axis=1)
    ax.plot(average_progression.index, average_progression.values, label=label)


def insight2(file_path='insight2.xlsx', release_dates_path='release_dates.csv'):
    # Cargar y preprocesar datos
    df, release_dates = load_data(file_path, release_dates_path)
    if df is None or release_dates is None:
        return
    
    df_long = preprocess_data(df, release_dates)

    # Calcular el ingreso total para cada película
    movie_revenue = df_long.groupby('Movie Name')['Revenue'].sum()

    # Inicializar el estado del botón en session state
    if 'filter_option' not in st.session_state:
        st.session_state.filter_option = 'all'
    
    # Botones para filtrar películas
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

    # Filtrar películas según la opción seleccionada
    if st.session_state.filter_option == 'less_than_45b':
        movies = movie_revenue[movie_revenue < 45e9].index
    elif st.session_state.filter_option == 'greater_than_45b':
        movies = movie_revenue[movie_revenue >= 45e9].index
    else:
        movies = df_long['Movie Name'].unique()

    # Manejar lista vacía de películas
    if len(movies) == 0:
        st.error("No movies match the selected filter. Please choose a different filter option.")
        return

    # Inicializar el índice de la película seleccionada en session state
    if 'selected_index' not in st.session_state:
        st.session_state.selected_index = 0

    # Asegurarse de que el índice esté dentro de los límites
    st.session_state.selected_index = min(st.session_state.selected_index, len(movies) - 1)

    # Botones de navegación para moverse entre películas
    col1, col2, col3 = st.columns([1, 5, 1])

    with col1:
        if st.button('Previous Movie'):
            st.session_state.selected_index = (st.session_state.selected_index - 1) % len(movies)

    with col3:
        if st.button('Next Movie'):
            st.session_state.selected_index = (st.session_state.selected_index + 1) % len(movies)

    # Dropdown para seleccionar una película
    selected_movie = st.selectbox('Select a movie to analyze:', movies, index=st.session_state.selected_index)

    # Filtrar datos para la película seleccionada
    df_movie = df_long[df_long['Movie Name'] == selected_movie]
    df_movie = df_movie[['Date', 'Revenue', 'Release Date']]
    df_movie.set_index('Date', inplace=True)
    df_movie.sort_index(inplace=True)

    # Obtener la fecha de estreno de la película seleccionada
    release_date = df_movie['Release Date'].iloc[0]
    st.write(f"Release date for {selected_movie}: {release_date.strftime('%d/%m/%y')}")

    # Definir el rango de fechas centrado alrededor de la fecha de estreno
    start_date = release_date - pd.Timedelta(days=7)
    end_date = release_date + pd.Timedelta(days=21)

    # Filtrar datos dentro del rango de fechas
    df_filtered = df_movie[(df_movie.index >= start_date) & (df_movie.index <= end_date)]

    # Añadir botón para sombrear fines de semana
    highlight_weekends = st.checkbox("Highlight Weekends", value=True)

    # Graficar las tendencias de ingresos con la línea roja de puntos para la fecha de estreno
    plot_revenue_trends(df_filtered, selected_movie, release_date, highlight_weekends)

    # Calcular y mostrar los ingresos para los primeros 7 días, los siguientes 7 días y los dos fines de semana
    first_week_revenue, second_week_revenue, first_weekend, second_weekend = calculate_revenue_periods(df_movie)

    # Función para formatear el cambio porcentual
    def format_percentage_change(value):
        if value is None:
            return "N/A"
        elif value < 0:
            return f"<span style='color: red;'>{value:.2f}%</span>"
        else:
            return f"<span style='color: green;'>{value:.2f}%</span>"

    # Mostrar los resultados con formato de cambio porcentual
    st.write(f"Total revenue in the first week after release: {first_week_revenue / 1e9:,.2f} billion VND")
    st.write(f"Total revenue in the second week after release: {second_week_revenue / 1e9:,.2f} billion VND")
    st.markdown(f"Change in revenue during the first week: {format_percentage_change(((second_week_revenue - first_week_revenue) / first_week_revenue * 100) if first_week_revenue > 0 else None)}", unsafe_allow_html=True)

    st.write("")  # Línea en blanco para separación

    st.write(f"Total revenue in the first weekend after release: {first_weekend / 1e9:,.2f} billion VND")
    st.write(f"Total revenue in the second weekend after release: {second_weekend / 1e9:,.2f} billion VND")
    st.markdown(f"Change in revenue during the second weekend: {format_percentage_change(((second_weekend - first_weekend) / first_weekend * 100) if first_weekend > 0 else None)}", unsafe_allow_html=True)


    # Analizar películas rentables
    st.write("**Profitable Movies Analysis**")
    analyze_profitable_movies(df_long)

    # Añadir esta línea en la función principal (insight2) para llamar a la nueva función
    analyze_movie_run_length(df_long)

if __name__ == "__main__":
    insight2()