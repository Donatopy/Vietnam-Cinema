import streamlit as st
from insight1 import load_and_process_data_insight1, plot_data_insight1
from insight2 import insight2
import pandas as pd

def main():
    st.title("Movie Revenue Insights")

    # Cargar datos
    df1 = load_and_process_data_insight1("insight1.xlsx")
    release_dates = pd.read_csv("release_dates.csv")
    release_dates['Release Date'] = pd.to_datetime(release_dates['Release Date'], format='%d/%m/%Y')
    release_dates['Year'] = release_dates['Release Date'].dt.year

    # Configurar pestañas
    tab1, tab2, tab3 = st.tabs(["Insight 1", "Insight 2", "Insight 3"])

    with tab1:
        st.header("Insight 1")
        st.subheader("Number of movies per year in each revenue range (0-5 billion, 10-20 billion, more than 100 billion etc.). VN movies have an average budget of 15 billion VND. Using a standard formula of 2.5x revenue to break even I want to see roughly how many are actually profitable from year to year.")
        plot_data_insight1(df1, release_dates)  # Pasar ambos DataFrames

    with tab2:
        st.header("Insight 2") 
        insight2()  # Llamar a la función del segundo insight
        
    with tab3:
        st.header("Insight 3")
        st.write("Insight 3 not implemented yet.")

if __name__ == "__main__":
    main()
