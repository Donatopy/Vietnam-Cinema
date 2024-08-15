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
        st.subheader("x")
        plot_data_insight1(df1, release_dates)  # Pasar ambos DataFrames

    with tab2:
        st.header("Insight 2") 
        insight2()  # Llamar a la función del segundo insight
        
    with tab3:
        st.header("Insight 3")
        st.write("Insight 3 not implemented yet.")

if __name__ == "__main__":
    main()
