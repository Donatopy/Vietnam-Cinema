import streamlit as st
from insight1 import load_and_process_data_insight1, plot_data_insight1
from insight2 import insight2
from insight3 import insight3
from insight4 import insight4
from insight5 import insight5
from insight6 import insight6
import pandas as pd

def main():
    st.title("Movie Revenue Insights")

    # Load data
    df1 = load_and_process_data_insight1("insight1.xlsx")
    release_dates = pd.read_csv("release_dates.csv")
    release_dates['Release Date'] = pd.to_datetime(release_dates['Release Date'], format='%d/%m/%Y')
    release_dates['Year'] = release_dates['Release Date'].dt.year

    # Configure tabs
    tab1, tab2, tab3, tab4, tab5, tab6= st.tabs(["Insight 1", "Insight 2", "Insight 3", "Insight 4", "Insight 5", "Insight 6"])

    with tab1:
        st.header("Insight 1")
        st.subheader("Number of movies per year in each revenue range (0-5 billion, 10-20 billion, more than 100 billion etc.). VN movies have an average budget of 15 billion VND. Using a standard formula of 2.5x revenue to break even I want to see roughly how many are actually profitable from year to year.")
        plot_data_insight1(df1, release_dates)

    with tab2:
        st.header("Insight 2") 
        insight2()
        
    with tab3:
        st.header("Insight 3")
        insight3()
        
    with tab4:
        st.header("Insight 4")
        insight4()
        
    with tab5:
        st.header("Insight 5")
        insight5()
        
    with tab6:
        st.header("Insight 6")
        insight6()

if __name__ == "__main__":
    main()
