import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def load_and_preprocess_data(file_path='insight2.xlsx', release_dates_path='release_dates.csv'):
    # Load data (reusing code from previous insights)
    df = pd.read_excel(file_path)
    release_dates = pd.read_csv(release_dates_path, parse_dates=['Release Date'], dayfirst=True)
    
    df_transposed = df.set_index('Movie Name').T
    df_transposed.columns.name = None
    df_transposed.reset_index(inplace=True)
    df_transposed.rename(columns={'index': 'Date'}, inplace=True)
    df_transposed['Date'] = pd.to_datetime(df_transposed['Date'])
    
    df_long = df_transposed.melt(id_vars=['Date'], var_name='Movie Name', value_name='Revenue')
    df_long['Revenue'] = pd.to_numeric(df_long['Revenue'], errors='coerce')
    df_long = df_long.merge(release_dates, on='Movie Name', how='left')
    
    return df_long

def calculate_features(df):
    results = []
    for movie in df['Movie Name'].unique():
        df_movie = df[df['Movie Name'] == movie].sort_values('Date')
        release_date = df_movie['Release Date'].iloc[0]
        
        first_week_revenue = df_movie[(df_movie['Date'] >= release_date) & 
                                      (df_movie['Date'] < release_date + pd.Timedelta(days=7))]['Revenue'].sum()
        
        first_weekend_revenue = df_movie[(df_movie['Date'].dt.dayofweek.isin([5, 6])) & 
                                         (df_movie['Date'] >= release_date) & 
                                         (df_movie['Date'] < release_date + pd.Timedelta(days=7))]['Revenue'].sum()
        
        second_week_revenue = df_movie[(df_movie['Date'] >= release_date + pd.Timedelta(days=7)) & 
                                       (df_movie['Date'] < release_date + pd.Timedelta(days=14))]['Revenue'].sum()
        
        second_weekend_revenue = df_movie[(df_movie['Date'].dt.dayofweek.isin([5, 6])) & 
                                          (df_movie['Date'] >= release_date + pd.Timedelta(days=7)) & 
                                          (df_movie['Date'] < release_date + pd.Timedelta(days=14))]['Revenue'].sum()
        
        total_revenue = df_movie['Revenue'].sum()
        
        results.append({
            'Movie Name': movie,
            'First Week Revenue': first_week_revenue / 1e9,
            'First Weekend Revenue': first_weekend_revenue / 1e9,
            'Second Week Revenue': second_week_revenue / 1e9,
            'Second Weekend Revenue': second_weekend_revenue / 1e9,
            'Total Revenue': total_revenue / 1e9
        })
    
    return pd.DataFrame(results)

def train_model(features_df):
    X = features_df[['Total Revenue']]
    y = features_df[['First Week Revenue', 'First Weekend Revenue', 'Second Week Revenue', 'Second Weekend Revenue']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    return model, r2

def predict_required_revenue(model, total_revenue):
    prediction = model.predict([[total_revenue]])
    return prediction[0]

def plot_prediction_vs_actual(features_df, model, highlight_point=None):
    actual_total = features_df['Total Revenue']
    predictions = model.predict(features_df[['Total Revenue']])
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    titles = ['First Week Revenue', 'First Weekend Revenue', 'Second Week Revenue', 'Second Weekend Revenue']
    
    for i, ax in enumerate(axes.flat):
        ax.scatter(actual_total, features_df[titles[i]], alpha=0.5, label='Actual')
        ax.scatter(actual_total, predictions[:, i], alpha=0.5, label='Predicted')
        if highlight_point:
            ax.scatter(highlight_point[0], highlight_point[i+1], color='green', s=100, edgecolor='black', zorder=5, label='User Prediction')
        ax.set_xlabel('Total Revenue (Billion VND)')
        ax.set_ylabel(f'{titles[i]} (Billion VND)')
        ax.set_title(f'Actual vs Predicted {titles[i]}')
        ax.legend()
    
    plt.tight_layout()
    return fig

def insight4():
    st.header("Insight 4: Required Revenue Predictor for First Two Weeks")
    st.write("This tool predicts the required revenue for the first two weeks to achieve a desired total revenue.")
    
    # Load and preprocess data
    df_long = load_and_preprocess_data()
    features_df = calculate_features(df_long)
    
    # Train the model
    model, r2 = train_model(features_df)
    
    st.write("Model Accuracy:")
    st.write(f"R-squared Score: {r2:.2f}")
    st.write(f"This means our prediction is about {r2*100:.0f}% accurate.")
    
    # User input for prediction
    st.subheader("Predict Required Revenue")
    st.write("Enter your desired total revenue to see what revenues you should aim for in the first two weeks.")
    
    total_revenue = st.number_input("Desired Total Revenue (Billion VND)", min_value=0.0, value=45.0, format="%.2f", key="total_revenue")
    
    if st.button("Calculate Required Revenues") or total_revenue:
        prediction = predict_required_revenue(model, total_revenue)
        st.write(f"To achieve a total revenue of {total_revenue:.2f} Billion VND, you should aim for:")
        st.write(f"First Week Revenue: {prediction[0]:.2f} Billion VND")
        st.write(f"First Weekend Revenue: {prediction[1]:.2f} Billion VND")
        st.write(f"Second Week Revenue: {prediction[2]:.2f} Billion VND")
        st.write(f"Second Weekend Revenue: {prediction[3]:.2f} Billion VND")
        
        # Add some context to the prediction
        avg_total_revenue = features_df['Total Revenue'].mean()
        avg_first_week = features_df['First Week Revenue'].mean()
        avg_first_weekend = features_df['First Weekend Revenue'].mean()
        avg_second_week = features_df['Second Week Revenue'].mean()
        avg_second_weekend = features_df['Second Weekend Revenue'].mean()
        
        st.write(f"\nFor reference:")
        st.write(f"- Average Total Revenue: {avg_total_revenue:.2f} Billion VND")
        st.write(f"- Average First Week Revenue: {avg_first_week:.2f} Billion VND")
        st.write(f"- Average First Weekend Revenue: {avg_first_weekend:.2f} Billion VND")
        st.write(f"- Average Second Week Revenue: {avg_second_week:.2f} Billion VND")
        st.write(f"- Average Second Weekend Revenue: {avg_second_weekend:.2f} Billion VND")
        
        if total_revenue > avg_total_revenue:
            st.write("Your target is above average. This may be challenging but potentially achievable for a successful movie.")
        else:
            st.write("Your target is below average. This should be more easily achievable.")
    
    # Plot prediction vs actual
    st.subheader("Prediction Performance")
    highlight_point = [total_revenue, prediction[0], prediction[1], prediction[2], prediction[3]]
    fig = plot_prediction_vs_actual(features_df, model, highlight_point=highlight_point)
    st.pyplot(fig)
    st.write("These graphs show how well our model predicts revenues based on total revenue. The closer the predicted points (orange) are to the actual points (blue), the more accurate our model is.")
    st.write("The large green point represents your prediction.")
    
    # Revenue breakdown
    st.subheader("Revenue Breakdown")
    avg_first_week_ratio = features_df['First Week Revenue'].mean() / features_df['Total Revenue'].mean()
    avg_first_weekend_ratio = features_df['First Weekend Revenue'].mean() / features_df['Total Revenue'].mean()
    avg_second_week_ratio = features_df['Second Week Revenue'].mean() / features_df['Total Revenue'].mean()
    avg_second_weekend_ratio = features_df['Second Weekend Revenue'].mean() / features_df['Total Revenue'].mean()
    
    st.write(f"On average:")
    st.write(f"- First week revenue accounts for {avg_first_week_ratio:.1%} of total revenue")
    st.write(f"- First weekend revenue accounts for {avg_first_weekend_ratio:.1%} of total revenue")
    st.write(f"- Second week revenue accounts for {avg_second_week_ratio:.1%} of total revenue")
    st.write(f"- Second weekend revenue accounts for {avg_second_weekend_ratio:.1%} of total revenue")
    st.write("This information can help you gauge if the predicted revenues are reasonable based on typical movie performance patterns.")

if __name__ == "__main__":
    insight4()