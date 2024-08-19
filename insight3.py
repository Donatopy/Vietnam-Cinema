import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def load_and_preprocess_data(file_path='insight2.xlsx', release_dates_path='release_dates.csv'):
    # Load data (reusing code from insight2)
    df = pd.read_excel(file_path)
    release_dates = pd.read_csv(release_dates_path, parse_dates=['Release Date'], dayfirst=True)
    
    # Preprocess data
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
        
        total_revenue = df_movie['Revenue'].sum()
        
        results.append({
            'Movie Name': movie,
            'First Week Revenue': first_week_revenue / 1e9,  # Convert to billions
            'First Weekend Revenue': first_weekend_revenue / 1e9,  # Convert to billions
            'Total Revenue': total_revenue / 1e9  # Convert to billions
        })
    
    return pd.DataFrame(results)

def train_model(features_df):
    X = features_df[['First Week Revenue', 'First Weekend Revenue']]
    y = features_df['Total Revenue']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2

def predict_revenue(model, first_week_revenue, first_weekend_revenue):
    prediction = model.predict([[first_week_revenue, first_weekend_revenue]])
    return prediction[0]

def plot_actual_vs_predicted(features_df, model, highlight_prediction=None):
    X = features_df[['First Week Revenue', 'First Weekend Revenue']]
    y_actual = features_df['Total Revenue']
    y_pred = model.predict(X)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_actual, y_pred, alpha=0.5)
    ax.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
    
    if highlight_prediction is not None:
        ax.scatter([highlight_prediction], [highlight_prediction], color='green', s=200, zorder=5)
    
    ax.set_xlabel('Actual Total Revenue (Billion VND)')
    ax.set_ylabel('Predicted Total Revenue (Billion VND)')
    ax.set_title('Actual vs Predicted Total Revenue')
    
    return fig

def insight3():
    st.header("Insight 3: Movie Revenue Prediction")
    st.write("This tool predicts a movie's total revenue based on its performance in the first week and first weekend.")
    
    # Load and preprocess data
    df_long = load_and_preprocess_data()
    features_df = calculate_features(df_long)
    
    # Train the model
    model, mse, r2 = train_model(features_df)
    
    st.write("Model Accuracy:")
    st.write(f"R-squared Score: {r2:.2f}")
    st.write(f"This means our prediction is about {r2*100:.0f}% accurate.")
    
    # User input for prediction
    st.subheader("Predict Total Revenue")
    st.write("Enter the movie's revenue for its first week and first weekend to get a prediction of its total revenue.")
    
    col1, col2 = st.columns(2)
    with col1:
        first_week_revenue = st.number_input("First Week Revenue (Billion VND)", min_value=0.0, value=21.73, format="%.2f")
    with col2:
        first_weekend_revenue = st.number_input("First Weekend Revenue (Billion VND)", min_value=0.0, value=7.95, format="%.2f")
    
    # Default prediction
    prediction = predict_revenue(model, first_week_revenue, first_weekend_revenue)
    st.write(f"Predicted Total Revenue: {prediction:.2f} Billion VND")
    
    # Add some context to the prediction
    avg_revenue = features_df['Total Revenue'].mean()
    if prediction > avg_revenue:
        st.write(f"This movie is predicted to perform above average. The average total revenue is {avg_revenue:.2f} Billion VND.")
    else:
        st.write(f"This movie is predicted to perform below average. The average total revenue is {avg_revenue:.2f} Billion VND.")
    
    if st.button("Update Prediction"):
        # This will recalculate the prediction with the current input values
        prediction = predict_revenue(model, first_week_revenue, first_weekend_revenue)
        st.write(f"Predicted Total Revenue: {prediction:.2f} Billion VND")
        
        # Add some context to the prediction
        if prediction > avg_revenue:
            st.write(f"This movie is predicted to perform above average! The average total revenue is {avg_revenue:.2f} Billion VND.")
        else:
            st.write(f"This movie is predicted to perform below average. The average total revenue is {avg_revenue:.2f} Billion VND.")
    
    # Plot actual vs predicted
    st.subheader("Actual vs Predicted Total Revenue")
    fig = plot_actual_vs_predicted(features_df, model, highlight_prediction=prediction)
    st.pyplot(fig)
    st.write("This graph shows how well our predictions match the actual total revenues. Points closer to the red line indicate more accurate predictions.")
    st.write("The large green point represents the current prediction.")
    
    # Display feature importance
    st.subheader("What Matters More?")
    feature_importance = pd.DataFrame({
        'Feature': ['First Week Revenue', 'First Weekend Revenue'],
        'Importance': abs(model.coef_)
    })
    feature_importance['Importance'] = feature_importance['Importance'] / feature_importance['Importance'].sum()
    st.write("Which is a stronger indicator of a movie's total revenue?")
    st.write(feature_importance)
    st.write(f"The first {'week' if feature_importance.iloc[0,1] > feature_importance.iloc[1,1] else 'weekend'} revenue seems to be a stronger predictor of total revenue.")

if __name__ == "__main__":
    insight3()
