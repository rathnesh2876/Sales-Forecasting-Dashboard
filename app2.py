import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(page_title="Customer Relationship Management | Sales Forecast", layout="wide", page_icon="📈")

# Custom CSS for better UI
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# Sidebar - File Upload
st.sidebar.header("📂 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload Sales CSV (e.g., from Kaggle)", type=["csv"])

if uploaded_file is not None:
    @st.cache_data
    def load_data(file):
        data = pd.read_csv(file)
        # Basic cleaning: remove empty rows
        data.dropna(how='all', inplace=True)
        return data

    df = load_data(uploaded_file)
    st.sidebar.success("✅ Dataset Loaded!")

    # Navigation
    section = st.sidebar.radio("Go to:", ["Data Overview", "Sales Forecast", "ML Classification"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("🛠️ **Developer:** Rathnesh M.S")

    # --- SECTION 1: DATA OVERVIEW ---
    if section == "Data Overview":
        st.title("📋 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

    # --- SECTION 2: SALES FORECAST (The Core Feature) ---
    elif section == "Sales Forecast":
        st.title("🚀 Future Sales Trend Analysis")
        
        # Column Selection UI
        st.warning("Please map the columns for forecasting:")
        c1, c2 = st.columns(2)
        date_col = c1.selectbox("Select Date Column", df.columns)
        target_col = c2.selectbox("Select Revenue/Sales Column", df.select_dtypes(include=np.number).columns)

        if st.button("Generate Forecast"):
            try:
                # Prepare data for Prophet
                forecast_df = df[[date_col, target_col]].copy()
                forecast_df[date_col] = pd.to_datetime(forecast_df[date_col], errors='coerce')
                forecast_df.dropna(subset=[date_col], inplace=True)
                
                # Group by date to handle multiple transactions per day
                forecast_df = forecast_df.groupby(date_col)[target_col].sum().reset_index()
                forecast_df.columns = ['ds', 'y']

                with st.spinner("🔮 Analyzing trends and predicting future..."):
                    m = Prophet(changepoint_prior_scale=0.05, daily_seasonality=False)
                    m.fit(forecast_df)
                    
                    # Create future dates (90 days)
                    future = m.make_future_dataframe(periods=90)
                    forecast = m.predict(future)

                # Plotting
                fig = go.Figure()
                # Actual Data
                fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['y'], name="Actual Sales", line=dict(color="#FF4B4B")))
                # Forecasted Data
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicted Trend", line=dict(color="#1f77b4", dash='dash')))
                # Confidence Interval
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(173,216,230,0.3)', name="Confidence range"))

                fig.update_layout(title="Sales Projection for Next 90 Days", hovermode="x unified", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

                # Metrics
                m1, m2, m3 = st.columns(3)
                last_val = forecast_df['y'].iloc[-1]
                future_val = forecast['yhat'].iloc[-1]
                growth = ((future_val - last_val)/last_val)*100
                
                m1.metric("Historical Total", f"${forecast_df['y'].sum():,.2f}")
                m2.metric("Predicted Value (90d)", f"${future_val:,.2f}")
                m3.metric("Projected Growth", f"{growth:+.2f}%")

            except Exception as e:
                st.error(f"Error during forecasting: {e}")
                st.info("Tip: Ensure your Date column is formatted correctly in the CSV.")
	
# --- NEW FEATURE: Top to Bottom Performance Analysis ---
        st.markdown("---")
        st.subheader("📊 Full Market Performance (Top to Bottom)")
        
        cat_cols = df.select_dtypes(include=['object']).columns
        num_cols = df.select_dtypes(include=[np.number]).columns

        if not cat_cols.empty and not num_cols.empty:
            c1, c2 = st.columns(2)
            group_col = c1.selectbox("Select Entity Column (Company/Category)", cat_cols, key="full_rank_cat")
            val_col = c2.selectbox("Select Value Column (Sales/Revenue)", num_cols, key="full_rank_val")

            # Complete Ranking Logic
            full_rank = df.groupby(group_col)[val_col].sum().sort_values(ascending=False).reset_index()
            
            # Display Top 5 and Bottom 5 in columns
            top_col, bot_col = st.columns(2)
            
            with top_col:
                st.markdown("### 📈 Top 5 Performers")
                for i in range(min(5, len(full_rank))):
                    st.write(f"**{i+1}. {full_rank.iloc[i][group_col]}**: ${full_rank.iloc[i][val_col]:,.2f}")

            with bot_col:
                st.markdown("### 📉 Bottom 5 Performers")
                for i in range(1, min(6, len(full_rank) + 1)):
                    st.write(f"**{len(full_rank)-i+1}. {full_rank.iloc[-i][group_col]}**: ${full_rank.iloc[-i][val_col]:,.2f}")

            # Expandable Full Ranking List
            with st.expander("View Full Ranking List (All Companies)"):
                st.table(full_rank.style.format({val_col: "${:,.2f}"}))
                
            # Visualization: Full Trend
            fig_full = px.bar(full_rank, x=group_col, y=val_col, 
                             title="Full Sales Distribution: Top to Bottom",
                             color=val_col, color_continuous_scale='RdYlGn')
            st.plotly_chart(fig_full, use_container_width=True)
            
        else:
            st.info("Please ensure your CSV contains categorical and numeric data.")
    # --- SECTION 3: ML CLASSIFICATION ---
    elif section == "ML Classification":
        st.title("🧪 Classification Model")
        st.write("Train a model to classify categories based on numeric features.")
        
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        if not numeric_df.empty:
            target = st.selectbox("Select Target Label", df.select_dtypes(include=['object']).columns)
            
            if st.button("Run Classification"):
                X = numeric_df
                le = LabelEncoder()
                y = le.fit_transform(df.loc[numeric_df.index, target])
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                model = LogisticRegression(max_iter=500)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                st.text("Model Results:")
                st.text(classification_report(y_test, y_pred))
        else:
            st.error("No numeric data available for classification.")

else:
    st.title("📊 Sales Forecasting Dashboard")
    st.info("Please upload a CSV file in the sidebar to begin your analysis.")
    # Show a placeholder image or instruction
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&q=80&w=1000", caption="Analyze your business trends efficiently.")