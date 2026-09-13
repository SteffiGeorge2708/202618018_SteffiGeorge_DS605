import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Configuration
st.set_page_config(
    page_title="NYC Airbnb Price Predictor", 
    page_icon="🗽", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Advanced Custom CSS (Large Typography & Premium Sidebar Styling)
st.markdown("""
    <style>
    /* Global Font & Background adjustments */
    html, body, [class*="css"] {
        font-size: 19px !important;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    .stApp {
        background-color: #f8f9fa;
    }

    /* Styled Sidebar Menu */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 2px solid #e9ecef !important;
        padding-top: 1rem;
    }

    /* Sidebar Title & Header Styling */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ff5a5f !important;
        font-size: 1.6rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    /* Sidebar Field Labels */
    [data-testid="stSidebar"] label {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: #2b2d42 !important;
        margin-top: 5px;
    }

    /* Inputs & Selectboxes styling */
    .stSelectbox div[data-baseweb="select"], div[data-baseweb="base-input"] input {
        font-size: 1.1rem !important;
        border-radius: 8px !important;
    }

    /* Main Section Typography */
    h1 {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        color: #1a1a1a !important;
    }

    .stSubheader, h2, h3 {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #2c3e50 !important;
    }

    /* Large Custom Price Display Card */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #fff5f5 100%);
        border-radius: 16px;
        padding: 32px;
        box-shadow: 0 8px 20px rgba(255, 90, 95, 0.12);
        border: 2px solid #ffe3e3;
        text-align: center;
        margin-bottom: 24px;
    }

    .card-title {
        font-size: 1.35rem !important;
        color: #555555;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .price-display {
        font-size: 60px !important;
        font-weight: 900 !important;
        color: #ff5a5f;
        margin-top: 8px;
        margin-bottom: 0px;
    }

    /* Primary CTA Button */
    .stButton > button {
        font-size: 1.35rem !important;
        font-weight: 800 !important;
        background-color: #ff5a5f !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(255, 90, 95, 0.3) !important;
        transition: all 0.2s ease-in-out;
    }

    .stButton > button:hover {
        background-color: #e0484d !important;
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load Data & Saved Pipeline
@st.cache_data
def load_data():
    df = pd.read_csv("AB_NYC_2019.csv")
    return df[(df["price"] >= 10) & (df["price"] <= 500)].copy()

@st.cache_resource
def load_pipeline():
    return joblib.load("models/airbnb_pipeline.pkl")

try:
    df_clean = load_data()
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"⚠️ Error loading model or dataset: {e}")
    st.stop()

# 4. Header Section
st.title("🗽 NYC Airbnb Nightly Price Predictor")
st.write("Configure your listing attributes in the sidebar menu to estimate optimal nightly rental rates.")
st.divider()

# 5. Redesigned Sidebar Menu
st.sidebar.markdown("### 🎛️ Listing Parameters")

neighbourhood_group = st.sidebar.selectbox(
    "Neighbourhood Group", 
    ["Manhattan", "Brooklyn", "Queens", "Staten Island", "Bronx"]
)

room_type = st.sidebar.selectbox(
    "Room Type", 
    ["Entire home/apt", "Private room", "Shared room"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📍 Location & Rules")

latitude = st.sidebar.number_input("Latitude", value=40.7128, format="%.4f")
longitude = st.sidebar.number_input("Longitude", value=-74.0060, format="%.4f")
minimum_nights = st.sidebar.slider("Minimum Nights Required", 1, 30, 2)
availability_365 = st.sidebar.slider("Annual Availability (Days)", 0, 365, 180)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Host & Review Metrics")

number_of_reviews = st.sidebar.number_input("Total Number of Reviews", value=10, min_value=0)
reviews_per_month = st.sidebar.number_input("Reviews Per Month", value=1.0, min_value=0.0, format="%.2f")
calculated_host_listings_count = st.sidebar.number_input("Host Total Listings", value=1, min_value=1)

# 6. Main Content Area (Two Columns)
col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    st.subheader("🎯 Price Valuation")
    
    if st.button("Calculate Nightly Rate", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            "neighbourhood_group": neighbourhood_group,
            "room_type": room_type,
            "latitude": latitude,
            "longitude": longitude,
            "minimum_nights": minimum_nights,
            "number_of_reviews": number_of_reviews,
            "reviews_per_month": reviews_per_month,
            "calculated_host_listings_count": calculated_host_listings_count,
            "availability_365": availability_365
        }])
        
        # Predict on log scale and transform back to dollars
        pred_log = pipeline.predict(input_data)[0]
        predicted_price = np.expm1(pred_log)
        
        # Prominent Result Display Card
        st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Estimated Nightly Price</div>
                <div class="price-display">${predicted_price:.2f}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Plot 1: Price Distribution (With enlarged text parameters)
        st.subheader("📊 Market Comparison")
        fig, ax = plt.subplots(figsize=(7, 4.2))
        
        # Set large matplotlib font parameters
        plt.rcParams.update({'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 15})
        sns.set_theme(style="whitegrid")
        
        sns.histplot(df_clean["price"], kde=True, ax=ax, color="#ff5a5f", bins=35, alpha=0.4)
        ax.axvline(predicted_price, color="#2b2d42", linestyle="--", linewidth=2.5, label=f"Predicted: ${predicted_price:.2f}")
        ax.set_title("Listing Position in NYC Airbnb Price Market", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Nightly Price ($)", fontsize=13, labelpad=8)
        ax.set_ylabel("Listings Count", fontsize=13, labelpad=8)
        ax.legend(fontsize=12)
        st.pyplot(fig)

with col2:
    st.subheader("📈 Model Analytics")
    
    try:
        regressor = pipeline.named_steps["regressor"]
        preprocessor = pipeline.named_steps["preprocessor"]
        
        num_cols = ["latitude", "longitude", "minimum_nights", "number_of_reviews", 
                    "reviews_per_month", "calculated_host_listings_count", "availability_365"]
        cat_cols = ["neighbourhood_group", "room_type"]
        
        onehot_cols = list(preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(cat_cols))
        all_features = num_cols + onehot_cols
        
        importances = regressor.feature_importances_
        feat_imp = pd.Series(importances, index=all_features).sort_values(ascending=False).head(7)
        
        # Plot 2: Feature Importance (With enlarged text parameters)
        st.markdown("##### Key Factors Influencing Predictions")
        fig2, ax2 = plt.subplots(figsize=(7, 4.8))
        
        sns.set_theme(style="whitegrid")
        sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax2, palette="Reds_r")
        ax2.set_title("Top Feature Importances (Random Forest)", fontsize=14, fontweight="bold", pad=12)
        ax2.set_xlabel("Relative Importance Weight", fontsize=13, labelpad=8)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        
        st.pyplot(fig2)
    except Exception as err:
        st.info("Train the pipeline model to view feature importances.")