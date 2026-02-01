import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from ml.pippline import model_training 
from sqlalchemy import text  # For raw SQL if needed, but we'll use ORM-style
from database_relational.db_main import sessionlocal


res_test, res_train, y_test, model, x, y_train, x_train, x_test = model_training()

# Page configuration (unchanged)
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (unchanged - all your styling remains exactly the same)
st.markdown("""
<style>
    /* [All your existing CSS unchanged] */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    /* ... rest of your CSS exactly as is ... */
</style>
""", unsafe_allow_html=True)

# Main title (unchanged)
st.markdown('<h1 class="main-title">📈 CUSTOMER CHURN PREDICTOR</h1>', unsafe_allow_html=True)

# Sidebar (unchanged - all inputs exactly the same)
with st.sidebar:
    st.markdown("## 📋 Customer Information")
    st.markdown("---")
    
    # Demographics
    st.markdown("### 👤 Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", ["yes", "No"])
    Partner = st.selectbox("Partner", ["yes", "No"])
    Dependents = st.selectbox("Dependents", ["yes", "No"])
    
    st.markdown("---")
    
    # Account Details
    st.markdown("### 📊 Account Details")
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, help="How long the customer has been with the company")
    MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, step=0.1)
    TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, max_value=5000.0, step=0.1)
    
    st.markdown("---")
    
    # Services
    st.markdown("### 📞 Services")
    PhoneService = st.selectbox("Phone Service", ["yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["yes", "No", "No phone"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["yes", "No", "No internet"])
    OnlineBackup = st.selectbox("Online Backup", ["yes", "No", "No internet"])
    DeviceProtection = st.selectbox("Device Protection", ["yes", "No", "No internet"])
    TechSupport = st.selectbox("Tech Support", ["yes", "No", "No internet"])
    StreamingTV = st.selectbox("Streaming TV", ["yes", "No", "No internet"])
    StreamingMovies = st.selectbox("Streaming Movies", ["yes", "No", "No internet"])
    
    st.markdown("---")
    
    # Billing
    st.markdown("### 💳 Billing")
    Contract = st.selectbox("Contract", ["month to month", "one year", "two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", 
                                 ["Electronic check", "Mailed check", 
                                  "Bank transfer (automatic)", "Credit card (automatic)"])

# Main content area (unchanged)
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Prediction Center")
    st.markdown("Click the button below to predict customer churn probability based on the provided information.")
    st.markdown('</div>', unsafe_allow_html=True)

# Predict button (unchanged)
x.columns = x.columns.astype(str)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_btn = st.button("🔮 Predict Churn", use_container_width=True)

# ONLY CHANGE: Replace raw SQL with SQLAlchemy (inside if predict_btn)
if predict_btn:
    # SQLAlchemy MySQL insertion (replaces your raw psycopg2 code)
    values = (gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, 
              InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, 
              StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, 
              MonthlyCharges, TotalCharges)
    
    # Use your session local
    db_session = sessionlocal()
    try:
        # SQLAlchemy text() for parameterized insert (MySQL compatible)
        insert_query = text("""
            INSERT INTO user_info (gender, SeniorCitizen, Partner, Dependents, tenure, 
                                   PhoneService, MultipleLines, InternetService, OnlineSecurity, 
                                   OnlineBackup, DeviceProtection, TechSupport, StreamingTV, 
                                   StreamingMovies, Contract, PaperlessBilling, PaymentMethod, 
                                   MonthlyCharges, TotalCharges)
            VALUES (:1, :2, :3, :4, :5, :6, :7, :8, :9, :10, 
                    :11, :12, :13, :14, :15, :16, :17, :18, :19)
        """)
        db_session.execute(insert_query, values)
        db_session.commit()
        st.success("✅ Customer data saved to MySQL database successfully!")
    except Exception as e:
        db_session.rollback()
        st.error(f"Database error: {str(e)}")
    finally:
        db_session.close()

    # [Rest of your prediction logic unchanged - prepare userinput, predict, display results]
    # Prepare user input
    userinput = pd.DataFrame(columns=x.columns)
    userinput.loc[0] = 0

    if(gender == "Male"):
        userinput["gender"] = 1
    else:
        userinput["gender"] = 0    

    if(SeniorCitizen == "yes"):
        userinput["SeniorCitizen"] = 1
    else:
        userinput["SeniorCitizen"] = 0 

    if(Partner == "yes"):
        userinput["Partner"] = 1
    else:
        userinput["Partner"] = 0 

    if(Dependents == "yes"):
        userinput["Dependents"] = 1
    else:
        userinput["Dependents"] = 0 

    userinput["tenure"] = tenure    

    if(PhoneService == "yes"):
        userinput["PhoneService"] = 1
    else:
        userinput["PhoneService"] = 0 

    if(PaperlessBilling == "yes"):
        userinput["PaperlessBilling"] = 1
    else:
        userinput["PaperlessBilling"] = 0  

    userinput["MonthlyCharges"] = MonthlyCharges
    userinput["TotalCharges"] = TotalCharges    

    for col, val in {
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaymentMethod": PaymentMethod
    }.items():
        col_name = f"{col}_{val}"
        if(col_name in userinput.columns):
            userinput[col_name] = 1  

    userinput.columns = (userinput.columns).astype(str)
    y_pred = model.predict(userinput)[0]
    st.session_state["prediction"] = y_pred
    st.session_state["user_input"] = userinput

    # Display prediction result (unchanged)
    st.markdown("---")
    
    if(y_pred == 0):
        st.markdown("""
        <div style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                    padding: 30px; border-radius: 15px; text-align: center; 
                    box-shadow: 0 8px 16px rgba(0,0,0,0.2); margin: 20px 0;'>
            <h2 style='color: white; margin: 0; font-size: 2rem;'>✅ Low Churn Risk</h2>
            <p style='color: white; margin: 10px 0 0 0; font-size: 1.2rem;'>
                This customer is likely to stay with the company
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); 
                    padding: 30px; border-radius: 15px; text-align: center; 
                    box-shadow: 0 8px 16px rgba(0,0,0,0.2); margin: 20px 0;'>
            <h2 style='color: white; margin: 0; font-size: 2rem;'>⚠️ High Churn Risk</h2>
            <p style='color: white; margin: 10px 0 0 0; font-size: 1.2rem;'>
                This customer is likely to leave the company
            </p>
        </div>
        """, unsafe_allow_html=True)

# [Rest of your code unchanged - Feature Importance, Model Performance, Footer all exactly the same]
# Feature Importance button
if "prediction" in st.session_state:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        feature_btn = st.button("📊 View Feature Importance", use_container_width=True)
    
    if feature_btn:
        y_pred = st.session_state["prediction"]
        userinput = st.session_state["user_input"]
        
        st.markdown("---")
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Feature Importance Analysis")
        
        feature_importance = pd.DataFrame({
            "feature": x.columns,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)
        
        top_feature = feature_importance.head(10)["feature"].values
        user_value = pd.DataFrame(userinput[top_feature].iloc[0]).values
        importance = feature_importance.head(10)["importance"].values

        fig, ax = plt.subplots(figsize=(10, 6))
        ax2 = ax.twinx()

        ax.barh(top_feature, importance, color='#667eea', alpha=0.7, label='Importance')
        ax2.plot(user_value, top_feature, color="#f45c43", marker='o', linewidth=2, markersize=8, label='User Value')
        
        ax.set_xlabel("Feature Importance", fontsize=12, fontweight='bold')
        ax2.set_xlabel("User Input Value", fontsize=12, fontweight='bold')
        ax.set_ylabel("Features", fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.title("Top 10 Features Influencing Prediction", fontsize=14, fontweight='bold', pad=20)
        st.pyplot(fig)
        
        st.markdown('</div>', unsafe_allow_html=True)

        if(y_pred == 1):
            st.markdown("---")
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("### 💡 Retention Recommendations")
            
            suggestion = []
            if(userinput["tenure"].iloc[0] < 12):
                suggestion.append("Offer long-term discount to increase tenure and build loyalty.")    
            if(userinput["MonthlyCharges"].iloc[0] > 80):
                suggestion.append("Provide flexible billing or cheaper plan options to reduce cost burden.")
            if("TechSupport_yes" in userinput.columns):
                tech_support_value = userinput["TechSupport_yes"].iloc[0]
            else:
                tech_support_value = 0            
            if(tech_support_value == 0):
                suggestion.append("Encourage using Tech Support for better service experience and satisfaction.") 
            if("OnlineSecurity_yes" in userinput.columns):
                OnlineSecurity_value = userinput["OnlineSecurity_yes"].iloc[0]
            else:
                OnlineSecurity_value = 0             
            if(OnlineSecurity_value == 0):
                suggestion.append("Offer security add-ons as part of loyalty program to increase value.")

            for i, s in enumerate(suggestion, 1):
                st.markdown(f"""
                <div style='background: #f8f9fa; padding: 15px; border-radius: 10px; 
                            margin: 10px 0; border-left: 4px solid #667eea;'>
                    <strong style='color: #667eea; font-size: 1.1rem;'>{i}.</strong> {s}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

# Model Performance Section (unchanged)
st.markdown("---")
st.markdown('<div class="info-card">', unsafe_allow_html=True)
st.markdown("### 🎯 Model Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{f1_score(y_train, res_train):.3f}</div>
        <div class="metric-label">F1 Score<br>(Training)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{f1_score(y_test, res_test):.3f}</div>
        <div class="metric-label">F1 Score<br>(Testing)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{accuracy_score(y_train, res_train):.3f}</div>
        <div class="metric-label">Accuracy<br>(Training)</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{accuracy_score(y_test, res_test):.3f}</div>
        <div class="metric-label">Accuracy<br>(Testing)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)