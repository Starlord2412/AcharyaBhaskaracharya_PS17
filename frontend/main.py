import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from ml.pippline import model_training 
import mysql.connector as mysql


res_test,res_train,y_test,model,x,y_train,x_train,x_test=model_training()
try:
 mydb=mysql.connect( host="localhost",
    user="root",
    password="Kalyani@190306",
    database="churn",
    auth_plugin='mysql_native_password')
 if(mydb.connect()):
     print("connection successfull!!")
except mysql.connector.Error as e:
    print("Error:",e)     
mc=mydb.cursor()


# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main gradient background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        border-right: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Title styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        padding: 20px;
        margin-bottom: 30px;
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin: 20px 0;
        border-left: 5px solid #667eea;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 15px;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    
    /* Sidebar header styling */
    [data-testid="stSidebar"] h2 {
        color: #667eea;
        font-weight: 700;
        font-size: 1.8rem;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 3px solid #667eea;
    }
    
    /* Input field styling */
    .stSelectbox label, .stNumberInput label {
        color: #2c3e50;
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* Success/Error message styling */
    .element-container div[data-testid="stMarkdownContainer"] > div[data-testid="stMarkdown"] {
        font-size: 1.1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #666;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Main title with custom styling
st.markdown('<h1 class="main-title">📈 CUSTOMER CHURN PREDICTOR</h1>', unsafe_allow_html=True)

# Sidebar for input
with st.sidebar:
    st.markdown("## 📋 Customer Information")
    st.markdown("---")
    
    # Demographic Information
    st.markdown("### 👤 Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", ["yes", "No"])
    Partner = st.selectbox("Partner", ["yes", "No"])
    Dependents = st.selectbox("Dependents", ["yes", "No"])
    
    st.markdown("---")
    
    # Account Information
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
    
    # Billing Information
    st.markdown("### 💳 Billing")
    Contract = st.selectbox("Contract", ["month to month", "one year", "two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", 
                                 ["Electronic check", "Mailed check", 
                                  "Bank transfer (automatic)", "Credit card (automatic)"])

# Main content area
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Prediction Center")
    st.markdown("Click the button below to predict customer churn probability based on the provided information.")
    st.markdown('</div>', unsafe_allow_html=True)

# Predict button
x.columns = x.columns.astype(str)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_btn = st.button("🔮 Predict Churn", use_container_width=True)

if predict_btn:
    # Database insertion
    querry = '''
    insert into user_info(gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,
DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges)
values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)''' 

    values = (gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
        DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges)

    mc.execute(querry, values)
    mydb.commit()

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

    # Display prediction result with enhanced styling
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

# Model Performance Section
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

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px; font-size: 0.9rem;'>
    <p>Built with ❤️ using Streamlit | Customer Churn Prediction System</p>
</div>
""", unsafe_allow_html=True)

mc.close()
mydb.close()