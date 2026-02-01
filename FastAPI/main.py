from fastapi import FastAPI,HTTPException,Depends,Query,Path
from typing import List,Annotated
from database_relational import model
from FastAPI.service import ChurnInput
from database_relational.db_main import engine, sessionlocal
from sqlalchemy.orm import session
from database_relational.auth import get_db
from fastapi.middleware.cors import CORSMiddleware
import joblib
from database_relational.auth import get_db 
from database_relational.model import user_data

app=FastAPI(title="Churn Predictor api")
#app.include_router(auth.router)
model.base.metadata.create_all(bind=engine)#creates tables

#middleware for cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#dependencies
dependency=Annotated[session,Depends(get_db)]  #dependency 

#load model 
model1 = joblib.load("mL/Random_forest_model.pkl") 


@app.get("/",response_model=dict)
async def home():
    return {"message":"This is test route"}    

@app.post("/post_name")#put dependency here
#creat any function for post
async def example_fun(user_ip:ChurnInput, db:dependency):
   if user_ip: 
    db_users=model.user_data(gender=user_ip.gender, SeniorCitizen=user_ip.SeniorCitizen,
                          Partner=user_ip.Partner, Dependents=user_ip.Dependents, tenure=user_ip.tenure, 
                          PhoneService=user_ip.PhoneService, MultipleLines=user_ip.MultipleLines, 
                          InternetService=user_ip.InternetService, OnlineSecurity=user_ip.OnlineSecurity, 
                          OnlineBackup=user_ip.OnlineBackup, DeviceProtection=user_ip.DeviceProtection, 
                          TechSupport=user_ip.TechSupport, StreamingTV=user_ip.StreamingTV, StreamingMovies=user_ip.StreamingMovies, 
                          Contract=user_ip.Contract, PaperlessBilling=user_ip.PaperlessBilling, PaymentMethod=user_ip.PaymentMethod, 
                          MonthlyCharges=user_ip.MonthlyCharges, TotalCharges=user_ip.TotalCharges)# sqlalchemy requireds **
    
    db.add(db_users)
    db.commit()
   
    db.refresh(db_users)
        
        # ✅ 2. Make prediction (add your model prediction logic here)
    prediction = model1.predict([[db_users.gender, db_users.SeniorCitizen, db_users.Partner, db_users.Dependents, db_users.tenure, 
                                      db_users.PhoneService, db_users.MultipleLines, db_users.InternetService, db_users.OnlineSecurity, 
                                      db_users.OnlineBackup, db_users.DeviceProtection, db_users.TechSupport, db_users.StreamingTV, 
                                      db_users.StreamingMovies, db_users.Contract, db_users.PaperlessBilling, db_users.PaymentMethod, 
                                      db_users.MonthlyCharges, db_users.TotalCharges]])[0]  # Your prediction function
        
        # ✅ 3. Return proper response
    return {
            "message": "Customer data saved successfully",
            "prediction": prediction,
            "customer_id": db_users.id
        }
        
   