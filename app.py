import streamlit as st
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Define F1 scores for each model
f1_scores = {
    "Logistic Regression": {"y1": 0.9856, "y2": 0.9759},
    "XGBoost Classifier": {"y1": 1.0, "y2": 1.0},
    "Random Forest Classifier": {"y1": 1.0, "y2": 1.0},
    "Support Vector Machine": {"y1": 1.0, "y2": 1.0},
    "AdaBoost Classifier": {"y1": 1.0, "y2": 1.0},
    "Bagging Classifier": {"y1": 0.9856, "y2": 0.9711},
    "Decision Tree": {"y1": 1.0, "y2": 1.0}
}

def display_predictions(prediction_y1, prediction_y2, model_choice):
    """
    Display the predictions for Y1 and Y2 based on the input values.
    
    Parameters:
    - prediction_y1 (list): Prediction result for Y1.
    - prediction_y2 (list): Prediction result for Y2.
    """
    
    # Title and Explanation
    st.markdown("## Predictions")
    st.write('Let\'s evaluate the responses based on the selected model!')
    
    # Questions
    st.markdown("### Questions")
    st.markdown("**Y1** = Would you feel comfortable discussing a mental health issue with your coworkers?")
    st.markdown("**Y2** = Would you feel comfortable discussing a mental health issue with your direct supervisor(s)?")
    
    # F1 Scores
    st.markdown("### Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**F1 Score for Y1**: {f1_scores[model_choice]['y1']}")
    with col2:
        st.markdown(f"**F1 Score for Y2**: {f1_scores[model_choice]['y2']}")
    
    # Predictions
    st.markdown("### Results")
    
    y1_result = "Needs Mental Health care" if prediction_y1[0] == 1 else "Does not need Mental Health care"
    y2_result = "High likelihood of mental health disorder" if prediction_y2[0] == 1 else "Low likelihood of mental health disorder"
    
    st.markdown(f"**Prediction for Y1**: <span style='color:red'>{y1_result}</span>", unsafe_allow_html=True)
    st.markdown(f"**Prediction for Y2**: <span style='color:yellow'>{y2_result}</span>", unsafe_allow_html=True)


# Usage:
# display_predictions([1], [0])

# Define the Streamlit app
def app():
    st.title("Mental Health Predictor")
    
    # Create a sidebar for model selection
    model_choice = st.sidebar.selectbox("Choose the Model", ["Logistic Regression", "XGBoost Classifier", "Random Forest Classifier", "Support Vector Machine", "AdaBoost Classifier", "Bagging Classifier", "Decision Tree"], index=0)

    
    # Demographics
    st.subheader("About You")
    
    # Age groups
    age_group_encoding = {"18-24":0, "25-29":1, "30-34":2, "35-39":3, "40-44":4, "45-49":5, "50-54":6, "55-59":7, "60-64":8, "65-69":9, "70-74":10, "75-79":11, "80-84":12, "85-89":13, "90-94":14, "95-99":15}
    age = st.selectbox("Age Group", list(age_group_encoding.keys()), index=0)
    encoded_age = age_group_encoding[age]
    
    # Gender
    gender_encoding = {"Male":0, "Female":1, "Non-Binary":2, "Prefer not to say":3, "Other":4}
    gender = st.selectbox("Gender", list(gender_encoding.keys()), index=0)
    encoded_gender = gender_encoding[gender]
    
    # Provided mapping
    country_mapping = {
        "United States of America": 6,
        "Brazil": 51,
        "Italy": 23,
        "Canada": 52,
        "Germany": 16,
        "India": 52,
        "Belarus": 16,
        "Macedonia": 51,
        "Slovenia": 52,
        "Albania": 52,
        "Austria": 51,
        "Kenya": 52,
        "Australia": 17,
        "Sao Tome and Principe": 52,
        "Vietnam": 52,
        "Indonesia": 52,
        "Switzerland": 47,
        "Finland": 45,
        "Turkey": 52,
        "Poland": 52,
        "United Kingdom": 52,
        "Nigeria": 6,
        "Bulgaria": 5,
        "Estonia": 39,
        "Colombia": 45,
        "Netherlands": 52,
        "Israel": 52,
        "Bangladesh": 52,
        "Greece": 52,
        "China": 52,
        "South Africa": 21,
        "Portugal": 51,
        "Pakistan": 52
    }

    # List of countries with "Other" option
    country_list = sorted(list(country_mapping.keys())) + ["Other"]

    # User selects a country
    country = st.selectbox("Country of Residence", country_list, index=0)

    # If "Other" is selected, get the country name from text input
    if country == "Other":
        other_country = st.text_input("Please specify your country:")
        if other_country:
            encoded_country = country_mapping.get(other_country, 99) # If not found in the mapping, default to 99
    else:
        encoded_country = country_mapping[country]

        
        
    # Work Environment
    st.subheader("Your Work Context")
    
    # Predefined categories for organization size
    num_employees_encoding = {
        "1-10": 0,
        "11-50": 1,
        "51-100": 2,
        "101-250": 3,
        "251-500": 4,
        "500+": 5
    }
    num_employees = st.selectbox("How many employees does your company or organization have?", list(num_employees_encoding.keys()), index=0)
    encoded_num_employees = num_employees_encoding[num_employees]
    
    mental_health_benefits_encoding = {"Yes": 1, "No": 0, "Don't Know": 2}
    mental_health_benefits = st.radio("Does your employer provide mental health benefits?", list(mental_health_benefits_encoding.keys()), index=0)
    encoded_mental_health_benefits = mental_health_benefits_encoding[mental_health_benefits]
    
    anonymity_protected_encoding = {"Yes": 1, "No": 0, "Don't Know": 2}
    anonymity_protected = st.radio("Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?", list(anonymity_protected_encoding.keys()), index=0)
    encoded_anonymity_protected = anonymity_protected_encoding[anonymity_protected]
    

    # Personal Experience
    st.subheader("Personal Mental Health")
    sought_treatment_encoding = {"Yes": 1, "No": 0}
    sought_treatment = st.radio("Have you ever sought treatment for a mental health disorder from a health professional?", list(sought_treatment_encoding.keys()), index=0)
    encoded_sought_treatment = sought_treatment_encoding[sought_treatment]
    
    diagnosed_condition_encoding = {"Yes": 1, "No": 0, "Prefer not to say": 2}
    diagnosed_condition = st.radio("Have you been diagnosed with a mental health condition by a medical professional?", list(diagnosed_condition_encoding.keys()), index=0)
    encoded_diagnosed_condition = diagnosed_condition_encoding[diagnosed_condition]
    
    if diagnosed_condition == "Yes":
        condition_description = st.text_area("If so, what condition(s) have you been diagnosed with?", "Type here...")

    # Perceptions
    st.subheader("Perceptions at the Workplace")
    physical_health_discussion_encoding = {"Yes": 1, "No": 0, "Maybe": 2}
    physical_health_discussion = st.radio("Do you think that discussing a physical health issue with your employer would have negative consequences?", list(physical_health_discussion_encoding.keys()), index=0)
    encoded_physical_health_discussion = physical_health_discussion_encoding[physical_health_discussion]
   
    mental_health_discussion_encoding = {"Yes": 1, "No": 0, "Maybe": 2}
    mental_health_discussion = st.radio("Do you think that discussing a mental health disorder with your employer would have negative consequences?", list(mental_health_discussion_encoding.keys()), index=0)
    encoded_mental_health_discussion = mental_health_discussion_encoding[mental_health_discussion]
    
    discuss_with_coworkers_encoding = {"Yes": 1, "No": 0, "Maybe": 2}
    discuss_with_coworkers = st.radio("Would you be willing to discuss a mental health issue with your coworkers?", list(discuss_with_coworkers_encoding.keys()), index=0)
    encoded_discuss_with_coworkers = discuss_with_coworkers_encoding[discuss_with_coworkers]

    # Work Interference
    st.subheader("Impact on Work")
    interference_treated_encoding = {"Rarely": 0, "Sometimes": 1, "Often": 2, "Always": 3}
    interference_treated = st.selectbox("How often do you feel that your mental health interferes with your work when being treated effectively?", list(interference_treated_encoding.keys()), index=0)
    encoded_interference_treated = interference_treated_encoding[interference_treated]
   
    interference_not_treated_encoding = {"Rarely": 0, "Sometimes": 1, "Often": 2, "Always": 3}
    interference_not_treated = st.selectbox("How often do you feel that your mental health interferes with your work when NOT being treated effectively?", list(interference_not_treated_encoding.keys()), index=0)
    encoded_interference_not_treated = interference_not_treated_encoding[interference_not_treated]

    # Remote Work
    st.subheader("Work Setting")
    
    remote_work_encoding = {"Yes": 1, "No": 0}
    remote_work = st.radio("Do you work remotely (outside of an office) at least 50% of the time?", ["Yes", "No"], index=0)
    encoded_remote_work = remote_work_encoding[remote_work]

    #Variables
    OpenlyIdentified_MH_Work = 0.0
    SelfEmployed = 0.0
    DiscussMH_PrevCoworker = encoded_mental_health_discussion
    DiscussMH_PrevEmployer = encoded_mental_health_discussion
    PrevCoworkerDiscussMH = 0.0
    PrevEmployerDiscussMH_Campaign = 1.0
    PrevEmployerMHResources = 0.0
    CurrentMHDisorder = encoded_diagnosed_condition
    FamilyHistoryMH = 1.0
    PreviousEmployers = 1.0
    KnowMHCareOptions = 1.0
    EmployerMHResources = 1.0
    EmployerMHBenefits = encoded_mental_health_benefits
    EmployerDiscussMH_Campaign = 0.0
    DiscussMH_Coworkers = encoded_mental_health_discussion
    DiscussMH_Employer = encoded_mental_health_discussion
    CoworkerDiscussMH = 0.0
    SoughtMH_Treatment = encoded_sought_treatment
    PastMHDisorder = 1.0
    ObservedSupportiveResponse = 1.0
    ObservedBadResponse = 1.0
    ObservationInfluenceReveal = 0.0
    PrevEmployerMHBenefits = encoded_mental_health_benefits
    CompanySize = encoded_num_employees
    ShareMH_FriendsFamily = encoded_discuss_with_coworkers
    RequestMH_LeaveEase = 4.0
    TeamReaction_KnowMH = 5.0
    MHInterferes_NOTTreated = encoded_interference_not_treated
    MHInterferes_Treated = encoded_interference_treated
    AnonymityMHResources = encoded_physical_health_discussion
    TechCompany = 1.0
    TechRole = 1.0
    PrevEmployerImportance_MH = 4.0
    PrevEmployerImportance_PH = encoded_physical_health_discussion
    EmployerImportance_MH = 5.0
    EmployerImportance_PH = encoded_physical_health_discussion
    TechIndustrySupport = 3.0
    PrevAnonymityMHResources = encoded_anonymity_protected
    PrevTechCompany = 1.0
    AwarePrevMHCare = 1.0
    LiveCountry = encoded_country
    WorkCountry = encoded_country
    Age = encoded_age
    Gender = encoded_gender
    DiscussPH_Interview = encoded_physical_health_discussion
    DiscussMH_Interview = 0.0
    ComfortDiscussMH_Coworkers = 1.0
    ComfortDiscussMH_Supervisor = 1.0
    ComfortDiscussPHvsMH = encoded_physical_health_discussion
    PrevComfortDiscussMH_Coworkers = encoded_discuss_with_coworkers
    PrevComfortDiscussMH_Supervisor = encoded_discuss_with_coworkers
    PrevComfortDiscussPHvsMH = encoded_physical_health_discussion

    # Combine input into an array for prediction. Make sure the order of features here matches the order the model expects.
    if st.button("Predict"):
        OpenlyIdentified_MH_Work :st.number_input('Are you openly identified at work as a person with a mental health issue?', value=0)
        SelfEmployed:st.number_input('Are you self-employed?', value=0)
        DiscussMH_PrevCoworker:st.number_input('Did you ever discuss your mental health with a previous coworker(s)?', value=0)
        DiscussMH_PrevEmployer:st.number_input('Did you ever discuss your mental health with your previous employer?', value=0)
        PrevCoworkerDiscussMH:st.number_input('Did you ever have a previous coworker discuss their or another coworkers mental health with you?', value=0)
        PrevEmployerDiscussMH_Campaign:st.number_input('Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?', value=0)
        PrevEmployerMHResources:st.number_input('Did your previous employers provide resources to learn more about mental health disorders and how to seek help?', value=0)
        CurrentMHDisorder:st.number_input('Do you currently have a mental health disorder?', value=0)
        FamilyHistoryMH:st.number_input('Do you have a family history of mental illness?', value=0)
        PreviousEmployers:st.number_input('Do you have previous employers?', value=0)
        KnowMHCareOptions:st.number_input('Do you know the options for mental health care available under your employer-provided health coverage?', value=0)
        EmployerMHResources:st.number_input('Does your employer offer resources to learn more about mental health disorders and options for seeking help?', value=0)
        EmployerMHBenefits:st.number_input('Does your employer provide mental health benefits as part of healthcare coverage?', value=0)
        EmployerDiscussMH_Campaign:st.number_input('Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?', value=0)
        DiscussMH_Coworkers:st.number_input('Have you ever discussed your mental health with coworkers?', value=0)
        DiscussMH_Employer:st.number_input('Have you ever discussed your mental health with your employer?', value=0)
        CoworkerDiscussMH:st.number_input('Have you ever had a coworker discuss their or another coworkers mental health with you?', value=0)
        SoughtMH_Treatment:st.number_input('Have you ever sought treatment for a mental health disorder from a mental health professional?', value=0)
        PastMHDisorder:st.number_input('Have you had a mental health disorder in the past?', value=0)
        ObservedSupportiveResponse:st.number_input('Have you observed or experienced a supportive or well handled response to a mental health issue in your current or previous workplace?', value=0)
        ObservedBadResponse:st.number_input('Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?', value=0)
        ObservationInfluenceReveal:st.number_input('Have your observations of how another individual who discussed a mental health issue made you less likely to reveal a mental health issue yourself in your current workplace?', value=0)
        PrevEmployerMHBenefits:st.number_input('Have your previous employers provided mental health benefits?', value=0)
        CompanySize:st.number_input('How many employees does your company or organization have?', value=0)
        ShareMH_FriendsFamily:st.number_input('How willing would you be to share with friends and family that you have a mental illness?', value=0)
        RequestMH_LeaveEase:st.number_input('If a mental health issue prompted you to request a medical leave from work, how easy or difficult would it be to ask for that leave?', value=0)
        TeamReaction_KnowMH:st.number_input('If they knew you suffered from a mental health disorder, how do you think that your team members/co-workers would react?', value=0)
        MHInterferes_NOTTreated:st.number_input('If you have a mental health disorder, how often do you feel that it interferes with your work when NOT being treated effectively (i.e., when you are experiencing symptoms)?', value=0)
        MHInterferes_Treated:st.number_input('If you have a mental health disorder, how often do you feel that it interferes with your work when being treated effectively?', value=0)
        AnonymityMHResources:st.number_input('Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?', value=0)
        TechCompany:st.number_input('Is your employer primarily a tech company/organization?', value=0)
        TechRole:st.number_input('Is your primary role within your company related to tech/IT?', value=0)
        PrevEmployerImportance_MH:st.number_input('Overall, how much importance did your previous employer place on mental health?', value=0)
        PrevEmployerImportance_PH:st.number_input('Overall, how much importance did your previous employer place on physical health?', value=0)
        EmployerImportance_MH:st.number_input('Overall, how much importance does your employer place on mental health?', value=0)
        EmployerImportance_PH:st.number_input('Overall, how much importance does your employer place on physical health?', value=0)
        TechIndustrySupport:st.number_input('Overall, how well do you think the tech industry supports employees with mental health issues?', value=0)
        PrevAnonymityMHResources:st.number_input('Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?', value=0)
        PrevTechCompany:st.number_input('Was your employer primarily a tech company/organization?', value=0)
        AwarePrevMHCare:st.number_input('Were you aware of the options for mental health care provided by your previous employers?', value=0)
        LiveCountry:st.number_input('What country do you live in?', value=0)
        WorkCountry:st.number_input('What country do you work in?', value=0)
        Age:st.number_input('What is your age?', value=0)
        Gender:st.number_input('What is your gender?', value=0)
        DiscussPH_Interview:st.number_input('What is your race?', value=0)
        DiscussMH_Interview:st.number_input('Would you be willing to bring up a physical health issue with a potential employer in an interview?', value=0)
        ComfortDiscussMH_Coworkers:st.number_input('Would you bring up your mental health with a potential employer in an interview?', value=0)
        ComfortDiscussMH_Supervisor:st.number_input('Would you feel more comfortable talking to your coworkers about your physical health or your mental health?', value=0)
        ComfortDiscussPHvsMH:st.number_input('Would you have been willing to discuss your mental health with your coworkers at previous employers?', value=0)
        PrevComfortDiscussMH_Coworkers:st.number_input('Would you have been willing to discuss your mental health with your direct supervisor(s)?', value=0)
        PrevComfortDiscussMH_Supervisor:st.number_input('Would you have felt more comfortable talking to your previous employer about your physical health or your mental health?', value=0)
        PrevComfortDiscussPHvsMH:st.number_input('Overall, how much importance did your previous employer place on physical health?', value=0)
            
        # Assuming all the necessary variables like Timestamp, OpenlyIdentified_MH_Work, etc. have been initialized previously
        input_data = np.array([OpenlyIdentified_MH_Work, SelfEmployed, DiscussMH_PrevCoworker,DiscussMH_PrevEmployer,PrevCoworkerDiscussMH,PrevEmployerDiscussMH_Campaign,PrevEmployerMHResources, CurrentMHDisorder,
                            FamilyHistoryMH, PreviousEmployers, KnowMHCareOptions, EmployerMHResources, EmployerMHBenefits, EmployerDiscussMH_Campaign, DiscussMH_Coworkers, DiscussMH_Employer, CoworkerDiscussMH,
                            SoughtMH_Treatment, PastMHDisorder, ObservedSupportiveResponse, ObservedBadResponse, ObservationInfluenceReveal, PrevEmployerMHBenefits, CompanySize, ShareMH_FriendsFamily, RequestMH_LeaveEase,
                            TeamReaction_KnowMH, MHInterferes_NOTTreated, MHInterferes_Treated, AnonymityMHResources, TechCompany, TechRole, PrevEmployerImportance_MH, PrevEmployerImportance_PH, EmployerImportance_MH,
                            EmployerImportance_PH, TechIndustrySupport, PrevAnonymityMHResources, PrevTechCompany, AwarePrevMHCare, LiveCountry, WorkCountry, Age, Gender, DiscussPH_Interview, DiscussMH_Interview, 
                            ComfortDiscussMH_Coworkers, ComfortDiscussMH_Supervisor, ComfortDiscussPHvsMH, PrevComfortDiscussMH_Coworkers, PrevComfortDiscussMH_Supervisor, PrevComfortDiscussPHvsMH]).reshape(1, -1)

        # Make predictions
        if model_choice == "Logistic Regression":
            # Logistic Regression Models
            logreg_model_y1 = joblib.load("logreg_model_y1.pkl")
            logreg_model_y2 = joblib.load("logreg_model_y2.pkl")

            prediction_y1 = logreg_model_y1.predict(input_data)
            prediction_y2 = logreg_model_y2.predict(input_data)
            display_predictions(prediction_y1, prediction_y2, model_choice)
            
        if model_choice == "XGBoost Classifier":
            # XGBoost Classifier Models
            xgboost_model_y1 = joblib.load("xgboost_model_y1.pkl")
            xgboost_model_y2 = joblib.load("xgboost_model_y2.pkl")
            
            prediction_y1 = xgboost_model_y1.predict(input_data)
            prediction_y2 = xgboost_model_y2.predict(input_data)
            display_predictions(prediction_y1, prediction_y2, model_choice)
            
        if model_choice == "Support Vector Machine":
            # Support Vector Machine Models
            svm_model_y1 = joblib.load("svm_model_y1.pkl")
            svm_model_y2 = joblib.load("svm_model_y2.pkl")

            prediction_y1 = svm_model_y1.predict(input_data)
            prediction_y2 = svm_model_y2.predict(input_data)
            display_predictions(prediction_y1, prediction_y2, model_choice)
            
        if model_choice == "Random Forest Classifier":
            # Random Forest Models
            random_forest_model_y1 = joblib.load("random_forest_model_y1.pkl")
            random_forest_model_y2 = joblib.load("random_forest_model_y2.pkl")

            prediction_y1 = random_forest_model_y1.predict(input_data)
            prediction_y2 = random_forest_model_y2.predict(input_data)
            display_predictions(prediction_y1, prediction_y2, model_choice)
            
        if model_choice == "Decision Tree":
            # Decision Tree Models
            decision_tree_model_y1 = joblib.load("decision_tree_model_y1.pkl")
            decision_tree_model_y2 = joblib.load("decision_tree_model_y2.pkl")

            prediction_y1 = decision_tree_model_y1.predict(input_data)
            prediction_y2 = decision_tree_model_y2.predict(input_data)
            display_predictions(prediction_y1, prediction_y2, model_choice)
            
        if model_choice == "Bagging Classifier":
            # Bagging Classifier Models
            bagging_model_y1 = joblib.load("bagging_model_y1.pkl")
            bagging_model_y2 = joblib.load("bagging_model_y2.pkl")
            
            prediction_y1 = bagging_model_y1.predict(input_data)
            prediction_y2 = bagging_model_y2.predict(input_data)
            display_predictions(prediction_y1, prediction_y2, model_choice)
    
        if model_choice == "AdaBoost Classifier":
            # AdaBoost Models
            ada_model_y1 = joblib.load("ada_model_y1.pkl")
            ada_model_y2 = joblib.load("ada_model_y2.pkl")
            
            prediction_y1 = ada_model_y1.predict(input_data)
            prediction_y2 = ada_model_y2.predict(input_data)
            display_predictions(prediction_y1, prediction_y2, model_choice)

                            
            # Optionally, add other sections like visualizations, insights, etc.

    # Run the app
if __name__ == "__main__":

    app()