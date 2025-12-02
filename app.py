import streamlit as st
import pandas as pd
import time
import io
import joblib

# --- 1. Load your ML Model (Dummy Function) ---
# In a real app, you would load your pickle/joblib model here.
# @st.cache_resource keeps the model in memory so it doesn't reload on every interaction.
@st.cache_resource
def load_model():
    # Example: model = joblib.load('my_model.pkl')
    # For now, we return a dummy string to simulate a loaded model
    model=joblib.load("final_model.pkl")
    return model

# --- 2. Prediction Logic ---
def predict_data(model, input_df):
    """

    Simulate a prediction. 
    Replace this logic with: predictions = model.predict(input_df)
    """
    df=input_df.copy()
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    if 'churn' in df.columns:
        df=df.drop(['churn'],axis=1,errors='ignore',inplace=True)
    df['total_calls'] = df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls'] + df['total_intl_calls']
    df['total_minutes'] = df['total_day_minutes'] + df['total_eve_minutes'] + df['total_night_minutes'] + df['total_intl_minutes']
    df['avg_call_duration'] = df['total_minutes'] / (df['total_calls'].replace(0,1))
    df['high_service_calls'] = (df['customer_service_calls'] > 3).astype(int)
    df['total_charge'] = df['total_day_charge'] + df['total_eve_charge'] + df['total_night_charge'] + df['total_intl_charge']
    df.drop(['total_day_minutes', 'total_eve_minutes', 'total_night_minutes', 'total_intl_minutes', 'total_minutes','total_eve_charge','total_day_charge'], axis= 1 , inplace= True)
    df.drop(["total_eve_calls", "total_night_calls", "total_calls", 'total_day_calls', "avg_call_duration", "area_code", "state" ], inplace= True, axis= 1)
    df['international_plan'] = df['international_plan'].map({'Yes': 1, 'No': 0})
    df['voice_mail_plan'] = df['voice_mail_plan'].map({'Yes': 1, 'No': 0})
    # Simulate processing time
    with st.spinner('Predicting...'):
        time.sleep(2) 
        
    # Example logic: Create a new 'Prediction' column based on input
    # (Here we just multiply a numeric column by 2 for demonstration)
    processed_data=df.copy()
    churn_prediction=model.predict(processed_data)
    results_df = input_df.copy()
    results_df['prediction'] = churn_prediction
    results_df['prediction'] = results_df['prediction'].map({
        0: 'Will Not Churn',
        1: 'Will Churn'
    })
        
    return results_df

# --- 3. Main App Layout ---
st.set_page_config(page_title="ML Prediction App", page_icon="🤖")

st.title("🤖 Batch Prediction App")
st.write("Upload a CSV, click Predict, and download the results.")

# A. UPLOAD BUTTON
uploaded_file = st.file_uploader("1. Upload your input CSV", type=["csv"])

if uploaded_file is not None:
    # Read the file to a dataframe
    input_df = pd.read_csv(uploaded_file)
    input_df= input_df.drop(['Churn'],axis=1,errors='ignore')
    st.write("Preview of Uploaded Data:")
    st.dataframe(input_df.head())

    # Initialize session state for results if it doesn't exist
    if 'result_df' not in st.session_state:
        st.session_state.result_df = None

    # B. PREDICT BUTTON
    # We use a button to trigger the model only when the user is ready
    if st.button("2. Predict Output"):
        model = load_model()
        
        try:
            # Run prediction
            result_df = predict_data(model, input_df)
            
            # Save result to session state so it persists
            st.session_state.result_df = result_df
            st.success("Prediction Complete!")
            
            # Show results
            st.write("Preview of Results:")
            st.dataframe(result_df.head())
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    # C. DOWNLOAD BUTTON
    # This shows up only if results are available in session state
    if st.session_state.result_df is not None:
        # Convert DataFrame to CSV for download
        csv_buffer = io.BytesIO()
        st.session_state.result_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="3. Download Result File",
            data=csv_data,
            file_name="predicted_results.csv",
            mime="text/csv"
        )