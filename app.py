
import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Student GPA Prediction",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------
# LOAD MODEL AND DATASET
# --------------------------------

MODEL_PATH = "model/best_student_performance_model.pkl"
DATA_PATH = "dataset/student_social_media.csv"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Please run evaluate_models.py first.")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("Dataset not found. Check your dataset folder.")
    st.stop()

model = load_model()
df = load_dataset()

# --------------------------------
# APPLICATION HEADER
# --------------------------------

st.title("🎓 Student Performance Prediction System")

st.write(
    "This Machine Learning application predicts a student's "
    "GPA based on their social media habits, study hours, "
    "sleep hours and other student characteristics."
)

st.divider()

# --------------------------------
# SIDEBAR
# --------------------------------

st.sidebar.title("About the Project")

st.sidebar.write(
    "This project uses supervised machine learning "
    "to predict university students' GPA."
)

st.sidebar.write("Algorithms evaluated:")
st.sidebar.write("- Linear Regression")
st.sidebar.write("- Random Forest Regressor")

st.sidebar.info(
    "The prediction is an estimate based on patterns "
    "in the training dataset. It is not a guaranteed GPA."
)

# --------------------------------
# INPUT FORM
# --------------------------------

st.subheader("Enter Student Information")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=10,
            max_value=100,
            value=20
        )

        gender = st.selectbox(
            "Gender",
            sorted(df["Gender"].dropna().unique())
        )

        department = st.selectbox(
            "Department",
            sorted(df["Department"].dropna().unique())
        )

        platform = st.selectbox(
            "Social Media Platform",
            sorted(df["Social Media Platform"].dropna().unique())
        )

    with col2:

        social_hours = st.number_input(
            "Daily Social Media Hours",
            min_value=0.0,
            max_value=24.0,
            value=3.0,
            step=0.5
        )

        study_hours = st.number_input(
            "Study Hours Per Day",
            min_value=0.0,
            max_value=24.0,
            value=4.0,
            step=0.5
        )

        sleep_hours = st.number_input(
            "Sleep Hours Per Day",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5
        )

        purpose = st.selectbox(
            "Social Media Purpose",
            sorted(df["Social Media Purpose"].dropna().unique())
        )

    submitted = st.form_submit_button(
        "Predict Student GPA",
        use_container_width=True
    )

# --------------------------------
# MAKE PREDICTION
# --------------------------------

if submitted:

    student_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Department": department,
        "Social Media Platform": platform,
        "Daily Social Media Hours": social_hours,
        "Study Hours Per Day": study_hours,
        "Sleep Hours Per Day": sleep_hours,
        "Social Media Purpose": purpose
    }])

    prediction = model.predict(student_data)[0]

    st.divider()

    st.subheader("Prediction Result")

    st.metric(
        label="Predicted GPA",
        value=f"{prediction:.2f}"
    )

    st.success(
        "GPA prediction completed successfully!"
    )

    st.caption(
        "This is a model estimate, not an official academic result."
    )

# --------------------------------
# PROJECT INFORMATION
# --------------------------------

st.divider()

st.subheader("About the Dataset")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))

col2.metric("Total Features", len(df.columns) - 1)

col3.metric(
    "Prediction Target",
    "GPA"
)

st.write(
    "Developed using Python, Pandas, Scikit-learn and Streamlit."
)
