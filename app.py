
from pathlib import Path
import pickle

import pandas as pd
import streamlit as st

from src.prediction import (
    validate_input,
    predict_performance,
    get_performance_category,
)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "student_performance_model.pkl"


# --------------------------------------------------
# Load the existing trained model
# --------------------------------------------------

@st.cache_resource
def load_trained_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


# --------------------------------------------------
# Application interface
# --------------------------------------------------

st.title("🎓 Student Performance Prediction System")

st.markdown(
    """
    Estimate a student's final score using the existing
    machine-learning model and academic information.
    """
)

st.info(
    "Enter the student details below, then select Predict Performance."
)

try:
    model = load_trained_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

st.subheader("Student Information")

with st.form("student_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
        )

        age = st.number_input(
            "Age",
            min_value=10,
            max_value=30,
            value=18,
            step=1,
        )

        study_hours = st.number_input(
            "Study hours per day",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5,
        )

        attendance = st.number_input(
            "Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=90.0,
            step=1.0,
        )

        previous_score = st.number_input(
            "Previous score",
            min_value=0.0,
            max_value=100.0,
            value=78.0,
            step=1.0,
        )

    with col2:
        assignment_score = st.number_input(
            "Assignment score",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=1.0,
        )

        midterm_score = st.number_input(
            "Midterm score",
            min_value=0.0,
            max_value=100.0,
            value=82.0,
            step=1.0,
        )

        extracurricular = st.selectbox(
            "Participates in extracurricular activities?",
            ["Yes", "No"],
        )

        internet_access = st.selectbox(
            "Has internet access?",
            ["Yes", "No"],
        )

        sleep_hours = st.number_input(
            "Sleep hours per day",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5,
        )

    submitted = st.form_submit_button(
        "Predict Performance",
        type="primary",
        use_container_width=True,
    )

# --------------------------------------------------
# Generate prediction
# --------------------------------------------------

if submitted:
    student_data = pd.DataFrame(
        [
            {
                "gender": gender,
                "age": age,
                "study_hours": study_hours,
                "attendance": attendance,
                "previous_score": previous_score,
                "assignment_score": assignment_score,
                "midterm_score": midterm_score,
                "extracurricular": extracurricular,
                "internet_access": internet_access,
                "sleep_hours": sleep_hours,
            }
        ]
    )

    try:
        validate_input(student_data)

        predicted_score = predict_performance(
            model,
            student_data,
        )

        category = get_performance_category(
            predicted_score,
        )

        st.divider()
        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Predicted Final Score",
                f"{predicted_score:.2f}/100",
            )

        with result_col2:
            st.metric(
                "Performance Category",
                category,
            )

        st.progress(
            min(max(float(predicted_score) / 100, 0.0), 1.0)
        )

        st.subheader("Student Details")
        st.dataframe(
            student_data,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "This is a model-based estimate, not a guaranteed "
            "academic outcome. Predictions depend on the training "
            "data and model quality."
        )

    except ValueError as error:
        st.error(f"Please check the input: {error}")

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()
st.caption("Student Performance Prediction System | Machine Learning Major Project")
