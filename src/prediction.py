import os
import pickle
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/student_performance_model.pkl"


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():
    """
    Load the trained machine learning model.
    """

    if not os.path.exists(MODEL_PATH):

        print(
            f"ERROR: Model file not found at: {MODEL_PATH}"
        )

        print(
            "\nPlease run model_training.py first."
        )

        return None

    try:

        with open(
            MODEL_PATH,
            "rb"
        ) as file:

            model = pickle.load(file)

        print(
            "Trained model loaded successfully."
        )

        return model

    except Exception as error:

        print(
            f"ERROR while loading model: {error}"
        )

        return None


# ============================================================
# GET STUDENT INPUT
# ============================================================

def get_student_input():
    """
    Collect new student information from the user.
    """

    print("\n")
    print("=" * 60)
    print("ENTER NEW STUDENT INFORMATION")
    print("=" * 60)

    # Gender
    gender = input(
        "\nGender (Male/Female): "
    ).strip()

    # Age
    age = int(
        input(
            "Age: "
        )
    )

    # Study hours
    study_hours = float(
        input(
            "Study Hours per Day: "
        )
    )

    # Attendance
    attendance = float(
        input(
            "Attendance Percentage: "
        )
    )

    # Previous score
    previous_score = float(
        input(
            "Previous Score: "
        )
    )

    # Assignment score
    assignment_score = float(
        input(
            "Assignment Score: "
        )
    )

    # Midterm score
    midterm_score = float(
        input(
            "Midterm Score: "
        )
    )

    # Extracurricular activities
    extracurricular = input(
        "Extracurricular Activities (Yes/No): "
    ).strip()

    # Internet access
    internet_access = input(
        "Internet Access (Yes/No): "
    ).strip()

    # Sleep hours
    sleep_hours = float(
        input(
            "Sleep Hours per Day: "
        )
    )

    # Create DataFrame
    student_data = pd.DataFrame([
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
            "sleep_hours": sleep_hours
        }
    ])

    return student_data


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_input(student_data):
    """
    Validate student input values.
    """

    # Age validation
    if not (
        10 <= student_data["age"].iloc[0] <= 30
    ):

        raise ValueError(
            "Age must be between 10 and 30."
        )

    # Study hours validation
    if not (
        0 <= student_data["study_hours"].iloc[0] <= 24
    ):

        raise ValueError(
            "Study hours must be between 0 and 24."
        )

    # Attendance validation
    if not (
        0 <= student_data["attendance"].iloc[0] <= 100
    ):

        raise ValueError(
            "Attendance must be between 0 and 100."
        )

    # Score validation
    score_columns = [
        "previous_score",
        "assignment_score",
        "midterm_score"
    ]

    for column in score_columns:

        value = student_data[column].iloc[0]

        if not (
            0 <= value <= 100
        ):

            raise ValueError(
                f"{column} must be between 0 and 100."
            )

    # Sleep validation
    if not (
        0 <= student_data["sleep_hours"].iloc[0] <= 24
    ):

        raise ValueError(
            "Sleep hours must be between 0 and 24."
        )

    # Gender validation
    gender = student_data["gender"].iloc[0].lower()

    if gender not in [
        "male",
        "female"
    ]:

        raise ValueError(
            "Gender must be Male or Female."
        )

    # Yes/No validation
    extracurricular = (
        student_data[
            "extracurricular"
        ].iloc[0].lower()
    )

    internet_access = (
        student_data[
            "internet_access"
        ].iloc[0].lower()
    )

    if extracurricular not in [
        "yes",
        "no"
    ]:

        raise ValueError(
            "Extracurricular must be Yes or No."
        )

    if internet_access not in [
        "yes",
        "no"
    ]:

        raise ValueError(
            "Internet Access must be Yes or No."
        )

    return True


# ============================================================
# PREDICT PERFORMANCE
# ============================================================

def predict_performance(
    model,
    student_data
):
    """
    Predict final score for a new student.
    """

    predicted_score = model.predict(
        student_data
    )[0]

    # Keep prediction within 0-100
    predicted_score = max(
        0,
        min(
            100,
            predicted_score
        )
    )

    return predicted_score


# ============================================================
# PERFORMANCE CATEGORY
# ============================================================

def get_performance_category(score):
    """
    Categorize predicted student performance.
    """

    if score >= 90:

        return "Excellent"

    elif score >= 80:

        return "Very Good"

    elif score >= 70:

        return "Good"

    elif score >= 60:

        return "Average"

    elif score >= 50:

        return "Below Average"

    else:

        return "Needs Improvement"


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(
    student_data,
    predicted_score
):
    """
    Display prediction result.
    """

    category = get_performance_category(
        predicted_score
    )

    print("\n")
    print("=" * 60)
    print("STUDENT PERFORMANCE PREDICTION RESULT")
    print("=" * 60)

    print(
        f"\nPredicted Final Score: "
        f"{predicted_score:.2f}/100"
    )

    print(
        f"Performance Category: "
        f"{category}"
    )

    print("\nStudent Details:")
    print("-" * 60)

    for column in student_data.columns:

        print(
            f"{column.replace('_', ' ').title()}: "
            f"{student_data[column].iloc[0]}"
        )

    print("=" * 60)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("=" * 60)

    # Load model
    model = load_model()

    if model is None:
        return

    try:

        # Get student information
        student_data = get_student_input()

        # Validate input
        validate_input(
            student_data
        )

        # Predict
        predicted_score = predict_performance(
            model,
            student_data
        )

        # Display result
        display_result(
            student_data,
            predicted_score
        )

    except ValueError as error:

        print(
            f"\nInput Error: {error}"
        )

    except Exception as error:

        print(
            f"\nUnexpected Error: {error}"
        )
        
# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()