import os
import sys
import subprocess


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# RUN PYTHON SCRIPT
# ============================================================

def run_script(script_name):
    """
    Run a Python script located inside the src directory.
    """

    script_path = os.path.join(
        PROJECT_ROOT,
        "src",
        script_name
    )

    if not os.path.exists(script_path):

        print(
            f"\nERROR: File not found:"
            f"\n{script_path}"
        )

        return False

    print("\n")
    print("=" * 70)
    print(f"Running: src/{script_name}")
    print("=" * 70)

    result = subprocess.run(
        [
            sys.executable,
            script_path
        ],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:

        print(
            f"\nERROR: src/{script_name} failed."
        )

        return False

    print(
        f"\nsrc/{script_name} completed successfully."
    )

    return True


# ============================================================
# DISPLAY PROJECT MENU
# ============================================================

def display_menu():

    print("\n")
    print("=" * 70)
    print("       STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("=" * 70)

    print("\nSelect an option:\n")

    print("1. Run Exploratory Data Analysis (EDA)")
    print("2. Train Machine Learning Models")
    print("3. Predict Student Performance")
    print("4. Run Complete Project")
    print("5. Exit")

    print("\n" + "=" * 70)


# ============================================================
# RUN EDA
# ============================================================

def run_eda():

    return run_script(
        "eda.py"
    )


# ============================================================
# RUN MODEL TRAINING
# ============================================================

def run_model_training():

    return run_script(
        "model_training.py"
    )


# ============================================================
# RUN PREDICTION
# ============================================================

def run_prediction():

    return run_script(
        "prediction.py"
    )


# ============================================================
# RUN COMPLETE PROJECT
# ============================================================

def run_complete_project():

    print("\n")
    print("=" * 70)
    print("STARTING COMPLETE PROJECT")
    print("=" * 70)

    # --------------------------------------------------------
    # Step 1: EDA
    # --------------------------------------------------------

    print("\n[1/3] Running EDA...")

    if not run_eda():

        print(
            "\nProject stopped because EDA failed."
        )

        return

    # --------------------------------------------------------
    # Step 2: Model Training
    # --------------------------------------------------------

    print(
        "\n[2/3] Training Machine Learning Models..."
    )

    if not run_model_training():

        print(
            "\nProject stopped because model training failed."
        )

        return

    # --------------------------------------------------------
    # Step 3: Prediction
    # --------------------------------------------------------

    print(
        "\n[3/3] Starting Prediction System..."
    )

    run_prediction()


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    while True:

        display_menu()

        choice = input(
            "Enter your choice (1-5): "
        ).strip()

        # ----------------------------------------------------
        # Option 1
        # ----------------------------------------------------

        if choice == "1":

            run_eda()

        # ----------------------------------------------------
        # Option 2
        # ----------------------------------------------------

        elif choice == "2":

            run_model_training()

        # ----------------------------------------------------
        # Option 3
        # ----------------------------------------------------

        elif choice == "3":

            run_prediction()

        # ----------------------------------------------------
        # Option 4
        # ----------------------------------------------------

        elif choice == "4":

            run_complete_project()

        # ----------------------------------------------------
        # Option 5
        # ----------------------------------------------------

        elif choice == "5":

            print(
                "\nThank you for using "
                "Student Performance Prediction System."
            )

            print("Goodbye!")

            break

        # ----------------------------------------------------
        # Invalid Option
        # ----------------------------------------------------

        else:

            print(
                "\nInvalid choice."
            )

            print(
                "Please enter a number between 1 and 5."
            )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()