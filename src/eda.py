import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = "data/student_performance.csv"
OUTPUT_DIR = "outputs/graphs"


# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

def load_data():
    """
    Load student performance dataset.
    """
    try:
        df = pd.read_csv(DATA_PATH)
        print("Dataset loaded successfully.")
        print(f"Dataset shape: {df.shape}")
        return df

    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATA_PATH}")
        return None


# --------------------------------------------------
# Final Score Distribution
# --------------------------------------------------

def final_score_distribution(df):
    """
    Plot distribution of final student scores.
    """

    plt.figure(figsize=(10, 6))

    sns.histplot(
        df["final_score"],
        bins=20,
        kde=True
    )

    plt.title("Distribution of Final Student Scores")
    plt.xlabel("Final Score")
    plt.ylabel("Number of Students")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "final_score_distribution.png"
        )
    )

    plt.show()
    plt.close()


# --------------------------------------------------
# Study Hours vs Final Score
# --------------------------------------------------

def study_hours_vs_score(df):
    """
    Analyze relationship between study hours
    and final score.
    """

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x="study_hours",
        y="final_score",
        hue="gender"
    )

    plt.title("Study Hours vs Final Score")
    plt.xlabel("Study Hours")
    plt.ylabel("Final Score")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "study_hours_vs_score.png"
        )
    )

    plt.show()
    plt.close()


# --------------------------------------------------
# Attendance vs Final Score
# --------------------------------------------------

def attendance_vs_score(df):
    """
    Analyze relationship between attendance
    and final score.
    """

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x="attendance",
        y="final_score"
    )

    plt.title("Attendance vs Final Score")
    plt.xlabel("Attendance (%)")
    plt.ylabel("Final Score")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "attendance_vs_score.png"
        )
    )

    plt.show()
    plt.close()


# --------------------------------------------------
# Previous Score vs Final Score
# --------------------------------------------------

def previous_score_vs_final_score(df):
    """
    Analyze relationship between previous score
    and final score.
    """

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x="previous_score",
        y="final_score"
    )

    plt.title("Previous Score vs Final Score")
    plt.xlabel("Previous Score")
    plt.ylabel("Final Score")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "previous_score_vs_final_score.png"
        )
    )

    plt.show()
    plt.close()


# --------------------------------------------------
# Correlation Heatmap
# --------------------------------------------------

def correlation_heatmap(df):
    """
    Generate correlation heatmap for numerical variables.
    """

    numerical_data = df.select_dtypes(
        include=["int64", "float64"]
    )

    correlation = numerical_data.corr()

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "correlation_heatmap.png"
        )
    )

    plt.show()
    plt.close()


# --------------------------------------------------
# Main EDA Function
# --------------------------------------------------

def run_eda():
    """
    Run complete exploratory data analysis.
    """

    df = load_data()

    if df is None:
        return

    print("\nGenerating EDA visualizations...")

    final_score_distribution(df)

    study_hours_vs_score(df)

    attendance_vs_score(df)

    previous_score_vs_final_score(df)

    correlation_heatmap(df)

    print("\nEDA completed successfully.")
    print(f"Graphs saved in: {OUTPUT_DIR}")


# --------------------------------------------------
# Program Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    run_eda()