import pandas as pd
# import numpy as np
import joblib
import streamlit as st

# Scikit-learn and Imbalanced-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. TRAINING + PREPROCESSING CODE
# ---------------------------------------------------------

def train_and_save_model(data_path: str = "./Expresso_churn_dataset.csv"):
    """
    Loads data, cleans it, fits a RandomForest with the
    same pipeline transformations, and saves artifacts.
    """
    # -----------------------
    # Load the dataset
    # -----------------------
    data = pd.read_csv(data_path)

    # Take the first 1000 rows (as you mentioned)
    # data = data.head(10000)

    print("Initial dataset shape:", data.shape)

    # -----------------------
    # Handle missing values
    # -----------------------
    # Drop columns with more than 50% missing values
    threshold = 0.5
    missing_percentage = data.isnull().sum() / len(data)
    data = data.loc[:, missing_percentage < threshold]

    # Separate numeric and non-numeric columns
    numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
    non_numeric_columns = data.select_dtypes(exclude=["float64", "int64"]).columns.tolist()

    # Fill numeric columns with median
    for col in numeric_columns:
        median_val = data[col].median()
        data[col] = data[col].fillna(median_val)

    # Fill non-numeric columns with mode
    for col in non_numeric_columns:
        mode_val = data[col].mode(dropna=True)
        if not mode_val.empty:
            data[col] = data[col].fillna(mode_val[0])
        else:
            data[col] = data[col].fillna("Unknown")

    # Remove duplicates
    data.drop_duplicates(inplace=True)
    print("Dataset shape after cleaning & dropping duplicates:", data.shape)

    # -----------------------
    # Define target
    # -----------------------
    target_column = "CHURN"
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    # -----------------------
    # Separate X and y
    # -----------------------
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # -----------------------
    # Identify high and low-cardinality columns
    # -----------------------
    cardinality_threshold = 50
    high_cardinality_cols = [
        col for col in non_numeric_columns
        if col in X.columns and X[col].nunique() > cardinality_threshold
    ]
    low_cardinality_cols = [
        col for col in non_numeric_columns
        if col in X.columns and X[col].nunique() <= cardinality_threshold
    ]

    # Update numeric_columns
    numeric_columns = [col for col in numeric_columns if col in X.columns]

    # Debugging: Check columns
    print("Columns in X:", X.columns)
    print("Numeric columns:", numeric_columns)
    print("Low-cardinality columns:", low_cardinality_cols)
    print("High-cardinality columns:", high_cardinality_cols)

    # -----------------------
    # Define ColumnTransformer
    # -----------------------
    numeric_transformer = Pipeline([
        ("scaler", RobustScaler())
    ])

    low_card_transformer = Pipeline([
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    high_card_transformer = Pipeline([
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_columns),
        ("low_card", low_card_transformer, low_cardinality_cols),
        ("high_card", high_card_transformer, high_cardinality_cols),
    ], remainder="drop")

    # -----------------------
    # Split data
    # -----------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=y)

    # Apply transformations
    X_train_processed = preprocessor.fit_transform(X_train)

    # Balance the target variable using RandomOverSampler
    oversampler = RandomOverSampler(random_state=42)
    X_train_bal, y_train_bal = oversampler.fit_resample(X_train_processed, y_train)

    # -----------------------
    # Train Random Forest Classifier
    # -----------------------
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train_bal, y_train_bal)

    # Transform X_test
    X_test_processed = preprocessor.transform(X_test)

    # -----------------------
    # Evaluate model
    # -----------------------
    y_pred = clf.predict(X_test_processed)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # -----------------------
    # Save preprocessor and model
    # -----------------------
    joblib.dump(preprocessor, "preprocessor.pkl")
    joblib.dump(clf, "churn_model.pkl")
    print("Saved 'preprocessor.pkl' and 'churn_model.pkl'.")

# ---------------------------------------------------------
# 2. STREAMLIT APP FOR INFERENCE
# ---------------------------------------------------------

def run_streamlit_app():
    st.title("Expresso Churn Prediction")

    # Load saved artifacts
    preprocessor = joblib.load("preprocessor.pkl")
    classifier = joblib.load("churn_model.pkl")

    # Retrieve column lists
    transformers = preprocessor.transformers_
    numeric_columns = transformers[0][2]
    low_cardinality_cols = transformers[1][2]
    high_cardinality_cols = transformers[2][2]

    # Input fields
    input_data = {}
    for col in numeric_columns:
        input_data[col] = st.number_input(f"Enter numeric value for {col}", value=0.0)

    for col in low_cardinality_cols:
        input_data[col] = st.text_input(f"Enter category for {col}", value="")

    for col in high_cardinality_cols:
        input_data[col] = st.text_input(f"Enter category for {col}", value="")

    # Predict button
    if st.button("Predict"):
        try:
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data])

            # Transform using the preprocessor
            X_input_processed = preprocessor.transform(input_df)

            # Predict churn probability
            probabilities = classifier.predict_proba(X_input_processed)[0]
            churn_probability = probabilities[1]  # Probability of "Yes" churn

            # Display prediction as a pie chart
            labels = ['No Churn', 'Churn']
            sizes = [1 - churn_probability, churn_probability]
            colors = ['#4CAF50', '#FF5722']
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
            st.pyplot(fig)

            # Debugging probabilities
            st.write(f"Churn Probability: {churn_probability:.2f}")

        except Exception as e:
            st.error(f"Error: {e}")


# ---------------------------------------------------------
# 3. MAIN ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    # Uncomment the next line to retrain the model
    train_and_save_model()

    # Run Streamlit app
    run_streamlit_app()
