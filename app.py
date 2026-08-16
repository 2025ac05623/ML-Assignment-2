
import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ------------------------------------------------
# Page configuration
# ------------------------------------------------

st.set_page_config(
    page_title="ML Classification Model Demo",
    page_icon="🤖",
    layout="wide"
)

# ------------------------------------------------
# Title
# ------------------------------------------------

st.title("ML Classification Model Demonstration")

st.write(
    "This application demonstrates five classification models "
    "trained on the UCI Breast Cancer Wisconsin dataset."
)

# ------------------------------------------------
# Model paths
# ------------------------------------------------

model_paths = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl"
}

# ------------------------------------------------
# Upload test data
# ------------------------------------------------

st.header("1. Upload Test Data")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

# ------------------------------------------------
# Model selection
# ------------------------------------------------

st.header("2. Select Classification Model")

selected_model = st.selectbox(
    "Choose a model:",
    list(model_paths.keys())
)

# ------------------------------------------------
# Process uploaded data
# ------------------------------------------------

if uploaded_file is not None:

    # Read uploaded CSV
    test_data = pd.read_csv(uploaded_file)

    st.success("Test data uploaded successfully!")

    # Display dataset information
    st.subheader("Uploaded Test Data")
    st.write(f"Rows: {test_data.shape[0]}")
    st.write(f"Columns: {test_data.shape[1]}")

    st.dataframe(test_data.head())

    # Check that target column exists
    if "Diagnosis" not in test_data.columns:

        st.error(
            "The uploaded CSV must contain a 'Diagnosis' column "
            "for evaluation."
        )

    else:

        # Separate features and target
        X_test_app = test_data.drop(columns=["Diagnosis"])
        y_test_app = test_data["Diagnosis"]

        # Load selected model
        model = joblib.load(model_paths[selected_model])

        # Make predictions
        y_pred_app = model.predict(X_test_app)

        # Get probability predictions for AUC
        y_prob_app = model.predict_proba(X_test_app)[:, 1]

        # ------------------------------------------------
        # Evaluation metrics
        # ------------------------------------------------

        st.header("3. Evaluation Metrics")

        accuracy_app = accuracy_score(y_test_app, y_pred_app)
        auc_app = roc_auc_score(y_test_app, y_prob_app)
        precision_app = precision_score(y_test_app, y_pred_app)
        recall_app = recall_score(y_test_app, y_pred_app)
        f1_app = f1_score(y_test_app, y_pred_app)
        mcc_app = matthews_corrcoef(y_test_app, y_pred_app)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Accuracy", f"{accuracy_app:.4f}")
            st.metric("Precision", f"{precision_app:.4f}")

        with col2:
            st.metric("AUC", f"{auc_app:.4f}")
            st.metric("Recall", f"{recall_app:.4f}")

        with col3:
            st.metric("F1 Score", f"{f1_app:.4f}")
            st.metric("MCC", f"{mcc_app:.4f}")

        # ------------------------------------------------
        # Confusion Matrix
        # ------------------------------------------------

        st.header("4. Confusion Matrix")

        cm = confusion_matrix(
            y_test_app,
            y_pred_app
        )

        cm_df = pd.DataFrame(
            cm,
            index=["Actual Benign", "Actual Malignant"],
            columns=["Predicted Benign", "Predicted Malignant"]
        )

        st.dataframe(cm_df)

        # ------------------------------------------------
        # Classification Report
        # ------------------------------------------------

        st.header("5. Classification Report")

        report = classification_report(
            y_test_app,
            y_pred_app,
            target_names=["Benign", "Malignant"]
        )

        st.text(report)

else:

    st.info(
        "Please upload test_data.csv to evaluate the selected model."
    )
