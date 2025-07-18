import streamlit as st
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

# Load XGBoost model and label encoder
model = xgb.Booster()
model.load_model("steel_model.json")

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Streamlit page config
st.set_page_config(page_title="Steel Fault Type Prediction", layout="centered")
st.title("🛠️ Steel Plate Fault Type Prediction")

# Feature columns used by the model
model_features = [
    "X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas",
    "X_Perimeter", "Y_Perimeter", "Sum_of_Luminosity", "Minimum_of_Luminosity", "Maximum_of_Luminosity",
    "Length_of_Conveyer", "TypeOfSteel_A300", "TypeOfSteel_A400", "Steel_Plate_Thickness", "Edges_Index",
    "Empty_Index", "Square_Index", "Outside_X_Index", "Edges_X_Index", "Edges_Y_Index",
    "Outside_Global_Index", "LogOfAreas", "Log_X_Index", "Log_Y_Index", "Orientation_Index",
    "Luminosity_Index", "SigmoidOfAreas"
]

# Default input values
default_values = {
    "X_Minimum": 42.0, "X_Maximum": 50.0, "Y_Minimum": 270900.0, "Y_Maximum": 270944.0,
    "Pixels_Areas": 267.0, "X_Perimeter": 17.0, "Y_Perimeter": 44.0, "Sum_of_Luminosity": 24220.0,
    "Minimum_of_Luminosity": 76.0, "Maximum_of_Luminosity": 108.0, "Length_of_Conveyer": 1687.0,
    "TypeOfSteel_A300": 1, "TypeOfSteel_A400": 0, "Steel_Plate_Thickness": 80.0, "Edges_Index": 0.0498,
    "Empty_Index": 0.2415, "Square_Index": 0.1818, "Outside_X_Index": 0.0047, "Edges_X_Index": 0.4706,
    "Edges_Y_Index": 1.0, "Outside_Global_Index": 1.0, "LogOfAreas": 2.4265, "Log_X_Index": 0.9031,
    "Log_Y_Index": 1.6435, "Orientation_Index": 0.8182, "Luminosity_Index": -0.2913, "SigmoidOfAreas": 0.5822
}

# Single input prediction form
with st.form("input_form"):
    st.subheader("🔧 Input Steel Plate Features")
    inputs = {}
    for name in model_features:
        if name.startswith("TypeOfSteel"):
            inputs[name] = st.selectbox(name, [0, 1], index=int(default_values[name]))
        else:
            inputs[name] = st.number_input(name, value=default_values[name], format="%.4f")

    submitted = st.form_submit_button("Predict")

# Handle single prediction
if submitted:
    input_df = pd.DataFrame([inputs], columns=model_features)
    dmatrix_input = xgb.DMatrix(input_df, feature_names=model_features)
    output = model.predict(dmatrix_input)
    predicted_index = int(np.argmax(output[0]))
    predicted_label = le.inverse_transform([predicted_index])[0]
    st.success(f"Predicted Fault Type: **{predicted_label}**")

# Separator
st.markdown("---")
st.header("Batch Prediction from CSV File")

# File uploader
uploaded_file = st.file_uploader("Upload CSV File with Feature Columns", type=["csv"])

if uploaded_file is not None:
    try:
        # Read uploaded CSV
        batch_df = pd.read_csv(uploaded_file)

        # Check for required columns
        missing_cols = [col for col in model_features if col not in batch_df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
        else:
            # Ensure correct column order
            batch_df = batch_df[model_features]

            # Create DMatrix
            dmatrix_batch = xgb.DMatrix(batch_df, feature_names=model_features)

            # Predict
            batch_preds = model.predict(dmatrix_batch)
            predicted_indices = np.argmax(batch_preds, axis=1)
            predicted_labels = le.inverse_transform(predicted_indices)

            # Add predictions
            result_df = batch_df.copy()
            result_df["Predicted_Fault_Type"] = predicted_labels

            # Show results
            st.success("Batch Prediction Complete")
            st.dataframe(result_df)

            # Downloadable results
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="steel_fault_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")
