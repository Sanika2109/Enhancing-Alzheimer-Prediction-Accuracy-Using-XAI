# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import seaborn as sns
import gradio as gr
from sklearn.inspection import PartialDependenceDisplay  # For PDP & ICE
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Optional: Initialize SHAP JS (useful in interactive notebooks)
shap.initjs()

# ---------------------------------------------------------
# 2. Load the Datasets
df_long = pd.read_csv('/content/drive/MyDrive/oasis_longitudinal.csv')
df_cross = pd.read_csv('/content/drive/MyDrive/oasis_cross-sectional.csv')

print("Longitudinal data shape:", df_long.shape)
print("Cross-sectional data shape:", df_cross.shape)

print("\nLongitudinal sample:")
print(df_long.head())
print("\nCross-sectional sample:")
print(df_cross.head())

# ---------------------------------------------------------
# 3. Preprocess & Harmonize the Datasets
# Rename cross-sectional "ID" to "Subject ID" and "Educ" to "EDUC"
df_cross.rename(columns={'ID': 'Subject ID', 'Educ': 'EDUC'}, inplace=True)

# *** Standardize Subject IDs ***
df_cross['Subject ID'] = df_cross['Subject ID'].str.split('_MR').str[0]
df_cross['Subject ID'] = df_cross['Subject ID'].str.replace("OAS1", "OAS2")

# ---------------------------------------------------------
# 4. Process the Longitudinal Data to Extract Features
df_long['Visit'] = pd.to_numeric(df_long['Visit'], errors='coerce')
df_long_sorted = df_long.sort_values(by=['Subject ID', 'Visit'])

df_long_features = df_long_sorted.groupby('Subject ID').agg(
    baseline_MMSE = ('MMSE', 'first'),
    latest_MMSE   = ('MMSE', 'last'),
    num_visits    = ('Subject ID', 'count')
).reset_index()

df_long_features['delta_MMSE'] = df_long_features['latest_MMSE'] - df_long_features['baseline_MMSE']

print("\nExtracted longitudinal features (sample):")
print(df_long_features.head())

# ---------------------------------------------------------
# 5. Merge Cross-Sectional and Longitudinal Features
df_merged = pd.merge(df_cross, df_long_features[['Subject ID', 'delta_MMSE']], on='Subject ID', how='left')
df_merged['delta_MMSE'].fillna(0, inplace=True)

if 'M/F' in df_merged.columns:
    df_merged['M/F'] = df_merged['M/F'].replace({'M': 1, 'F': 0})
    print("\nConverted M/F values to numeric (1 for Male, 0 for Female).")

print("\nMerged data sample:")
print(df_merged.head())

# ---------------------------------------------------------
# 6. Handle Missing Values Based on Domain Knowledge
for col in ['EDUC', 'SES', 'MMSE', 'CDR', 'Delay']:
    df_merged[col] = df_merged[col].fillna(df_merged[col].median())

# ---------------------------------------------------------
# 7. Define the Target and Select Features for Modeling
df_merged['target'] = df_merged['CDR'].apply(lambda x: 1 if x > 0 else 0)
selected_features = ['Age', 'M/F', 'MMSE', 'eTIV', 'nWBV', 'delta_MMSE']

X = df_merged[selected_features]
X = X.fillna(X.mean())
y = df_merged['target']

print("\nSelected features (sample):")
print(X.head())

# ---------------------------------------------------------
# 8. Train-Validation-Test Split and Cross Validation
# First, split 80% for training+validation and 20% for testing.
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Then, split training+validation into 75% training and 25% validation.
# Result: ~60% training, ~20% validation, ~20% testing.
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# ---------------------------------------------------------
# 9. Model Training using XGBoost and Cross Validation
# Initialize XGBoost model with basic parameters (adjust as needed)
model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Perform 5-fold cross validation on the training set
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
print("\nCross Validation Scores on Training Data:", cv_scores)
print("Mean Cross Validation Score:", cv_scores.mean())

# Evaluate on the validation set
y_val_pred = model.predict(X_val)
print("\nValidation Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nValidation Classification Report:")
print(classification_report(y_val, y_val_pred))

# Evaluate on the test set
y_pred = model.predict(X_test)
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nTest Classification Report:")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------
# 10. Explainability with SHAP
# XGBoost is tree-based, so we can use the TreeExplainer.
explainer = shap.TreeExplainer(model)

# ---------------------------------------------------------
# 11. Function to Generate SHAP, PDP, and ICE Explanations
def generate_explanations(age, gender, mmse, etiv, nwbv, delta_mmse):
    """
    This function computes:
      - A SHAP force plot for the given instance with an annotated title that
        clearly indicates the classification (Alzheimer's vs. Non-Alzheimer's)
        and the predicted probability.
      - A set of 1D Partial Dependence Plots (PDPs) with ICE curves for key features.
    """
    # Basic input validation
    if age <= 0:
        return "Invalid input: Age must be positive.", "Invalid input: Age must be positive."
    if mmse < 0 or mmse > 30:
        return "Invalid input: MMSE must be between 0 and 30.", "Invalid input: MMSE must be between 0 and 30."
    if etiv <= 0:
        return "Invalid input: eTIV must be positive.", "Invalid input: eTIV must be positive."
    if nwbv <= 0 or nwbv > 1:
        return "Invalid input: nWBV must be between 0 and 1.", "Invalid input: nWBV must be between 0 and 1."

    # Convert gender input ("Male"/"Female") to numeric value
    gender_val = 1 if gender == "Male" else 0

    # Create the input instance as a numpy array then convert to a DataFrame
    instance = np.array([[age, gender_val, mmse, etiv, nwbv, delta_mmse]])
    instance_df = pd.DataFrame(instance, columns=selected_features)

    # Compute the predicted probability for the positive class (Alzheimer's)
    predicted_proba = model.predict_proba(instance_df)[0][1]
    predicted_label = "Alzheimer's" if predicted_proba >= 0.5 else "Non-Alzheimer's"

    ### SHAP Explanation ###
    instance_shap_values = explainer.shap_values(instance_df)
    # For classifiers, TreeExplainer returns a list (one array per class) - use the positive class.
    if isinstance(instance_shap_values, list):
        shap_val = instance_shap_values[1].squeeze()
        expected_value = explainer.expected_value[1]
    else:
        shap_val = instance_shap_values.squeeze()
        expected_value = explainer.expected_value

    # Generate the SHAP force plot in matplotlib mode
    shap.force_plot(expected_value, shap_val, feature_names=selected_features, matplotlib=True)
    plt.title("Prediction: {} (Probability: {:.2f})".format(predicted_label, predicted_proba))
    plt.savefig("shap_plot_dynamic.png", bbox_inches="tight")
    plt.close()

    ### Partial Dependence Plots (PDP) with ICE ###
    pdp_features = ['Age', 'MMSE', 'nWBV', 'delta_MMSE']
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    PartialDependenceDisplay.from_estimator(
        model,
        X_train,
        features=pdp_features,
        ax=axs,
        kind='both'
    )
    plt.suptitle("Partial Dependence Plots with ICE Curves", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("pdp_ice_explanation.png", bbox_inches="tight")
    plt.close()
 # SHAP Bar Plot for Given Input
    plt.figure(figsize=(8, 5))
    shap.bar_plot(shap_val, feature_names=selected_features, show=False)
    plt.title("SHAP Feature Importance for Given Input")
    plt.savefig("shap_bar_dynamic.png", bbox_inches="tight")
    plt.close()

    return "shap_plot_dynamic.png", "pdp_ice_explanation.png", "shap_bar_dynamic.png"

# Gradio Interface for Explanations
iface_explanations = gr.Interface(
    fn=generate_explanations,
    inputs=[
        gr.Number(label="Age", value=70),
        gr.Radio(choices=["Male", "Female"], label="Gender", value="Male"),
        gr.Number(label="MMSE", value=25),
        gr.Number(label="eTIV", value=1500),
        gr.Number(label="nWBV", value=0.75),
        gr.Number(label="Delta MMSE", value=0.0)
    ],
    outputs=[
        gr.Image(label="SHAP Force Plot"),
        gr.Image(label="PDP with ICE Curves"),
        gr.Image(label="SHAP Feature Importance")
    ],
    title="Alzheimer's Prediction using Explainable AI techniques - SHAP, PDP, and ICE (XGB With CV)",
    description="Modify the inputs to see SHAP explanations, Partial Dependence Plots with ICE for local & global interpretations."
)

iface_explanations.launch(share=True)
