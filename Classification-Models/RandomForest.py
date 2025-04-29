# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import seaborn as sns
import gradio as gr
from sklearn.inspection import PartialDependenceDisplay  # For PDP & ICE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

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
# For example, if cross-sectional IDs look like "OAS1_0001_MR1" and longitudinal IDs look like "OAS2_0001",
# first remove the "_MR..." suffix from cross-sectional IDs:
df_cross['Subject ID'] = df_cross['Subject ID'].str.split('_MR').str[0]
# Then replace "OAS1" with "OAS2" to match the longitudinal IDs:
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
# For subjects without matching longitudinal data, fill missing delta_MMSE with 0
df_merged['delta_MMSE'].fillna(0, inplace=True)

# Convert Gender column to numeric; here we assume the column is named "M/F"
if 'M/F' in df_merged.columns:
    df_merged['M/F'] = df_merged['M/F'].replace({'M': 1, 'F': 0})
    print("\nConverted M/F values to numeric (1 for Male, 0 for Female).")

print("\nMerged data sample:")
print(df_merged.head())

# ---------------------------------------------------------
# 6. Handle Missing Values Based on Domain Knowledge
# The columns that typically have missing values are: EDUC, SES, MMSE, CDR, Delay.
# We'll use the median for these numeric columns.
for col in ['EDUC', 'SES', 'MMSE', 'CDR', 'Delay']:
    df_merged[col] = df_merged[col].fillna(df_merged[col].median())

# ---------------------------------------------------------
# 7. Define the Target and Select Features for Modeling
# Here, target is set based on CDR (if CDR > 0, target = 1; else 0)
df_merged['target'] = df_merged['CDR'].apply(lambda x: 1 if x > 0 else 0)
selected_features = ['Age', 'M/F', 'MMSE', 'eTIV', 'nWBV', 'delta_MMSE']

X = df_merged[selected_features]
# If there are any remaining missing values, fill them with the column mean
X = X.fillna(X.mean())
y = df_merged['target']

print("\nSelected features (sample):")
print(X.head())

# ---------------------------------------------------------
# 8. Train-Validation-Test Split and Model Training
# Split data into training (60%), validation (20%), and test (20%) sets.
# First, split into training+validation (80%) and test (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Then, split training+validation into training (60%) and validation (20%)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

print("Training set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Test set shape:", X_test.shape)

# Define hyperparameter grid for RandomForestClassifier
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

# Use GridSearchCV with 5-fold cross validation on the training set
grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                           param_grid,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score on training set:", grid_search.best_score_)

# Set the model to the best estimator from GridSearchCV
best_model = grid_search.best_estimator_

# Evaluate on the validation set
y_val_pred = best_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation set Accuracy:", val_accuracy)
print("\nValidation Set Classification Report:")
print(classification_report(y_val, y_val_pred))

# Evaluate on the test set
y_test_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test set Accuracy:", test_accuracy)
print("\nTest Set Classification Report:")
print(classification_report(y_test, y_test_pred))

# For subsequent explainability, use the best model
model = best_model

# ---------------------------------------------------------
# 9. Explainability with SHAP
explainer = shap.TreeExplainer(model)

# ---------------------------------------------------------
# 10. Function to Generate SHAP, PDP, and ICE Explanations
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
    # Determine predicted label using threshold of 0.5
    predicted_label = "Alzheimer's" if predicted_proba >= 0.5 else "Non-Alzheimer's"

    ### SHAP Explanation ###
    instance_shap_values = explainer.shap_values(instance_df)
    # For classifiers, SHAP returns a list of arrays (one per class)
    if isinstance(instance_shap_values, list):
        # Use the SHAP values corresponding to the positive class
        shap_val = instance_shap_values[1].squeeze()[:, 1]
        expected_value = explainer.expected_value[1]
    else:
        shap_val = instance_shap_values.squeeze()[:, 1]
        expected_value = explainer.expected_value[1]

    # Generate the SHAP force plot in matplotlib mode
    shap.force_plot(expected_value, shap_val, feature_names=selected_features,  matplotlib=True)
    # Annotate the plot with a title indicating the prediction
    plt.title("Prediction: {} (Probability: {:.2f})".format(predicted_label, predicted_proba))
    plt.savefig("shap_plot_dynamic.png", bbox_inches="tight")
    plt.close()

    ### Partial Dependence Plots (PDP) with ICE ###
    pdp_features = ['Age', 'MMSE', 'nWBV', 'delta_MMSE']
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    PartialDependenceDisplay.from_estimator(
        model,
        X_train,  # using training data for PDP generation
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
    title="Alzheimer's Prediction using Explainable AI techniques - SHAP, PDP, and ICE (RF Classifier with CV)",
    description="Modify the inputs to see SHAP explanations, Partial Dependence Plots with ICE for local & global interpretations."
)

iface_explanations.launch(share=True)
