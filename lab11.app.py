import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib

# Load dataset
data = pd.read_excel('lab_11_bridge_data.xlsx', sheet_name='lab_11_bridge_data')

# Split into train+val and test sets (10 samples)
test_data = data.sample(n=50, random_state=42)
train_val_data = data.drop(test_data.index)

# Features and target
X_test = test_data.drop('Max_Load_Tons', axis=1)
y_test = test_data['Max_Load_Tons']

# Selected Features Setup (unchanged)
selected_features = ['Span_ft', 'Deck_Width_ft', 'Age_Years', 'Num_Lanes', 'Material', 'Condition_Rating']
target = 'Max_Load_Tons'
X_selected = train_val_data[selected_features]
y_selected = train_val_data[target]

# Select relevant features and target variable
features = ["Span_ft", "Deck_Width_ft", "Age_Years", "Num_Lanes"]
target = "Max_Load_Tons"

# Drop rows with missing values
data = data[features + [target]].dropna()

# Split data into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print(f"Model Mean Squared Error: {mse}")

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse:,.2f}Tons ")

# All Features Setup with Imputation
X_all = train_val_data.drop('Max_Load_Tons', axis=1)
y_all = train_val_data['Max_Load_Tons']

# If "Order" is the index, reset it so that it becomes a column
if X_all.index.name == 'Order' or 'Order' not in X_all.columns:
    X_all = X_all.reset_index()

# Identify numerical and categorical columns (now including "Order" if it exists)
categorical_cols = ['Material', 'Condition_Rating']
numerical_cols = ['Span_ft', 'Deck_Width_ft', 'Num_Lanes']

# Compute default values for each feature in X_all
default_values = {}
for col in X_all.columns:
    if col in numerical_cols:
        default_values[col] = X_all[col].median()
    elif col in categorical_cols:
        default_values[col] = X_all[col].mode().iloc[0]
    else:
        default_values[col] = ''  # or another appropriate default

# Convert the dictionary to a one-row DataFrame
default_all_df = pd.DataFrame([default_values])
# Reindex to ensure exact column order as in X_all
default_all_df = default_all_df.reindex(columns=X_all.columns)

# Save the complete default row to a CSV file
default_all_df.to_csv('default_all_features.csv', index=False)

# Define pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(sparse=True, handle_unknown='ignore'))  
])

# Combine pipelines
preprocessor_selected = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])

X_selected = train_val_data[numerical_cols + categorical_cols]  
y_selected = train_val_data['Max_Load_Tons']

X_selected_processed = preprocessor_selected.fit_transform(X_selected)

# Combine pipelines
preprocessor_all = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])

# Apply preprocessing to data
X_selected_processed = preprocessor_all.fit_transform(X_selected)
X_all_processed = preprocessor_all.fit_transform(X_all)

# Now split the data
X_train_selected, X_val_selected, y_train_selected, y_val_selected = train_test_split(
    X_selected_processed, y_selected, test_size=0.2, random_state=42)

X_train_all, X_val_all, y_train_all, y_val_all = train_test_split(
    X_all_processed, y_all, test_size=0.2, random_state=42)

# Create the test set for all features
X_test_all = X_all_processed[len(X_train_all) + len(X_val_all):]
y_test_all = y_all[len(y_train_all) + len(y_val_all):]

# STEP TWO: MODEL DEVELOPMENT

# Set up early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

### Modified ANN with Selected Features
model_selected = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train_selected.shape[1],),
                       kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1)
])
model_selected.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size'))

history_selected = model_selected.fit(
    X_train_selected, y_train_selected,
    validation_data=(X_val_selected, y_val_selected),
    epochs=200, batch_size=32, callbacks=[early_stop], verbose=0)

model_selected.save('model_selected.h5')

### Modified ANN with All Features with Imputation
model_all = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train_all.shape[1],),
                       kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.Dense(1)
])
model_all.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size'))

history_all = model_all.fit(
    X_train_all, y_train_all,
    validation_data=(X_val_all, y_val_all),
    epochs=300, batch_size=32, callbacks=[early_stop], verbose=0)

model_all.save('model_all.h5')

# Save the preprocessing pipelines
joblib.dump(preprocessor_selected, 'preprocessor_selected.pkl')
joblib.dump(preprocessor_all, 'preprocessor_all.pkl')

# Train linear regression on selected features
lr = LinearRegression()
lr.fit(X_selected_processed, y_selected)

# STEP THREE: EVALUATION AND COMPARISON

### Prepare Test Data
X_test_selected = preprocessor_selected.transform(test_data[selected_features])
X_test_all = preprocessor_all.transform(test_data)

# Predictions from ANN (selected features)
y_pred_selected = model_selected.predict(X_test_selected).flatten()

# Predictions from ANN (all features)
y_pred_all = model_all.predict(X_test_all).flatten()

# Predictions from linear regression
y_pred_lr = lr.predict(X_test_selected).flatten()

# Create comparison DataFrame with actual and predicted prices
results = pd.DataFrame({
    'Max_Load_Tons': y_test.values,
    'ANN (Selected)': y_pred_selected.flatten(),
    'ANN (All)': y_pred_all.flatten(),
    'Linear Regression': y_pred_lr.flatten()
})

# Calculate percentage differences
results['ANN (Selected) % Diff'] = ((results['ANN (Selected)'] - results['Max_Load_Tons']) / results['Max_Load_Tons']) * 100
results['ANN (All) % Diff'] = ((results['ANN (All)'] - results['Max_Load_Tons']) / results['Max_Load_Tons']) * 100
results['Linear Regression % Diff'] = ((results['Linear Regression'] - results['Max_Load_Tons']) / results['Max_Load_Tons']) * 100

# Calculate MAPE for each model
mape_ann_selected = results['ANN (Selected) % Diff'].abs().mean()
mape_ann_all = results['ANN (All) % Diff'].abs().mean()
mape_lr = results['Linear Regression % Diff'].abs().mean()

# Create a DataFrame for MAPE comparison
mape_table = pd.DataFrame({
    'Model': ['ANN (Selected)', 'ANN (All)', 'Linear Regression'],
    'MAPE (%)': [mape_ann_selected, mape_ann_all, mape_lr]
})

# Style the MAPE table for better visualization
styled_mape_table = mape_table.style.format({'MAPE (%)': "{:.2f}%"}) \
                                    .highlight_min(subset='MAPE (%)', color='lightgreen')

# Display the styled table
display(styled_mape_table)

# Print MAPE values and identify the best model
print(f"MAPE for ANN (Selected): {mape_ann_selected:.2f}%")
print(f"MAPE for ANN (All): {mape_ann_all:.2f}%")
print(f"MAPE for Linear Regression: {mape_lr:.2f}%")

best_model = mape_table.loc[mape_table['MAPE (%)'].idxmin(), 'Model']
best_mape = mape_table['MAPE (%)'].min()
print(f"The best model based on MAPE is: {best_model} with MAPE of {best_mape:.2f}%")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

# Load the trained models and preprocessing pipelines
model_selected = keras.models.load_model('model_selected.h5')
model_all = keras.models.load_model('model_all.h5')
preprocessor_selected = joblib.load('preprocessor_selected.pkl')
preprocessor_all = joblib.load('preprocessor_all.pkl')

# Streamlit app title
st.title("Bridge Load Prediction App")

# Define the form for user input
st.sidebar.header("Bridge Data Input")
span = st.sidebar.number_input("Span (ft)", min_value=0, value=50)
deck_width = st.sidebar.number_input("Deck Width (ft)", min_value=0, value=20)
age = st.sidebar.number_input("Bridge Age (Years)", min_value=0, value=50)
num_lanes = st.sidebar.number_input("Number of Lanes", min_value=0, value=2)
material = st.sidebar.selectbox("Material", ["Concrete", "Steel", "Wood"])
condition = st.sidebar.selectbox("Condition Rating", ["Good", "Fair", "Poor"])

# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    'Span_ft': [span],
    'Deck_Width_ft': [deck_width],
    'Age_Years': [age],
    'Num_Lanes': [num_lanes],
    'Material': [material],
    'Condition_Rating': [condition]
})

# Process the input data using the preprocessor
input_data_selected = preprocessor_selected.transform(input_data)
input_data_all = preprocessor_all.transform(input_data)

# Predict the maximum load using both models
prediction_selected = model_selected.predict(input_data_selected)
prediction_all = model_all.predict(input_data_all)

# Display predictions
st.subheader("Predicted Maximum Load")
st.write(f"Using Selected Features: {prediction_selected[0][0]:.2f} Tons")
st.write(f"Using All Features: {prediction_all[0][0]:.2f} Tons")
