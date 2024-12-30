import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_excel("wine_quality_regression.xlsx")  # Update with your dataset path

# Preprocess the data
X = data.drop(columns=["quality", "Id"])  # Drop the target and ID column
y = data["quality"]

# Split and scale the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
}

# Save the trained models to .pkl files
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)  # Train the model
    with open(f"{model_name.lower().replace(' ', '_')}_model.pkl", "wb") as file:
        pickle.dump(model, file)  # Save the model to a file

# Save the scaler
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("Models and scaler saved successfully!")
