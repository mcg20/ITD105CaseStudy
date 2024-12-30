import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Load pre-trained models and scaler
models = {
    "Random Forest": pickle.load(open("random_forest_model.pkl", "rb")),
    "Linear Regression": pickle.load(open("linear_regression_model.pkl", "rb")),
    "Decision Tree": pickle.load(open("decision_tree_model.pkl", "rb")),
}
scaler = pickle.load(open("scaler.pkl", "rb"))

# Streamlit app layout
st.set_page_config(page_title="Wine Quality Prediction", layout="wide", page_icon="🍷")

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #f3f3f3, #d7d7d7);
    }
    h1 {
        text-align: center;
        font-family: 'Arial Black', sans-serif;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar for input features
st.sidebar.header("🔢 Input Features")
fixed_acidity = st.sidebar.number_input("Fixed Acidity", value=7.4, step=0.1)
volatile_acidity = st.sidebar.number_input("Volatile Acidity", value=0.7, step=0.01)
citric_acid = st.sidebar.number_input("Citric Acid", value=0.0, step=0.01)
residual_sugar = st.sidebar.number_input("Residual Sugar", value=1.9, step=0.1)
chlorides = st.sidebar.number_input("Chlorides", value=0.076, step=0.001)
free_sulfur_dioxide = st.sidebar.number_input("Free Sulfur Dioxide", value=11.0, step=1.0)
total_sulfur_dioxide = st.sidebar.number_input("Total Sulfur Dioxide", value=34.0, step=1.0)
density = st.sidebar.number_input("Density", value=0.9978, step=0.0001)
pH = st.sidebar.number_input("pH", value=3.51, step=0.01)
sulphates = st.sidebar.number_input("Sulphates", value=0.56, step=0.01)
alcohol = st.sidebar.number_input("Alcohol", value=9.4, step=0.1)
actual_quality = st.sidebar.number_input("Actual Wine Quality (Optional):", value=5.0, step=0.1)

# Prepare input data
columns = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]
input_features = pd.DataFrame(
    [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
      chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
      density, pH, sulphates, alcohol]],
    columns=columns
)

input_scaled = scaler.transform(input_features)

# Title and SDG
st.title("🍷 Predicting Wine Quality for Sustainable Consumption Using Machine Learning Models")
st.markdown(
    """
### 🌍 Aligning with SDG Goal 12:
**"Ensure sustainable consumption and production patterns."**
"""
)

if st.sidebar.button("🔮 Predict"):

    # Get predictions from all models
    predictions = {name: round(model.predict(input_scaled)[0], 2) for name, model in models.items()}

    # Display predictions
    st.subheader("🗊 Predictions")
    for name, prediction in predictions.items():
        st.write(f"**{name}:** {prediction} ⭐")

     # Calculate MAE
    if actual_quality > 0:
        mae_scores = {name: round(abs(actual_quality - pred), 2) for name, pred in predictions.items()}
        st.subheader("📉 MAE Scores")
        for name, mae in mae_scores.items():
            st.write(f"**{name}:** {mae}")
    else:
        st.warning("Actual quality not provided. MAE scores cannot be calculated.")


    

    # MAE Chart
    valid_mae = {name: mae for name, mae in mae_scores.items() if isinstance(mae, float)}
    if valid_mae:
        fig_mae = px.bar(
            x=list(valid_mae.keys()),
            y=list(valid_mae.values()),
            labels={"x": "Model", "y": "Mean Absolute Error"},
            title="MAE Comparison",
            text_auto=True,
            color=list(valid_mae.values()),
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_mae, use_container_width=True)

    # Bar Chart of Predictions
    st.subheader("🗊 Bar Chart of Predictions")
    fig_predictions = px.bar(
        x=list(predictions.keys()),
        y=list(predictions.values()),
        labels={"x": "Model", "y": "Predicted Quality"},
        title="Predicted Quality by Model",
        text_auto=True,
        color=list(predictions.values()),
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig_predictions, use_container_width=True)
    
    # Dynamic Bar Chart for Feature Contributions Based on Input
st.subheader("📊 Bar Chart of Feature Contributions for Prediction")

if hasattr(models["Random Forest"], "feature_importances_"):
    # Extract feature importances from the Random Forest model
    feature_importances = models["Random Forest"].feature_importances_

    # Calculate weighted contributions dynamically based on input features
    weighted_contributions = {
        columns[i]: feature_importances[i] * input_features.iloc[0][columns[i]]
        for i in range(len(columns))
    }

    # Normalize the contributions to focus on relative impact
    total_contribution = sum(weighted_contributions.values())
    normalized_contributions = {
        feature: (value / total_contribution) * 100 if total_contribution > 0 else 0
        for feature, value in weighted_contributions.items()
    }

    # Create a DataFrame for the bar chart
    contribution_data = pd.DataFrame({
        "Feature": list(normalized_contributions.keys()),
        "Contribution (%)": list(normalized_contributions.values())
    }).sort_values(by="Contribution (%)", ascending=False)

    # Plot the bar chart
    fig_bar = px.bar(
        contribution_data,
        x="Contribution (%)",
        y="Feature",
        orientation="h",
        title="Feature Contributions Based on Input for Prediction",
        labels={"Contribution (%)": "Contribution (%)", "Feature": "Features"},
        text_auto=True,
        color="Contribution (%)",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.warning("Feature importances not available for the Random Forest model.")



    # Enhanced Line Chart
    if actual_quality > 0:
        st.subheader("🖭 Line Chart: Predictions vs Actual")
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=list(predictions.keys()),
            y=list(predictions.values()),
            mode="lines+markers",
            name="Predicted",
            line=dict(color="blue", width=3),
        ))
        fig_line.add_trace(go.Scatter(
            x=list(predictions.keys()),
            y=[actual_quality] * len(predictions),
            mode="lines+markers",
            name="Actual",
            line=dict(dash="dash", color="red", width=3),
        ))
        fig_line.update_layout(
            title="Predictions vs Actual Quality",
            xaxis_title="Model",
            yaxis_title="Wine Quality",
            template="plotly_white"
        )
        st.plotly_chart(fig_line, use_container_width=True)

    # Heatmap Visualization
    st.subheader("🔥 3D Heatmap Visualization")
    random_data = pd.DataFrame(
        np.random.uniform(0, 1, size=(100, len(columns))), columns=columns
    )
    random_data_scaled = scaler.transform(random_data)
    random_data["Predicted Quality"] = models["Random Forest"].predict(random_data_scaled)
    fig_heatmap = px.scatter_3d(
        random_data,
        x="alcohol",
        y="sulphates",
        z="pH",
        color="Predicted Quality",
        title="3D Heatmap of Selected Features and Predicted Quality",
        labels={"color": "Predicted Quality"},
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

   # Dynamic Pie Chart
st.subheader("📈 Dynamic Pie Chart of Feature Contributions")

if hasattr(models["Random Forest"], "feature_importances_"):
    # Extract feature importances
    feature_importances = models["Random Forest"].feature_importances_

    # Calculate weighted contributions dynamically based on input features
    weighted_contributions = {
        columns[i]: feature_importances[i] * input_features.iloc[0][columns[i]]
        for i in range(len(columns))
    }

    # Normalize the contributions to percentages (sum up to 100%)
    total_contribution = sum(weighted_contributions.values())
    normalized_contributions = {
        feature: (value / total_contribution) * 100 if total_contribution > 0 else 0
        for feature, value in weighted_contributions.items()
    }

    # Create a Pie Chart with dynamic contributions
    fig_pie = px.pie(
        names=list(normalized_contributions.keys()),
        values=list(normalized_contributions.values()),
        title="Dynamic Feature Contributions to Wine Quality",
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Dynamic Scatter Plot of Feature Contributions
    st.subheader("🔁 Dynamic Scatter Plot of Feature Contributions")
    if hasattr(models["Random Forest"], "feature_importances_"):
        feature_importances = models["Random Forest"].feature_importances_
        feature_contributions = {
            columns[i]: feature_importances[i] for i in range(len(columns))
        }
        scatter_data = pd.DataFrame({
            "Feature": list(feature_contributions.keys()),
            "Importance": list(feature_contributions.values()),
            "Input Value": [input_features.iloc[0][feature] for feature in feature_contributions.keys()],
            "Weighted Contribution": [
                feature_contributions[feature] * input_features.iloc[0][feature]
                for feature in feature_contributions.keys()
            ],
        })
        fig_scatter = px.scatter(
            scatter_data,
            x="Input Value",
            y="Weighted Contribution",
            size="Importance",
            color="Feature",
            hover_data=["Feature", "Input Value", "Importance", "Weighted Contribution"],
            labels={"x": "Input Value", "y": "Weighted Contribution"},
            title="Scatter Plot of Feature Contributions vs Input Values",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# Save Model Section
st.subheader("📂 Save Model")
model_to_save = st.selectbox("Select a Model to Save", options=models.keys())
if st.button("Save Selected Model"):
    with open(f"{model_to_save.replace(' ', '_')}_saved.pkl", "wb") as f:
        pickle.dump(models[model_to_save], f)
    st.success(f"Model '{model_to_save}' saved successfully!")
