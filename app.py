import streamlit as st
import pandas as pd
import pickle
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Load model pipeline
@st.cache_resource
def load_model():
    with open("model_pipeline.pkl", "rb") as f:
        return pickle.load(f)

pipeline = load_model()
model_only = pipeline.named_steps["model"]
preprocessor = pipeline.named_steps["preprocessing"]

# Load dataset for dropdown inputs
df = pd.read_csv("MY1995-2023-Fuel-Consumption-Ratings.csv").dropna()

st.title("🚗 CO₂ Emission Predictor")
st.markdown("Input vehicle details to predict CO₂ emissions (g/km).")

# User inputs
make = st.selectbox("Make", sorted(df["Make"].unique()))
filtered_models = df[df["Make"] == make]["Model"].unique()
model = st.selectbox("Model", sorted(filtered_models))

engine_size = st.number_input("Engine Size (L)", min_value=0.0, value=2.0, step=0.1)
cylinders = st.number_input("Cylinders", min_value=2, value=4, step=1)
fuel_cons_city = st.number_input("Fuel Consumption City (L/100km)", min_value=0.0, value=10.0, step=0.1)
fuel_cons_hwy = st.number_input("Fuel Consumption Hwy (L/100km)", min_value=0.0, value=7.0, step=0.1)
comb = st.number_input("Combined Fuel Consumption (L/100km)", min_value=0.0, value=8.5, step=0.1)
comb_mpg = st.number_input("Combined MPG", min_value=0, value=30, step=1)
fuel_type = st.selectbox("Fuel Type", sorted(df["FuelType"].unique()))

# Move explanation method selection outside the predict button's if statement
explanation_method = st.selectbox("Choose explanation method", ["SHAP", "LIME"])

input_data = pd.DataFrame([{
    "EngineSize_L": engine_size,
    "Cylinders": cylinders,
    "FuelConsCity_L100km": fuel_cons_city,
    "FuelConsHwy_L100km": fuel_cons_hwy,
    "Comb_L100km": comb,
    "Comb_mpg": comb_mpg,
    "FuelType": fuel_type
}])

# Try to get proper feature names from the preprocessor
try:
    feature_names = preprocessor.get_feature_names_out()
except:
    # Create more readable feature names as fallback
    numeric_cols = ["Engine Size (L)", "Cylinders", "City Fuel (L/100km)", 
                   "Highway Fuel (L/100km)", "Combined (L/100km)", "Combined MPG"]
    fuel_types = sorted(df["FuelType"].unique())
    feature_names = numeric_cols + [f"Fuel: {ft}" for ft in fuel_types]

# Create a mapping of technical names to display names if needed
display_names = {
    "EngineSize_L": "Engine Size (L)",
    "Cylinders": "Cylinders",
    "FuelConsCity_L100km": "City Fuel (L/100km)",
    "FuelConsHwy_L100km": "Highway Fuel (L/100km)",
    "Comb_L100km": "Combined (L/100km)",
    "Comb_mpg": "Combined MPG"
}

# Use session state to store prediction and transformed data
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
    st.session_state.input_transformed = None
    st.session_state.readable_feature_names = None

if st.button("🔍 Predict CO₂ Emission"):
    # Store prediction and transformed data in session state
    st.session_state.prediction = pipeline.predict(input_data)[0]
    st.session_state.input_transformed = preprocessor.transform(input_data)
    
    # Try to create readable feature names
    try:
        feature_names = preprocessor.get_feature_names_out()
        
        # Create more readable versions of the feature names
        readable_names = []
        for name in feature_names:
            if name.startswith(('onehotencoder__', 'standardscaler__')):
                # Extract the base feature name after the transformer prefix
                base_name = name.split('__', 1)[1]
                
                # Handle fuel type features that might have format like "FuelType_X"
                if base_name.startswith('FuelType_'):
                    fuel = base_name.split('_')[1]
                    readable_names.append(f"Fuel: {fuel}")
                else:
                    # Use our display mapping if available
                    readable_names.append(display_names.get(base_name, base_name))
            else:
                readable_names.append(name)
        
        st.session_state.readable_feature_names = readable_names
    except:
        # Fallback to generic names
        st.session_state.readable_feature_names = [f"Feature {i+1}" for i in range(st.session_state.input_transformed.shape[1])]

# Display prediction if available
if st.session_state.prediction is not None:
    st.success(f"Predicted CO₂ Emission: **{st.session_state.prediction:.2f} g/km**")
    
    # Check which explanation method is selected and display it
    if explanation_method == "SHAP":
        st.subheader("🔍 SHAP Feature Importance")
        
        # Create better colors for SHAP
        colors = ["#ff5a5f", "#767676", "#007a87"]
        cmap = LinearSegmentedColormap.from_list("shap", colors)
        
        # Cache SHAP explainer
        @st.cache_resource
        def get_shap_explainer(_model):
            return shap.Explainer(_model)
        
        shap_explainer = get_shap_explainer(model_only)
        shap_values = shap_explainer(st.session_state.input_transformed)
        
        # Create SHAP summary plot with proper feature names
        feature_names_to_use = st.session_state.readable_feature_names or feature_names
        
        # Create a waterfall plot for this specific prediction
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        plt.title("Impact of Each Feature on CO₂ Emission Prediction", fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Also add a bar plot for clearer feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        shap_importance = np.abs(shap_values.values).mean(0)
        idx = np.argsort(shap_importance)
        plt.barh(np.array(feature_names_to_use)[idx[-10:]], shap_importance[idx[-10:]], color=colors[2])
        plt.title("Top 10 Most Important Features", fontsize=14)
        plt.xlabel("Mean |SHAP Value|", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show a beeswarm plot for all features' impacts
        st.write("### All Features' Impact Distribution")
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.beeswarm(shap_values, max_display=10, show=False)
        plt.title("Distribution of Feature Impacts on CO₂ Emission", fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
    
    elif explanation_method == "LIME":
        st.subheader("🟢 LIME Feature Importance")
        
        try:
            # Sample and transform training data for LIME
            X_sample = df[["EngineSize_L", "Cylinders", "FuelConsCity_L100km",
                          "FuelConsHwy_L100km", "Comb_L100km", "Comb_mpg", "FuelType"]].sample(
                min(500, len(df)), random_state=42)
            
            X_sample_transformed = preprocessor.transform(X_sample)
            
            # Create prediction function for LIME
            def predict_fn(x):
                return model_only.predict(x)
            
            # Create LIME explainer with proper feature names
            feature_names_to_use = st.session_state.readable_feature_names or feature_names
            
            explainer_lime = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_sample_transformed,
                feature_names=feature_names_to_use,
                mode="regression",
                random_state=42
            )
            
            # Generate explanation
            exp = explainer_lime.explain_instance(
                data_row=st.session_state.input_transformed[0],
                predict_fn=predict_fn,
                num_features=10
            )
            
            # Get the explanation data
            lime_data = pd.DataFrame(exp.as_list(), columns=['Feature', 'Impact'])
            
            # Sort by absolute impact for visualization
            lime_data['Abs_Impact'] = lime_data['Impact'].abs()
            lime_data = lime_data.sort_values('Abs_Impact', ascending=False)
            
            # Create colors based on impact direction
            colors = ['#007a87' if x > 0 else '#ff5a5f' for x in lime_data['Impact']]
            
            # Create a more attractive horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(lime_data['Feature'], lime_data['Impact'], color=colors)
            
            # Add a vertical line at x=0
            ax.axvline(x=0, color='#767676', linestyle='-', alpha=0.3)
            
            # Add labels and title
            ax.set_xlabel('Impact on CO₂ Emission (g/km)')
            ax.set_title('LIME Explanation: Feature Impact on CO₂ Prediction', fontsize=14)
            
            # Add annotations to the end of each bar
            for i, impact in enumerate(lime_data['Impact']):
                ax.text(
                    impact + (0.5 if impact >= 0 else -0.5), 
                    i, 
                    f"{impact:.2f}", 
                    va='center'
                )
            
            # Improve aesthetics
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Display explanation details in a table
            st.write("### Detailed Feature Impact")
            st.write("Positive values increase CO₂ emissions, negative values decrease emissions.")
            
            lime_display = pd.DataFrame(exp.as_list(), columns=['Feature', 'Impact'])
            lime_display = lime_display.sort_values('Impact', ascending=False)
            
            # Style the dataframe
            def highlight_impact(val):
                color = '#00000' if val > 0 else '#ffcccb' if val < 0 else ''
                return f'background-color: {color}'
            
            styled_lime = lime_display.style.applymap(highlight_impact, subset=['Impact'])
            st.dataframe(styled_lime, use_container_width=True)
            
            # Show feature values for context
            st.write("### Your Vehicle's Feature Values")
            input_display = pd.DataFrame({
                'Feature': ['Engine Size', 'Cylinders', 'City Fuel Consumption', 
                           'Highway Fuel Consumption', 'Combined Consumption', 'Combined MPG', 'Fuel Type'],
                'Value': [engine_size, cylinders, fuel_cons_city, 
                         fuel_cons_hwy, comb, comb_mpg, fuel_type]
            })
            st.dataframe(input_display, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating LIME explanation: {str(e)}")
            st.write("Please try using SHAP explanation instead.")