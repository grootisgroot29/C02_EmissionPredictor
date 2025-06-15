

  <h1>CO₂ Emission Predictor WebApp</h1>
  <p>This is a machine learning-powered web application that predicts CO₂ emissions of vehicles based on technical specifications. It also provides <strong>explainable AI</strong> features using SHAP and LIME to interpret predictions.</p>

  <h2>Dataset</h2>
  <p>
    Dataset: <code>MY1995-2023 Fuel Consumption Ratings</code><br>
    Contains fuel consumption data and CO₂ emissions for vehicles sold in Canada.
  </p>
  <p><strong>Features used:</strong></p>
  <ul>
    <li>EngineSize_L</li>
    <li>Cylinders</li>
    <li>FuelConsCity_L100km</li>
    <li>FuelConsHwy_L100km</li>
    <li>Comb_L100km</li>
    <li>Comb_mpg</li>
    <li>FuelType</li>
  </ul>
  <p><strong>Target:</strong> CO2Emission_g_km</p>

  <h2>Model Pipeline</h2>
  <ul>
    <li>Preprocessing using <code>StandardScaler</code> and <code>OneHotEncoder</code></li>
    <li>Model: <code>RandomForestRegressor</code> with 100 estimators</li>
    <li>Train-test split: 80/20</li>
    <li>Pipeline created using scikit-learn's <code>Pipeline</code> and <code>ColumnTransformer</code></li>
    <li>Serialized using <code>pickle</code> to <code>model_pipeline.pkl</code></li>
  </ul>

  <h2>Web Interface (Streamlit)</h2>
  <ul>
    <li>Interactive UI built with <code>Streamlit</code></li>
    <li>Users can select vehicle specs (Make, Model, Engine Size, etc.)</li>
    <li>Predicts CO₂ emissions upon input</li>
    <li>Provides model explainability via SHAP and LIME</li>
    <li>Visuals generated using matplotlib</li>
  </ul>

  <h2>Explainability Features</h2>
  <ul>
    <li><strong>SHAP:</strong> Shows global feature importance using Shapley values</li>
    <li><strong>LIME:</strong> Explains individual predictions with a local surrogate model</li>
    <li><strong>PDP vs ICE:</strong> PDP shows average effects, ICE reveals individual variation</li>
  </ul>

  <h2>Deployment</h2>
  <p>The application is deployed on <strong>Render</strong> and is accessible through a live web URL.</p>

  <h2>Files Included</h2>
  <ul>
    <li><code>app.py</code> – Streamlit web application</li>
    <li><code>model_pipeline.pkl</code> – Serialized ML pipeline</li>
    <li><code>MY1995-2023-Fuel-Consumption-Ratings.csv</code> – Dataset</li>
    <li><code>X_test.csv</code> – Sample test data</li>
  </ul>

  <h2>Run Locally</h2>
  <pre><code>Step 1: Install requirements
pip install streamlit pandas scikit-learn shap lime matplotlib

Step 2: Run the app
streamlit run app.py
  </code></pre>

  <h2>Summary</h2>
  <p>This project demonstrates how to combine machine learning with explainable AI to create transparent and user-friendly predictive systems. By using SHAP, LIME, PDP, and ICE, the app offers valuable insight into model behavior and fosters trust in predictions.</p>

</body>
</html>
