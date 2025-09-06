from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pandas as pd
import numpy as np
import joblib
import os


app = Flask(__name__)
app.secret_key = "dev-secret-key"  # for flash messages


# Load Random Forest model and training metadata once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_models", "best_model_random_forest.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "saved_models", "scaler.joblib")
FEATURE_NAMES_PATH = os.path.join(os.path.dirname(__file__), "saved_models", "feature_names.joblib")
DATA_PATH = os.path.join(os.path.dirname(__file__), "EV_Predictive_Maintenance_Dataset_15min.csv")

# Load the Random Forest model
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Random Forest model loaded successfully!")
except Exception as e:
    model = None
    print(f"❌ Failed to load Random Forest model: {e}")

# Load the scaler
try:
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler loaded successfully!")
except Exception as e:
    scaler = None
    print(f"❌ Failed to load scaler: {e}")

# Load feature names
try:
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    print("✅ Feature names loaded successfully!")
except Exception as e:
    feature_names = None
    print(f"❌ Failed to load feature names: {e}")

# Load base dataset for reference
try:
    base_df = pd.read_csv(DATA_PATH)
    print("✅ Base dataset loaded successfully!")
except Exception as e:
    base_df = None
    print(f"❌ Failed to load base dataset: {e}")


def preprocess_input(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Simple preprocessing function for Random Forest model
    """
    df = df_in.copy()
    
    # Remove target columns if they exist
    target_columns = ['Failure_Probability', 'Maintenance_Type', 'RUL', 'Component_Health_Score', 'TTF']
    for col in target_columns:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Remove timestamp column as it's not used in training
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])
    
    # Ensure we have the same features as training data
    if feature_names is not None:
        # Add missing features with median values from training data
        for feature in feature_names:
            if feature not in df.columns:
                if base_df is not None and feature in base_df.columns:
                    df[feature] = base_df[feature].median()
                else:
                    df[feature] = 0
        
        # Keep only the features used in training
        df = df[feature_names]
    
    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df


@app.route("/")
def index():
    # Use existing index.html if present; otherwise fall back to predict page
    try:
        return render_template("index.html")
    except Exception:
        return redirect(url_for("predict"))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", table_html=None, metrics=None)

    # POST: handle file upload and run predictions
    if model is None or scaler is None:
        flash("Model or scaler not available. Please check server logs.")
        return redirect(url_for("predict"))

    file = request.files.get("file")
    if not file or file.filename == "":
        flash("Please select a CSV file to upload.")
        return redirect(url_for("predict"))

    try:
        input_df = pd.read_csv(file)
    except Exception as e:
        flash(f"Failed to read CSV: {e}")
        return redirect(url_for("predict"))

    # Keep a copy for display
    display_df = input_df.copy()

    # Preprocess the data
    X = preprocess_input(input_df)
    
    # Scale the features using the same scaler used in training
    X_scaled = scaler.transform(X)

    # Make predictions
    try:
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]  # Probability of failure
    except Exception as e:
        flash(f"Prediction failed: {e}")
        return redirect(url_for("predict"))

    # Attach results for preview
    result_df = display_df.copy()
    result_df["Prediction"] = y_pred
    result_df["Failure_Probability_Pred"] = y_proba

    # Simple metrics summary
    metrics = {
        "rows": int(len(result_df)),
        "failures_predicted": int((y_pred == 1).sum()),
        "avg_failure_probability": float(np.mean(y_proba)) if len(y_proba) else 0.0,
        "model_used": "Random Forest"
    }

    # Show only first 50 rows to keep UI light
    preview = result_df.head(50)
    table_html = preview.to_html(classes="table table-striped table-sm", index=False)

    return render_template("predict.html", table_html=table_html, metrics=metrics)


@app.route("/form", methods=["GET", "POST"])
def form_predict():
    if model is None or scaler is None or base_df is None:
        flash("Model, scaler, or base dataset not available. Please check server logs.")
        return redirect(url_for("index"))

    # Get important features for the form (top 10 most important features)
    if feature_names is not None:
        form_features = feature_names[:10]  # Use first 10 features for simplicity
    else:
        form_features = ['SoC', 'SoH', 'Battery_Voltage', 'Battery_Current', 'Battery_Temperature', 
                        'Motor_Temperature', 'Motor_Vibration', 'Motor_Torque', 'Motor_RPM', 'Power_Consumption']

    # Build form schema
    dtypes = {}
    defaults = {}
    for col in form_features:
        if col in base_df.columns:
            dtypes[col] = "number"
            defaults[col] = float(base_df[col].median())
        else:
            dtypes[col] = "number"
            defaults[col] = 0.0

    if request.method == "GET":
        return render_template("form_predict.html", feature_types=dtypes, defaults=defaults, result=None)

    # POST: read form values and make prediction
    row = {}
    for feat in form_features:
        val = request.form.get(feat, "")
        try:
            row[feat] = float(val) if val else defaults[feat]
        except ValueError:
            row[feat] = defaults[feat]

    input_df = pd.DataFrame([row])
    X = preprocess_input(input_df)
    X_scaled = scaler.transform(X)

    try:
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]
        
        result = {
            "prediction": int(y_pred[0]),
            "probability": float(y_proba[0]),
            "prediction_text": "FAILURE LIKELY" if y_pred[0] == 1 else "NO FAILURE",
            "confidence": "HIGH" if y_proba[0] > 0.8 or y_proba[0] < 0.2 else "MEDIUM"
        }
    except Exception as e:
        flash(f"Prediction failed: {e}")
        return redirect(url_for("form_predict"))

    return render_template("form_predict.html", feature_types=dtypes, defaults=defaults, result=result)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Simple API endpoint for predictions
    Expects JSON with feature values
    """
    if model is None or scaler is None:
        return jsonify({"error": "Model or scaler not available"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        X = preprocess_input(input_df)
        X_scaled = scaler.transform(X)
        
        # Make prediction
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]
        
        result = {
            "prediction": int(y_pred[0]),
            "failure_probability": float(y_proba[0]),
            "prediction_text": "FAILURE LIKELY" if y_pred[0] == 1 else "NO FAILURE",
            "confidence": "HIGH" if y_proba[0] > 0.8 or y_proba[0] < 0.2 else "MEDIUM",
            "model": "Random Forest"
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/info")
def api_info():
    """
    API endpoint to get model information
    """
    info = {
        "model_type": "Random Forest Classifier",
        "target": "Failure_Probability (Binary Classification)",
        "accuracy": "81.67%",
        "features_count": len(feature_names) if feature_names else 0,
        "status": "Ready" if model is not None else "Not Loaded"
    }
    return jsonify(info)


if __name__ == "__main__":
    print("🚀 Starting EV Predictive Maintenance Flask App...")
    print("📊 Model: Random Forest (Failure Probability Prediction)")
    print("🔗 Available endpoints:")
    print("   - / : Home page")
    print("   - /predict : File upload prediction")
    print("   - /form : Manual input prediction")
    print("   - /api/predict : API endpoint")
    print("   - /api/info : Model information")
    app.run(debug=True)