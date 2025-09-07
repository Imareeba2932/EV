from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.secret_key = "dev-secret-key-change-in-production"  # for flash messages
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ev_predict.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# Define models directly in app.py to avoid circular imports
class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    def set_password(self, password):
        """Hash and set the password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if provided password matches the hash"""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class PredictionHistory(db.Model):
    __tablename__ = 'prediction_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    prediction_type = db.Column(db.String(50), nullable=False)  # 'form' or 'file'
    prediction_result = db.Column(db.Text, nullable=False)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('predictions', lazy=True))
    
    def __repr__(self):
        return f'<PredictionHistory {self.id}>'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize database tables
def init_db():
    with app.app_context():
        db.create_all()
        print("✅ Database tables created/verified")

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
    return render_template("index.html")
   


@app.route("/predict", methods=["GET", "POST"])
@login_required
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
@login_required
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


# Authentication Routes
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")
    
    # Handle registration
    username = request.form.get("username", "").strip()
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")
    first_name = request.form.get("first_name", "").strip()
    last_name = request.form.get("last_name", "").strip()
    
    # Validation
    errors = []
    
    if not username or len(username) < 3:
        errors.append("Username must be at least 3 characters long.")
    
    if not email or "@" not in email:
        errors.append("Please enter a valid email address.")
    
    if not password or len(password) < 6:
        errors.append("Password must be at least 6 characters long.")
    
    if password != confirm_password:
        errors.append("Passwords do not match.")
    
    if not first_name or not last_name:
        errors.append("First name and last name are required.")
    
    # Check if user already exists
    if User.query.filter_by(username=username).first():
        errors.append("Username already exists.")
    
    if User.query.filter_by(email=email).first():
        errors.append("Email already registered.")
    
    if errors:
        for error in errors:
            flash(error, "error")
        return render_template("register.html")
    
    # Create new user
    try:
        user = User(
            username=username,
            email=email,
            first_name=first_name,
            last_name=last_name
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash("Registration successful! Please log in.", "success")
        return redirect(url_for("login"))
    
    except Exception as e:
        db.session.rollback()
        flash("Registration failed. Please try again.", "error")
        return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")
    
    # Handle login
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    remember = bool(request.form.get("remember"))
    
    if not username or not password:
        flash("Please enter both username and password.", "error")
        return render_template("login.html")
    
    user = User.query.filter_by(username=username).first()
    
    if user and user.check_password(password):
        if user.is_active:
            login_user(user, remember=remember)
            flash(f"Welcome back, {user.first_name}!", "success")
            
            # Redirect to next page or dashboard
            next_page = request.args.get("next")
            return redirect(next_page) if next_page else redirect(url_for("index"))
        else:
            flash("Your account has been deactivated.", "error")
    else:
        flash("Invalid username or password.", "error")
    
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for("index"))


@app.route("/profile")
@login_required
def profile():
    # Get user's prediction history
    predictions = PredictionHistory.query.filter_by(user_id=current_user.id)\
        .order_by(PredictionHistory.created_at.desc()).limit(10).all()
    
    return render_template("profile.html", predictions=predictions)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    print("🚀 Starting EV Predictive Maintenance Flask App...")
    print("📊 Model: Random Forest (Failure Probability Prediction)")
    print("🔗 Available endpoints:")
    print("   - / : Home page")
    print("   - /register : User registration")
    print("   - /login : User login")
    print("   - /logout : User logout")
    print("   - /profile : User profile")
    print("   - /predict : File upload prediction (requires login)")
    print("   - /form : Manual input prediction (requires login)")
    print("   - /api/predict : API endpoint")
    print("   - /api/info : Model information")
    
    # Initialize database after all models are loaded
    init_db()
    
    app.run(debug=True)