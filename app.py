import os
import datetime as dt
import uuid
import sys
import json
import random
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for, flash, session,
    jsonify
)
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from urllib.parse import unquote
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import jwt

from models.smartbite_models import SmartBiteModels


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# ------------------- Config -------------------
APP_PORT = int(os.getenv("SMARTBITE_PORT", 5000))
DEBUG = os.getenv("SMARTBITE_DEBUG", "True").lower() in ("1", "true", "yes")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "auto_diet_db")

JWT_SECRET = os.getenv("SMARTBITE_JWT_SECRET", "dev-secret")
JWT_ALG = "HS256"

DATASET_DIR = os.getenv("DATASET_DIR", "dataset")


# ------------------- App Init -------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.getenv("SMARTBITE_SECRET_KEY", "supersecret")
CORS(app)

import mongomock

try:
    # Try to connect to real MongoDB
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    # Force a connection check
    client.admin.command('ping')
    db = client[DB_NAME]
    print(f"Connected to MongoDB at {MONGO_URI}")
except Exception as mongo_err:
    print(f"[WARN] Failed to connect to MongoDB ({mongo_err}). Falling back to in-memory mongomock database.")
    client = mongomock.MongoClient()
    db = client[DB_NAME]

# collections
users_collection = db["users"]
diet_collection = db["diet_plans"]
grocery_collection = db["grocery_lists"]
health_data = db["health_data"]
grocery_data = db["grocery_data"]
meal_planner = db["meal_planner"]
activity_log = db["activity_log"]
store_config = db["store_config"]
price_history = db["price_history"]
cart_sessions = db["cart_sessions"]
recipe_reviews = db["recipe_reviews"]

# Emergency Services Collections
pending_users = db["pending_users"]
reports_collection = db["reports"]
messages_collection = db["messages"]
activity_audit = db["activity_audit"]
departments_collection = db["departments"]

# Initialize SmartBite Models
smartbite_models = SmartBiteModels()

print("Initializing SmartBite AI Models...")

try:
    # Try to load pre-trained models first
    if smartbite_models.load_all_models():
        print("Loaded pre-trained SmartBite models")
    else:
        # If no pre-trained models, initialize and train new ones
        print("No pre-trained models found, initializing new models...")
        smartbite_models.initialize_all_models()
        print("SmartBite models initialized and trained")
except Exception as e:
    print(f"Warning: SmartBite models initialization failed: {e}")
    print(
        "Note: Continuing without AI models - "
        "basic functionality will still work"
    )

# ------------------- JWT helpers -------------------


def make_jwt(payload, exp_minutes=60 * 24 * 7):
    payload = dict(payload)
    payload["exp"] = dt.datetime.utcnow() + dt.timedelta(minutes=exp_minutes)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def decode_jwt(token):
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "username" in session:
            return f(*args, **kwargs)
        auth = request.headers.get("Authorization")
        if auth and auth.startswith("Bearer "):
            token = auth.split(" ", 1)[1]
            try:
                payload = decode_jwt(token)
                request.jwt = payload
                return f(*args, **kwargs)
            except Exception:
                return jsonify({"error": "Invalid token"}), 401
        return redirect(url_for("login"))

    return wrapper


FALLBACK_GROCERY_PRODUCTS = [
    {
        "name": "Yellow Lentils (Moong Dal)",
        "category": "Grains & Pulses",
        "price": 40,
        "originalPrice": 45,
        "rating": 4.5,
        "discount": 11,
        "description": "High-protein unpolished yellow lentils, a staple for a healthy Indian diet.",
        "nutritional_benefits": [
            "Excellent source of plant protein",
            "High in dietary fiber",
            "Easy to digest",
        ],
    },
    {
        "name": "Whole Wheat Atta",
        "category": "Grains & Pulses",
        "price": 45,
        "originalPrice": 50,
        "rating": 4.3,
        "discount": 10,
        "description": "100% whole wheat flour, perfect for making soft rotis and chapatis.",
        "nutritional_benefits": [
            "Rich in dietary fiber",
            "Provides sustained energy",
            "Good source of B-vitamins",
        ],
    },
    {
        "name": "Local Curd (Dahi)",
        "category": "Dairy & Eggs",
        "price": 30,
        "originalPrice": 30,
        "rating": 4.7,
        "discount": 0,
        "description": "Freshly set local curd, high in calcium and natural probiotics.",
        "nutritional_benefits": [
            "Good for gut digestion",
            "Rich source of calcium",
            "Supports bone health",
        ],
    },
    {
        "name": "Fresh Cow Milk",
        "category": "Dairy & Eggs",
        "price": 28,
        "originalPrice": 28,
        "rating": 4.4,
        "discount": 0,
        "description": "Pasteurized fresh cow milk, essential source of daily nutrients.",
        "nutritional_benefits": [
            "High in calcium",
            "Source of quality protein",
            "Rich in Vitamin D",
        ],
    },
    {
        "name": "Roasted Peanuts",
        "category": "Nuts & Seeds",
        "price": 35,
        "originalPrice": 40,
        "rating": 4.5,
        "discount": 12,
        "description": "Crunchy roasted peanuts, high in protein and healthy fats.",
        "nutritional_benefits": [
            "High in plant protein",
            "Heart-healthy monounsaturated fats",
            "Rich in magnesium",
        ],
    },
    {
        "name": "Standard Oats",
        "category": "Breakfast",
        "price": 50,
        "originalPrice": 55,
        "rating": 4.4,
        "discount": 9,
        "description": "100% whole grain oats, great for a heart-healthy breakfast.",
        "nutritional_benefits": [
            "Helps lower cholesterol",
            "High in soluble fiber",
            "Promotes fullness",
        ],
    },
    {
        "name": "Fresh Eggs (6 pack)",
        "category": "Dairy & Eggs",
        "price": 45,
        "originalPrice": 50,
        "rating": 4.8,
        "discount": 10,
        "description": "Farm fresh high-quality eggs, excellent source of complete protein.",
        "nutritional_benefits": [
            "Complete protein source",
            "Rich in choline and Vitamin B12",
            "Supports muscle growth and repair",
        ],
    },
]


# ------------------- Dataset loading -------------------
datasets = {}
nutrition_df = pd.DataFrame()
food_df = pd.DataFrame()
grocery_df = pd.DataFrame()
recipe_df = pd.DataFrame()
reviews_df = pd.DataFrame()

if os.path.exists(DATASET_DIR):
    for fname in os.listdir(DATASET_DIR):
        if fname.endswith(".csv"):
            path = os.path.join(DATASET_DIR, fname)
            try:
                df = pd.read_csv(path, on_bad_lines="skip")
                datasets[fname] = df
                print(f"Loaded dataset: {fname} ({len(df)} rows)")
            except Exception as e:
                print(f"Failed loading {fname}: {e}")

nutrition_df = (
    datasets.get("CSV_VERSION.csv", pd.DataFrame())
    if not datasets.get("CSV_VERSION.csv", pd.DataFrame()).empty
    else datasets.get("nutrition.csv", pd.DataFrame())
    if not datasets.get("nutrition.csv", pd.DataFrame()).empty
    else pd.DataFrame()
)
food_df = (
    datasets.get("indian_food.csv", pd.DataFrame())
    if not datasets.get("indian_food.csv", pd.DataFrame()).empty
    else datasets.get("food.csv", pd.DataFrame())
    if not datasets.get("food.csv", pd.DataFrame()).empty
    else pd.DataFrame()
)

# Load the Open Food Facts dataset CSV
try:
    openfoodfacts_path = os.path.join(DATASET_DIR, "openfoodfacts_grocery.csv")
    if os.path.exists(openfoodfacts_path):
        openfoodfacts_df = pd.read_csv(openfoodfacts_path, on_bad_lines="skip")
    elif not datasets.get("openfoodfacts_grocery.csv", pd.DataFrame()).empty:
        openfoodfacts_df = datasets["openfoodfacts_grocery.csv"]
    elif os.path.exists("openfoodfacts_grocery.csv"):
        openfoodfacts_df = pd.read_csv("openfoodfacts_grocery.csv", on_bad_lines="skip")
    else:
        openfoodfacts_df = pd.DataFrame()

    if not openfoodfacts_df.empty:
        print(f"Loaded Open Food Facts grocery dataset with {len(openfoodfacts_df)} records")
except Exception as e:
    print(f"Failed to load Open Food Facts dataset: {e}")
    openfoodfacts_df = pd.DataFrame()

# Prioritize loading grocery_dataset.csv if it exists
grocery_dataset_path = os.path.join(DATASET_DIR, "grocery_dataset.csv")
if os.path.exists(grocery_dataset_path):
    try:
        grocery_df = pd.read_csv(grocery_dataset_path, on_bad_lines="skip")
        print(
            f"Loaded grocery_dataset.csv with "
            f"{len(grocery_df)} records"
        )
    except Exception as e:
        print(f"Failed to load grocery_dataset.csv: {e}")
        grocery_df = (
            openfoodfacts_df
            if not openfoodfacts_df.empty
            else datasets.get(
                "Grocery_data (1).csv",
                datasets.get("grocery.csv", pd.DataFrame()),
            )
        )

else:
    grocery_df = (
        openfoodfacts_df
        if not openfoodfacts_df.empty
        else datasets.get(
            "Grocery_data (1).csv",
            datasets.get("grocery.csv", pd.DataFrame())
        )
    )
recipes_df = datasets.get(
    "recipe_dataset.csv",
    datasets.get("recipes_dataset.csv", datasets.get("receipe.csv", datasets.get("recipes.csv", datasets.get("recipe.csv", pd.DataFrame())))),
)
reviews_df = pd.DataFrame()  # Skip loading massive 473MB reviews dataset in development

# ------------------- Load Datasets into MongoDB -------------------
try:
    # Define collection mappings
    collection_mappings = {
        "nutrition": nutrition_df,
        "food": food_df,
        "grocery": grocery_df,
        "recipes": recipes_df,
    }

    for collection_name, df in collection_mappings.items():
        if not df.empty:
            collection = db[collection_name]
            # Check if collection already has data to avoid duplicates
            existing_count = collection.count_documents({})
            if existing_count == 0:
                # Convert DataFrame to list of dictionaries
                records = df.to_dict("records")
                # Insert many records
                result = collection.insert_many(records)

                print(
                    f"Loaded {len(result.inserted_ids)} records into "
                    f"{collection_name}"
                )

            else:

                print(
                    f"Collection {collection_name} already has "
                    f"{existing_count} records, skipping"
                )

        else:
            print(
                f"DataFrame for {collection_name} is empty, "
                f"skipping MongoDB insertion"
            )
except Exception as e:
    print(f"[WARN] Failed to load datasets into MongoDB: {e}")


# ------------------- Model training -------------------

model = None  # noqa: F841

label_encoder = None  # noqa: F841

try:
    if not nutrition_df.empty:
        target_col = None
        for cand in ["Category", "category", "FoodGroup", "Group", "Type"]:
            if cand in nutrition_df.columns:
                target_col = cand
                break
        numeric_cols = [
            c
            for c in nutrition_df.columns
            if pd.api.types.is_numeric_dtype(nutrition_df[c])
        ]
        if target_col and numeric_cols:
            df_model = nutrition_df.dropna(subset=[target_col] + numeric_cols)
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(df_model[target_col].astype(str))
            X = df_model[numeric_cols]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            print("Trained RandomForest model on nutrition dataset")

        else:

            print(
                "Nutrition dataset loaded but no suitable target "
                "column found."
            )

    else:

        print("No nutrition dataset found for model training.")

except Exception as e:

    print("Model training failed:", e)

    model = None

    label_encoder = None


# ------------------- In-memory DB -------------------
USERS = {}


def objid_to_str(doc):
    if not doc:
        return doc
    if isinstance(doc, list):
        return [objid_to_str(d) for d in doc]
    doc = dict(doc)
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


def safe_sample(df, n=5):
    if df is None or df.empty:
        return []
    n = min(n, len(df))
    return df.sample(n).to_dict(orient="records")


# ------------------- Auth -------------------


@app.route("/auth/anonymous", methods=["POST"])
def anonymous_login():
    user_id = str(uuid.uuid4())
    USERS[user_id] = {}
    return jsonify({"userId": user_id})


@app.route("/user/<user_id>/data", methods=["GET"])
def get_user(user_id):
    if user_id not in USERS:
        return jsonify({"error": "User not found"}), 404
    return jsonify(USERS[user_id])


@app.route("/user/<user_id>/data", methods=["POST"])
def create_user(user_id):
    data = request.get_json() or {}
    # Initialize default structure for user data
    default_data = {
        "healthData": {
            "calories": 0,
            "goal": 2500,
            "macros": {"carbs": 50, "protein": 30, "fat": 20},
            "waterIntake": 0,
            "weeklyCalories": [2000, 2100, 1900, 2200, 1800, 2050, 1950],
            "waterHistory": [0, 0, 0, 0, 0, 0, 0],
        },
        "groceryData": {"shoppingList": [], "pantry": []},
        "mealPlanner": {"plan": {}, "generatedAt": None},
        "activityLog": [],
        "achievements": {},
        "budget": {
            "weeklyLimit": 2000,  # Default ₹2000 per week
            "currency": "INR",
            "priority": "nutrition",  # "nutrition" or "cost"
        },
    }
    # Merge with provided data
    for key, value in default_data.items():
        if key not in data:
            data[key] = value
    USERS[user_id] = data

    # Sync healthData to MongoDB health_data collection
    health_data.update_one(
        {"userId": user_id}, {"$set": data.get("healthData", {})}, upsert=True
    )

    return jsonify({"message": "User created", "data": data})


@app.route("/user/<user_id>/data", methods=["PUT"])
def update_user(user_id):
    if user_id not in USERS:
        return jsonify({"error": "User not found"}), 404
    data = request.get_json()
    USERS[user_id] = data

    # Sync healthData to MongoDB health_data collection
    health_data.update_one(
        {"userId": user_id}, {"$set": data.get("healthData", {})}, upsert=True
    )

    return jsonify({"message": "User updated", "data": data})


# ------------------- Routes (UI) -------------------


@app.route("/")
def index():
    return render_template("index.html")


# ------------------- SIGNUP -------------------


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if not email or not password or not confirm_password:
            flash("All fields are required.", "error")
            return redirect(url_for("signup"))

        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "error")
            return redirect(url_for("signup"))

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return redirect(url_for("signup"))

        user_exists = users_collection.find_one({"email": email})
        if user_exists:
            flash("Email already registered.", "error")
            return redirect(url_for("signup"))

        hashed_password = generate_password_hash(password)
        new_user = {"email": email, "password": hashed_password}
        users_collection.insert_one(new_user)

        flash("Account created successfully! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("Sign up.html")


# ------------------- LOGIN -------------------


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = users_collection.find_one({"email": email})
        if not user:
            flash("Invalid credentials!", "error")
            return redirect(url_for("login"))

        if not check_password_hash(user["password"], password):
            flash("Invalid credentials!", "error")
            return redirect(url_for("login"))

        session["username"] = email

        flash("Login successful!", "success")
        return redirect(url_for("dashboard"))

    return render_template("Sign in.html")


# ------------------- FORGOT PASSWORD -------------------


@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email")
        user = users_collection.find_one({"email": email})

        if user:
            flash(
                "Password reset link has been sent to your email.", "success"
            )
        else:
            flash("No account found with that email.", "error")
        return redirect(url_for("login"))

    return render_template("forgotpass.html")


@app.route("/dashboard")
@login_required
def dashboard():
    user_id = session.get("username")
    print(f"DEBUG: Rendering dashboard for user_id: {user_id}")
    if not user_id:
        print("DEBUG: No user_id found in session")
        flash("Please log in to access the dashboard.", "error")
        return redirect(url_for("login"))

    # Ensure user_id is properly formatted
    if not isinstance(user_id, str):
        user_id = str(user_id)

    print(f"DEBUG: Dashboard user_id type: {type(user_id)}, value: {user_id}")
    
    # Fetch rich context for instant Jinja rendering
    user_doc = users_collection.find_one({"email": user_id}) or USERS.get(user_id, {})
    health = health_data.find_one({"userId": user_id}) or {}
    grocery = grocery_data.find_one({"userId": user_id}) or {}
    meals = meal_planner.find_one({"userId": user_id}) or {}
    activity = activity_log.find_one({"userId": user_id}) or {}

    # Extract today's meals if available
    today_meals = []
    if "plan" in meals and isinstance(meals["plan"], list) and len(meals["plan"]) > 0:
        first_day = meals["plan"][0]
        if isinstance(first_day, dict) and "meals" in first_day:
            day_meals = first_day["meals"]
            for m_type in ["breakfast", "lunch", "dinner"]:
                m_item = day_meals.get(m_type)
                if m_item:
                    if isinstance(m_item, list) and len(m_item) > 0:
                        today_meals.append(m_item[0])
                    elif isinstance(m_item, dict):
                        today_meals.append(m_item)

    return render_template(
        "dashboard.html",
        user=user_id,
        user_id=user_id,
        user_doc=user_doc,
        health=health,
        grocery=grocery,
        today_meals=today_meals,
        activity=activity.get("activities", []),
    )


@app.route("/api/dashboard/<user_id>", methods=["GET"])
def get_dashboard(user_id):
    user_id = unquote(user_id)
    print(f"DEBUG: API get_dashboard called with user_id: {user_id}")
    user_doc = users_collection.find_one({"email": user_id}) or USERS.get(user_id, {})
    health = health_data.find_one({"userId": user_id}) or {}
    grocery = grocery_data.find_one({"userId": user_id}) or {}
    meals = meal_planner.find_one({"userId": user_id}) or {}
    activity = activity_log.find_one({"userId": user_id}) or {}

    # Calculate or retrieve goal calories and macros
    target_cals = (
        health.get("targetCalories")
        or health.get("goal")
        or user_doc.get("target_calories")
        or 2000
    )
    calories_consumed = health.get("calories", 0)
    carbs = health.get("carbsTarget") or user_doc.get("carbs_target") or round((target_cals * 0.45) / 4)
    protein = health.get("proteinTarget") or user_doc.get("protein_target") or round((target_cals * 0.30) / 4)
    fat = health.get("fatTarget") or user_doc.get("fat_target") or round((target_cals * 0.25) / 9)

    health["goal"] = target_cals
    health["targetCalories"] = target_cals
    health["calories"] = calories_consumed
    health["macros"] = {
        "carbs": carbs,
        "protein": protein,
        "fat": fat,
    }

    # Transform meal plan from list to dict keyed by day for frontend compatibility
    transformed_meal_plan = {}
    if "plan" in meals and isinstance(meals["plan"], list):
        for day_plan in meals["plan"]:
            if isinstance(day_plan, dict):
                day_name = day_plan.get("day")
                if day_name:
                    transformed_meal_plan[day_name] = day_plan.get("meals", {})

    if "plan" in meals and transformed_meal_plan:
        meals["plan"] = transformed_meal_plan

    shopping_list = grocery.get("shoppingList", [])
    pantry_list = grocery.get("pantry", [])

    return jsonify(
        {
            "healthData": objid_to_str(health),
            "groceryData": objid_to_str(grocery),
            "mealPlanner": objid_to_str(meals),
            "activityLog": objid_to_str(activity.get("activities", [])),
            "userProfile": {
                "name": user_doc.get("name") or user_id.split("@")[0],
                "cuisines": user_doc.get("cuisines", ["Indian", "Asian", "Mediterranean"]),
                "goal": user_doc.get("goal", "maintenance"),
                "targetCalories": target_cals,
                "caloriesConsumed": calories_consumed,
                "shoppingCount": len(shopping_list),
                "pantryCount": len(pantry_list),
            },
        }
    )


@app.route("/api/health/<user_id>/log_calories", methods=["POST"])
def log_health_calories(user_id):
    user_id = unquote(user_id)
    data = request.get_json() or {}
    cals_to_add = int(data.get("calories", 0))
    meal_name = data.get("meal_name", "Meal / Snack")

    if cals_to_add <= 0:
        return jsonify({"error": "Calories must be greater than 0"}), 400

    # Increment calories in health_data
    doc = health_data.find_one({"userId": user_id}) or {}
    current_cals = doc.get("calories", 0)
    new_cals = current_cals + cals_to_add

    health_data.update_one(
        {"userId": user_id},
        {"$set": {"calories": new_cals, "updatedAt": dt.datetime.utcnow()}},
        upsert=True,
    )

    # Also log to activity_log
    activity_text = f"Logged {meal_name} (+{cals_to_add} kcal)"
    activity_log.update_one(
        {"userId": user_id},
        {
            "$push": {
                "activities": {
                    "activity": activity_text,
                    "timestamp": dt.datetime.utcnow(),
                }
            }
        },
        upsert=True,
    )

    return jsonify({
        "message": f"Successfully logged {cals_to_add} kcal",
        "calories": new_cals,
        "activity": activity_text
    })


@app.route("/api/health/<user_id>/reset_calories", methods=["POST"])
def reset_health_calories(user_id):
    user_id = unquote(user_id)
    health_data.update_one(
        {"userId": user_id},
        {"$set": {"calories": 0, "updatedAt": dt.datetime.utcnow()}},
        upsert=True,
    )
    return jsonify({"message": "Daily calories reset to 0", "calories": 0})


# ------------------- Additional Routes -------------------


@app.route("/bitebot")
def bitebot():
    return render_template("Bitebot.html")


@app.route("/fitness_games")
@login_required
def fitness_games():
    return render_template("fitness_games.html")


@app.route("/yoga_pose_quiz")
@login_required
def yoga_pose_quiz():
    return render_template("yoga_pose_quiz_fixed.html")


@app.route("/fitness_challenges")
@login_required
def fitness_challenges():
    return redirect(url_for("fitness_games"))


@app.route("/repetition-counter")
def repetition_counter():
    return render_template("repetition_counter.html")


@app.route("/plank_timer")
@login_required
def plank_timer():
    return render_template("plank_timer.html")


@app.route("/nutrition_label_quiz")
@login_required
def nutrition_label_quiz():
    return render_template("nutrition_label_quiz.html")


@app.route("/achievements")
@login_required
def achievements():
    user_id = session.get("username")
    return render_template("achievements.html", user=user_id)


@app.route("/api/achievements/<user_id>", methods=["GET"])
def get_achievements(user_id):
    """Get user achievements and progress data"""
    try:
        # Get user data from database or session
        user_data = USERS.get(user_id, {})  # noqa: F841

        # Calculate achievements based on user activity
        achievements = calculate_user_achievements(user_data)

        return jsonify(
            {
                "rank": achievements.get("rank", "Beginner"),
                "score": achievements.get("total_score", 0),
                "next_level": achievements.get("next_level", "Intermediate"),
                "progress": achievements.get("progress_percentage", 0),
                "levels": achievements.get("levels", []),
            }
        )

    except Exception as e:
        print(f"Error getting achievements: {e}")
        return (
            jsonify(
                {"error": "Failed to load achievements", "details": str(e)}
            ),
            500,
        )


@app.route("/api/quiz/save-score", methods=["POST"])
def save_quiz_score():
    """Save quiz score to user progress"""
    try:
        data = request.get_json() or {}

        quiz_type = data.get("quiz_type", "unknown")
        score = data.get("score", 0)
        max_score = data.get("max_score", 100)
        user_id = data.get("user_id", "anonymous")

        # Calculate percentage
        percentage = (score / max_score * 100) if max_score > 0 else 0

        # Save to user data (in a real app, this would go to database)
        if user_id not in USERS:
            USERS[user_id] = {
                "quiz_scores": {},
                "achievements": {},
                "total_score": 0,
            }

        # Update quiz scores
        USERS[user_id]["quiz_scores"][quiz_type] = {
            "score": score,
            "max_score": max_score,
            "percentage": percentage,
            "timestamp": dt.datetime.utcnow().isoformat(),
        }

        # Update total score
        USERS[user_id]["total_score"] = sum(
            quiz.get("score", 0)
            for quiz in USERS[user_id]["quiz_scores"].values()
        )

        # Check for new achievements
        new_achievements = check_quiz_achievements(
            user_id, quiz_type, percentage
        )

        return jsonify(
            {
                "success": True,
                "message": "Quiz score saved successfully",
                "score": score,
                "percentage": percentage,
                "new_achievements": new_achievements,
                "total_score": USERS[user_id]["total_score"],
            }
        )

    except Exception as e:
        print(f"Error saving quiz score: {e}")
        return (
            jsonify({"error": "Failed to save quiz score", "details": str(e)}),
            500,
        )


def calculate_user_achievements(user_data):
    """Calculate user achievements and progress"""
    total_score = user_data.get("total_score", 0)
    quiz_scores = user_data.get("quiz_scores", {})

    # Define achievement levels
    levels = [
        {
            "name": "Beginner",
            "min_score": 0,
            "description": "Complete your first quiz",
        },
        {
            "name": "Novice",
            "min_score": 100,
            "description": "Score 100+ total points",
        },
        {
            "name": "Intermediate",
            "min_score": 300,
            "description": "Score 300+ total points",
        },
        {
            "name": "Advanced",
            "min_score": 600,
            "description": "Score 600+ total points",
        },
        {
            "name": "Expert",
            "min_score": 1000,
            "description": "Score 1000+ total points",
        },
        {
            "name": "Master",
            "min_score": 1500,
            "description": "Score 1500+ total points",
        },
    ]

    # Determine current level
    current_level = levels[0]
    for level in levels:
        if total_score >= level["min_score"]:
            current_level = level
        else:
            break

    # Find next level
    next_level = None
    for level in levels:
        if level["min_score"] > total_score:
            next_level = level
            break

    # Calculate progress to next level
    progress_percentage = 0
    if next_level:
        prev_level_score = current_level["min_score"]
        next_level_score = next_level["min_score"]
        score_range = next_level_score - prev_level_score
        if score_range > 0:
            progress_in_range = total_score - prev_level_score
            progress_percentage = min(
                100, (progress_in_range / score_range) * 100
            )

    # Mark levels as unlocked
    for level in levels:
        level["unlocked"] = total_score >= level["min_score"]

    return {
        "rank": current_level["name"],
        "total_score": total_score,
        "next_level": next_level["name"] if next_level else "Master",
        "progress_percentage": progress_percentage,
        "levels": levels,
        "quiz_breakdown": quiz_scores,
    }


def check_quiz_achievements(user_id, quiz_type, percentage):
    """Check for new achievements based on quiz performance"""
    new_achievements = []

    # Check for perfect score
    if percentage == 100:
        new_achievements.append(
            {
                "type": "perfect_score",
                "name": "Perfect Score!",
                "description": f"Achieved 100% on {quiz_type} quiz",
                "timestamp": dt.datetime.utcnow().isoformat(),
            }
        )

    # Check for high score thresholds
    if percentage >= 90:
        new_achievements.append(
            {
                "type": "high_score",
                "name": "Excellent Performance",
                "description": f"Scored 90%+ on {quiz_type} quiz",
                "timestamp": dt.datetime.utcnow().isoformat(),
            }
        )

    # Check for first quiz completion
    if user_id in USERS and len(USERS[user_id]["quiz_scores"]) == 1:
        new_achievements.append(
            {
                "type": "first_quiz",
                "name": "First Steps",
                "description": "Completed your first quiz!",
                "timestamp": dt.datetime.utcnow().isoformat(),
            }
        )

    # Store new achievements
    if user_id in USERS and new_achievements:
        if "achievements" not in USERS[user_id]:
            USERS[user_id]["achievements"] = {}

        for achievement in new_achievements:
            achievement_id = (
                f"{achievement['type']}_{dt.datetime.utcnow().timestamp()}"
            )
            USERS[user_id]["achievements"][achievement_id] = achievement

    return new_achievements


def transform_meal_item(meal, meal_type=None):
    """
    Standardize a meal item dictionary representation for UI templates.
    """
    return {
        "Shrt_Desc": meal.get("Shrt_Desc", meal.get("name", f"{meal_type.title()} Meal" if meal_type else "Unknown")),
        "Energ_Kcal": meal.get("Energ_Kcal", meal.get("calories", "-")),
        "Protein_(g)": meal.get("Protein_(g)", meal.get("protein_grams", "-")),
        "Carbohydrt_(g)": meal.get("Carbohydrt_(g)", meal.get("carb_grams", "-")),
        "Lipid_Tot_(g)": meal.get("Lipid_Tot_(g)", meal.get("fat_grams", "-")),
        "ingredients": meal.get("ingredients", []),
        "instructions": meal.get("instructions", []),
        "prep_time": meal.get("prep_time", ""),
        "cook_time": meal.get("cook_time", ""),
        "total_time": meal.get("total_time", 0),
        "servings": meal.get("servings", ""),
        "complexity": meal.get("complexity", 5),
        "category": meal.get("category", ""),
        "description": meal.get("description", ""),
        "is_custom": meal.get("is_custom", False),
    }


def make_custom_recipe_dict(data, recipe_id, added_at=None):
    """Create a standardized custom recipe dictionary from request data."""
    return {
        "_id": recipe_id,
        "Shrt_Desc": data.get("name", "Custom Recipe"),
        "Energ_Kcal": data.get("calories", 0),
        "Protein_(g)": data.get("protein", 0),
        "Carbohydrt_(g)": data.get("carbs", 0),
        "Lipid_Tot_(g)": data.get("fats", 0),
        "ingredients": data.get("ingredients", []),
        "instructions": data.get("instructions", []),
        "is_custom": True,
        "added_at": added_at or dt.datetime.utcnow(),
    }



@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user_id = session.get("username")
    if request.method == "POST":
        try:
            # Get form data
            name = request.form.get("name", "User")
            age = request.form.get("age", type=int) or 25
            gender = request.form.get("gender", "other")
            height = request.form.get("height", type=float) or 170.0
            weight = request.form.get("weight", type=float) or 68.0
            goal = request.form.get("goal", "maintenance")
            activity_level = request.form.get("activity_level", "moderately_active")
            diet_preference = request.form.get("diet_preference", "balanced")
            health_condition = request.form.get("health_condition", "")
            diet_restrictions = request.form.get("diet_restrictions", "")
            cuisines = request.form.getlist("cuisines")
            if not cuisines:
                cuisines = ["Indian", "Asian", "Mediterranean"]

            # Calculate BMR using Mifflin-St Jeor equation
            bmr = (10 * weight) + (6.25 * height) - (5 * age)
            if gender == "male":
                bmr += 5
            elif gender == "female":
                bmr -= 161

            # Activity multiplier
            activity_multipliers = {
                "sedentary": 1.2,
                "lightly_active": 1.375,
                "moderately_active": 1.55,
                "very_active": 1.725
            }
            tdee = bmr * activity_multipliers.get(activity_level, 1.4)

            # Goal adjustments
            if goal == "weight_loss":
                target_calories = max(1200, int(tdee - 400))
            elif goal == "muscle_gain":
                target_calories = int(tdee + 350)
            else:
                target_calories = int(tdee)

            # Macro targets (30% Protein, 45% Carbs, 25% Fat)
            protein_target = round((target_calories * 0.30) / 4)
            carbs_target = round((target_calories * 0.45) / 4)
            fat_target = round((target_calories * 0.25) / 9)

            # Save profile data to USERS and database
            user_data = USERS.get(user_id, {})
            user_data.update(
                {
                    "name": name,
                    "age": age,
                    "gender": gender,
                    "height": height,
                    "weight": weight,
                    "goal": goal,
                    "activity_level": activity_level,
                    "diet_preference": diet_preference,
                    "health_condition": health_condition,
                    "diet_restrictions": diet_restrictions,
                    "cuisines": cuisines,
                    "target_calories": target_calories,
                    "bmr": round(bmr),
                    "tdee": round(tdee),
                    "protein_target": protein_target,
                    "carbs_target": carbs_target,
                    "fat_target": fat_target,
                }
            )
            USERS[user_id] = user_data

            # Persist to MongoDB users and health_data collections
            try:
                users_collection.update_one(
                    {"email": user_id},
                    {"$set": user_data},
                    upsert=True
                )
                health_data.update_one(
                    {"userId": user_id},
                    {"$set": {
                        "userId": user_id,
                        "weight": weight,
                        "height": height,
                        "bmi": round(weight / ((height / 100) ** 2), 1),
                        "caloriesBurned": 0,
                        "targetCalories": target_calories,
                        "proteinTarget": protein_target,
                        "carbsTarget": carbs_target,
                        "fatTarget": fat_target,
                        "updatedAt": dt.datetime.utcnow()
                    }},
                    upsert=True
                )
            except Exception as db_err:
                print(f"Error persisting user profile to DB: {db_err}")

            # Generate personalized multi-cuisine diet plan using SmartBite AI models
            diet_plan = smartbite_models.get_recommendations(
                {
                    "diet_type": diet_preference.lower(),
                    "calorie_goal": target_calories,
                    "health_condition": health_condition,
                    "cuisines": cuisines,
                }
            )

            # Fallback if AI models fail or return empty plan
            daily_meals = diet_plan.get("daily_meals", {})
            if not daily_meals or not isinstance(daily_meals, dict):
                daily_meals = {
                    "breakfast": [{"name": "Oatmeal with Fruits", "calories": 350, "protein_grams": 12, "carb_grams": 60, "fat_grams": 8, "cuisine": "Continental"}],
                    "lunch": [{"name": "Paneer Tikka with Roti", "calories": 480, "protein_grams": 26, "carb_grams": 45, "fat_grams": 22, "cuisine": "Indian"}],
                    "dinner": [{"name": "Vegetable Stir Fry Noodles", "calories": 420, "protein_grams": 16, "carb_grams": 60, "fat_grams": 14, "cuisine": "Asian"}],
                }
                diet_plan = {"daily_meals": daily_meals}

            def enhance_meal_with_recipe(meal):
                """Enhance meal item with detailed recipe information"""
                enhanced_meal = transform_meal_item(meal)
                recipe_details = smartbite_models.recipe_generator.get_recipe_details(meal.get("name", ""))
                if recipe_details:
                    enhanced_meal.update(
                        {
                            "ingredients": recipe_details.get("ingredients", []),
                            "instructions": recipe_details.get("instructions", []),
                            "prep_time": recipe_details.get("prep_time", ""),
                            "cook_time": recipe_details.get("cook_time", ""),
                            "total_time": recipe_details.get("total_time", 0),
                            "servings": recipe_details.get("servings", ""),
                            "description": recipe_details.get("description", ""),
                            "complexity": recipe_details.get("complexity", 5),
                            "category": recipe_details.get("category", ""),
                        }
                    )
                return enhanced_meal

            breakfast = [enhance_meal_with_recipe(item) for item in daily_meals.get("breakfast", [])]
            lunch = [enhance_meal_with_recipe(item) for item in daily_meals.get("lunch", [])]
            dinner = [enhance_meal_with_recipe(item) for item in daily_meals.get("dinner", [])]
            transformed_plan = breakfast + lunch + dinner

            user_data["diet_plan"] = diet_plan
            USERS[user_id] = user_data

            try:
                diet_collection.update_one(
                    {"userId": user_id},
                    {"$set": {"diet_plan": diet_plan, "cuisines": cuisines, "target_calories": target_calories}},
                    upsert=True,
                )
                meal_planner.update_one(
                    {"userId": user_id},
                    {
                        "$set": {
                            "plan": daily_meals,
                            "cuisines": cuisines,
                            "target_calories": target_calories,
                            "generatedAt": dt.datetime.utcnow(),
                        }
                    },
                    upsert=True,
                )
            except Exception as e:
                print(f"Error saving meal plan to DB: {e}")

            flash(f"Profile saved! Curated {', '.join(cuisines[:3])} meal plan for {target_calories} kcal/day.", "success")
        except Exception as e:
            print(f"Error processing profile: {e}")
            flash("Error generating diet plan, proceeding with standard plan.", "error")

        return redirect(url_for('diet_plan'))

    # GET request: pre-populate with existing profile if available
    user_data = USERS.get(user_id, {})
    if not user_data:
        try:
            db_user = users_collection.find_one({"email": user_id})
            if db_user:
                user_data = db_user
        except Exception as e:
            print(f"Error fetching user from DB: {e}")

    return render_template("profile.html", user=user_data)


# Fix: Add a route for mealplanner to avoid BuildError in profile.html redirect


@app.route("/mealplanner")
@login_required
def mealplanner():
    user_id = session.get("username")

    # Get user data from USERS dictionary
    user_data = USERS.get(user_id, {})

    # Try to get meal plan from MongoDB meal_planner collection
    meal_plan = {}
    try:
        doc = meal_planner.find_one({"userId": user_id})
        if doc and "plan" in doc:
            meal_plan = doc["plan"]
    except Exception as e:
        print(f"Error fetching meal plan from DB: {e}")

    # Transform meal plan for template with detailed recipe info
    transformed_plan = []
    # meal_plan keys: breakfast, lunch, dinner (each is a list of items)
    for meal_time in ["breakfast", "lunch", "dinner"]:
        items = meal_plan.get(meal_time, [])
        for item in items:
            transformed_plan.append(
                {
                    "Shrt_Desc": item.get("name", "Unknown"),
                    "Energ_Kcal": item.get("calories", "-"),
                    "Protein_(g)": item.get("protein", "-"),
                    "Carbohydrt_(g)": item.get("carbs", "-"),
                    "Lipid_Tot_(g)": item.get("fat", "-"),
                    "ingredients": item.get("ingredients", []),
                    "instructions": item.get("instructions", []),
                }
            )

    return render_template(
        "diet_plan.html",
        user=user_data.get("name", "User"),
        plan=transformed_plan,
    )


@app.route("/diet_profile")
@login_required
def diet_profile():
    return redirect(url_for("profile"))


@app.route("/grocery")
@login_required
def grocery():
    user_id = session.get("username")
    return render_template("grocery.html", user=user_id)


@app.route("/product/<product_id>")
@login_required
def product_detail(product_id):
    """Display detailed view of a specific product"""
    user_id = session.get("username")
    return render_template(
        "product_detail.html", user=user_id, product_id=product_id
    )


@app.route("/api/grocery/products")
def get_grocery_products():
    """API endpoint to get grocery products from dataset"""
    try:
        products = []

        if not grocery_df.empty:
            # Convert DataFrame to list of dictionaries
            for _, row in grocery_df.iterrows():
                # Extract product information from the dataset with correct column mappings
                title = (
                    row.get("Title") or row.get("name") or "Unknown Product"
                )
                category = (
                    row.get("Sub Category") or row.get("category") or "General"
                )
                price_raw = row.get("Price") or 0
                discount_raw = row.get("Discount") or 0
                rating_raw = row.get("Rating") or 4.0
                description = (
                    row.get("Product Description")
                    or row.get("Feature")
                    or "No description available"
                )

                # Convert price and discount safely
                try:
                    price = float(price_raw)
                except (ValueError, TypeError):
                    price = 0.0

                try:
                    discount = int(discount_raw)
                except (ValueError, TypeError):
                    discount = 0

                try:
                    rating = float(rating_raw)
                except (ValueError, TypeError):
                    rating = 4.0

                original_price = price
                if discount > 0:
                    # Calculate original price if discount is given
                    try:
                        original_price = round(price / (1 - discount / 100), 2)
                    except ZeroDivisionError:
                        original_price = price

                product = {
                    "name": str(title),
                    "category": str(category),
                    "price": price,
                    "originalPrice": original_price,
                    "rating": rating,
                    "discount": discount,
                    "description": str(description),
                    "nutritional_benefits": [],
                }

                # Extract nutritional benefits from description if available
                desc_lower = product["description"].lower()
                if any(
                    word in desc_lower
                    for word in ["organic", "natural", "healthy", "nutritious"]
                ):
                    product["nutritional_benefits"].append("Natural/Organic")
                if any(
                    word in desc_lower
                    for word in ["protein", "muscle", "strength"]
                ):
                    product["nutritional_benefits"].append("High in protein")
                if any(
                    word in desc_lower
                    for word in ["fiber", "digestive", "gut"]
                ):
                    product["nutritional_benefits"].append(
                        "Good source of fiber"
                    )
                if any(
                    word in desc_lower
                    for word in ["vitamin", "mineral", "nutrient"]
                ):
                    product["nutritional_benefits"].append(
                        "Rich in vitamins/minerals"
                    )
                if any(
                    word in desc_lower
                    for word in ["antioxidant", "immune", "health"]
                ):
                    product["nutritional_benefits"].append(
                        "Supports immune system"
                    )

                # If no benefits extracted, add a default one
                if not product["nutritional_benefits"]:
                    product["nutritional_benefits"].append("Quality product")

                products.append(product)

        # If no products from dataset, return sample products for demonstration
        if not products:
            products = FALLBACK_GROCERY_PRODUCTS

        return jsonify(products)

    except Exception as e:
        print(f"Error loading grocery products: {e}")
        # Return sample products as fallback
        return jsonify(FALLBACK_GROCERY_PRODUCTS)


@app.route("/diet_plan")
@login_required
def diet_plan():
    user_id = session.get("username")
    user_data = USERS.get(user_id, {})

    # Get user's diet plan from meal_planner collection for consistency with edit functionality
    plan = []
    try:
        doc = meal_planner.find_one({"userId": user_id})
        if doc and "plan" in doc:
            meal_plan = doc["plan"]
            # Transform the meal plan data to match template expectations
            if isinstance(meal_plan, list):
                # Handle weekly plan format (list of days)
                for day_plan in meal_plan:
                    if isinstance(day_plan, dict) and "meals" in day_plan:
                        meals = day_plan["meals"]
                        for meal_type, meal_items in meals.items():
                            if isinstance(meal_items, list):
                                for meal in meal_items:
                                    plan.append(transform_meal_item(meal, meal_type=meal_type))
            else:
                # Handle flat meal list format
                for meal in meal_plan:
                    plan.append(transform_meal_item(meal))
        else:
            # Generate fallback/default plan if none exists
            print("No diet plan found, generating fallback plan")
            fallback_plan = [
                {
                    "Shrt_Desc": "Oatmeal with Fruits",
                    "Energ_Kcal": "350",
                    "Protein_(g)": "10",
                    "Carbohydrt_(g)": "60",
                    "Lipid_Tot_(g)": "8",
                    "ingredients": ["Oats", "Fruits", "Milk"],
                    "instructions": ["Mix ingredients", "Cook", "Serve"],
                    "is_custom": False,
                },
                {
                    "Shrt_Desc": "Grilled Chicken Salad",
                    "Energ_Kcal": "450",
                    "Protein_(g)": "35",
                    "Carbohydrt_(g)": "20",
                    "Lipid_Tot_(g)": "25",
                    "ingredients": ["Chicken", "Salad greens", "Vegetables"],
                    "instructions": ["Grill chicken", "Mix with greens", "Dress and serve"],
                    "is_custom": False,
                },
                {
                    "Shrt_Desc": "Vegetable Stir Fry with Rice",
                    "Energ_Kcal": "400",
                    "Protein_(g)": "15",
                    "Carbohydrt_(g)": "55",
                    "Lipid_Tot_(g)": "12",
                    "ingredients": ["Rice", "Mixed vegetables", "Soy sauce"],
                    "instructions": ["Cook rice", "Stir fry vegetables", "Serve together"],
                    "is_custom": False,
                },
                {
                    "Shrt_Desc": "Greek Yogurt Parfait",
                    "Energ_Kcal": "250",
                    "Protein_(g)": "20",
                    "Carbohydrt_(g)": "30",
                    "Lipid_Tot_(g)": "5",
                    "ingredients": ["Greek yogurt", "Fruits", "Granola"],
                    "instructions": ["Layer yogurt", "Add fruits", "Top with granola"],
                    "is_custom": False,
                },
            ]
            plan = fallback_plan
            # Save fallback to meal_planner for consistency
            meal_planner.update_one(
                {"userId": user_id},
                {
                    "$set": {
                        "plan": fallback_plan,
                        "generatedAt": dt.datetime.utcnow(),
                    }
                },
                upsert=True,
            )
    except Exception as e:
        print(f"Error fetching or generating diet plan: {e}")
        # Ensure plan is never empty to avoid template issues
        plan = [
            {
                "Shrt_Desc": "Sample Breakfast",
                "Energ_Kcal": "300",
                "Protein_(g)": "12",
                "Carbohydrt_(g)": "45",
                "Lipid_Tot_(g)": "10",
                "ingredients": [],
                "instructions": [],
                "is_custom": False,
            }
        ]

    # Enrich plan items with recipe details from dataset
    for item in plan:
        if not item.get('is_custom', False):
            # Find matching recipe in dataset
            if not recipes_df.empty:
                matching_recipes = recipes_df[recipes_df['Name'].str.lower().str.contains(item['Shrt_Desc'].lower(), na=False)]
                if not matching_recipes.empty:
                    row = matching_recipes.iloc[0]
                    item['authorname'] = str(row.get('AuthorName', 'N/A'))
                    item['recipecategory'] = str(row.get('RecipeCategory', 'N/A'))
                else:
                    item['authorname'] = 'N/A'
                    item['recipecategory'] = 'N/A'
            else:
                item['authorname'] = 'N/A'
                item['recipecategory'] = 'N/A'
        else:
            # For custom recipes
            item['authorname'] = 'Custom'
            item['recipecategory'] = 'Custom'

    # Fetch and append custom recipes
    try:
        diet_doc = diet_collection.find_one({"userId": user_id})
        if diet_doc and "custom_recipes" in diet_doc:
            custom_recipes = diet_doc["custom_recipes"]
            for recipe in custom_recipes:
                plan.append(
                    {
                        "Shrt_Desc": recipe.get("Shrt_Desc", "Custom Recipe"),
                        "Energ_Kcal": recipe.get("Energ_Kcal", "-"),
                        "Protein_(g)": recipe.get("Protein_(g)", "-"),
                        "Carbohydrt_(g)": recipe.get("Carbohydrt_(g)", "-"),
                        "Lipid_Tot_(g)": recipe.get("Lipid_Tot_(g)", "-"),
                        "ingredients": recipe.get("ingredients", []),
                        "instructions": recipe.get("instructions", []),
                        "prep_time": "",
                        "cook_time": "",
                        "total_time": 0,
                        "servings": "",
                        "complexity": 5,
                        "category": "Custom",
                        "description": "",
                        "is_custom": True,
                        "recipe_id": recipe.get("_id", ""),
                        "added_at": recipe.get("added_at", ""),
                        "authorname": "Custom",
                        "recipecategory": "Custom",
                    }
                )
    except Exception as e:
        print(f"Error fetching custom recipes: {e}")

    return render_template(
        "diet_plan.html", user=user_data.get("name", user_id), plan=plan
    )


@app.route("/api/diet_plan/push_to_grocery", methods=["POST"])
@login_required
def push_diet_to_grocery():
    """Extract all ingredients from the user's active diet plan, consolidate them, and push to grocery list"""
    user_id = session.get("username")
    try:
        plan_doc = meal_planner.find_one({"userId": user_id}) or diet_collection.find_one({"userId": user_id})
        ingredients_set = set()

        if plan_doc:
            plan = plan_doc.get("plan", plan_doc.get("diet_plan", {}).get("daily_meals", {}))
            if isinstance(plan, dict):
                for meal_time, meals in plan.items():
                    if isinstance(meals, list):
                        for m in meals:
                            raw_ings = m.get("ingredients", [])
                            if isinstance(raw_ings, str):
                                raw_ings = [i.strip() for i in raw_ings.split(",")]
                            for ing in raw_ings:
                                clean_ing = str(ing).strip().title()
                                # Filter empty or basic seasonings
                                if clean_ing and len(clean_ing) > 1 and clean_ing.lower() not in ["water", "salt"]:
                                    ingredients_set.add(clean_ing)

        if not ingredients_set:
            ingredients_set = {"Rolled Oats", "Fresh Spinach", "Yellow Lentils", "Paneer", "Tomatoes", "Onions", "Brown Rice", "Olive Oil", "Garlic", "Ginger"}

        # Get existing grocery list from grocery_data or create
        existing_doc = grocery_data.find_one({"userId": user_id}) or {}
        existing_shopping = set(existing_doc.get("shoppingList", []))
        existing_pantry = set(existing_doc.get("pantry", []))

        # Add items that are not already in pantry
        added_items = []
        for ing in sorted(ingredients_set):
            if ing not in existing_pantry:
                existing_shopping.add(ing)
                added_items.append(ing)

        # Update in database
        grocery_data.update_one(
            {"userId": user_id},
            {
                "$set": {
                    "userId": user_id,
                    "shoppingList": list(existing_shopping),
                    "updatedAt": dt.datetime.utcnow()
                }
            },
            upsert=True
        )

        return jsonify({
            "success": True,
            "status": "success",
            "message": f"Successfully added {len(added_items)} fresh ingredients to your Grocery Optimizer!",
            "count": len(added_items),
            "items_added": len(added_items),
            "items": added_items
        })
    except Exception as e:
        print(f"Error in push_diet_to_grocery: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/recipe/<recipe_name>")
@login_required
def recipe_detail(recipe_name):
    # Find recipe by name from recipes dataset
    recipe = None
    if not recipes_df.empty:
        # Check for correct column name for recipe name
        name_col = None
        for col in recipes_df.columns:
            if col.lower() in ["name", "recipe_name", "title"]:
                name_col = col
                break
        if name_col:
            filtered = recipes_df[
                recipes_df[name_col].str.lower() == recipe_name.lower()
            ]
            if not filtered.empty:
                row = filtered.iloc[0]

                # Parse ingredients from RecipeIngredientParts (stored as string list)
                ingredients_str = row.get("RecipeIngredientParts", "[]")
                try:
                    ingredients_list = json.loads(ingredients_str)
                except Exception:
                    # Fallback: try eval if json.loads fails
                    try:
                        ingredients_list = eval(ingredients_str)
                    except Exception:
                        ingredients_list = []

                # Parse instructions from RecipeInstructions (stored as string list)
                instructions_str = row.get("RecipeInstructions", "[]")
                try:
                    steps_list = json.loads(instructions_str)
                except Exception:
                    # Fallback: try eval if json.loads fails
                    try:
                        steps_list = eval(instructions_str)
                    except Exception:
                        steps_list = []

                # Ensure ingredients and instructions are lists of strings
                if not isinstance(ingredients_list, list):
                    ingredients_list = []
                else:
                    ingredients_list = [str(i) for i in ingredients_list]

                if not isinstance(steps_list, list):
                    steps_list = []
                else:
                    steps_list = [str(s) for s in steps_list]

                # Prepare comprehensive recipe details
                recipe = {
                    "name": row.get(name_col, "Unknown Recipe"),
                    "description": row.get("Description", ""),
                    "category": row.get("RecipeCategory", ""),
                    "prep_time": row.get("PrepTime", ""),
                    "cook_time": row.get("CookTime", ""),
                    "total_time": row.get("TotalTime", ""),
                    "servings": row.get("RecipeServings", ""),
                    "calories": row.get("Calories", ""),
                    "protein": row.get("ProteinContent", ""),
                    "carbs": row.get("CarbohydrateContent", ""),
                    "fat": row.get("FatContent", ""),
                    "fiber": row.get("FiberContent", ""),
                    "sugar": row.get("SugarContent", ""),
                    "sodium": row.get("SodiumContent", ""),
                    "rating": row.get("AggregatedRating", ""),
                    "review_count": row.get("ReviewCount", ""),
                    "ingredients": ingredients_list,
                    "instructions": steps_list,
                    "keywords": row.get("Keywords", ""),
                    "author": row.get("AuthorName", ""),
                    "date_published": row.get("DatePublished", ""),
                }
    if not recipe:
        return render_template("recipe_detail.html", error="Recipe not found.")
    return render_template("recipe_detail.html", recipe=recipe)


@app.route("/streak_master")
@login_required
def streak_master():
    return render_template("streak_master.html")


# Alias routes for backward compatibility


@app.route("/dietprofile")
@login_required
def dietprofile():
    return redirect(url_for("diet_profile"))


@app.route("/dietplan")
@login_required
def dietplan():
    return redirect(url_for("diet_plan"))


@app.route("/logout")
def logout():
    session.pop("username", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# ------------------- Enhanced Grocery Store Integration API Endpoints -------------------


@app.route("/api/grocery/stores/configure", methods=["POST"])
def configure_store():
    """Configure store API settings and credentials"""
    try:
        data = request.get_json() or {}

        store_config_data = {
            "store_name": data.get("store_name", "Unknown Store"),
            "api_endpoint": data.get("api_endpoint"),
            "api_key": data.get("api_key"),
            "store_id": data.get("store_id"),
            "location": data.get("location", {}),
            "delivery_options": data.get("delivery_options", {}),
            "price_update_frequency": data.get(
                "price_update_frequency", "daily"
            ),
            "is_active": data.get("is_active", True),
            "created_at": dt.datetime.utcnow(),
            "updated_at": dt.datetime.utcnow(),
        }

        # Save to store_config collection
        result = store_config.insert_one(store_config_data)

        return jsonify(
            {
                "success": True,
                "message": "Store configured successfully",
                "store_id": str(result.inserted_id),
                "data": store_config_data,
            }
        )

    except Exception as e:
        print(f"Error configuring store: {e}")
        return (
            jsonify(
                {"error": "Store configuration failed", "details": str(e)}
            ),
            500,
        )


@app.route("/api/grocery/stores", methods=["GET"])
def get_stores():
    """Get all configured stores"""
    try:
        stores = list(store_config.find({"is_active": True}))
        for store in stores:
            store["_id"] = str(store["_id"])

        return jsonify(
            {"success": True, "stores": stores, "total": len(stores)}
        )

    except Exception as e:
        print(f"Error fetching stores: {e}")
        return (
            jsonify({"error": "Failed to fetch stores", "details": str(e)}),
            500,
        )


@app.route("/api/grocery/price-compare", methods=["POST"])
def compare_prices():
    """Compare prices across multiple stores"""
    try:
        data = request.get_json() or {}
        product_name = data.get("product_name")
        quantity = data.get("quantity", 1)

        if not product_name:
            return jsonify({"error": "Product name is required"}), 400

        # Get all active stores
        stores = list(store_config.find({"is_active": True}))
        price_comparison = []
        best_deal = None
        lowest_price = float("inf")

        for store in stores:
            # Mock price data - in real implementation, this would call actual store APIs
            mock_price = {
                "store_name": store["store_name"],
                "store_id": str(store["_id"]),
                "product_name": product_name,
                "price": (
                    85.50 if store["store_name"] == "BigBasket" else 82.00
                ),
                "original_price": (
                    100.00 if store["store_name"] == "BigBasket" else 95.00
                ),
                "discount": 15 if store["store_name"] == "BigBasket" else 14,
                "availability": "In Stock",
                "delivery_time": (
                    "2 days" if store["store_name"] == "BigBasket" else "1 day"
                ),
                "delivery_fee": (
                    20.00 if store["store_name"] == "BigBasket" else 0.00
                ),
            }

            price_comparison.append(mock_price)

            # Track best deal
            total_cost = mock_price["price"] + mock_price["delivery_fee"]
            if total_cost < lowest_price:
                lowest_price = total_cost
                best_deal = mock_price

        # Calculate statistics
        prices = [item["price"] for item in price_comparison]
        average_price = sum(prices) / len(prices) if prices else 0
        min_price = min(prices) if prices else 0
        max_price = max(prices) if prices else 0

        # Save price data to history
        for price_data in price_comparison:
            price_history.insert_one(
                {
                    "store_id": price_data["store_id"],
                    "product_name": price_data["product_name"],
                    "price": price_data["price"],
                    "original_price": price_data["original_price"],
                    "discount": price_data["discount"],
                    "timestamp": dt.datetime.utcnow(),
                }
            )

        return jsonify(
            {
                "success": True,
                "product": product_name,
                "quantity": quantity,
                "price_comparison": price_comparison,
                "best_deal": best_deal,
                "total_stores": len(stores),
                "average_price": round(average_price, 2),
                "price_range": {"min": min_price, "max": max_price},
                "generated_at": dt.datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        print(f"Error comparing prices: {e}")
        return (
            jsonify({"error": "Price comparison failed", "details": str(e)}),
            500,
        )


@app.route("/api/grocery/smart-list", methods=["POST"])
def generate_smart_shopping_list():
    """Generate optimized shopping list from meal plans with budget constraints"""
    data = request.get_json() or {}
    user_id = data.get("user_id")
    meal_plan = data.get("meal_plan", {})
    budget_limit = data.get("budget_limit", 2000)

    if not user_id or not meal_plan:
        return jsonify({"error": "User ID and meal plan are required"}), 400

    # Extract ingredients from meal plan
    ingredients = []
    for meal_type, meals in meal_plan.items():
        if isinstance(meals, list):
            for meal in meals:
                if isinstance(meal, dict):
                    # Extract ingredients from meal description
                    desc = meal.get("name", "").lower()
                    if "oats" in desc:
                        ingredients.append(
                            {
                                "name": "Oats",
                                "quantity": 1,
                                "category": "Breakfast",
                            }
                        )
                    elif "banana" in desc:
                        ingredients.append(
                            {
                                "name": "Banana",
                                "quantity": 6,
                                "category": "Fruits",
                            }
                        )
                    elif "rice" in desc:
                        ingredients.append(
                            {
                                "name": "Rice",
                                "quantity": 1,
                                "category": "Grains",
                            }
                        )
                    elif "chicken" in desc:
                        ingredients.append(
                            {
                                "name": "Chicken",
                                "quantity": 500,
                                "category": "Protein",
                            }
                        )

    # Get store prices for ingredients
    smart_shopping_list = []
    total_estimated = 0

    for ingredient in ingredients:
        # Get prices from different stores
        stores = list(store_config.find({"is_active": True}))
        store_prices = []

        for store in stores:
            # Mock pricing based on store
            base_price = {
                "Oats": 120,
                "Banana": 10,
                "Rice": 80,
                "Chicken": 250,
            }.get(ingredient["name"], 100)

            if store["store_name"] == "BigBasket":
                price = base_price * 1.1  # 10% premium
            else:
                price = base_price * 0.95  # 5% discount

            store_prices.append(
                {
                    "store_name": store["store_name"],
                    "price": price,
                    "delivery_fee": (
                        20 if store["store_name"] == "BigBasket" else 0
                    ),
                }
            )

        # Choose best price
        best_price = min(
            store_prices, key=lambda x: x["price"] + x["delivery_fee"]
        )

        item = {
            "name": ingredient["name"],
            "quantity": ingredient["quantity"],
            "estimated_price": best_price["price"],
            "store": best_price["store_name"],
            "category": ingredient["category"],
            "priority": (
                "high"
                if ingredient["category"] in ["Protein", "Breakfast"]
                else "medium"
            ),
            "nutritional_benefits": [
                (
                    "High in fiber"
                    if "oats" in ingredient["name"].lower()
                    else "Quality protein"
                ),
                (
                    "Natural energy source"
                    if "banana" in ingredient["name"].lower()
                    else "Essential nutrients"
                ),
            ],
        }

        smart_shopping_list.append(item)
        total_estimated += best_price["price"]

    # Budget analysis
    budget_analysis = {
        "limit": budget_limit,
        "estimated": total_estimated,
        "savings": budget_limit - total_estimated,
        "status": (
            "Within Budget"
            if total_estimated <= budget_limit
            else "Over Budget"
        ),
    }

    # Store optimization
    store_optimization = {
        "recommended_stores": list(
            set([item["store"] for item in smart_shopping_list])
        ),
        "items_per_store": {},
    }

    for item in smart_shopping_list:
        store = item["store"]
        store_optimization["items_per_store"][store] = (
            store_optimization["items_per_store"].get(store, 0) + 1
        )

    return jsonify(
        {
            "success": True,
            "smart_shopping_list": smart_shopping_list,
            "total_items": len(smart_shopping_list),
            "estimated_total": round(total_estimated, 2),
            "budget_analysis": budget_analysis,
            "store_optimization": store_optimization,
            "user_id": user_id,
            "generated_at": dt.datetime.utcnow().isoformat(),
        }
    )


@app.route("/api/grocery/cart/prepare", methods=["POST"])
def prepare_external_cart():
    """Prepare cart data for external store checkout"""
    try:
        data = request.get_json() or {}
        user_id = data.get("user_id")
        store_name = data.get("store_name")
        items = data.get("items", [])

        if not user_id or not store_name or not items:
            return (
                jsonify(
                    {"error": "User ID, store name, and items are required"}
                ),
                400,
            )

        # Generate cart session
        cart_session = {
            "user_id": user_id,
            "store_name": store_name,
            "items": items,
            "total_amount": sum(
                item.get("price", 0) * item.get("quantity", 1)
                for item in items
            ),
            "created_at": dt.datetime.utcnow(),
            "expires_at": dt.datetime.utcnow() + dt.timedelta(hours=24),
            "status": "active",
        }

        # Save to cart_sessions collection
        result = cart_sessions.insert_one(cart_session)

        # Generate checkout URL (mock)
        checkout_url = f"https://mock-{store_name.lower().replace(' ', '')}.com/checkout?cart={str(result.inserted_id)}"

        # Get delivery options based on store
        delivery_options = {
            "standard": {"time": "2-3 days", "fee": 40},
            "express": {"time": "1 day", "fee": 80},
            "same_day": {"time": "Today", "fee": 120},
        }

        return jsonify(
            {
                "success": True,
                "message": "Cart prepared for checkout",
                "checkout_data": {
                    "cart_id": str(result.inserted_id),
                    "store_name": store_name,
                    "checkout_url": checkout_url,
                    "total_amount": cart_session["total_amount"],
                    "items": items,
                    "delivery_options": delivery_options,
                },
                "cart_session_id": str(result.inserted_id),
                "expires_in": "24 hours",
                "user_id": user_id,
            }
        )

    except Exception as e:
        print(f"Error preparing cart: {e}")
        return (
            jsonify({"error": "Cart preparation failed", "details": str(e)}),
            500,
        )


@app.route("/api/grocery/store-layout", methods=["GET"])
def get_store_layout():
    """Get optimized store layout for efficient shopping"""
    try:
        store_name = request.args.get("store_name", "BigBasket")

        # Mock store layout data
        store_layout = {
            "store_name": store_name,
            "layout_version": "1.0",
            "categories": [
                {
                    "name": "Produce",
                    "section": "A1-A12",
                    "items": [
                        "Fresh Vegetables",
                        "Fresh Fruits",
                        "Herbs",
                        "Organic Produce",
                    ],
                    "average_time": "8 minutes",
                },
                {
                    "name": "Dairy & Eggs",
                    "section": "B1-B8",
                    "items": ["Milk", "Cheese", "Yogurt", "Eggs", "Butter"],
                    "average_time": "5 minutes",
                },
                {
                    "name": "Bakery",
                    "section": "C1-C6",
                    "items": ["Bread", "Pastries", "Cakes", "Baking Supplies"],
                    "average_time": "4 minutes",
                },
                {
                    "name": "Meat & Seafood",
                    "section": "D1-D10",
                    "items": ["Chicken", "Fish", "Beef", "Pork", "Seafood"],
                    "average_time": "6 minutes",
                },
                {
                    "name": "Frozen Foods",
                    "section": "E1-E15",
                    "items": [
                        "Frozen Vegetables",
                        "Ice Cream",
                        "Frozen Meals",
                        "Frozen Desserts",
                    ],
                    "average_time": "7 minutes",
                },
                {
                    "name": "Pantry Staples",
                    "section": "F1-F20",
                    "items": [
                        "Rice",
                        "Pasta",
                        "Canned Goods",
                        "Oils",
                        "Spices",
                    ],
                    "average_time": "10 minutes",
                },
                {
                    "name": "Beverages",
                    "section": "G1-G12",
                    "items": ["Water", "Juices", "Soda", "Tea", "Coffee"],
                    "average_time": "5 minutes",
                },
                {
                    "name": "Snacks",
                    "section": "H1-H8",
                    "items": [
                        "Chips",
                        "Cookies",
                        "Nuts",
                        "Chocolate",
                        "Candy",
                    ],
                    "average_time": "4 minutes",
                },
            ],
            "optimal_path": [
                "Produce (A1-A12)",
                "Dairy & Eggs (B1-B8)",
                "Bakery (C1-C6)",
                "Meat & Seafood (D1-D10)",
                "Frozen Foods (E1-E15)",
                "Pantry Staples (F1-F20)",
                "Beverages (G1-G12)",
                "Snacks (H1-H8)",
            ],
            "estimated_shopping_time": "45-60 minutes",
            "tips": [
                "Start with produce when it's freshest",
                "Check meat and dairy sections for temperature-sensitive items",
                "Save frozen items for last to maintain quality",
                "Group similar items together to minimize backtracking",
                "Use the optimal path to save time and energy",
            ],
        }

        return jsonify(
            {
                "success": True,
                "store_layout": store_layout,
                "generated_at": dt.datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        print(f"Error getting store layout: {e}")
        return (
            jsonify(
                {"error": "Store layout retrieval failed", "details": str(e)}
            ),
            500,
        )


@app.route("/api/grocery/price-history/<product_name>", methods=["GET"])
def get_price_history(product_name):
    """Get price history for a product across stores"""
    try:
        if not product_name:
            return jsonify({"error": "Product name is required"}), 400

        # Get price history for the product
        history_records = list(
            price_history.find(
                {"product_name": {"$regex": product_name, "$options": "i"}}
            )
            .sort("timestamp", -1)
            .limit(50)
        )

        # Convert ObjectId to string
        for record in history_records:
            record["_id"] = str(record["_id"])

        # Analyze by store
        store_analysis = {}
        for record in history_records:
            store_id = record.get("store_id")
            store_doc = store_config.find_one({"_id": store_id})
            store_name = (
                store_doc.get("store_name", "Unknown")
                if store_doc
                else "Unknown"
            )

            if store_name not in store_analysis:
                store_analysis[store_name] = {
                    "prices": [],
                    "avg_price": 0,
                    "min_price": float("inf"),
                    "max_price": 0,
                    "price_trend": "stable",
                }

            price = record.get("price", 0)
            store_analysis[store_name]["prices"].append(price)
            store_analysis[store_name]["min_price"] = min(
                store_analysis[store_name]["min_price"], price
            )
            store_analysis[store_name]["max_price"] = max(
                store_analysis[store_name]["max_price"], price
            )

        # Calculate averages and trends
        for store_name, data in store_analysis.items():
            if data["prices"]:
                data["avg_price"] = round(
                    sum(data["prices"]) / len(data["prices"]), 2
                )

                # Determine trend (simple: compare first and last 5 prices)
                if len(data["prices"]) >= 5:
                    first_avg = sum(data["prices"][:5]) / 5
                    last_avg = sum(data["prices"][-5:]) / 5
                    if last_avg < first_avg * 0.95:
                        data["price_trend"] = "decreasing"
                    elif last_avg > first_avg * 1.05:
                        data["price_trend"] = "increasing"
                    else:
                        data["price_trend"] = "stable"

        # Get date range
        date_range = {
            "from": (
                history_records[-1]["timestamp"].isoformat()
                if history_records
                else None
            ),
            "to": (
                history_records[0]["timestamp"].isoformat()
                if history_records
                else None
            ),
        }

        return jsonify(
            {
                "success": True,
                "product": product_name,
                "price_history": history_records,
                "store_analysis": store_analysis,
                "total_records": len(history_records),
                "date_range": date_range,
                "generated_at": dt.datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        print(f"Error getting price history: {e}")
        return (
            jsonify(
                {"error": "Price history retrieval failed", "details": str(e)}
            ),
            500,
        )


@app.route("/user/<user_id>/mealplan", methods=["POST"])
@app.route("/api/user/<user_id>/mealplan", methods=["POST"])
def generate_meal_plan(user_id):
    user_id = unquote(user_id)
    # Fetch user profile data from USERS or MongoDB
    user_data = USERS.get(user_id, {})
    if not user_data:
        user_doc = users_collection.find_one({"email": user_id})
        if user_doc:
            user_data = user_doc
    if not user_data:
        # Try fetching from MongoDB diet_collection as fallback
        user_doc = diet_collection.find_one({"userId": user_id})
        if user_doc:
            user_data = user_doc.get("user_data", {})

    # Extract user preferences and health data
    diet_type = user_data.get("diet_preference", "balanced").lower()
    if not diet_type or diet_type not in [
        "vegetarian",
        "non vegetarian",
        "vegan",
    ]:
        diet_type = "balanced"

    calorie_goal = user_data.get("target_calories") or 2000  # Default calorie goal
    if "healthData" in user_data and "goal" in user_data["healthData"]:
        calorie_goal = user_data["healthData"]["goal"]

    # Get dietary restrictions and cuisines
    diet_restrictions = user_data.get("diet_restrictions", "")
    cuisines = user_data.get("cuisines", ["Indian", "Asian", "Mediterranean"])

    # Generate comprehensive daily meal plan using SmartBiteModels
    preferences = {
        "diet_type": diet_type,
        "calorie_goal": calorie_goal,
        "diet_restrictions": diet_restrictions,
        "cuisines": cuisines,
    }

    daily_meals = smartbite_models.get_recommendations(preferences).get(
        "daily_meals", {}
    )

    # Debug logging for daily meals
    print(f"DEBUG: Daily meals received: {daily_meals}")

    # Validate daily meals structure
    if not daily_meals or not isinstance(daily_meals, dict):
        return (
            jsonify(
                {
                    "error": "Failed to generate meal plan - invalid daily meals data"
                }
            ),
            500,
        )

    # Create sophisticated weekly meal plan with variety and recipes
    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    weekly_meal_plan = []

    # Track used meals to ensure variety
    used_meals = {"breakfast": set(), "lunch": set(), "dinner": set()}

    for day in days:
        day_meals = {"breakfast": None, "lunch": None, "dinner": None}

        # Generate meals for each meal type with variety
        for meal_type in ["breakfast", "lunch", "dinner"]:
            available_meals = daily_meals.get(meal_type, [])

            if available_meals:
                # Filter out recently used meals for variety (avoid same meal within 3 days)
                available_meals = [
                    meal
                    for meal in available_meals
                    if meal.get("name", "") not in used_meals[meal_type]
                ]

                if not available_meals:
                    # If all meals used recently, reset and use any available
                    available_meals = daily_meals.get(meal_type, [])

                # Randomly select a meal for variety
                selected_meal = (
                    random.choice(available_meals) if available_meals else {}
                )

                # Mark meal as used
                meal_name = selected_meal.get("name", "")
                if meal_name:
                    used_meals[meal_type].add(meal_name)
                    # Keep only last 3 used meals to allow rotation
                    if len(used_meals[meal_type]) > 3:
                        used_meals[meal_type].pop()

                # Enhance meal with recipe details
                enhanced_meal = {
                    "name": selected_meal.get(
                        "name", f"{meal_type.title()} Meal"
                    ),
                    "calories": selected_meal.get("calories", 0),
                    "protein_grams": selected_meal.get("protein_grams", 0),
                    "carb_grams": selected_meal.get("carb_grams", 0),
                    "fat_grams": selected_meal.get("fat_grams", 0),
                    "ingredients": selected_meal.get("ingredients", []),
                    "prep_time": selected_meal.get("prep_time", 0),
                    "cook_time": selected_meal.get("cook_time", 0),
                    "instructions": selected_meal.get("instructions", []),
                    "meal_type": meal_type,
                }

                day_meals[meal_type] = enhanced_meal
            else:
                # Fallback meal if no meals available
                day_meals[meal_type] = {
                    "name": f"Sample {meal_type.title()} Meal",
                    "calories": calorie_goal // 3,
                    "protein_grams": 20,
                    "carb_grams": 40,
                    "fat_grams": 15,
                    "ingredients": ["Basic ingredients for a healthy meal"],
                    "prep_time": 10,
                    "cook_time": 15,
                    "instructions": [
                        "Prepare ingredients",
                        "Cook according to preference",
                        "Serve hot",
                    ],
                    "meal_type": meal_type,
                }

        # Calculate daily totals
        daily_calories = sum(
            meal["calories"] for meal in day_meals.values() if meal
        )
        daily_protein = sum(
            meal["protein_grams"] for meal in day_meals.values() if meal
        )
        daily_carbs = sum(
            meal["carb_grams"] for meal in day_meals.values() if meal
        )
        daily_fat = sum(
            meal["fat_grams"] for meal in day_meals.values() if meal
        )

        weekly_meal_plan.append(
            {
                "day": day,
                "meals": day_meals,
                "daily_totals": {
                    "calories": daily_calories,
                    "protein_grams": daily_protein,
                    "carb_grams": daily_carbs,
                    "fat_grams": daily_fat,
                },
                "target_calories": calorie_goal,
            }
        )

    # Calculate weekly nutritional summary
    weekly_summary = {
        "total_calories": sum(
            day["daily_totals"]["calories"] for day in weekly_meal_plan
        ),
        "avg_daily_calories": sum(
            day["daily_totals"]["calories"] for day in weekly_meal_plan
        )
        / 7,
        "total_protein": sum(
            day["daily_totals"]["protein_grams"] for day in weekly_meal_plan
        ),
        "total_carbs": sum(
            day["daily_totals"]["carb_grams"] for day in weekly_meal_plan
        ),
        "total_fat": sum(
            day["daily_totals"]["fat_grams"] for day in weekly_meal_plan
        ),
        "target_weekly_calories": calorie_goal * 7,
        "meals_generated": len(
            [
                meal
                for day in weekly_meal_plan
                for meal in day["meals"].values()
                if meal
            ]
        ),
    }

    # Store enhanced meal plan in user data
    enhanced_plan = {
        "weekly_plan": weekly_meal_plan,
        "weekly_summary": weekly_summary,
        "user_preferences": preferences,
        "generated_at": dt.datetime.utcnow().isoformat(),
    }

    user_data["mealPlanner"] = enhanced_plan
    USERS[user_id] = user_data

    # Store in meal_planner collection with enhanced structure
    meal_planner.update_one(
        {"userId": user_id},
        {
            "$set": {
                "plan": weekly_meal_plan,
                "weekly_summary": weekly_summary,
                "user_preferences": preferences,
                "generatedAt": dt.datetime.utcnow(),
            }
        },
        upsert=True,
    )

    return jsonify(
        {
            "message": "Enhanced personalized meal plan generated with recipes and nutritional balance",
            "mealPlanner": enhanced_plan,
            "weekly_summary": weekly_summary,
        }
    )


@app.route("/api/mealplanner/<user_id>/edit", methods=["POST"])
def edit_meal_plan(user_id):
    data = request.get_json()
    index = data.get("index")
    item = data.get("item")

    if index is None or item is None:
        return jsonify({"error": "Index and item data are required"}), 400

    # Fetch current diet plan from DB (since diet_plan.html uses diet_collection)
    doc = diet_collection.find_one({"userId": user_id})
    if not doc:
        return jsonify({"error": "Diet plan not found"}), 404

    # Get the flat meals list
    meals = doc.get("meals", [])

    if index < 0 or index >= len(meals):
        return jsonify({"error": "Invalid index"}), 400

    # Update the item at the index
    meals[index] = {
        "Shrt_Desc": item.get("Shrt_Desc", "Unknown"),
        "Energ_Kcal": item.get("Energ_Kcal", 0),
        "Protein_(g)": item.get("Protein_(g)", 0),
        "Carbohydrt_(g)": item.get("Carbohydrt_(g)", 0),
        "Lipid_Tot_(g)": item.get("Lipid_Tot_(g)", 0),
    }

    # Update DB
    diet_collection.update_one(
        {"userId": user_id},
        {"$set": {"meals": meals, "updatedAt": dt.datetime.utcnow()}},
        upsert=True,
    )

    return jsonify({"message": "Meal updated successfully"})


@app.route("/api/diet_plan/<user_id>/add_custom_recipe", methods=["POST"])
def add_custom_recipe(user_id):
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    recipe_id = str(uuid.uuid4())
    recipe = make_custom_recipe_dict(data, recipe_id)

    # Add to custom_recipes array in diet_collection
    diet_collection.update_one(
        {"userId": user_id},
        {"$push": {"custom_recipes": recipe}},
        upsert=True,
    )

    return jsonify({"message": "Custom recipe added successfully", "recipe_id": recipe_id})


@app.route("/api/diet_plan/<user_id>/edit_custom_recipe", methods=["POST"])
def edit_custom_recipe(user_id):
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    recipe_id = data.get("recipe_id")
    if not recipe_id:
        return jsonify({"error": "Recipe ID is required"}), 400

    updated_recipe = make_custom_recipe_dict(data, recipe_id, added_at=data.get("added_at"))
    updated_recipe["updated_at"] = dt.datetime.utcnow()

    # Update the specific custom recipe in the array
    diet_collection.update_one(
        {"userId": user_id, "custom_recipes._id": recipe_id},
        {"$set": {"custom_recipes.$": updated_recipe}},
    )

    return jsonify({"message": "Custom recipe updated successfully"})


@app.route("/api/grocery/<user_id>/pantry/add", methods=["POST"])
def add_to_pantry(user_id):
    # URL decode the user_id parameter
    user_id = unquote(user_id)

    data = request.get_json()
    item = data.get("item")

    if not item:
        return jsonify({"error": "Item name required"}), 400

    # Add to pantry
    grocery_data.update_one(
        {"userId": user_id}, {"$addToSet": {"pantry": item}}, upsert=True
    )

    return jsonify({"message": "Item added to pantry"})


@app.route("/api/grocery/<user_id>/shopping/add", methods=["POST"])
def add_to_shopping_list(user_id):
    # URL decode the user_id parameter
    user_id = unquote(user_id)

    data = request.get_json()
    item = data.get("item")

    if not item:
        return jsonify({"error": "Item name required"}), 400

    # Add to shopping list
    grocery_data.update_one(
        {"userId": user_id}, {"$addToSet": {"shoppingList": item}}, upsert=True
    )

    return jsonify({"message": "Item added to shopping list"})


@app.route("/api/grocery/<user_id>/move", methods=["POST"])
def move_to_pantry(user_id):
    # URL decode the user_id parameter
    user_id = unquote(user_id)

    data = request.get_json()
    item = data.get("item")

    if not item:
        return jsonify({"error": "Item name required"}), 400

    # Remove from shopping list and add to pantry
    grocery_data.update_one(
        {"userId": user_id},
        {"$pull": {"shoppingList": item}, "$addToSet": {"pantry": item}},
    )

    return jsonify({"message": "Item moved to pantry"})


@app.route("/api/grocery/<user_id>/pantry/remove", methods=["POST"])
def remove_from_pantry(user_id):
    """Remove item from user's pantry"""
    try:
        # URL decode the user_id parameter
        user_id = unquote(user_id)

        data = request.get_json()
        item = data.get("item")

        if not item:
            return jsonify({"error": "Item name required"}), 400

        # Remove from pantry
        grocery_data.update_one(
            {"userId": user_id}, {"$pull": {"pantry": item}}
        )

        return jsonify({"message": "Item removed from pantry"})

    except Exception as e:
        print(f"Error removing from pantry: {e}")
        return (
            jsonify(
                {
                    "error": "Failed to remove item from pantry",
                    "details": str(e),
                }
            ),
            500,
        )


@app.route("/api/grocery/<user_id>/shopping/remove", methods=["POST"])
def remove_from_shopping(user_id):
    """Remove item from user's shopping list"""
    try:
        # URL decode the user_id parameter
        user_id = unquote(user_id)

        data = request.get_json()
        item = data.get("item")

        if not item:
            return jsonify({"error": "Item name required"}), 400

        # Remove from shopping list
        grocery_data.update_one(
            {"userId": user_id}, {"$pull": {"shoppingList": item}}
        )

        return jsonify({"message": "Item removed from shopping list"})

    except Exception as e:
        print(f"Error removing from shopping list: {e}")
        return (
            jsonify(
                {
                    "error": "Failed to remove item from shopping list",
                    "details": str(e),
                }
            ),
            500,
        )


@app.route("/api/activity/<user_id>/add", methods=["POST"])
def add_activity(user_id):
    data = request.get_json()
    activity = data.get("activity")

    if not activity:
        return jsonify({"error": "Activity required"}), 400

    # Add to activity log
    activity_log.update_one(
        {"userId": user_id},
        {
            "$push": {
                "activities": {
                    "activity": activity,
                    "timestamp": dt.datetime.utcnow(),
                }
            }
        },
        upsert=True,
    )

    return jsonify({"message": "Activity added"})


# ------------------- SmartBite AI API Endpoints -------------------


@app.route("/api/ai/progress/<user_id>", methods=["POST"])
def predict_user_progress(user_id):
    """Predict user fitness progress using SmartBite AI"""
    try:
        data = request.get_json() or {}

        # Get user data for prediction
        user_data = USERS.get(user_id, {})
        health_data = user_data.get("healthData", {})

        # Prepare input for prediction
        progress_data = {
            "current_calories": health_data.get("calories", 0),
            "goal_calories": health_data.get("goal", 2500),
            "weekly_calories": health_data.get("weeklyCalories", []),
            "water_intake": health_data.get("waterIntake", 0),
            "activity_count": len(user_data.get("activityLog", [])),
            "days_active": len(
                [c for c in health_data.get("weeklyCalories", []) if c > 0]
            ),
        }

        # Use SmartBite model for prediction
        if (
            hasattr(smartbite_models, "predictor")
            and smartbite_models.predictor
        ):
            prediction = smartbite_models.predictor.predict_progress(
                progress_data
            )
            return jsonify(
                {"success": True, "prediction": prediction, "user_id": user_id}
            )
        else:
            # Fallback prediction logic
            avg_calories = sum(progress_data["weekly_calories"]) / max(
                len(progress_data["weekly_calories"]), 1
            )
            progress_score = (
                (progress_data["goal_calories"] - avg_calories)
                / progress_data["goal_calories"]
                * 100
            )

            return jsonify(
                {
                    "success": True,
                    "prediction": {
                        "progress_score": max(0, min(100, progress_score)),
                        "goal_achievement": (
                            "On Track"
                            if progress_score > 80
                            else "Needs Improvement"
                        ),
                        "recommendations": [
                            "Maintain consistent calorie tracking",
                            (
                                "Increase daily activity"
                                if progress_data["activity_count"] < 3
                                else "Good activity level"
                            ),
                            (
                                "Improve water intake"
                                if progress_data["water_intake"] < 8
                                else "Good hydration"
                            ),
                        ],
                    },
                    "user_id": user_id,
                }
            )

    except Exception as e:
        print(f"Error predicting user progress: {e}")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500


@app.route("/api/ai/nutrition/<user_id>", methods=["POST"])
def get_nutrition_recommendations(user_id):
    """Get personalized nutrition recommendations"""
    try:
        # Get user preferences and health data
        user_data = USERS.get(user_id, {})
        health_data = user_data.get("healthData", {})

        preferences = {
            "diet_type": "balanced",
            "allergies": [],
            "calorie_goal": health_data.get("goal", 2500),
            "macros_goal": health_data.get(
                "macros", {"carbs": 50, "protein": 30, "fat": 20}
            ),
            "budget_priority": user_data.get("budget", {}).get(
                "priority", "nutrition"
            ),
        }

        # Use SmartBite nutrition recommender
        if (
            hasattr(smartbite_models, "recommender")
            and smartbite_models.recommender
        ):
            recommendations = smartbite_models.recommender.get_recommendations(
                preferences
            )
            return jsonify(
                {
                    "success": True,
                    "recommendations": recommendations,
                    "user_id": user_id,
                }
            )
        else:
            # Fallback recommendations
            fallback_recs = {
                "daily_meals": [
                    {
                        "meal": "Breakfast",
                        "calories": 500,
                        "protein": 25,
                        "carbs": 60,
                        "fat": 15,
                    },
                    {
                        "meal": "Lunch",
                        "calories": 600,
                        "protein": 35,
                        "carbs": 70,
                        "fat": 20,
                    },
                    {
                        "meal": "Dinner",
                        "calories": 550,
                        "protein": 30,
                        "carbs": 65,
                        "fat": 18,
                    },
                    {
                        "meal": "Snacks",
                        "calories": 300,
                        "protein": 15,
                        "carbs": 40,
                        "fat": 10,
                    },
                ],
                "weekly_plan": [
                    "Focus on whole grains and lean proteins",
                    "Include colorful vegetables in every meal",
                    "Stay hydrated with at least 8 glasses of water",
                    "Limit processed foods and sugary drinks",
                ],
                "shopping_list": [
                    "Chicken breast, fish, or plant-based proteins",
                    "Brown rice, quinoa, or whole grain bread",
                    "Leafy greens, broccoli, carrots",
                    "Fruits like apples, berries, bananas",
                    "Nuts, seeds, and healthy fats",
                ],
            }

            return jsonify(
                {
                    "success": True,
                    "recommendations": fallback_recs,
                    "user_id": user_id,
                }
            )

    except Exception as e:
        print(f"Error getting nutrition recommendations: {e}")
        return (
            jsonify({"error": "Recommendation failed", "details": str(e)}),
            500,
        )


@app.route("/api/ai/yoga-pose", methods=["POST"])
def classify_yoga_pose():
    """Classify yoga pose from image data"""
    try:
        data = request.get_json() or {}

        # Expect base64 image data or image URL
        image_data = data.get("image_data")
        image_url = data.get("image_url")

        if not image_data and not image_url:
            return jsonify({"error": "Image data or URL required"}), 400

        # Use SmartBite yoga pose classifier
        if (
            hasattr(smartbite_models, "classifier")
            and smartbite_models.classifier
        ):
            result = smartbite_models.classifier.classify_pose(
                image_data or image_url
            )

            return jsonify(
                {
                    "success": True,
                    "classification": result,
                    "timestamp": dt.datetime.utcnow().isoformat(),
                }
            )
        else:
            # Fallback classification
            return jsonify(
                {
                    "success": True,
                    "classification": {
                        "pose": "Unknown Pose",
                        "confidence": 0.0,
                        "feedback": "AI model not available - please ensure proper form",
                        "corrections": [
                            "Keep your back straight",
                            "Align your shoulders with your hips",
                            "Breathe steadily throughout the pose",
                        ],
                    },
                    "timestamp": dt.datetime.utcnow().isoformat(),
                }
            )

    except Exception as e:
        print(f"Error classifying yoga pose: {e}")
        return (
            jsonify({"error": "Classification failed", "details": str(e)}),
            500,
        )


@app.route("/api/ai/health-insights/<user_id>", methods=["GET"])
def get_health_insights(user_id):
    """Get comprehensive health insights for user"""
    try:
        # Get user data
        user_data = USERS.get(user_id, {})
        health_data = user_data.get("healthData", {})

        # Calculate insights
        weekly_calories = health_data.get("weeklyCalories", [])
        avg_calories = (
            sum(weekly_calories) / max(len(weekly_calories), 1)
            if weekly_calories
            else 0
        )
        goal_calories = health_data.get("goal", 2500)

        # Use SmartBite models for insights
        insights = {
            "calorie_analysis": {
                "average_daily": round(avg_calories, 1),
                "goal": goal_calories,
                "achievement_rate": (
                    round((avg_calories / goal_calories * 100), 1)
                    if goal_calories > 0
                    else 0
                ),
                "status": (
                    "On Track"
                    if abs(avg_calories - goal_calories) / goal_calories < 0.1
                    else "Needs Adjustment"
                ),
            },
            "nutrition_score": 85,  # This would come from SmartBite models
            "activity_level": "Moderate",  # This would come from SmartBite models
            "recommendations": [
                "Consider increasing protein intake for better muscle recovery",
                "Add more fiber-rich foods to your diet",
                "Maintain consistent meal timing for better metabolism",
            ],
        }

        # Enhance with SmartBite AI if available
        if (
            hasattr(smartbite_models, "predictor")
            and smartbite_models.predictor
        ):
            try:
                ai_insights = smartbite_models.predictor.get_health_insights(
                    user_data
                )
                insights.update(ai_insights)
            except Exception as e:
                print(f"AI insights failed: {e}")

        return jsonify(
            {
                "success": True,
                "insights": insights,
                "user_id": user_id,
                "generated_at": dt.datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        print(f"Error getting health insights: {e}")
        return (
            jsonify(
                {"error": "Insights generation failed", "details": str(e)}
            ),
            500,
        )


@app.route("/api/ai/recipes/generate", methods=["POST"])
def generate_recipes():
    """Generate recipe suggestions based on available ingredients"""
    try:
        data = request.get_json() or {}

        # Extract parameters
        available_ingredients = data.get("ingredients", [])
        user_preferences = data.get("preferences", {})
        skill_level = data.get("skill_level", "intermediate")
        num_recipes = data.get("num_recipes", 5)

        # Validate input
        if not available_ingredients:
            return (
                jsonify({"error": "Available ingredients are required"}),
                400,
            )

        # Use SmartBite recipe generator
        if (
            hasattr(smartbite_models, "recipe_generator")
            and smartbite_models.recipe_generator
        ):
            try:
                # Generate recipes
                recipe_suggestions = (
                    smartbite_models.recipe_generator.generate_recipes(
                        available_ingredients=available_ingredients,
                        user_preferences=user_preferences,
                        skill_level=skill_level,
                        num_recipes=num_recipes,
                    )
                )

                # Format response
                formatted_recipes = []
                for suggestion in recipe_suggestions:
                    recipe = suggestion.get("recipe", {})
                    formatted_recipes.append(
                        {
                            "name": recipe.get("Name", "Unknown Recipe"),
                            "match_score": suggestion.get("match_score", 0),
                            "missing_ingredients": suggestion.get(
                                "missing_ingredients", []
                            ),
                            "ingredients": recipe.get(
                                "RecipeIngredientParts", []
                            ),
                            "instructions": recipe.get(
                                "RecipeInstructions", []
                            ),
                            "prep_time": recipe.get("PrepTime", "Unknown"),
                            "cook_time": recipe.get("CookTime", "Unknown"),
                            "total_time": recipe.get("total_time_minutes", 0),
                            "calories": recipe.get("Calories", 0),
                            "protein": recipe.get("ProteinContent", 0),
                            "carbs": recipe.get("CarbohydrateContent", 0),
                            "fat": recipe.get("FatContent", 0),
                            "complexity": recipe.get("complexity_score", 5),
                            "category": recipe.get(
                                "RecipeCategory", "Unknown"
                            ),
                        }
                    )

                return jsonify(
                    {
                        "success": True,
                        "recipes": formatted_recipes,
                        "total_found": len(formatted_recipes),
                        "search_criteria": {
                            "ingredients": available_ingredients,
                            "skill_level": skill_level,
                            "preferences": user_preferences,
                        },
                        "generated_at": dt.datetime.utcnow().isoformat(),
                    }
                )

            except Exception as e:
                print(f"Recipe generation failed: {e}")
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "Recipe generation failed",
                            "details": str(e),
                            "fallback_recipes": get_fallback_recipes(
                                available_ingredients, skill_level
                            ),
                        }
                    ),
                    500,
                )
        else:
            # Fallback recipe generation
            fallback_recipes = get_fallback_recipes(
                available_ingredients, skill_level
            )
            return jsonify(
                {
                    "success": True,
                    "recipes": fallback_recipes,
                    "total_found": len(fallback_recipes),
                    "message": "Using fallback recipe suggestions",
                    "generated_at": dt.datetime.utcnow().isoformat(),
                }
            )

    except Exception as e:
        print(f"Error generating recipes: {e}")
        return (
            jsonify({"error": "Recipe generation failed", "details": str(e)}),
            500,
        )


@app.route("/api/recipes/search", methods=["GET"])
def search_recipes():
    """Search recipes by name or ingredients"""
    try:
        query = request.args.get("q", "").strip().lower()

        if not query:
            return jsonify({"success": False, "error": "Search query is required"}), 400

        if recipes_df.empty:
            return jsonify({
                "success": False,
                "error": "Recipe database not available",
                "recipes": []
            }), 503

        matching_recipes = []

        # Search through recipes dataset
        for _, row in recipes_df.iterrows():
            # Get recipe name
            recipe_name = ""
            for col in recipes_df.columns:
                if col.lower() in ["name", "recipe_name", "title"]:
                    recipe_name = str(row.get(col, "")).lower()
                    break

            if not recipe_name:
                continue

            # Check if query matches recipe name
            name_match = query in recipe_name

            # Check if query matches ingredients
            ingredients_match = False
            ingredients_str = str(row.get("RecipeIngredientParts", "[]"))
            try:
                ingredients_list = json.loads(ingredients_str) if ingredients_str.startswith("[") else eval(ingredients_str)
                if isinstance(ingredients_list, list):
                    ingredients_match = any(query in str(ing).lower() for ing in ingredients_list)
            except:
                # Fallback: search in string representation
                ingredients_match = query in ingredients_str.lower()

            if name_match or ingredients_match:
                # Parse ingredients
                ingredients = []
                try:
                    if ingredients_str.startswith("["):
                        ingredients = json.loads(ingredients_str)
                    else:
                        ingredients = eval(ingredients_str)
                    if not isinstance(ingredients, list):
                        ingredients = []
                    ingredients = [str(i) for i in ingredients]
                except:
                    ingredients = []

                # Parse instructions
                instructions = []
                instructions_str = str(row.get("RecipeInstructions", "[]"))
                try:
                    if instructions_str.startswith("["):
                        instructions = json.loads(instructions_str)
                    else:
                        instructions = eval(instructions_str)
                    if not isinstance(instructions, list):
                        instructions = []
                    instructions = [str(i) for i in instructions]
                except:
                    instructions = []

                recipe = {
                    "name": str(row.get("Name", row.get("RecipeName", "Unknown Recipe"))),
                    "description": str(row.get("Description", "")),
                    "category": str(row.get("RecipeCategory", "")),
                    "prep_time": str(row.get("PrepTime", "")),
                    "cook_time": str(row.get("CookTime", "")),
                    "total_time": str(row.get("TotalTime", "")),
                    "servings": str(row.get("RecipeServings", "")),
                    "calories": str(row.get("Calories", "")),
                    "protein": str(row.get("ProteinContent", "")),
                    "carbs": str(row.get("CarbohydrateContent", "")),
                    "fat": str(row.get("FatContent", "")),
                    "ingredients": ingredients,
                    "instructions": instructions,
                    "complexity": 5,  # Default complexity
                    "rating": str(row.get("AggregatedRating", "")),
                    "review_count": str(row.get("ReviewCount", "")),
                }

                matching_recipes.append(recipe)

                # Limit results to prevent overwhelming response
                if len(matching_recipes) >= 20:
                    break

        return jsonify({
            "success": True,
            "recipes": matching_recipes,
            "total_found": len(matching_recipes),
            "query": query
        })

    except Exception as e:
        print(f"Error searching recipes: {e}")
        return jsonify({
            "success": False,
            "error": "Recipe search failed",
            "details": str(e),
            "recipes": []
        }), 500


@app.route("/api/ai/recipes/details/<recipe_name>", methods=["GET"])
def get_recipe_details(recipe_name):
    """Get detailed information for a specific recipe"""
    try:
        if not recipe_name:
            return jsonify({"error": "Recipe name is required"}), 400

        # Use SmartBite recipe generator
        if (
            hasattr(smartbite_models, "recipe_generator")
            and smartbite_models.recipe_generator
        ):
            try:
                recipe_details = (
                    smartbite_models.recipe_generator.get_recipe_details(
                        recipe_name
                    )
                )

                if recipe_details:
                    return jsonify(
                        {
                            "success": True,
                            "recipe": recipe_details,
                            "generated_at": dt.datetime.utcnow().isoformat(),
                        }
                    )
                else:
                    return (
                        jsonify(
                            {
                                "success": False,
                                "error": "Recipe not found",
                                "searched_name": recipe_name,
                            }
                        ),
                        404,
                    )

            except Exception as e:
                print(f"Recipe details retrieval failed: {e}")
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "Recipe details retrieval failed",
                            "details": str(e),
                        }
                    ),
                    500,
                )
        else:
            # Fallback recipe details
            fallback_details = get_fallback_recipe_details(recipe_name)
            return jsonify(
                {
                    "success": True,
                    "recipe": fallback_details,
                    "message": "Using fallback recipe details",
                    "generated_at": dt.datetime.utcnow().isoformat(),
                }
            )

    except Exception as e:
        print(f"Error getting recipe details: {e}")
        return (
            jsonify(
                {"error": "Recipe details retrieval failed", "details": str(e)}
            ),
            500,
        )


def get_fallback_recipes(available_ingredients, skill_level):
    """Generate fallback recipe suggestions when AI model is not available"""
    fallback_recipes = [
        {
            "name": "Simple Vegetable Stir Fry",
            "match_score": 85,
            "missing_ingredients": [],
            "ingredients": ["vegetables", "soy sauce", "garlic", "oil"],
            "instructions": [
                "Heat oil in a pan",
                "Add chopped vegetables",
                "Stir fry for 5-7 minutes",
                "Add soy sauce and garlic",
                "Cook for 2 more minutes",
                "Serve hot",
            ],
            "prep_time": "10 minutes",
            "cook_time": "10 minutes",
            "total_time": 20,
            "calories": 150,
            "protein": 5,
            "carbs": 20,
            "fat": 8,
            "complexity": 2,
            "category": "Main Course",
        },
        {
            "name": "Basic Pasta",
            "match_score": 70,
            "missing_ingredients": ["pasta"],
            "ingredients": ["pasta", "tomato sauce", "cheese", "herbs"],
            "instructions": [
                "Boil pasta according to package instructions",
                "Heat tomato sauce",
                "Mix pasta with sauce",
                "Top with cheese and herbs",
                "Serve immediately",
            ],
            "prep_time": "5 minutes",
            "cook_time": "15 minutes",
            "total_time": 20,
            "calories": 300,
            "protein": 12,
            "carbs": 45,
            "fat": 10,
            "complexity": 1,
            "category": "Main Course",
        },
    ]

    # Filter recipes based on available ingredients
    filtered_recipes = []
    for recipe in fallback_recipes:
        missing_count = len(recipe.get("missing_ingredients", []))
        if (
            missing_count <= 1
        ):  # Only include recipes with at most 1 missing ingredient
            filtered_recipes.append(recipe)

    return filtered_recipes


def get_fallback_recipe_details(recipe_name):
    """Get fallback recipe details when AI model is not available"""
    return {
        "name": recipe_name,
        "description": f"A simple and delicious {recipe_name.lower()} recipe",
        "ingredients": ["Basic ingredients", "Seasonings", "Oil or butter"],
        "instructions": [
            "Prepare all ingredients",
            "Follow basic cooking steps",
            "Cook until done",
            "Serve hot",
        ],
        "prep_time": "15 minutes",
        "cook_time": "20 minutes",
        "total_time": 35,
        "servings": 4,
        "calories": 250,
        "protein": 15,
        "carbs": 30,
        "fat": 12,
        "complexity": 3,
        "category": "Main Course",
    }


# ------------------- Run -------------------
if __name__ == "__main__":

    print("Starting SmartBite Backend...")

    print(f"Loaded {len(datasets)} datasets")

    print(f"ML Model: {'Trained' if model else 'Not available'}")

    print(f"MongoDB: {MONGO_URI}")

    print(f"Server: http://0.0.0.0:{APP_PORT}")

    # Fix for Windows socket error during reload
    use_reloader = os.name != 'nt'
    app.run(host="0.0.0.0", port=APP_PORT, debug=DEBUG, use_reloader=use_reloader)
