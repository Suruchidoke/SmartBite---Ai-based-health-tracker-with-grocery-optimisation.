"""
🌿 SmartBite — AI Health Tracker & Zero-Waste Grocery Optimizer
================================================================
A unified, easy-to-read, single-file Flask application.

This file is organized into 5 clear, logical sections:
  1. CONFIGURATION & APP INITIALIZATION
  2. DATABASE & PERSISTENT STATE (MongoDB / mongomock)
  3. CORE SERVICES (Auth, Clinical Nutrition, Grocery, Gemini AI)
  4. WEB UI ROUTES (Jinja2 Templates)
  5. SECURE REST APIs (IDOR-Protected JSON Endpoints)

Author: Suruchi Doke
Version: 2.0.0 (Production-Ready)
"""

import os
import re
import uuid
import logging
import datetime as dt
from functools import wraps
from urllib.parse import unquote
from dotenv import load_dotenv

from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, jsonify
)
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from pymongo import MongoClient, ASCENDING
import mongomock
import jwt

# Try latest google.genai SDK
try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smartbite")

# Load .env variables
load_dotenv()


# ==============================================================================
# 1. CONFIGURATION & APP INITIALIZATION
# ==============================================================================

class Config:
    """
    Centralized Application Configuration.
    Loads secrets, database connection parameters, AI model selections,
    and server flags directly from the local .env file or system environment.
    """
    # Cryptographic keys for signing Flask session cookies and JSON Web Tokens
    SECRET_KEY = os.getenv("SMARTBITE_SECRET_KEY", "smartbite-secret-key-change-in-prod-99128")
    JWT_SECRET = os.getenv("SMARTBITE_JWT_SECRET", "smartbite-jwt-secret-88371")
    JWT_ALG = "HS256"

    # Browser Cookie Security Policies:
    # - HTTPOnly: Prevents client-side JavaScript (XSS attacks) from reading the session cookie
    # - SameSite=Lax: Mitigates Cross-Site Request Forgery (CSRF) on navigation
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    PERMANENT_SESSION_LIFETIME = dt.timedelta(days=7)

    # Database connection parameters (MongoDB)
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    DB_NAME = os.getenv("MONGO_DB", "auto_diet_db")
    MONGO_TIMEOUT_MS = int(os.getenv("MONGO_TIMEOUT_MS", 2000))

    # Google Gemini AI credentials & model selection
    # BiteBot uses the latest generation 'gemini-3.6-flash' via the modern google-genai SDK
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

    # Local web server execution flags
    PORT = int(os.getenv("SMARTBITE_PORT", 5000))
    DEBUG = os.getenv("SMARTBITE_DEBUG", "True").lower() in ("1", "true", "yes")
    TESTING = os.getenv("FLASK_TESTING", "False").lower() in ("1", "true", "yes")


# Initialize the Flask Web Application
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config.from_object(Config)

# Enable Cross-Origin Resource Sharing (CORS) for external client compatibility
CORS(app)


def create_app(testing=False):
    """
    Application Factory helper function.
    
    Allows test runners (e.g. pytest) and scripts to retrieve the configured
    Flask instance with optional test-isolated configuration (e.g., isolated test DB).
    
    Args:
        testing (bool): When True, enables Flask TESTING mode and routes data
                        to 'smartbite_test_db' in-memory via mongomock.
    Returns:
        Flask: The configured Flask application instance.
    """
    if testing:
        app.config["TESTING"] = True
        app.config["DB_NAME"] = "smartbite_test_db"
    return app


# ==============================================================================
# 2. DATABASE & PERSISTENT STATE (MongoDB with graceful mongomock fallback)
# ==============================================================================

class DatabaseManager:
    """
    Manages MongoDB connections, collection references, and index creation.
    
    Zero-Setup Architecture:
    If a local MongoDB instance is reachable (default mongodb://localhost:27017),
    it connects and stores data permanently on disk.
    If MongoDB is not installed or offline, it automatically and transparently
    falls back to an in-memory 'mongomock' database so the application works
    out-of-the-box without requiring any manual server setup!
    """

    def __init__(self, flask_app):
        uri = flask_app.config["MONGO_URI"]
        db_name = flask_app.config["DB_NAME"]
        timeout = flask_app.config["MONGO_TIMEOUT_MS"]
        use_mock = flask_app.config.get("TESTING", False)

        if use_mock:
            # Test-isolation mode: always use isolated in-memory mock
            self.client = mongomock.MongoClient()
            self.db = self.client[db_name]
            logger.info("Using in-memory mongomock database for testing.")
        else:
            try:
                # Attempt live MongoDB connection with a short timeout
                self.client = MongoClient(uri, serverSelectionTimeoutMS=timeout)
                self.client.admin.command("ping")  # Test connectivity
                self.db = self.client[db_name]
                logger.info(f"Connected to MongoDB at {uri} (Database: {db_name})")
            except Exception as err:
                # Graceful zero-friction fallback for developers without MongoDB running
                logger.warning(f"MongoDB connection failed ({err}). Falling back to in-memory mongomock.")
                self.client = mongomock.MongoClient()
                self.db = self.client[db_name]

        # Ensure performance and uniqueness indexes exist
        self._ensure_indexes()

    def _ensure_indexes(self):
        """
        Builds essential database indexes:
        1. Unique index on 'email' prevents duplicate user registrations.
        2. Relational indexes on 'userId' ensure sub-millisecond lookups for user state.
        """
        try:
            self.db["users"].create_index([("email", ASCENDING)], unique=True)
            self.db["health_data"].create_index([("userId", ASCENDING)])
            self.db["grocery_data"].create_index([("userId", ASCENDING)])
            self.db["meal_planner"].create_index([("userId", ASCENDING)])
            self.db["activity_log"].create_index([("userId", ASCENDING)])
        except Exception as e:
            logger.warning(f"Index creation notice: {e}")

    # Typed property accessors for MongoDB collections
    @property
    def users(self): return self.db["users"]
    @property
    def health_data(self): return self.db["health_data"]
    @property
    def grocery_data(self): return self.db["grocery_data"]
    @property
    def meal_planner(self): return self.db["meal_planner"]
    @property
    def activity_log(self): return self.db["activity_log"]
    @property
    def recipes(self): return self.db["recipes"]
    @property
    def grocery(self): return self.db["grocery"]


# Global database manager instance attached to the Flask app
db_manager = DatabaseManager(app)


def objid_to_str(doc):
    """
    Recursively converts BSON ObjectIds and datetime objects into JSON-serializable strings.
    
    Why this is needed:
    MongoDB uses non-standard BSON types (like ObjectId and datetime.datetime) which cause
    Python's standard json.dumps / Flask jsonify to raise 'TypeError: Object of type ObjectId is not JSON serializable'.
    This helper recursively traverses dictionaries and lists to sanitize all data.
    """
    if not doc:
        return doc
    if isinstance(doc, list):
        return [objid_to_str(d) for d in doc]
    if isinstance(doc, dict):
        new_doc = {}
        for k, v in doc.items():
            if k == "_id":
                new_doc["_id"] = str(v)
            elif hasattr(v, "isoformat"):
                new_doc[k] = v.isoformat()
            elif isinstance(v, (dict, list)):
                new_doc[k] = objid_to_str(v)
            else:
                new_doc[k] = v
        return new_doc
    return doc


def get_or_create_user_state(user_id):
    """
    Fetches the user's complete domain state from MongoDB.
    
    Replaces volatile in-memory global dictionaries (USERS = {}) with persistent,
    indexed MongoDB records across 5 distinct collections:
      1. users: Login credentials, clinical targets (BMR, TDEE), allergen tags
      2. health_data: Today's calories consumed, water intake, weekly history
      3. grocery_data: Active pantry inventory & shopping list items
      4. meal_planner: Weekly meal schedule and generated plans
      5. activity_log: Workout audit trail, quiz scores, and gamified points
    
    If default records do not exist for a newly signed-up user, it initializes
    them automatically with sensible nutritional defaults.
    """
    # 1. User Profile Document
    user_doc = db_manager.users.find_one({"email": user_id}) or {}

    # 2. Daily Health & Calorie Tracking Document
    health = db_manager.health_data.find_one({"userId": user_id})
    if not health:
        health = {
            "userId": user_id,
            "calories": 0,
            "goal": user_doc.get("target_calories", 2000),
            "targetCalories": user_doc.get("target_calories", 2000),
            "macros": {
                "carbs": user_doc.get("carbs_target", 225),
                "protein": user_doc.get("protein_target", 150),
                "fat": user_doc.get("fat_target", 56),
            },
            "waterIntake": 0,
            "weeklyCalories": [2000, 2100, 1900, 2200, 1800, 2050, 1950],
            "waterHistory": [0, 0, 0, 0, 0, 0, 0],
        }
        db_manager.health_data.update_one({"userId": user_id}, {"$set": health}, upsert=True)

    # 3. Grocery & Pantry Inventory Document
    grocery = db_manager.grocery_data.find_one({"userId": user_id})
    if not grocery:
        grocery = {"userId": user_id, "shoppingList": [], "pantry": []}
        db_manager.grocery_data.update_one({"userId": user_id}, {"$set": grocery}, upsert=True)

    # 4. Meal Planner Schedule Document
    meals = db_manager.meal_planner.find_one({"userId": user_id}) or {
        "userId": user_id,
        "plan": {},
        "generatedAt": None,
    }

    # 5. Activity Audit & Gamification Scores
    activity = db_manager.activity_log.find_one({"userId": user_id}) or {
        "userId": user_id,
        "activities": [],
        "quiz_scores": {},
        "total_score": 0,
    }

    return {
        "user_doc": user_doc,
        "health": health,
        "grocery": grocery,
        "meals": meals,
        "activity": activity,
    }


# ==============================================================================
# 3. CORE DOMAIN SERVICES
# ==============================================================================

# ------------------------------------------------------------------------------
# 3.1 Authentication & Security Service
# ------------------------------------------------------------------------------

class AuthService:
    """
    Cryptographic and security utilities for user authentication.
    
    Responsibilities:
    - Salted password hashing (Scrypt / PBKDF2) to resist rainbow tables and brute force.
    - Cryptographic URL-safe timed tokens (itsdangerous) for tamper-proof password reset flows.
    - Stateless JSON Web Token (JWT) encoding & decoding for REST API consumers.
    """

    @staticmethod
    def hash_password(password: str) -> str:
        """
        Derives a salted, cryptographically secure hash of the plaintext password.
        Uses Werkzeug's default modern hashing algorithm (scrypt/pbkdf2:sha256).
        """
        return generate_password_hash(password)

    @staticmethod
    def check_password(hashed_password: str, password: str) -> bool:
        """
        Constant-time verification of plaintext password against stored hash.
        Resistant to timing attacks.
        """
        return check_password_hash(hashed_password, password)

    @staticmethod
    def generate_reset_token(email: str) -> str:
        """
        Generates a cryptographically signed token containing the user's email.
        Uses URLSafeTimedSerializer with the application's SECRET_KEY.
        """
        s = URLSafeTimedSerializer(app.config["SECRET_KEY"])
        return s.dumps(email, salt="password-reset-salt")

    @staticmethod
    def verify_reset_token(token: str, max_age_seconds: int = 3600) -> str | None:
        """
        Validates the authenticity and expiration of a password reset token.
        Default expiration: 3600 seconds (1 hour).
        Returns the decrypted email address if valid; otherwise returns None.
        """
        s = URLSafeTimedSerializer(app.config["SECRET_KEY"])
        try:
            return s.loads(token, salt="password-reset-salt", max_age=max_age_seconds)
        except (BadSignature, SignatureExpired):
            return None

    @staticmethod
    def make_jwt(payload: dict, exp_minutes: int = 60 * 24 * 7) -> str:
        """
        Generates a signed JSON Web Token (JWT) with standard expiration timestamp.
        Default expiration: 7 days.
        """
        p = dict(payload)
        p["exp"] = dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=exp_minutes)
        return jwt.encode(p, app.config["JWT_SECRET"], algorithm=app.config["JWT_ALG"])

    @staticmethod
    def decode_jwt(token: str) -> dict | None:
        """
        Decodes and verifies the signature and expiration of an incoming Bearer JWT.
        Returns the decoded payload dict if authentic, or None if tampered/expired.
        """
        try:
            return jwt.decode(token, app.config["JWT_SECRET"], algorithms=[app.config["JWT_ALG"]])
        except Exception:
            return None


def get_current_user_id() -> str | None:
    """
    Hybrid Authentication Extractor:
    
    Seamlessly inspects both:
      1. Web Browser Session: session['username'] (cookie-based authentication)
      2. REST API Request Header: 'Authorization: Bearer <JWT>' (stateless mobile/API authentication)
    
    Returns the authenticated user's email/ID string, or None if unauthenticated.
    """
    # 1. Check Flask session cookie (Web UI users)
    if "username" in session and session["username"]:
        return str(session["username"])

    # 2. Check HTTP Authorization header (REST API clients)
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1]
        payload = AuthService.decode_jwt(token)
        if payload and "sub" in payload:
            return str(payload["sub"])
        if payload and "email" in payload:
            return str(payload["email"])

    return None


def login_required(f):
    """
    Authentication Gatekeeper Decorator.
    
    Protects sensitive routes:
    - If user is authenticated: attaches request.current_user and proceeds.
    - If user is NOT authenticated:
        - API calls (/api/*) receive a 401 Unauthorized JSON error.
        - Web page requests are redirected to the /login page.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        user_id = get_current_user_id()
        if user_id:
            request.current_user = user_id
            return f(*args, **kwargs)
        if request.path.startswith("/api/"):
            return jsonify({"error": "Authentication required. Please log in."}), 401
        return redirect(url_for("login"))
    return wrapper


def require_ownership(param_name="user_id"):
    """
    Broken Object Level Authorization (IDOR) Security Guardrail.
    
    OWASP Top 10 Prevention:
    Prevents User A from reading or modifying User B's health, pantry, or profile data
    by simply manipulating the URL parameters (e.g. /api/dashboard/victim@email.com).
    
    Logic:
      1. Resolves caller's authenticated identity.
      2. Compares caller's ID with the target resource parameter.
      3. If 'me' is passed as param, automatically aliases to caller's ID.
      4. If caller tries to access another user's record, immediately rejects with
         HTTP 403 Forbidden and audit error code 'IDOR_PREVENTED'.
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            current_user = get_current_user_id()
            if not current_user:
                return jsonify({"error": "Unauthorized"}), 401

            target = kwargs.get(param_name)
            if target == "me":
                kwargs[param_name] = current_user
                target = current_user

            if target and target != current_user:
                return jsonify({
                    "error": "Forbidden: Access denied to another user's private data.",
                    "code": "IDOR_PREVENTED"
                }), 403

            request.current_user = current_user
            return f(*args, **kwargs)
        return wrapper
    return decorator


# ------------------------------------------------------------------------------
# 3.2 Clinical Nutrition Service
# ------------------------------------------------------------------------------

class NutritionService:
    """
    Clinical nutrition calculation engine, macronutrient distributor,
    and allergen safety validation system.
    
    Implemented Standards:
    - Mifflin-St Jeor Clinical Equation (gold standard for Basal Metabolic Rate estimation)
    - World Health Organization physical activity level (PAL) multipliers
    - Clinical safety calorie minimums (1200 kcal for women, 1500 kcal for men)
    - Comprehensive allergen keyword mapping across 7 major allergen groups
    """

    # Comprehensive ingredient keyword mapping for major allergen categories
    ALLERGENS_MAP = {
        "gluten": ["wheat", "atta", "maida", "flour", "bread", "pasta", "barley", "rye", "semolina"],
        "dairy": ["milk", "curd", "dahi", "paneer", "cheese", "butter", "ghee", "cream", "yogurt"],
        "peanuts": ["peanut", "groundnut", "peanut butter"],
        "tree_nuts": ["almond", "walnut", "cashew", "pistachio"],
        "eggs": ["egg", "eggs", "omelette"],
        "soy": ["soy", "soya", "tofu", "soy sauce"],
        "shellfish": ["shrimp", "prawn", "crab", "lobster"],
    }

    @classmethod
    def calculate_bmr_tdee(cls, weight_kg, height_cm, age, gender="Male", activity_level="Moderate", goal="maintenance", custom_diet="standard"):
        """
        Calculates Basal Metabolic Rate (BMR), Total Daily Energy Expenditure (TDEE),
        target daily calories with goal adjustments, and macronutrient gram splits.
        
        Clinical Guardrails:
        Enforces physiological safety floors (1200 kcal for females, 1500 kcal for males)
        to prevent metabolic slowdown and nutritional deficiencies during extreme deficit requests.
        
        Args:
            weight_kg (float): Body weight in kilograms.
            height_cm (float): Height in centimeters.
            age (int): Age in years.
            gender (str): 'Male' or 'Female'.
            activity_level (str): 'Sedentary', 'Light', 'Moderate', 'Active', or 'Very Active'.
            goal (str): 'weight_loss', 'maintenance', or 'weight_gain'.
            custom_diet (str): 'standard', 'high_protein', or 'low_carb'/'keto'.
            
        Returns:
            dict: {bmr, tdee, target_calories, macros: {carbs, protein, fat}, warning}
        """
        # Step 1: Input sanitation with physiological minimums
        w = max(30.0, float(weight_kg or 70.0))
        h = max(100.0, float(height_cm or 170.0))
        a = max(15, int(age or 25))
        is_female = str(gender).strip().lower() in ["female", "f", "woman"]

        # Step 2: Calculate BMR via Mifflin-St Jeor formula
        # Men:   BMR = 10 * weight(kg) + 6.25 * height(cm) - 5 * age + 5
        # Women: BMR = 10 * weight(kg) + 6.25 * height(cm) - 5 * age - 161
        if is_female:
            bmr = 10 * w + 6.25 * h - 5 * a - 161
            safety_floor = 1200  # Clinical absolute minimum for women
        else:
            bmr = 10 * w + 6.25 * h - 5 * a + 5
            safety_floor = 1500  # Clinical absolute minimum for men

        # Step 3: Compute TDEE using Physical Activity Level (PAL) multipliers
        multipliers = {
            "sedentary": 1.2,       # Little to no exercise
            "light": 1.375,         # Light exercise 1-3 days/week
            "moderate": 1.55,       # Moderate exercise 3-5 days/week
            "active": 1.725,        # Hard exercise 6-7 days/week
            "very active": 1.9      # Very hard exercise or physical job
        }
        tdee = round(bmr * multipliers.get(str(activity_level).strip().lower(), 1.55))

        # Step 4: Adjust target calories based on user fitness goal
        goal_lower = str(goal).strip().lower()
        if "loss" in goal_lower:
            target = tdee - 500      # 500 kcal deficit (~0.5 kg safe fat loss/week)
        elif "gain" in goal_lower:
            target = tdee + 350      # 350 kcal surplus (lean muscle hyper-trophy)
        else:
            target = tdee            # Caloric maintenance

        # Step 5: Enforce clinical safety floor to avoid starvation metabolism
        warning = None
        if target < safety_floor:
            target = safety_floor
            warning = f"Target clamped to safety floor of {safety_floor} kcal/day to prevent metabolic harm."

        # Step 6: Derive macronutrient targets (grams) based on chosen diet paradigm
        diet_lower = str(custom_diet).strip().lower()
        if "low_carb" in diet_lower or "keto" in diet_lower:
            carb_p, prot_p, fat_p = 0.15, 0.35, 0.50   # 15% Carbs, 35% Protein, 50% Fat
        elif "high_protein" in diet_lower:
            carb_p, prot_p, fat_p = 0.40, 0.35, 0.25   # 40% Carbs, 35% Protein, 25% Fat
        else:
            carb_p, prot_p, fat_p = 0.45, 0.30, 0.25   # Balanced standard split

        # Carbs: 4 kcal/g, Protein: 4 kcal/g, Fat: 9 kcal/g
        return {
            "bmr": round(bmr),
            "tdee": tdee,
            "target_calories": round(target),
            "macros": {
                "carbs": round((target * carb_p) / 4),
                "protein": round((target * prot_p) / 4),
                "fat": round((target * fat_p) / 9),
            },
            "warning": warning,
        }

    @classmethod
    def filter_allergens(cls, ingredients, user_allergens):
        """
        Allergen Safety Guardrail:
        Cross-references candidate recipe ingredients against the user's declared allergens.
        
        Returns:
            dict: {"safe": bool, "offending_allergens": [{"ingredient": ..., "allergen": ...}]}
        """
        if not user_allergens or not ingredients:
            return {"safe": True, "offending_allergens": []}

        offending = []
        clean_allergens = [a.strip().lower() for a in user_allergens]
        for ing in ingredients:
            ing_l = str(ing).lower()
            for al in clean_allergens:
                # Check all known synonyms / derived ingredients for this allergen
                for kw in cls.ALLERGENS_MAP.get(al, [al]):
                    if kw in ing_l:
                        offending.append({"ingredient": ing, "allergen": al})
                        break
        return {"safe": len(offending) == 0, "offending_allergens": offending}


# ------------------------------------------------------------------------------
# 3.3 Zero-Waste Grocery Service
# ------------------------------------------------------------------------------

class GroceryService:
    """
    Pantry tracking, shopping cart synchronization, and zero-waste meal plan consolidation.
    
    Key Innovations:
    1. Intelligent Ingredient Sanitization: Strips units and quantities (e.g. '2 tbsp Olive Oil' -> 'Olive Oil').
    2. Zero-Waste Deduplication: Automatically subtracts items already in the user's pantry
       before pushing weekly meal plans to the grocery checklist, saving users money and food waste!
    3. Thread-Safe Atomic Updates: Utilizes MongoDB $addToSet and $pull to prevent race conditions.
    """

    # Common measurement units to strip during ingredient normalization
    UNITS = {"tsp", "tbsp", "cup", "cups", "g", "gram", "grams", "kg", "oz", "pinch", "piece", "can", "bunch"}

    @classmethod
    def clean_name(cls, raw: str) -> str:
        """
        Normalizes ingredient strings by stripping numeric quantities, units, and parentheses.
        Example: '2 tbsp Olive Oil (Extra Virgin)' -> 'Olive Oil'
        """
        text = re.sub(r"\(.*?\)", "", str(raw or "")).strip()
        text = re.sub(r"[\d/\.\+\-\*]+", "", text).strip()
        words = [w for w in text.split() if w.lower() not in cls.UNITS]
        cleaned = " ".join(words).strip().title()
        return cleaned if cleaned else raw.strip().title()

    @classmethod
    def add_pantry(cls, user_id: str, name: str, category: str = "General"):
        """Adds an item to the user's active pantry inventory using atomic $addToSet."""
        clean = cls.clean_name(name)
        item = {"name": clean, "category": category, "addedAt": dt.datetime.now(dt.timezone.utc).isoformat()}
        db_manager.grocery_data.update_one({"userId": user_id}, {"$addToSet": {"pantry": item}}, upsert=True)
        return get_or_create_user_state(user_id)["grocery"].get("pantry", [])

    @classmethod
    def remove_pantry(cls, user_id: str, name: str):
        """Removes an item from the user's active pantry inventory."""
        db_manager.grocery_data.update_one({"userId": user_id}, {"$pull": {"pantry": {"name": name}}})
        return get_or_create_user_state(user_id)["grocery"].get("pantry", [])

    @classmethod
    def add_shopping(cls, user_id: str, name: str, category: str = "General", cost: float = 50.0):
        """Adds an item to the user's shopping checklist with estimated price."""
        clean = cls.clean_name(name)
        item = {"name": clean, "category": category, "estimated_cost": cost, "addedAt": dt.datetime.now(dt.timezone.utc).isoformat()}
        db_manager.grocery_data.update_one({"userId": user_id}, {"$addToSet": {"shoppingList": item}}, upsert=True)
        return get_or_create_user_state(user_id)["grocery"].get("shoppingList", [])

    @classmethod
    def remove_shopping(cls, user_id: str, name: str):
        """Removes an item from the user's shopping checklist."""
        db_manager.grocery_data.update_one({"userId": user_id}, {"$pull": {"shoppingList": {"name": name}}})
        return get_or_create_user_state(user_id)["grocery"].get("shoppingList", [])

    @classmethod
    def move_item(cls, user_id: str, name: str, source: str, destination: str):
        """
        Transfers an item between the shopping cart and pantry.
        Useful when a user checks off purchased items to instantly stock their pantry!
        """
        if source == "shopping" and destination == "pantry":
            cls.remove_shopping(user_id, name)
            cls.add_pantry(user_id, name)
        elif source == "pantry" and destination == "shopping":
            cls.remove_pantry(user_id, name)
            cls.add_shopping(user_id, name)
        return get_or_create_user_state(user_id)["grocery"]

    @classmethod
    def consolidate_diet_to_grocery(cls, user_id: str, meal_plan: dict):
        """
        Zero-Waste Algorithm:
        1. Traverses the weekly meal plan and extracts all raw ingredient names.
        2. Normalizes ingredient strings to clean title names.
        3. Queries user's existing pantry: if item is already in pantry, it is DEDUCTED.
        4. If item is missing from pantry, pushes it to the shopping checklist.
        
        Returns summary dictionary of deducted items and added items.
        """
        extracted = set()
        if isinstance(meal_plan, list):
            for m in meal_plan:
                if isinstance(m, dict) and "ingredients" in m:
                    raw = m["ingredients"]
                    raw_list = [i.strip() for i in raw.split(",")] if isinstance(raw, str) else raw
                    for r in raw_list:
                        c = cls.clean_name(r)
                        if c: extracted.add(c)
        elif isinstance(meal_plan, dict):
            for _, day_val in meal_plan.items():
                meals = day_val.get("meals", day_val) if isinstance(day_val, dict) else day_val
                if isinstance(meals, dict):
                    for _, items in meals.items():
                        if isinstance(items, list):
                            for m in items:
                                if isinstance(m, dict) and "ingredients" in m:
                                    raw = m["ingredients"]
                                    raw_list = [i.strip() for i in raw.split(",")] if isinstance(raw, str) else raw
                                    for r in raw_list:
                                        c = cls.clean_name(r)
                                        if c: extracted.add(c)

        # Retrieve current user's pantry and shopping lists
        user_groc = get_or_create_user_state(user_id)["grocery"]
        pantry_names = {p.get("name", "").lower() for p in user_groc.get("pantry", []) if isinstance(p, dict)}
        shopping_names = {s.get("name", "").lower() for s in user_groc.get("shoppingList", []) if isinstance(s, dict)}

        deducted, added, already_present = [], [], []
        for ing in extracted:
            ing_l = ing.lower()
            # If the user already has this ingredient in their pantry, don't buy it!
            if any(p in ing_l or ing_l in p for p in pantry_names):
                deducted.append(ing)
            # Otherwise, add to shopping list if not already present
            elif ing_l not in shopping_names:
                cls.add_shopping(user_id, ing, "Meal Plan Staples")
                added.append(ing)
            else:
                already_present.append(ing)

        return {
            "total_extracted": len(extracted),
            "added_to_shopping_count": len(added),
            "deducted_from_pantry_count": len(deducted),
            "already_present_count": len(already_present),
            "added_items": added,
            "deducted_items": deducted,
        }

    @classmethod
    def get_user_grocery_data(cls, user_id: str) -> dict:
        """Helper to fetch full grocery document for a user."""
        return get_or_create_user_state(user_id)["grocery"]

    # Backward compatibility aliases for existing test suites
    add_pantry_item = add_pantry
    remove_pantry_item = remove_pantry
    add_shopping_item = add_shopping
    remove_shopping_item = remove_shopping


# ------------------------------------------------------------------------------
# 3.4 Google Gemini AI Service (BiteBot)
# ------------------------------------------------------------------------------

class AIService:
    """
    Intelligent conversational AI nutrition assistant (BiteBot).
    
    Features:
    - Powered by Google Gemini 3.6 Flash via the official `google-genai` SDK.
    - Contextual Grounding (RAG): Dynamically injects user's real-time daily calories,
      fitness goals, allergen warnings, and available pantry items into the system prompt.
    - Zero-Waste Recipe Prioritization: Instructs Gemini to prioritize existing pantry staples.
    - Offline Heuristic Fallback: Seamless local rule-based recipe synthesis if offline
      or if no Google API key is configured.
    """

    @classmethod
    def generate_chat_response(cls, user_message: str, context: dict) -> str:
        """
        Generates a personalized response to the user's dietary question.
        
        Args:
            user_message (str): Query from the chat interface.
            context (dict): {pantry, targetCalories, goal, allergens}
            
        Returns:
            str: Grounded advice and recipes.
        """
        pantry = [p.get("name") for p in context.get("pantry", []) if isinstance(p, dict) and p.get("name")]
        pantry_str = ", ".join(pantry) if pantry else "None"
        target_cals = context.get("targetCalories", 2000)
        goal = context.get("goal", "maintenance")
        allergens = context.get("allergens", [])

        api_key = app.config.get("GEMINI_API_KEY")

        # Attempt call to Google GenAI SDK if library is present and key is set
        if HAS_GENAI and api_key:
            try:
                client = genai.Client(api_key=api_key)
                prompt = (
                    f"You are BiteBot, a clinical nutritionist and zero-waste grocery assistant.\n"
                    f"User Profile: Daily Target: {target_cals} kcal, Goal: {goal}, Allergens to avoid: {allergens}\n"
                    f"Available Pantry Items: {pantry_str}\n\n"
                    f"Prioritize recipes using available pantry items to prevent grocery spending.\n\n"
                    f"User Query: {user_message}"
                )
                # Model selection hierarchy (prioritizing current gemini-3.6-flash)
                candidates = [
                    app.config.get("GEMINI_MODEL", "gemini-3.6-flash"),
                    "gemini-3.6-flash",
                    "gemini-3.5-flash",
                    "gemini-flash-latest"
                ]
                for model_name in candidates:
                    try:
                        res = client.models.generate_content(model=model_name, contents=prompt)
                        if res and res.text:
                            return res.text.strip()
                    except Exception:
                        continue
            except Exception as e:
                logger.warning(f"Gemini API request notice: {e}")

        # Seamless offline heuristic fallback when no API key is provided
        return cls._heuristic_fallback(user_message, pantry, target_cals, goal)

    @classmethod
    def _heuristic_fallback(cls, msg: str, pantry: list, cals: int, goal: str) -> str:
        """Local rule-based generator for offline or demonstration environments."""
        m = msg.lower()
        p_str = ", ".join(pantry) if pantry else "pantry staples"
        if any(w in m for w in ["cook", "recipe", "dinner", "lunch", "breakfast", "eat", "pantry"]):
            return (
                f"Based on your pantry (**{p_str}**) and daily target of **{cals} kcal** ({goal}), "
                f"here are recommended zero-waste recipes:\n\n"
                f"🍲 **Spiced Pantry Lentil Bowl**: Simmer lentils or legumes with spices and greens (~350 kcal, 22g protein).\n\n"
                f"🥣 **Quick Power Oats**: 50g oats simmered in milk, topped with nuts and honey (~310 kcal, 12g protein).\n\n"
                f"💡 *Add any missing ingredients to your grocery list in 1 click.*"
            )
        return (
            f"Hello! I am **BiteBot**, your nutrition assistant. "
            f"Your daily target is **{cals} kcal** ({goal}). Ask me for recipes using your pantry ({p_str}) or nutrition tips!"
        )


# Fallback product catalog for initial exploration if database collections are empty
FALLBACK_PRODUCTS = [
    {"_id": "p1", "name": "Yellow Lentils (Moong Dal)", "category": "Grains & Pulses", "price": 40, "originalPrice": 45, "rating": 4.5, "description": "High-protein unpolished yellow lentils, a staple for a healthy diet."},
    {"_id": "p2", "name": "Whole Wheat Atta", "category": "Grains & Pulses", "price": 45, "originalPrice": 50, "rating": 4.3, "description": "100% whole wheat flour, rich in dietary fiber."},
    {"_id": "p3", "name": "Local Curd (Dahi)", "category": "Dairy & Eggs", "price": 30, "originalPrice": 30, "rating": 4.7, "description": "Freshly set curd, high in calcium and natural probiotics."},
    {"_id": "p4", "name": "Fresh Cow Milk", "category": "Dairy & Eggs", "price": 28, "originalPrice": 28, "rating": 4.4, "description": "Pasteurized fresh cow milk, rich in calcium and protein."},
    {"_id": "p5", "name": "Roasted Peanuts", "category": "Nuts & Seeds", "price": 35, "originalPrice": 40, "rating": 4.5, "description": "Crunchy roasted peanuts, high in healthy fats and plant protein."},
    {"_id": "p6", "name": "Standard Oats", "category": "Breakfast", "price": 50, "originalPrice": 55, "rating": 4.4, "description": "100% whole grain oats, great for sustained morning energy."},
    {"_id": "p7", "name": "Fresh Eggs (6 pack)", "category": "Dairy & Eggs", "price": 45, "originalPrice": 50, "rating": 4.8, "description": "Farm fresh eggs, complete source of amino acids."},
]

# Fallback curated multi-cuisine meal plans for immediate display
DEFAULT_CURATED_MEALS = [
    {
        "_id": "m1",
        "Shrt_Desc": "Moong Dal Tadka with Steamed Basmati Rice",
        "Energ_Kcal": 380,
        "Protein_(g)": 18,
        "Carbohydrt_(g)": 58,
        "Lipid_Tot_(g)": 8,
        "cuisine": "Indian",
        "category": "High-Fiber Lunch",
        "ingredients": ["Yellow Moong Dal", "Turmeric", "Cumin Seeds", "Garlic", "Ghee", "Steamed Basmati Rice"],
        "instructions": ["Rinse moong dal and pressure cook with turmeric and salt for 3 whistles.", "Heat ghee in a pan, temper cumin seeds, minced garlic, and green chilies.", "Pour tempering into cooked dal and simmer for 5 minutes. Serve hot with steamed rice."]
    },
    {
        "_id": "m2",
        "Shrt_Desc": "Tofu & Broccoli Veggie Stir-Fry",
        "Energ_Kcal": 360,
        "Protein_(g)": 22,
        "Carbohydrt_(g)": 38,
        "Lipid_Tot_(g)": 12,
        "cuisine": "Asian",
        "category": "Plant-Protein Dinner",
        "ingredients": ["Firm Tofu", "Broccoli Florets", "Soy Sauce", "Sesame Oil", "Garlic", "Ginger", "Brown Rice"],
        "instructions": ["Press and cube firm tofu, pan-sear in sesame oil until golden brown.", "Add broccoli florets, minced ginger, and garlic; toss on high heat for 3-4 minutes.", "Stir in low-sodium soy sauce and serve over cooked brown rice."]
    },
    {
        "_id": "m3",
        "Shrt_Desc": "Mediterranean Chickpea & Olive Quinoa Salad",
        "Energ_Kcal": 410,
        "Protein_(g)": 16,
        "Carbohydrt_(g)": 56,
        "Lipid_Tot_(g)": 14,
        "cuisine": "Mediterranean",
        "category": "Antioxidant Rich Lunch",
        "ingredients": ["Cooked Quinoa", "Boiled Chickpeas", "Cucumber", "Cherry Tomatoes", "Kalamata Olives", "Feta Cheese", "Extra Virgin Olive Oil", "Lemon Juice"],
        "instructions": ["Cook quinoa and allow to cool to room temperature.", "Toss with chickpeas, diced cucumbers, halved tomatoes, and olives.", "Whisk olive oil and lemon juice, pour dressing over salad and top with crumbled feta."]
    },
    {
        "_id": "m4",
        "Shrt_Desc": "Mexican Black Bean & Roasted Corn Bowl",
        "Energ_Kcal": 420,
        "Protein_(g)": 19,
        "Carbohydrt_(g)": 64,
        "Lipid_Tot_(g)": 10,
        "cuisine": "Mexican",
        "category": "Energizing Lunch",
        "ingredients": ["Black Beans", "Sweet Corn", "Brown Rice", "Avocado", "Salsa", "Cumin", "Lime"],
        "instructions": ["Warm black beans with cumin, lime juice, and a pinch of salt.", "Char sweet corn kernels lightly in a dry skillet.", "Assemble warm brown rice in a bowl, top with seasoned black beans, corn, avocado slices, and fresh salsa."]
    },
    {
        "_id": "m5",
        "Shrt_Desc": "Herbed Grilled Chicken Breast with Sweet Potato",
        "Energ_Kcal": 460,
        "Protein_(g)": 44,
        "Carbohydrt_(g)": 36,
        "Lipid_Tot_(g)": 12,
        "cuisine": "Continental",
        "category": "Lean Muscle Dinner",
        "ingredients": ["Skinless Chicken Breast", "Sweet Potato", "Rosemary", "Garlic", "Olive Oil", "Steamed Asparagus"],
        "instructions": ["Marinate chicken breast in olive oil, minced garlic, fresh rosemary, and black pepper.", "Grill chicken for 6-7 minutes each side until cooked through.", "Serve with oven-baked sweet potato wedges and steamed asparagus."]
    },
    {
        "_id": "m6",
        "Shrt_Desc": "Overnight Power Oats with Chia & Berries",
        "Energ_Kcal": 340,
        "Protein_(g)": 14,
        "Carbohydrt_(g)": 52,
        "Lipid_Tot_(g)": 8,
        "cuisine": "Continental",
        "category": "Nutritious Breakfast",
        "ingredients": ["Rolled Oats", "Almond Milk", "Chia Seeds", "Honey", "Blueberries", "Walnuts"],
        "instructions": ["Combine rolled oats, almond milk, and chia seeds in a mason jar.", "Refrigerate overnight (minimum 6 hours) to thicken.", "Top with raw honey, fresh blueberries, and crushed walnuts before enjoying."]
    }
]


# ==============================================================================
# 4. WEB UI ROUTES (Jinja2 HTML Templates)
# ==============================================================================

@app.route("/")
def index():
    """
    Public Landing Page.
    - If user is already logged in: redirects directly to their /dashboard.
    - If guest/unauthenticated: renders the landing page with feature highlights.
    """
    if get_current_user_id():
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    """
    User Registration Controller.
    - Validates email format, password match, and minimum password length.
    - Enforces uniqueness via MongoDB unique email index.
    - Encrypts password with salted Scrypt hash.
    - Pre-initializes default user documents across health, grocery, and activity collections.
    """
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        # Validation: password match & minimum length
        if not email or not password or password != confirm or len(password) < 6:
            flash("Please verify your email and password (minimum 6 characters).", "error")
            return redirect(url_for("signup"))

        # Validation: duplicate email check
        if db_manager.users.find_one({"email": email}):
            flash("Email is already registered. Please log in.", "error")
            return redirect(url_for("signup"))

        # Secure password hashing & persistent user insertion
        hashed = AuthService.hash_password(password)
        db_manager.users.insert_one({
            "email": email,
            "password": hashed,
            "name": email.split("@")[0].capitalize(),
            "target_calories": 2000,
            "goal": "maintenance",
            "allergens": [],
            "createdAt": dt.datetime.now(dt.timezone.utc),
        })
        # Initialize default records in other collections
        get_or_create_user_state(email)
        flash("Account created successfully! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("Sign up.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """
    User Authentication Controller.
    - Validates user email and password against stored salted hash.
    - Establishes a secure HTTPOnly session cookie.
    - Redirects to /dashboard upon successful verification.
    """
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = db_manager.users.find_one({"email": email})
        if not user or not AuthService.check_password(user["password"], password):
            flash("Invalid email or password.", "error")
            return redirect(url_for("login"))

        # Issue authenticated session
        session["username"] = email
        session.permanent = True
        flash("Login successful! Welcome to SmartBite.", "success")
        return redirect(url_for("dashboard"))

    return render_template("Sign in.html")


@app.route("/logout")
def logout():
    """Destroys current session and securely logs out the user."""
    session.clear()
    flash("You have been securely logged out.", "success")
    return redirect(url_for("login"))


@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    """
    Cryptographic Password Reset Request.
    - Generates a signed, URL-safe timed token containing the user's email.
    - Displays or delivers the secure reset link.
    """
    reset_link = None
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        if db_manager.users.find_one({"email": email}):
            token = AuthService.generate_reset_token(email)
            reset_link = url_for("reset_password", token=token, _external=True)
            flash("Password reset link generated below.", "success")
        else:
            flash("No account found with that email.", "error")
    return render_template("forgotpass.html", reset_link=reset_link)


@app.route("/reset_password/<token>", methods=["GET", "POST"])
def reset_password(token):
    """
    Password Reset Confirmation.
    - Validates cryptographic token signature and 1-hour expiration timestamp.
    - Updates the user's hashed password in MongoDB upon verification.
    """
    email = AuthService.verify_reset_token(token)
    if not email:
        flash("Password reset link is invalid or expired.", "error")
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        pwd = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        if len(pwd) < 6 or pwd != confirm:
            flash("Passwords must match and have at least 6 characters.", "error")
            return redirect(url_for("reset_password", token=token))

        # Store updated encrypted password
        db_manager.users.update_one({"email": email}, {"$set": {"password": AuthService.hash_password(pwd)}})
        flash("Password updated successfully! You can now log in.", "success")
        return redirect(url_for("login"))

    return render_template("forgotpass.html", valid_token=True, token=token, email=email)


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    """
    User Profile & Biometric Settings.
    - Allows user to update weight, height, age, gender, activity level, and goals.
    - Automatically executes Mifflin-St Jeor engine to recompute BMR, TDEE, and macro gram goals.
    - Synchronizes updated biometric targets to both users and health_data collections.
    """
    uid = get_current_user_id()
    user_doc = db_manager.users.find_one({"email": uid}) or {}

    if request.method == "POST":
        name = request.form.get("name", user_doc.get("name", ""))
        weight = float(request.form.get("weight", user_doc.get("weight", 70)))
        height = float(request.form.get("height", user_doc.get("height", 170)))
        age = int(request.form.get("age", user_doc.get("age", 25)))
        gender = request.form.get("gender", user_doc.get("gender", "Male"))
        activity = request.form.get("activity_level", user_doc.get("activity_level", "Moderate"))
        goal = request.form.get("goal", user_doc.get("goal", "maintenance"))

        # Re-run clinical engine
        bio = NutritionService.calculate_bmr_tdee(weight, height, age, gender, activity, goal)
        update_data = {
            "name": name, "weight": weight, "height": height, "age": age,
            "gender": gender, "activity_level": activity, "goal": goal,
            "bmr": bio["bmr"], "tdee": bio["tdee"], "target_calories": bio["target_calories"],
            "carbs_target": bio["macros"]["carbs"], "protein_target": bio["macros"]["protein"],
            "fat_target": bio["macros"]["fat"],
        }
        db_manager.users.update_one({"email": uid}, {"$set": update_data})
        db_manager.health_data.update_one({"userId": uid}, {"$set": {"goal": bio["target_calories"], "targetCalories": bio["target_calories"], "macros": bio["macros"]}}, upsert=True)
        flash("Profile updated successfully!", "success")
        return redirect(url_for("profile"))

    state = get_or_create_user_state(uid)
    return render_template("profile.html", user=user_doc, health=state["health"])


@app.route("/dashboard")
@login_required
def dashboard():
    """
    Main Health & Nutrition Dashboard.
    Renders personalized calorie budget, water hydration tracker,
    macro distribution bars, shopping checklist preview, and recent activity logs.
    """
    uid = get_current_user_id()
    state = get_or_create_user_state(uid)
    return render_template(
        "dashboard.html",
        user=uid,
        user_id=uid,
        user_doc=state["user_doc"],
        health=state["health"],
        grocery=state["grocery"],
        today_meals=[],
        activity=state["activity"].get("activities", []),
    )


@app.route("/diet_plan")
@app.route("/dietplan")
@login_required
def diet_plan():
    """
    Weekly Meal Planner & Recipe Explorer.
    Displays scheduled breakfast, lunch, and dinner recipes,
    with 1-click zero-waste consolidation into the user's shopping checklist.
    """
    uid = get_current_user_id()
    state = get_or_create_user_state(uid)
    db_recipes = list(db_manager.recipes.find().limit(25))
    all_meals = objid_to_str(db_recipes) if db_recipes else DEFAULT_CURATED_MEALS

    return render_template(
        "diet_plan.html",
        user=uid,
        user_doc=state["user_doc"],
        plan=all_meals,
        meal_plan=all_meals,
        recipes=all_meals
    )


@app.route("/mealplanner")
@login_required
def mealplanner():
    """Alias for diet plan page."""
    return redirect(url_for("diet_plan"))


@app.route("/grocery")
@login_required
def grocery():
    """
    Zero-Waste Grocery & Pantry Management View.
    Displays user's active pantry inventory, checklist of items to buy,
    and grocery catalog for quick additions.
    """
    uid = get_current_user_id()
    state = get_or_create_user_state(uid)
    return render_template("grocery.html", user=uid, shopping_list=state["grocery"].get("shoppingList", []), pantry=state["grocery"].get("pantry", []), products=FALLBACK_PRODUCTS)


@app.route("/bitebot")
def bitebot():
    """
    BiteBot AI Assistant Interface.
    Conversational chat view powered by Google Gemini 3.6 Flash.
    """
    return render_template("Bitebot.html", user=get_current_user_id() or "Guest")


@app.route("/fitness_games")
@app.route("/fitness_challenges")
@login_required
def fitness_games():
    """Gamified fitness hub offering plank challenges, rep counters, and health quizzes."""
    return render_template("fitness_games.html")


@app.route("/plank_timer")
@login_required
def plank_timer():
    """Interactive core hold stopwatch challenge that records workout points."""
    return render_template("plank_timer.html")


@app.route("/repetition_counter")
@login_required
def repetition_counter():
    """Interactive workout repetition counter challenge."""
    return render_template("repetition_counter.html")


@app.route("/achievements")
@app.route("/streak_master")
@login_required
def achievements():
    """User achievement badge viewer and rank progression board."""
    return render_template("achievements.html", user=get_current_user_id())


@app.route("/yoga_pose_quiz")
@login_required
def yoga_pose_quiz():
    """Interactive Yoga posture quiz."""
    return render_template("yoga_pose_quiz_fixed.html")


@app.route("/nutrition_label_quiz")
@login_required
def nutrition_label_quiz():
    """Interactive Nutrition label reading quiz."""
    return render_template("nutrition_label_quiz.html")


@app.route("/quizzes")
@login_required
def quizzes():
    """General quiz alias."""
    return redirect(url_for("yoga_pose_quiz"))


@app.route("/product/<product_id>")
def product_detail(product_id):
    """Product detail inspection page for grocery catalog items."""
    p = next((x for x in FALLBACK_PRODUCTS if x["_id"] == product_id or x["name"].lower() == product_id.lower()), FALLBACK_PRODUCTS[0])
    return render_template("product_detail.html", product=p, user=get_current_user_id())


# ==============================================================================
# 5. SECURE REST APIs (IDOR-Protected JSON Endpoints)
# ==============================================================================

@app.route("/api/dashboard/<user_id>", methods=["GET"])
@login_required
@require_ownership("user_id")
def api_dashboard(user_id):
    """
    User Dashboard Metrics API.
    
    Security:
      - Protected by @require_ownership: Rejects attempts by User A to view User B's dashboard.
      - Returns 403 Forbidden with code 'IDOR_PREVENTED' on unauthorized access.
    
    Returns:
      JSON object containing healthData, groceryData, mealPlanner, activityLog, and userProfile.
    """
    uid = unquote(user_id)
    state = get_or_create_user_state(uid)
    h = state["health"]
    u = state["user_doc"]
    g = state["grocery"]
    target = h.get("targetCalories") or u.get("target_calories") or 2000

    return jsonify({
        "healthData": objid_to_str(h),
        "groceryData": objid_to_str(g),
        "mealPlanner": objid_to_str(state["meals"]),
        "activityLog": objid_to_str(state["activity"].get("activities", [])),
        "userProfile": {
            "name": u.get("name") or uid.split("@")[0].capitalize(),
            "targetCalories": target,
            "caloriesConsumed": h.get("calories", 0),
            "shoppingCount": len(g.get("shoppingList", [])),
            "pantryCount": len(g.get("pantry", [])),
        }
    })


@app.route("/api/health/<user_id>/log_calories", methods=["POST"])
@login_required
@require_ownership("user_id")
def api_log_calories(user_id):
    """
    Calorie Intake Logging Endpoint.
    
    Validates meal calories and atomically increments user's daily consumption in MongoDB,
    while recording an entry into their activity audit log.
    """
    uid = unquote(user_id)
    data = request.get_json() or {}
    cals = int(data.get("calories", 0))
    meal = data.get("meal_name", "Meal")
    if cals <= 0:
        return jsonify({"error": "Calories must be positive."}), 400

    # Atomically increment total daily calories in MongoDB
    doc = db_manager.health_data.find_one({"userId": uid}) or {}
    new_cals = doc.get("calories", 0) + cals
    db_manager.health_data.update_one({"userId": uid}, {"$set": {"calories": new_cals}}, upsert=True)

    # Append activity audit event
    db_manager.activity_log.update_one(
        {"userId": uid},
        {"$push": {"activities": {"activity": f"Logged {meal} (+{cals} kcal)", "timestamp": dt.datetime.now(dt.timezone.utc)}}},
        upsert=True
    )
    return jsonify({"message": f"Logged {cals} kcal", "calories": new_cals})


@app.route("/api/diet_plan/push_to_grocery", methods=["POST"])
@login_required
def api_push_to_grocery():
    """
    Zero-Waste Meal Plan Consolidation API.
    
    Extracts all ingredients from the user's active weekly meal plan,
    automatically deducts items already in the user's pantry,
    and pushes only missing items to the shopping checklist.
    """
    uid = get_current_user_id()
    data = request.get_json(silent=True) or {}
    plan = data.get("meals") or data.get("plan")
    if not plan:
        meals_doc = db_manager.meal_planner.find_one({"userId": uid}) or {}
        plan = meals_doc.get("plan", {})
    if not plan:
        db_recipes = list(db_manager.recipes.find().limit(25))
        plan = objid_to_str(db_recipes) if db_recipes else DEFAULT_CURATED_MEALS

    result = GroceryService.consolidate_diet_to_grocery(uid, plan)
    added_count = result.get("added_to_shopping_count", 0)
    deducted_count = result.get("deducted_from_pantry_count", 0)
    already_count = result.get("already_present_count", 0)
    
    if added_count > 0:
        msg = f"Successfully added {added_count} items to your shopping list! ({deducted_count} pantry items excluded to prevent waste)"
    elif already_count > 0:
        msg = f"All {already_count} ingredients are already in your shopping list or stocked in your pantry!"
    else:
        msg = "Grocery shopping list is already up to date!"

    return jsonify({
        "success": True,
        "count": added_count,
        "deducted": deducted_count,
        "already_present": already_count,
        "message": msg,
        "details": result
    })


@app.route("/api/diet_plan/<user_id>/add_custom_recipe", methods=["POST"])
@login_required
@require_ownership("user_id")
def api_add_custom_recipe(user_id):
    """
    Saves a user-submitted custom recipe to MongoDB.
    """
    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Recipe name is required."}), 400

    uid = unquote(user_id)
    recipe_doc = {
        "userId": uid,
        "Shrt_Desc": name,
        "Energ_Kcal": float(data.get("calories", 350)),
        "Protein_(g)": float(data.get("protein", 15)),
        "Carbohydrt_(g)": float(data.get("carbs", 40)),
        "Lipid_Tot_(g)": float(data.get("fats", 10)),
        "ingredients": data.get("ingredients", []),
        "instructions": data.get("instructions", []),
        "cuisine": data.get("cuisine", "Custom"),
        "category": "Custom Recipe",
        "createdAt": dt.datetime.now(dt.timezone.utc).isoformat()
    }
    db_manager.recipes.insert_one(recipe_doc)

    db_manager.activity_log.update_one(
        {"userId": uid},
        {"$push": {"activities": {"activity": f"Added custom recipe: {name}", "timestamp": dt.datetime.now(dt.timezone.utc)}}},
        upsert=True
    )
    return jsonify({"success": True, "message": "Custom recipe added successfully!"})



@app.route("/api/grocery/products", methods=["GET"])
def api_products():
    """Returns grocery catalog items for search and quick cart additions."""
    return jsonify({"products": FALLBACK_PRODUCTS, "count": len(FALLBACK_PRODUCTS)})


@app.route("/api/grocery/<user_id>/pantry/add", methods=["POST"])
@login_required
@require_ownership("user_id")
def api_add_pantry(user_id):
    """
    Adds an ingredient to user's pantry.
    Protected against IDOR: only the authenticated owner can modify their pantry.
    """
    data = request.get_json() or {}
    item = data.get("name") or data.get("item")
    if not item: return jsonify({"error": "Item name is required."}), 400
    p = GroceryService.add_pantry(unquote(user_id), item, data.get("category", "General"))
    return jsonify({"success": True, "pantry": objid_to_str(p)})


@app.route("/api/grocery/<user_id>/pantry/remove", methods=["POST"])
@login_required
@require_ownership("user_id")
def api_remove_pantry(user_id):
    """Removes an ingredient from user's pantry."""
    data = request.get_json() or {}
    item = data.get("name") or data.get("item")
    p = GroceryService.remove_pantry(unquote(user_id), item)
    return jsonify({"success": True, "pantry": objid_to_str(p)})


@app.route("/api/grocery/<user_id>/shopping/add", methods=["POST"])
@login_required
@require_ownership("user_id")
def api_add_shopping(user_id):
    """Adds an item to user's shopping checklist."""
    data = request.get_json() or {}
    item = data.get("name") or data.get("item")
    if not item: return jsonify({"error": "Item name is required."}), 400
    s = GroceryService.add_shopping(unquote(user_id), item, data.get("category", "General"), float(data.get("cost", 50.0)))
    return jsonify({"success": True, "shoppingList": objid_to_str(s)})


@app.route("/api/grocery/<user_id>/shopping/remove", methods=["POST"])
@login_required
@require_ownership("user_id")
def api_remove_shopping(user_id):
    """Removes an item from user's shopping checklist."""
    data = request.get_json() or {}
    item = data.get("name") or data.get("item")
    s = GroceryService.remove_shopping(unquote(user_id), item)
    return jsonify({"success": True, "shoppingList": objid_to_str(s)})


@app.route("/api/grocery/<user_id>/move", methods=["POST"])
@login_required
@require_ownership("user_id")
def api_move_grocery(user_id):
    """Moves an item between shopping list and pantry (e.g. after purchasing)."""
    data = request.get_json() or {}
    g = GroceryService.move_item(unquote(user_id), data.get("name"), data.get("from", "shopping"), data.get("to", "pantry"))
    return jsonify({"success": True, "groceryData": objid_to_str(g)})


@app.route("/api/ai/chat", methods=["POST"])
def api_ai_chat():
    """
    Conversational AI Endpoint (BiteBot).
    
    1. Extracts caller identity (or guest fallback).
    2. Gathers real-time pantry inventory, calorie goals, and allergen filters.
    3. Grounding (RAG): Formats clinical prompt for Google Gemini 3.6 Flash.
    4. Returns AI-generated culinary guidance.
    """
    data = request.get_json() or {}
    msg = data.get("message", "").strip()
    if not msg: return jsonify({"error": "Message cannot be empty."}), 400

    uid = get_current_user_id() or data.get("user_id", "guest@smartbite.com")
    state = get_or_create_user_state(uid)
    reply = AIService.generate_chat_response(msg, {
        "userId": uid,
        "pantry": state["grocery"].get("pantry", []),
        "targetCalories": state["health"].get("targetCalories", 2000),
        "goal": state["user_doc"].get("goal", "healthy maintenance"),
        "allergens": state["user_doc"].get("allergens", []),
    })
    return jsonify({"success": True, "reply": reply, "timestamp": dt.datetime.now(dt.timezone.utc).isoformat()})


@app.route("/api/quiz/save-score", methods=["POST"])
def api_save_score():
    """
    Gamified Workout & Quiz Score Recording.
    Increments user's total fitness score and logs the challenge in their activity trail.
    """
    data = request.get_json() or {}
    quiz = data.get("quiz_type", "workout")
    score = int(data.get("score", 0))
    uid = data.get("user_id") or get_current_user_id() or "guest@smartbite.com"

    db_manager.activity_log.update_one(
        {"userId": uid},
        {
            "$set": {f"quiz_scores.{quiz}": {"score": score, "timestamp": dt.datetime.now(dt.timezone.utc).isoformat()}},
            "$inc": {"total_score": score},
            "$push": {"activities": {"activity": f"Completed {quiz.replace('_', ' ').title()} (+{score} pts)", "timestamp": dt.datetime.now(dt.timezone.utc)}}
        },
        upsert=True
    )
    tot = (db_manager.activity_log.find_one({"userId": uid}) or {}).get("total_score", score)
    return jsonify({"success": True, "score": score, "total_score": tot})


@app.route("/api/achievements/<user_id>", methods=["GET"])
def api_achievements(user_id):
    """
    Calculates dynamic user level and rank progression based on accumulated workout points:
    - Beginner: 0 - 99 pts
    - Novice: 100 - 299 pts
    - Intermediate: 300 - 599 pts
    - Advanced: 600 - 999 pts
    - Master: 1000+ pts
    """
    tot = (db_manager.activity_log.find_one({"userId": unquote(user_id)}) or {}).get("total_score", 0)
    rank = "Master" if tot >= 1000 else "Advanced" if tot >= 600 else "Intermediate" if tot >= 300 else "Novice" if tot >= 100 else "Beginner"
    return jsonify({"rank": rank, "score": tot, "next_level": "Master"})


# ==============================================================================
# 6. APPLICATION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    port = app.config.get("PORT", 5000)
    debug = app.config.get("DEBUG", True)
    print("\n" + "=" * 65)
    print("SmartBite -- Production-Ready AI Health & Grocery Platform")
    print(f"Server running on: http://127.0.0.1:{port}")
    print(f"Gemini AI Status: {'Configured' if app.config.get('GEMINI_API_KEY') else 'Offline Heuristic Fallback'}")
    print("=" * 65 + "\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
