
"""
SmartBite AI Models - Core AI functionality for SmartBite
======================================================

This module contains the main SmartBite AI models for:
1. User Progress Prediction
2. Nutrition Recommendations
3. Meal Planning
4. Health Insights

Author: SmartBite Development Team
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# USER PROGRESS PREDICTOR
# =============================================================================


class UserProgressPredictor:
    """
    AI model for predicting user fitness progress and providing
    personalized recommendations.
    """

    def __init__(self):
        self.scaler = None
        self.models = {}
        self.feature_columns = [
            "age",
            "weight",
            "height",
            "bmi",
            "activity_level",
            "workout_frequency",
            "avg_workout_duration",
            "calories_burned",
            "diet_compliance",
            "sleep_hours",
            "stress_level",
            "motivation_level",
            "gender_encoded",
            "goal_type_encoded",
        ]

    def generate_synthetic_data(self, n_users=1000):
        """Generate synthetic user data for training"""
        np.random.seed(42)

        data = {
            "user_id": range(1, n_users + 1),
            "age": np.random.randint(18, 65, n_users),
            "weight": np.random.uniform(45, 120, n_users),
            "height": np.random.uniform(150, 200, n_users),
            "gender": np.random.choice(["Male", "Female"], n_users),
            "activity_level": np.random.choice(
                ["Sedentary", "Light", "Moderate", "Active", "Very Active"],
                n_users,
            ),
            "workout_frequency": np.random.randint(0, 7, n_users),
            "avg_workout_duration": np.random.randint(20, 120, n_users),
            "calories_burned": np.random.randint(100, 800, n_users),
            "diet_compliance": np.random.uniform(0, 1, n_users),
            "sleep_hours": np.random.uniform(4, 12, n_users),
            "stress_level": np.random.randint(1, 10, n_users),
            "motivation_level": np.random.randint(1, 10, n_users),
            "goal_type": np.random.choice(
                [
                    "weight_loss",
                    "muscle_gain",
                    "fitness_improvement",
                    "maintenance",
                ],
                n_users,
            ),
            "start_date": [
                datetime.now() - timedelta(days=np.random.randint(0, 365))
                for _ in range(n_users)
            ],
        }

        df = pd.DataFrame(data)

        # Calculate BMI
        df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)

        # Generate target variables
        df["weight_loss_4weeks"] = self._calculate_weight_loss(df)
        df["fitness_score_improvement"] = self._calculate_fitness_improvement(
            df
        )
        df["goal_achievement_rate"] = self._calculate_goal_achievement(df)

        return df

    def _calculate_weight_loss(self, df):
        """Calculate weight loss predictions"""
        base_loss = -0.5

        activity_multiplier = {
            "Sedentary": 0.8,
            "Light": 1.0,
            "Moderate": 1.2,
            "Active": 1.4,
            "Very Active": 1.6,
        }

        age_multiplier = 1.0 - (df["age"] - 25) * 0.01
        diet_multiplier = df["diet_compliance"] * 0.5 + 0.5
        calories_factor = df["calories_burned"] / 500

        weekly_loss = (
            base_loss
            * df["activity_level"].apply(
                lambda x: activity_multiplier.get(x, 1.0)
            )
            * age_multiplier.clip(0.5, 1.5)
            * diet_multiplier
            * calories_factor
        )

        return weekly_loss * 4

    def _calculate_fitness_improvement(self, df):
        """Calculate fitness score improvement"""
        base_improvement = 5

        frequency_bonus = df["workout_frequency"] * 2
        duration_bonus = (df["avg_workout_duration"] - 30) * 0.1
        motivation_bonus = (df["motivation_level"] - 5) * 1.5
        age_penalty = ((df["age"] - 35) * 0.2).clip(lower=0)

        return (
            base_improvement
            + frequency_bonus
            + duration_bonus
            + motivation_bonus
            - age_penalty
        )

    def _calculate_goal_achievement(self, df):
        """Calculate goal achievement probability"""
        base_probability = 0.3

        activity_score = df["activity_level"].apply(
            lambda x: {
                "Sedentary": 0,
                "Light": 0.1,
                "Moderate": 0.2,
                "Active": 0.3,
                "Very Active": 0.4,
            }.get(x, 0.2)
        )

        consistency_bonus = df["workout_frequency"] * 0.05
        motivation_bonus = (df["motivation_level"] - 5) * 0.05
        diet_bonus = df["diet_compliance"] * 0.2
        sleep_bonus = ((df["sleep_hours"] - 7) * 0.02).clip(lower=0)

        probability = (
            base_probability
            + activity_score
            + consistency_bonus
            + motivation_bonus
            + diet_bonus
            + sleep_bonus
        )

        return np.clip(probability, 0, 1)

    def prepare_features(self, df):
        """Prepare features for training"""
        data = df.copy()

        # Encode categorical variables
        le_gender = LabelEncoder()
        le_activity = LabelEncoder()
        le_goal = LabelEncoder()

        data["gender_encoded"] = le_gender.fit_transform(data["gender"])
        data["activity_encoded"] = le_activity.fit_transform(
            data["activity_level"]
        )
        data["goal_type_encoded"] = le_goal.fit_transform(data["goal_type"])

        # Select features
        feature_cols = [
            "age",
            "weight",
            "height",
            "bmi",
            "gender_encoded",
            "activity_encoded",
            "workout_frequency",
            "avg_workout_duration",
            "calories_burned",
            "diet_compliance",
            "sleep_hours",
            "stress_level",
            "motivation_level",
            "goal_type_encoded",
        ]

        X = data[feature_cols]

        # Scale features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled, data

    def train_models(self, data=None):
        """Train models for multiple prediction tasks"""
        if data is None:
            print("Generating synthetic data...")
            data = self.generate_synthetic_data(2000)

        X_scaled, processed_data = self.prepare_features(data)

        # Define target variables
        targets = {
            "weight_loss_4weeks": processed_data["weight_loss_4weeks"],
            "fitness_score_improvement": processed_data[
                "fitness_score_improvement"
            ],
            "goal_achievement_rate": processed_data["goal_achievement_rate"],
        }

        # Train separate models for each target
        for target_name, y in targets.items():
            print(f"Training model for {target_name}...")

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            # Model selection
            models = {
                "linear_regression": LinearRegression(),
                "random_forest": RandomForestRegressor(
                    n_estimators=100, random_state=42
                ),
                "gradient_boosting": GradientBoostingRegressor(
                    n_estimators=100, random_state=42
                ),
            }

            best_model = None
            best_score = -float("inf")

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if target_name == "goal_achievement_rate":
                    score = r2_score(y_test, y_pred)
                else:
                    score = -mean_squared_error(y_test, y_pred)

                if score > best_score:
                    best_score = score
                    best_model = model

            self.models[target_name] = best_model
            print(f"  Best model: {type(best_model).__name__}")

        return True

    def predict_user_progress(self, user_data):
        """Predict user progress for all metrics"""
        if not self.models:
            return None

        X_scaled, _ = self.prepare_features(pd.DataFrame([user_data]))
        predictions = {}

        for target_name, model in self.models.items():
            pred = model.predict(X_scaled)[0]
            predictions[target_name] = max(0, pred)

        return predictions

    def get_health_insights(self, user_data):
        """Get comprehensive health insights"""
        predictions = self.predict_user_progress(user_data)
        health_data = user_data.get("healthData", {})

        weekly_calories = health_data.get("weeklyCalories", [])
        avg_calories = (
            sum(weekly_calories) / max(len(weekly_calories), 1)
            if weekly_calories
            else 0
        )
        goal_calories = health_data.get("goal", 2500)

        return {
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
            "predictions": predictions,
            "nutrition_score": 85,
            "activity_level": "Moderate",
            "recommendations": [
                "Consider increasing protein intake for better muscle recovery",
                "Add more fiber-rich foods to your diet",
                "Maintain consistent meal timing for better metabolism",
            ],
        }


# =============================================================================
# NUTRITION RECOMMENDER
# =============================================================================


class NutritionRecommender:
    """
    Nutrition recommendation system using Indian food dataset.
    """

    def __init__(self, recipe_generator=None):
        self.food_data = None
        self.vectorizer = None
        self.scaler = None
        self.diet_classifier = None
        self.calorie_predictor = None
        self.similarity_matrix = None
        self.recipe_generator = recipe_generator

    def load_and_preprocess_data(self, csv_path="dataset/indian_food.csv"):
        """Load and preprocess the Indian food dataset"""
        try:
            self.food_data = pd.read_csv(csv_path)
            print(f"Loaded {len(self.food_data)} food items")

            # Clean the data
            self.food_data = self._clean_data()
            print(f"After cleaning: {len(self.food_data)} food items")

            return True
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            print(f"Error loading data: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error loading data: {e}")
            return False

    def _clean_data(self):
        """Clean and preprocess the food data"""
        df = self.food_data.copy()

        # Handle missing values
        df = df.dropna(subset=["name", "ingredients"])

        # Fill missing values with appropriate defaults
        df["diet"] = df["diet"].fillna("vegetarian")
        df["course"] = df["course"].fillna("main course")
        df["flavor_profile"] = df["flavor_profile"].fillna("spicy")
        df["state"] = df["state"].fillna("Unknown")
        df["region"] = df["region"].fillna("Unknown")

        # Convert time values to numeric
        df["prep_time"] = pd.to_numeric(
            df["prep_time"], errors="coerce"
        ).fillna(0)
        df["cook_time"] = pd.to_numeric(
            df["cook_time"], errors="coerce"
        ).fillna(0)

        # Clean ingredients text
        df["ingredients"] = df["ingredients"].apply(self._clean_ingredients)

        # Add nutritional values
        df = self._add_nutritional_info(df)

        return df

    def _clean_ingredients(self, ingredients):
        """Clean ingredients text"""
        if pd.isna(ingredients):
            return ""

        ingredients = re.sub(r"[^\w\s,]", "", str(ingredients))
        ingredients = " ".join(ingredients.split())
        return ingredients.lower()

    def _add_nutritional_info(self, df):
        """Add nutritional information"""
        nutrition_info = {
            "vegetarian": {
                "calories": 250,
                "protein": 8,
                "carbs": 30,
                "fat": 10,
                "fiber": 5,
            },
            "non vegetarian": {
                "calories": 300,
                "protein": 20,
                "carbs": 25,
                "fat": 15,
                "fiber": 3,
            },
        }

        def get_nutrition_value(diet_type, nutrient):
            if pd.isna(diet_type):
                if nutrient == "calories":
                    return 275
                elif nutrient == "protein":
                    return 14
                elif nutrient == "carbs":
                    return 27
                elif nutrient == "fat":
                    return 12
                else:
                    return 4
            else:
                diet_str = str(diet_type).strip().lower()
                default_values = {
                    "calories": 275,
                    "protein": 14,
                    "carbs": 27,
                    "fat": 12,
                    "fiber": 4,
                }

                for key, values in nutrition_info.items():
                    if key in diet_str:
                        return values.get(nutrient, default_values[nutrient])

                return default_values[nutrient]

        df["estimated_calories"] = df["diet"].apply(
            lambda x: get_nutrition_value(x, "calories")
        )
        df["estimated_protein"] = df["diet"].apply(
            lambda x: get_nutrition_value(x, "protein")
        )
        df["estimated_carbs"] = df["diet"].apply(
            lambda x: get_nutrition_value(x, "carbs")
        )
        df["estimated_fat"] = df["diet"].apply(
            lambda x: get_nutrition_value(x, "fat")
        )
        df["estimated_fiber"] = df["diet"].apply(
            lambda x: get_nutrition_value(x, "fiber")
        )

        return df

    def train_models(self):
        """Train recommendation models"""
        if self.food_data is None:
            print("No data loaded!")
            return False

        self._prepare_text_features()
        self._create_similarity_matrix()

        return True

    def _prepare_text_features(self):
        """Prepare text features for vectorization"""
        self.food_data["text_features"] = (
            self.food_data["name"]
            + " "
            + self.food_data["ingredients"]
            + " "
            + self.food_data["course"]
            + " "
            + self.food_data["flavor_profile"]
        )

        self.vectorizer = TfidfVectorizer(
            max_features=1000, stop_words="english", ngram_range=(1, 2)
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.food_data["text_features"]
        )

    def _create_similarity_matrix(self):
        """Create similarity matrix for content-based filtering"""
        self.similarity_matrix = cosine_similarity(
            self.tfidf_matrix, self.tfidf_matrix
        )

    def get_recommendations(self, preferences):
        """Get personalized nutrition recommendations with multi-cuisine support"""
        diet_type = preferences.get("diet_type", "balanced")
        calorie_goal = preferences.get("calorie_goal", 2000)
        health_condition = preferences.get("health_condition", None)
        cuisines = preferences.get("cuisines", [])
        if not cuisines and preferences.get("cuisine"):
            cuisines = [preferences.get("cuisine")]

        # Generate meal plan
        meal_plan = self._generate_meal_plan(
            diet_type, calorie_goal, health_condition=health_condition, cuisines=cuisines
        )

        return {
            "daily_meals": meal_plan,
            "calorie_goal": calorie_goal,
            "diet_type": diet_type,
            "cuisines": cuisines,
            "weekly_plan": [
                "Focus on whole grains, lean proteins, and healthy fats",
                "Enjoy balanced flavors from your preferred cuisines",
                "Stay hydrated with at least 2.5L of water daily",
                "Consolidate weekly ingredients to minimize food waste",
            ],
            "shopping_list": [
                "Cuisine-tailored herbs and spice blends",
                "Fresh seasonal vegetables and produce",
                "Quality plant-based and lean animal proteins",
                "Complex carbohydrate bases (rice, whole wheat, noodles)",
            ],
        }

    def _generate_meal_plan(
        self, diet_type, total_calories, meals_per_day=3, health_condition=None, cuisines=None
    ):
        """Generate meal plan with health condition and multi-cuisine support"""
        target_cal_per_meal = total_calories / meals_per_day

        def detect_recipe_cuisine(row):
            text = f"{row.get('Keywords', '')} {row.get('Name', '')} {row.get('Description', '')}".lower()
            if any(k in text for k in ["indian", "curry", "tikka", "dal", "paneer", "masala", "roti", "biryani", "chana", "sambar", "dosa", "idli", "khichdi"]):
                return "Indian"
            elif any(k in text for k in ["chinese", "japanese", "thai", "vietnamese", "korean", "ramen", "noodle", "stir fry", "satay", "pad thai", "sushi", "wonton", "tofu", "teriyaki", "miso"]):
                return "Asian"
            elif any(k in text for k in ["mediterranean", "greek", "italian", "olive", "pasta", "salad", "hummus", "pesto", "feta", "risotto", "bruschetta"]):
                return "Mediterranean"
            elif any(k in text for k in ["mexican", "taco", "fajita", "salsa", "burrito", "enchilada", "bean", "guacamole", "quesadilla"]):
                return "Mexican"
            return "Continental"

        # Try to use recipes from RecipeGenerator if available
        if self.recipe_generator and self.recipe_generator.recipes_data is not None:
            # Filter recipes by diet type and health condition
            filter_prefs = {"diet_type": diet_type}
            if health_condition:
                filter_prefs["health_condition"] = health_condition
            filtered_recipes = self.recipe_generator._filter_by_dietary_restrictions(
                filter_prefs
            )

            # Assign cuisine to each recipe
            filtered_recipes = filtered_recipes.copy()
            filtered_recipes["detected_cuisine"] = filtered_recipes.apply(detect_recipe_cuisine, axis=1)

            # If user selected specific cuisines, prioritize them
            if cuisines:
                norm_cuisines = [str(c).lower().strip() for c in cuisines if str(c).lower() != "all"]
                if norm_cuisines:
                    cuisine_matched = filtered_recipes[
                        filtered_recipes["detected_cuisine"].str.lower().isin(norm_cuisines)
                    ]
                    if len(cuisine_matched) >= 3:
                        filtered_recipes = cuisine_matched

            # Select meals for each meal time
            meal_plan = {"breakfast": [], "lunch": [], "dinner": []}
            meal_times = ["breakfast", "lunch", "dinner"]
            meal_keywords = {
                "breakfast": ["breakfast", "morning", "oat", "cereal", "toast", "egg", "pancake", "smoothie", "yogurt", "upma", "dosa", "idli", "waffle"],
                "lunch": ["lunch", "sandwich", "salad", "soup", "rice", "noodle", "wrap", "bowl", "curry", "dal", "pasta", "satay", "pad thai"],
                "dinner": ["dinner", "chicken", "beef", "fish", "pasta", "curry", "stew", "roast", "grill", "tikka", "ramen", "stir fry", "paneer"]
            }

            for meal_time in meal_times:
                keywords = meal_keywords[meal_time]
                # Filter recipes by keywords in name or category
                matched_pool = filtered_recipes[
                    filtered_recipes["Name"].str.lower().apply(
                        lambda x: any(k in str(x).lower() for k in keywords)
                    )
                ]

                # If specific keyword match is small, use general filtered pool
                if len(matched_pool) < 2:
                    matched_pool = filtered_recipes

                if not matched_pool.empty:
                    pool = matched_pool.copy()
                    pool["calorie_diff"] = abs(pool["Calories"] - target_cal_per_meal)
                    selected_recipes = pool.sort_values("calorie_diff").head(3)

                    selected_meals = []
                    for _, recipe in selected_recipes.iterrows():
                        selected_meals.append({
                            "name": recipe["Name"],
                            "calories": int(recipe["Calories"]),
                            "protein_grams": round(float(recipe.get("ProteinContent", 0)), 1),
                            "carb_grams": round(float(recipe.get("CarbohydrateContent", 0)), 1),
                            "fat_grams": round(float(recipe.get("FatContent", 0)), 1),
                            "ingredients": recipe["RecipeIngredientParts"] if isinstance(recipe["RecipeIngredientParts"], list) else [str(recipe["RecipeIngredientParts"])],
                            "instructions": recipe["RecipeInstructions"] if isinstance(recipe["RecipeInstructions"], list) else [str(recipe["RecipeInstructions"])],
                            "prep_time": str(recipe.get("PrepTime", "15m")),
                            "cook_time": str(recipe.get("CookTime", "20m")),
                            "cuisine": recipe.get("detected_cuisine", "Continental"),
                        })
                    meal_plan[meal_time] = selected_meals
                else:
                    meal_plan[meal_time] = self._get_fallback_meals_for_time(
                        target_cal_per_meal, meal_time, health_condition
                    )

            return meal_plan
        elif self.food_data is not None:
            # Original food-based logic
            filtered = self.food_data[self.food_data["diet"] == diet_type]
            if filtered.empty:
                filtered = self.food_data  # fallback to all foods

            # Apply health condition filters
            if health_condition:
                filtered = self._apply_health_condition_filters(
                    filtered, health_condition
                )

            # Select meals for each meal time
            meal_plan = {"breakfast": [], "lunch": [], "dinner": []}
            meal_times = ["breakfast", "lunch", "dinner"]

            for meal_time in meal_times:
                # Filter by course type if available
                meal_options = filtered[
                    filtered["course"].str.contains(
                        meal_time, case=False, na=False
                    )
                ]

                if meal_options.empty:
                    meal_options = filtered  # fallback to all filtered foods

                # Shuffle meal options to increase variety
                meal_options = meal_options.sample(frac=1).reset_index(
                    drop=True
                )

                # Select meals close to target calories per meal
                selected_meals = []
                calories_accum = 0

                for _, meal in meal_options.iterrows():
                    meal_calories = meal.get("estimated_calories", 0)
                    if (
                        calories_accum + meal_calories <= target_cal_per_meal
                        or not selected_meals
                    ):
                        selected_meals.append(
                            {
                                "name": meal["name"],
                                "calories": meal_calories,
                                "protein_grams": meal.get(
                                    "estimated_protein", 0
                                ),
                                "carb_grams": meal.get("estimated_carbs", 0),
                                "fat_grams": meal.get("estimated_fat", 0),
                                "ingredients": meal.get(
                                    "ingredients", ""
                                ).split(","),
                                "prep_time": meal.get("prep_time", 0),
                                "cook_time": meal.get("cook_time", 0),
                            }
                        )
                        calories_accum += meal_calories

                    if calories_accum >= target_cal_per_meal:
                        break

                meal_plan[meal_time] = selected_meals

            return meal_plan
        else:
            # Fallback meal plan when no data
            return self._get_fallback_meal_plan(
                target_cal_per_meal, health_condition
            )

    def _apply_health_condition_filters(self, df, health_condition):
        """Apply filters based on health condition"""
        if health_condition == "diabetes":
            # Low carb, high fiber foods
            return df[df["estimated_carbs"] < 30]
        elif health_condition == "hypertension":
            # Low sodium foods (assuming low salt in ingredients)
            return df[
                ~df["ingredients"].str.contains(
                    "salt|soy sauce|sodium", case=False
                )
            ]
        elif health_condition == "thyroid":
            # Foods rich in iodine and selenium
            return df[
                df["ingredients"].str.contains(
                    "seafood|fish|nuts|seeds", case=False
                )
            ]
        elif health_condition == "pcos":
            # Low glycemic index foods
            return df[df["estimated_carbs"] < 25]
        elif health_condition == "high_cholesterol":
            # Low fat, high fiber foods
            return df[(df["estimated_fat"] < 15) & (df["estimated_fiber"] > 5)]
        return df

    def _get_fallback_meal_plan(
        self, target_cal_per_meal, health_condition=None
    ):
        """Fallback meal plan when no dataset is available"""
        if health_condition == "diabetes":
            return {
                "breakfast": [
                    {
                        "name": "Boiled eggs with whole wheat toast",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 18,
                        "carb_grams": 15,
                        "fat_grams": 8,
                        "ingredients": ["eggs", "whole wheat bread", "black pepper"],
                        "prep_time": 5,
                        "cook_time": 5,
                    },
                    {
                        "name": "Curd (Dahi) with oats",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 12,
                        "carb_grams": 25,
                        "fat_grams": 6,
                        "ingredients": ["curd", "oats", "chia seeds"],
                        "prep_time": 5,
                        "cook_time": 0,
                    },
                    {
                        "name": "Peanut butter toast",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 10,
                        "carb_grams": 20,
                        "fat_grams": 12,
                        "ingredients": ["whole wheat bread", "peanut butter"],
                        "prep_time": 5,
                        "cook_time": 0,
                    },
                ],
                "lunch": [
                    {
                        "name": "Lentil soup (Dal) with brown rice",
                        "calories": target_cal_per_meal * 0.4,
                        "protein_grams": 22,
                        "carb_grams": 45,
                        "fat_grams": 6,
                        "ingredients": ["yellow lentils", "brown rice", "onion", "tomato"],
                        "prep_time": 10,
                        "cook_time": 20,
                    },
                    {
                        "name": "Sprouted moong salad",
                        "calories": target_cal_per_meal * 0.4,
                        "protein_grams": 18,
                        "carb_grams": 30,
                        "fat_grams": 4,
                        "ingredients": ["moong sprouts", "cucumber", "tomato", "lemon"],
                        "prep_time": 10,
                        "cook_time": 0,
                    },
                    {
                        "name": "Vegetable khichdi",
                        "calories": target_cal_per_meal * 0.4,
                        "protein_grams": 14,
                        "carb_grams": 50,
                        "fat_grams": 5,
                        "ingredients": ["rice", "moong dal", "mixed seasonal vegetables"],
                        "prep_time": 10,
                        "cook_time": 15,
                    },
                ],
                "dinner": [
                    {
                        "name": "Tofu/Paneer scramble with roti",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 20,
                        "carb_grams": 30,
                        "fat_grams": 10,
                        "ingredients": ["tofu or paneer", "onion", "tomato", "whole wheat roti"],
                        "prep_time": 10,
                        "cook_time": 10,
                    },
                    {
                        "name": "Egg bhurji (scrambled) with roti",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 22,
                        "carb_grams": 25,
                        "fat_grams": 12,
                        "ingredients": ["eggs", "onion", "tomato", "whole wheat roti"],
                        "prep_time": 5,
                        "cook_time": 10,
                    },
                    {
                        "name": "Chickpea (Chana) curry with brown rice",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 16,
                        "carb_grams": 40,
                        "fat_grams": 8,
                        "ingredients": ["chickpeas", "spices", "brown rice"],
                        "prep_time": 15,
                        "cook_time": 25,
                    },
                ],
            }
        elif health_condition == "hypertension":
            return {
                "breakfast": [
                    {
                        "name": "Oatmeal porridge with banana",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 8,
                        "carb_grams": 45,
                        "fat_grams": 5,
                        "ingredients": ["oats", "milk", "banana"],
                        "prep_time": 5,
                        "cook_time": 5,
                    },
                    {
                        "name": "Peanut butter banana toast",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 12,
                        "carb_grams": 40,
                        "fat_grams": 12,
                        "ingredients": ["whole wheat bread", "peanut butter", "banana"],
                        "prep_time": 5,
                        "cook_time": 0,
                    },
                    {
                        "name": "Local Curd (Dahi) with seasonal fruits",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 10,
                        "carb_grams": 30,
                        "fat_grams": 6,
                        "ingredients": ["curd", "banana", "papaya"],
                        "prep_time": 5,
                        "cook_time": 0,
                    },
                ],
                "lunch": [
                    {
                        "name": "Lentil stew with vegetables",
                        "calories": target_cal_per_meal * 0.4,
                        "protein_grams": 18,
                        "carb_grams": 40,
                        "fat_grams": 5,
                        "ingredients": ["yellow lentils", "carrots", "spinach", "herbs"],
                        "prep_time": 10,
                        "cook_time": 20,
                    },
                    {
                        "name": "Egg bhurji (scrambled) with roti",
                        "calories": target_cal_per_meal * 0.4,
                        "protein_grams": 22,
                        "carb_grams": 25,
                        "fat_grams": 12,
                        "ingredients": ["eggs", "onion", "tomato", "whole wheat roti"],
                        "prep_time": 5,
                        "cook_time": 10,
                    },
                    {
                        "name": "Sprouted moong salad",
                        "calories": target_cal_per_meal * 0.4,
                        "protein_grams": 18,
                        "carb_grams": 30,
                        "fat_grams": 4,
                        "ingredients": ["moong sprouts", "cucumber", "tomato", "lemon"],
                        "prep_time": 10,
                        "cook_time": 0,
                    },
                ],
                "dinner": [
                    {
                        "name": "Chickpea (Chana) salad",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 15,
                        "carb_grams": 35,
                        "fat_grams": 6,
                        "ingredients": ["boiled chickpeas", "cucumber", "onion", "tomato", "lemon"],
                        "prep_time": 10,
                        "cook_time": 0,
                    },
                    {
                        "name": "Tofu/Paneer scramble with roti",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 20,
                        "carb_grams": 30,
                        "fat_grams": 10,
                        "ingredients": ["tofu or paneer", "onion", "tomato", "whole wheat roti"],
                        "prep_time": 10,
                        "cook_time": 10,
                    },
                    {
                        "name": "Vegetable khichdi",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 12,
                        "carb_grams": 45,
                        "fat_grams": 4,
                        "ingredients": ["rice", "moong dal", "mixed seasonal vegetables"],
                        "prep_time": 10,
                        "cook_time": 15,
                    },
                ],
            }
        elif health_condition == "thyroid":
            return {
                "breakfast": [
                    {
                        "name": "Curd (Dahi) with peanuts and honey",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 15,
                        "carb_grams": 20,
                        "fat_grams": 12,
                        "ingredients": ["curd", "peanuts", "honey"],
                        "prep_time": 5,
                        "cook_time": 0,
                    },
                    {
                        "name": "Oatmeal with sliced apple",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 10,
                        "carb_grams": 40,
                        "fat_grams": 5,
                        "ingredients": ["oats", "milk", "apple"],
                        "prep_time": 5,
                        "cook_time": 5,
                    },
                    {
                        "name": "Eggs with spinach bhaji",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 18,
                        "carb_grams": 10,
                        "fat_grams": 12,
                        "ingredients": ["eggs", "spinach", "mustard oil"],
                        "prep_time": 10,
                        "cook_time": 5,
                    },
                ],
                "lunch": [
                    {
                        "name": "Moong Dal with roti and salad",
                        "calories": target_cal_per_meal * 0.4,
                        "protein_grams": 20,
                        "carb_grams": 45,
                        "fat_grams": 6,
                        "ingredients": ["moong dal", "whole wheat roti", "cucumber", "tomato"],
                        "prep_time": 15,
                        "cook_time": 15,
                    },
                    {
                        "name": "Egg curry with rice",
                        "calories": target_cal_per_meal * 0.4,
                        "protein_grams": 24,
                        "carb_grams": 50,
                        "fat_grams": 12,
                        "ingredients": ["eggs", "spices", "rice"],
                        "prep_time": 15,
                        "cook_time": 20,
                    },
                    {
                        "name": "Chickpea (Chana) curry with roti",
                        "calories": target_cal_per_meal * 0.4,
                        "protein_grams": 18,
                        "carb_grams": 45,
                        "fat_grams": 8,
                        "ingredients": ["chickpeas", "whole wheat roti", "onion", "tomato"],
                        "prep_time": 15,
                        "cook_time": 25,
                    },
                ],
                "dinner": [
                    {
                        "name": "Vegetable khichdi with curd",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 14,
                        "carb_grams": 45,
                        "fat_grams": 5,
                        "ingredients": ["rice", "moong dal", "seasonal vegetables", "curd"],
                        "prep_time": 10,
                        "cook_time": 15,
                    },
                    {
                        "name": "Paneer scramble (Bhurji) with roti",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 22,
                        "carb_grams": 25,
                        "fat_grams": 12,
                        "ingredients": ["paneer", "onion", "tomato", "roti"],
                        "prep_time": 10,
                        "cook_time": 10,
                    },
                    {
                        "name": "Sprouted moong salad",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 16,
                        "carb_grams": 25,
                        "fat_grams": 4,
                        "ingredients": ["moong sprouts", "cucumber", "tomato", "lemon juice"],
                        "prep_time": 10,
                        "cook_time": 0,
                    },
                ],
            }
        elif health_condition == "pcos":
            return {
                "breakfast": [
                    {
                        "name": "Oatmeal porridge with flax seeds",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 10,
                        "carb_grams": 35,
                        "fat_grams": 6,
                        "ingredients": ["oats", "milk", "flax seeds"],
                        "prep_time": 5,
                        "cook_time": 5,
                    },
                    {
                        "name": "Boiled eggs and roasted peanuts",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 18,
                        "carb_grams": 15,
                        "fat_grams": 14,
                        "ingredients": ["eggs", "peanuts"],
                        "prep_time": 5,
                        "cook_time": 5,
                    },
                    {
                        "name": "Moong Dal chilla (pancake)",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 12,
                        "carb_grams": 25,
                        "fat_grams": 5,
                        "ingredients": ["moong dal batter", "onion", "green chili"],
                        "prep_time": 10,
                        "cook_time": 10,
                    },
                ],
                "lunch": [
                    {
                        "name": "Sprouted moong salad with curd",
                        "calories": target_cal_per_meal * 0.4,
                        "protein_grams": 18,
                        "carb_grams": 30,
                        "fat_grams": 6,
                        "ingredients": ["moong sprouts", "cucumber", "tomato", "curd"],
                        "prep_time": 10,
                        "cook_time": 0,
                    },
                    {
                        "name": "Paneer curry with brown rice",
                        "calories": target_cal_per_meal * 0.4,
                        "protein_grams": 22,
                        "carb_grams": 40,
                        "fat_grams": 12,
                        "ingredients": ["paneer", "spices", "brown rice"],
                        "prep_time": 15,
                        "cook_time": 20,
                    },
                    {
                        "name": "Lentil soup (Dal) with whole wheat roti",
                        "calories": target_cal_per_meal * 0.4,
                        "protein_grams": 20,
                        "carb_grams": 45,
                        "fat_grams": 6,
                        "ingredients": ["yellow lentils", "whole wheat roti", "salad"],
                        "prep_time": 15,
                        "cook_time": 15,
                    },
                ],
                "dinner": [
                    {
                        "name": "Chickpea curry with whole wheat roti",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 16,
                        "carb_grams": 40,
                        "fat_grams": 8,
                        "ingredients": ["chickpeas", "spices", "whole wheat roti"],
                        "prep_time": 15,
                        "cook_time": 25,
                    },
                    {
                        "name": "Egg bhurji (scrambled) with salad",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 22,
                        "carb_grams": 15,
                        "fat_grams": 12,
                        "ingredients": ["eggs", "onion", "tomato", "cucumber"],
                        "prep_time": 5,
                        "cook_time": 10,
                    },
                    {
                        "name": "Tofu stir fry with seasonal veggies",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 18,
                        "carb_grams": 20,
                        "fat_grams": 8,
                        "ingredients": ["tofu", "broccoli", "carrots", "soy sauce"],
                        "prep_time": 10,
                        "cook_time": 10,
                    },
                ],
            }
        elif health_condition == "high_cholesterol":
            return {
                "breakfast": [
                    {
                        "name": "Oatmeal porridge with sliced apple",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 8,
                        "carb_grams": 45,
                        "fat_grams": 4,
                        "ingredients": ["oats", "milk", "apple"],
                        "prep_time": 10,
                        "cook_time": 5,
                    },
                    {
                        "name": "Whole wheat toast with peanut butter",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 10,
                        "carb_grams": 40,
                        "fat_grams": 8,
                        "ingredients": ["whole wheat bread", "peanut butter"],
                        "prep_time": 5,
                        "cook_time": 0,
                    },
                    {
                        "name": "Local Curd (Dahi) with honey",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 12,
                        "carb_grams": 25,
                        "fat_grams": 5,
                        "ingredients": ["curd", "honey", "chia seeds"],
                        "prep_time": 5,
                        "cook_time": 0,
                    },
                ],
                "lunch": [
                    {
                        "name": "Vegetable & Lentil soup",
                        "calories": target_cal_per_meal * 0.4,
                        "protein_grams": 14,
                        "carb_grams": 35,
                        "fat_grams": 4,
                        "ingredients": ["lentils", "carrots", "spinach", "herbs"],
                        "prep_time": 15,
                        "cook_time": 20,
                    },
                    {
                        "name": "Sprouted moong salad",
                        "calories": target_cal_per_meal * 0.4,
                        "protein_grams": 18,
                        "carb_grams": 30,
                        "fat_grams": 4,
                        "ingredients": ["moong sprouts", "cucumber", "tomato", "lemon juice"],
                        "prep_time": 10,
                        "cook_time": 0,
                    },
                    {
                        "name": "Chickpea (Chana) salad",
                        "calories": target_cal_per_meal * 0.4,
                        "protein_grams": 16,
                        "carb_grams": 35,
                        "fat_grams": 5,
                        "ingredients": ["boiled chickpeas", "onion", "tomato", "cucumber", "lemon"],
                        "prep_time": 15,
                        "cook_time": 0,
                    },
                ],
                "dinner": [
                    {
                        "name": "Tofu scramble with whole wheat roti",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 18,
                        "carb_grams": 30,
                        "fat_grams": 6,
                        "ingredients": ["tofu", "onion", "tomato", "whole wheat roti"],
                        "prep_time": 10,
                        "cook_time": 15,
                    },
                    {
                        "name": "Vegetable khichdi",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 12,
                        "carb_grams": 45,
                        "fat_grams": 4,
                        "ingredients": ["rice", "moong dal", "seasonal vegetables"],
                        "prep_time": 15,
                        "cook_time": 20,
                    },
                    {
                        "name": "Moong Dal soup with roti",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 14,
                        "carb_grams": 30,
                        "fat_grams": 4,
                        "ingredients": ["moong dal", "herbs", "whole wheat roti"],
                        "prep_time": 15,
                        "cook_time": 15,
                    },
                ],
            }
        else:
            return {
                "breakfast": [
                    {
                        "name": "Oatmeal porridge with banana",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 8,
                        "carb_grams": 45,
                        "fat_grams": 5,
                        "ingredients": ["oats", "milk", "banana"],
                        "prep_time": 10,
                        "cook_time": 5,
                    },
                    {
                        "name": "Scrambled eggs with whole wheat toast",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 18,
                        "carb_grams": 35,
                        "fat_grams": 10,
                        "ingredients": ["eggs", "whole wheat bread", "vegetable oil"],
                        "prep_time": 10,
                        "cook_time": 5,
                    },
                    {
                        "name": "Local Curd (Dahi) with honey and peanuts",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 12,
                        "carb_grams": 25,
                        "fat_grams": 8,
                        "ingredients": ["curd", "honey", "peanuts"],
                        "prep_time": 10,
                        "cook_time": 0,
                    },
                ],
                "lunch": [
                    {
                        "name": "Egg bhurji (scrambled) with whole wheat roti",
                        "calories": target_cal_per_meal * 0.4,
                        "protein_grams": 24,
                        "carb_grams": 30,
                        "fat_grams": 12,
                        "ingredients": ["eggs", "onion", "tomato", "whole wheat roti"],
                        "prep_time": 15,
                        "cook_time": 10,
                    },
                    {
                        "name": "Lentil soup (Dal) with brown rice",
                        "calories": target_cal_per_meal * 0.4,
                        "protein_grams": 20,
                        "carb_grams": 45,
                        "fat_grams": 6,
                        "ingredients": ["yellow lentils", "brown rice", "spices"],
                        "prep_time": 15,
                        "cook_time": 15,
                    },
                    {
                        "name": "Chickpea (Chana) salad",
                        "calories": target_cal_per_meal * 0.4,
                        "protein_grams": 16,
                        "carb_grams": 35,
                        "fat_grams": 6,
                        "ingredients": ["boiled chickpeas", "onion", "cucumber", "tomato", "lemon"],
                        "prep_time": 10,
                        "cook_time": 0,
                    },
                ],
                "dinner": [
                    {
                        "name": "Paneer curry with whole wheat roti",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 22,
                        "carb_grams": 35,
                        "fat_grams": 12,
                        "ingredients": ["paneer", "spices", "whole wheat roti"],
                        "prep_time": 10,
                        "cook_time": 15,
                    },
                    {
                        "name": "Vegetable khichdi",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 12,
                        "carb_grams": 45,
                        "fat_grams": 4,
                        "ingredients": ["rice", "moong dal", "mixed seasonal vegetables"],
                        "prep_time": 15,
                        "cook_time": 15,
                    },
                    {
                        "name": "Chickpea (Chana) curry with brown rice",
                        "calories": target_cal_per_meal * 0.3,
                        "protein_grams": 16,
                        "carb_grams": 40,
                        "fat_grams": 8,
                        "ingredients": ["chickpeas", "spices", "brown rice"],
                        "prep_time": 10,
                        "cook_time": 20,
                    },
                ],
            }


# =============================================================================
# HYDRATION RECOMMENDER
# =============================================================================


class HydrationRecommender:
    """
    AI-powered hydration recommender that provides personalized water intake recommendations
    based on user activity level, climate, health conditions, and other factors.
    """

    def __init__(self):
        self.base_water_intake = 2000  # Base daily water intake in ml
        self.activity_multipliers = {
            "sedentary": 1.0,
            "light": 1.2,
            "moderate": 1.4,
            "active": 1.6,
            "very_active": 1.8,
        }
        self.climate_multipliers = {
            "cold": 0.9,
            "moderate": 1.0,
            "warm": 1.2,
            "hot": 1.4,
            "very_hot": 1.6,
        }
        self.health_condition_multipliers = {
            "diabetes": 1.3,  # Higher water needs for blood sugar regulation
            "hypertension": 1.1,  # Slightly higher for medication effects
            "kidney_disease": 0.8,  # Lower intake for kidney protection
            "heart_disease": 1.0,  # Normal intake
            "thyroid": 1.2,  # Higher for metabolism
            "pregnancy": 1.3,  # Higher for amniotic fluid and blood volume
            "breastfeeding": 1.5,  # Higher for milk production
            "fever": 1.4,  # Higher for temperature regulation
            "diarrhea": 1.5,  # Higher for fluid replacement
            "none": 1.0,
        }

    def calculate_daily_water_goal(self, user_profile):
        """
        Calculate personalized daily water intake goal based on user profile.

        Args:
            user_profile (dict): User profile containing age, weight, activity level, etc.

        Returns:
            dict: Daily water goal and recommendations
        """
        # Extract user data
        age = user_profile.get("age", 30)
        weight = user_profile.get("weight", 70)  # in kg
        activity_level = user_profile.get("activity_level", "moderate").lower()
        climate = user_profile.get("climate", "moderate").lower()
        health_conditions = user_profile.get("health_conditions", [])
        gender = user_profile.get("gender", "other").lower()

        # Base calculation using weight (30ml per kg is a common rule)
        weight_based_intake = weight * 30

        # Apply activity multiplier
        activity_multiplier = self.activity_multipliers.get(
            activity_level, 1.0
        )
        adjusted_intake = weight_based_intake * activity_multiplier

        # Apply climate multiplier
        climate_multiplier = self.climate_multipliers.get(climate, 1.0)
        adjusted_intake *= climate_multiplier

        # Apply health condition multipliers
        health_multiplier = 1.0
        for condition in health_conditions:
            condition_multiplier = self.health_condition_multipliers.get(
                condition.lower(), 1.0
            )
            health_multiplier = max(
                health_multiplier, condition_multiplier
            )  # Take the highest multiplier

        adjusted_intake *= health_multiplier

        # Age adjustments
        if age < 18:
            adjusted_intake *= 0.9  # Slightly less for teenagers
        elif age > 65:
            adjusted_intake *= 0.95  # Slightly less for elderly

        # Gender adjustments (slight differences)
        if gender == "female":
            adjusted_intake *= 0.95
        elif gender == "male":
            adjusted_intake *= 1.05

        # Ensure reasonable bounds
        daily_goal = max(1500, min(4000, adjusted_intake))

        return {
            "daily_goal_ml": round(daily_goal),
            "daily_goal_ounces": round(daily_goal * 0.033814, 1),
        "daily_goal_glasses": round(daily_goal / 250, 1),
            "factors": {
                "weight_based": round(weight_based_intake),
                "activity_multiplier": activity_multiplier,
                "climate_multiplier": climate_multiplier,
                "health_multiplier": health_multiplier,
            },
        }

    def get_hydration_schedule(
        self, daily_goal_ml, wake_time="06:00", sleep_time="22:00"
    ):
        """
        Create a personalized hydration schedule throughout the day.

        Args:
            daily_goal_ml (int): Daily water goal in ml
            wake_time (str): Wake up time in HH:MM format
            sleep_time (str): Sleep time in HH:MM format

        Returns:
            list: Hourly hydration schedule
        """
        # Calculate awake hours
        wake_hour = int(wake_time.split(":")[0])
        sleep_hour = int(sleep_time.split(":")[0])
        awake_hours = (
            sleep_hour - wake_hour
            if sleep_hour > wake_hour
            else 24 - wake_hour + sleep_hour
        )

        # Distribute water intake throughout awake hours
        # More water in morning and afternoon, less in evening
        schedule = []
        remaining_water = daily_goal_ml

        for hour in range(wake_hour, wake_hour + awake_hours):
            hour_24 = hour % 24

            if 6 <= hour_24 <= 10:  # Morning (6-10)
                intake = daily_goal_ml * 0.25  # 25% in morning
            elif 11 <= hour_24 <= 15:  # Midday (11-15)
                intake = daily_goal_ml * 0.35  # 35% midday
            elif 16 <= hour_24 <= 20:  # Afternoon/Evening (16-20)
                intake = daily_goal_ml * 0.30  # 30% afternoon
            else:  # Late night (21-5)
                intake = daily_goal_ml * 0.10  # 10% evening

            intake = min(intake, remaining_water)
            remaining_water -= intake

            schedule.append(
                {
                    "hour": hour_24,
                    "time_range": f"{hour_24:02d}:00-{hour_24+1:02d}:00",
                    "recommended_intake_ml": round(intake),
                    "recommended_intake_oz": round(intake * 0.033814, 1),
                    "cumulative_ml": round(daily_goal_ml - remaining_water),
                }
            )

        return schedule

    def get_hydration_tips(self, user_profile, current_intake_ml=0):
        """
        Get personalized hydration tips based on user profile and current intake.

        Args:
            user_profile (dict): User profile information
            current_intake_ml (int): Current day's water intake in ml

        Returns:
            dict: Hydration tips and reminders
        """
        daily_goal = self.calculate_daily_water_goal(user_profile)
        goal_ml = daily_goal["daily_goal_ml"]

        progress_percentage = (
            (current_intake_ml / goal_ml) * 100 if goal_ml > 0 else 0
        )

        tips = []

        # General tips
        tips.append(
            "Drink water throughout the day, don't wait until you're thirsty"
        )
        tips.append("Carry a reusable water bottle for easy access")

        # Activity-based tips
        activity_level = user_profile.get("activity_level", "moderate").lower()
        if activity_level in ["active", "very_active"]:
            tips.append("Drink extra water before, during, and after exercise")
            tips.append(
                "Replace each pound of sweat loss with 16-24 oz of water"
            )

        # Climate-based tips
        climate = user_profile.get("climate", "moderate").lower()
        if climate in ["warm", "hot", "very_hot"]:
            tips.append("Increase water intake in hot weather")
            tips.append("Drink water even if you don't feel thirsty in heat")

        # Health condition tips
        health_conditions = user_profile.get("health_conditions", [])
        if "diabetes" in [h.lower() for h in health_conditions]:
            tips.append(
                "Monitor blood sugar levels when increasing water intake"
            )
        if "kidney_disease" in [h.lower() for h in health_conditions]:
            tips.append(
                "Consult your doctor about appropriate water intake levels"
            )
        if "pregnancy" in [h.lower() for h in health_conditions]:
            tips.append(
                "Stay well-hydrated for amniotic fluid and blood volume"
            )

        # Progress-based tips
        if progress_percentage < 50:
            tips.append("Set reminders on your phone to drink water regularly")
            tips.append(
                "Keep water visible - place bottles in areas you frequent"
            )
        elif progress_percentage >= 80:
            tips.append("Great job staying hydrated! Keep up the good work")

        return {
            "progress_percentage": round(progress_percentage, 1),
            "remaining_ml": max(0, goal_ml - current_intake_ml),
            "tips": tips[:5],  # Limit to 5 most relevant tips
            "next_reminder": self._calculate_next_reminder(
                current_intake_ml, goal_ml
            ),
        }

    def _calculate_next_reminder(self, current_intake, goal_ml):
        """Calculate when the next hydration reminder should be"""
        progress = current_intake / goal_ml if goal_ml > 0 else 0

        if progress < 0.3:
            return "Drink your first glass of water for the day!"
        elif progress < 0.6:
            return "Time for your mid-morning water break"
        elif progress < 0.8:
            return "Afternoon hydration check - drink up!"
        else:
            return "Almost at your daily goal - keep it up!"

    def track_hydration_streak(self, water_history):
        """
        Track hydration consistency and streaks.

        Args:
            water_history (list): List of daily water intake for past days

        Returns:
            dict: Streak information and achievements
        """
        if not water_history:
            return {
                "current_streak": 0,
                "longest_streak": 0,
                "achievements": [],
            }

        # Assume goal is met if intake >= 80% of typical goal (adjustable)
        typical_goal = 2000  # This could be personalized
        goal_threshold = typical_goal * 0.8

        current_streak = 0
        longest_streak = 0
        temp_streak = 0

        for intake in reversed(
            water_history
        ):  # Check from most recent backwards
            if intake >= goal_threshold:
                temp_streak += 1
                current_streak = temp_streak
            else:
                break

        # Calculate longest streak
        temp_streak = 0
        for intake in water_history:
            if intake >= goal_threshold:
                temp_streak += 1
                longest_streak = max(longest_streak, temp_streak)
            else:
                temp_streak = 0

        achievements = []
        if current_streak >= 7:
            achievements.append("Week Warrior - 7 day hydration streak!")
        if current_streak >= 30:
            achievements.append("Hydration Hero - 30 day streak!")
        if longest_streak >= 50:
            achievements.append("Water Master - 50+ day record!")

        return {
            "current_streak": current_streak,
            "longest_streak": longest_streak,
            "achievements": achievements,
        }


# =============================================================================
# RECIPE GENERATOR
# =============================================================================


class RecipeGenerator:
    """
    AI-powered recipe generator that suggests recipes based on available ingredients,
    dietary preferences, and user skill level.
    """

    def __init__(self):
        self.recipes_data = None
        self.ingredient_index = {}
        self.complexity_levels = {
            "beginner": {
                "max_steps": 5,
                "max_time": 30,
                "simple_techniques": True,
            },
            "intermediate": {
                "max_steps": 8,
                "max_time": 60,
                "moderate_techniques": True,
            },
            "advanced": {
                "max_steps": 15,
                "max_time": 120,
                "complex_techniques": True,
            },
        }
        # Load recipe dataset on initialization
        self.load_recipes_data()

    def load_recipes_data(self, csv_path="dataset/recipes_dataset.csv"):
        """Load and preprocess the recipes dataset"""
        try:
            self.recipes_data = pd.read_csv(csv_path)
            print(f"Loaded {len(self.recipes_data)} recipes")

            # Clean and preprocess the data
            self.recipes_data = self._clean_recipes_data()
            print(f"After cleaning: {len(self.recipes_data)} recipes")

            # Build ingredient index for faster matching
            self._build_ingredient_index()

            return True
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            print(f"Error loading recipes data: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error loading recipes data: {e}")
            return False

    def _clean_recipes_data(self):
        """Clean and preprocess recipes data"""
        df = self.recipes_data.copy()

        # Handle missing values
        df = df.dropna(subset=["Name", "RecipeIngredientParts"])

        # Fill missing nutritional values with defaults
        nutritional_cols = [
            "Calories",
            "FatContent",
            "SaturatedFatContent",
            "CholesterolContent",
            "SodiumContent",
            "CarbohydrateContent",
            "FiberContent",
            "SugarContent",
            "ProteinContent",
        ]

        for col in nutritional_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Clean ingredient parts
        df["RecipeIngredientParts"] = df["RecipeIngredientParts"].apply(
            self._clean_ingredients_list
        )

        # Clean instructions
        if "RecipeInstructions" not in df.columns:
            if "recipe" in df.columns:
                df["RecipeInstructions"] = df["recipe"]
            elif "Instructions" in df.columns:
                df["RecipeInstructions"] = df["Instructions"]
            else:
                df["RecipeInstructions"] = ""

        df["RecipeInstructions"] = df["RecipeInstructions"].apply(
            self._clean_instructions
        )

        # Add complexity score
        df["complexity_score"] = df.apply(self._calculate_complexity, axis=1)

        # Add preparation time in minutes
        df["total_time_minutes"] = df.apply(self._calculate_total_time, axis=1)

        return df

    def _clean_ingredients_list(self, ingredients):
        """Clean ingredients list"""
        if pd.isna(ingredients):
            return []

        try:
            # Handle string representation of list
            if isinstance(ingredients, str):
                # Remove brackets and quotes, split by comma
                ingredients = (
                    ingredients.strip("c()")
                    .replace('"', "")
                    .replace("'", "")
                )
                ingredients_list = [
                    ing.strip()
                    for ing in ingredients.split(",")
                    if ing.strip()
                ]
                return ingredients_list
        except Exception:
            pass

        return []

    def _clean_instructions(self, instructions):
        """Clean instructions"""
        if pd.isna(instructions):
            return []

        try:
            if isinstance(instructions, list):
                return instructions
            if isinstance(instructions, str):
                cleaned = (
                    instructions.strip("c()")
                    .replace('"', "")
                    .replace("'", "")
                    .strip()
                )
                if not cleaned:
                    return []
                if '","' in instructions:
                    instructions_list = [
                        inst.strip()
                        for inst in instructions.split('","')
                        if inst.strip()
                    ]
                elif "." in cleaned:
                    instructions_list = [
                        inst.strip()
                        for inst in cleaned.split(".")
                        if inst.strip()
                    ]
                else:
                    instructions_list = [cleaned]
                return instructions_list
        except Exception:
            pass

        return []

    def _calculate_complexity(self, row):
        """Calculate recipe complexity score"""
        score = 0

        # Number of ingredients
        ingredients = row.get("RecipeIngredientParts", [])
        if isinstance(ingredients, list):
            score += min(
                len(ingredients) * 0.5, 5
            )  # Max 5 points for ingredients

        # Number of steps
        instructions = row.get("RecipeInstructions", [])
        if isinstance(instructions, list):
            score += min(len(instructions) * 0.3, 4)  # Max 4 points for steps

        # Cooking time
        total_time = row.get("total_time_minutes", 0)
        if total_time > 60:
            score += 2
        elif total_time > 30:
            score += 1

        # Special techniques (if detectable from description)
        description = str(row.get("Description", "")).lower()
        if any(
            word in description
            for word in [
                "sous vide",
                "ferment",
                "debone",
                "fillet",
                "emulsify",
            ]
        ):
            score += 2

        return min(score, 10)  # Max complexity score of 10

    def _calculate_total_time(self, row):
        """Calculate total time in minutes"""
        total_time = 0

        # Prep time
        prep_time = row.get("PrepTime", "")
        if prep_time:
            total_time += self._parse_time_to_minutes(prep_time)

        # Cook time
        cook_time = row.get("CookTime", "")
        if cook_time:
            total_time += self._parse_time_to_minutes(cook_time)

        return total_time

    def _parse_time_to_minutes(self, time_str):
        """Parse time string to minutes"""
        if not time_str or pd.isna(time_str):
            return 0

        try:
            # Handle formats like "PT30M", "30M", "1H30M", etc.
            time_str = (
                str(time_str)
                .upper()
                .replace("PT", "")
                .replace("H", "H")
                .replace("M", "M")
            )

            hours = 0
            minutes = 0

            if "H" in time_str:
                parts = time_str.split("H")
                hours = int(parts[0]) if parts[0].isdigit() else 0
                time_str = parts[1] if len(parts) > 1 else ""

            if "M" in time_str:
                minutes = int(time_str.replace("M", ""))

            return hours * 60 + minutes
        except Exception:
            return 0

    def _build_ingredient_index(self):
        """Build ingredient index for faster matching"""
        if self.recipes_data is None:
            return

        self.ingredient_index = {}

        for idx, row in self.recipes_data.iterrows():
            ingredients = row.get("RecipeIngredientParts", [])
            if isinstance(ingredients, list):
                for ingredient in ingredients:
                    ingredient = ingredient.lower().strip()
                    if ingredient not in self.ingredient_index:
                        self.ingredient_index[ingredient] = []
                    self.ingredient_index[ingredient].append(idx)

    def generate_recipes(
        self,
        available_ingredients,
        user_preferences=None,
        skill_level="intermediate",
        num_recipes=5,
    ):
        """
        Generate recipe suggestions based on available ingredients and user preferences

        Args:
            available_ingredients: List of ingredients available to the user
            user_preferences: Dict with dietary preferences and restrictions
            skill_level: 'beginner', 'intermediate', or 'advanced'
            num_recipes: Number of recipes to suggest

        Returns:
            List of recipe suggestions with match scores
        """
        if self.recipes_data is None:
            return []

        # Normalize available ingredients
        available_ingredients = [
            ing.lower().strip() for ing in available_ingredients
        ]

        # Filter by dietary restrictions
        filtered_recipes = self._filter_by_dietary_restrictions(
            user_preferences
        )

        # Score recipes based on ingredient matching
        scored_recipes = []
        for idx, recipe in filtered_recipes.iterrows():
            match_score = self._calculate_match_score(
                recipe, available_ingredients, skill_level
            )
            if (
                match_score > 0
            ):  # Only include recipes with at least some ingredient matches
                scored_recipes.append(
                    {
                        "recipe": recipe,
                        "match_score": match_score,
                        "missing_ingredients": self._get_missing_ingredients(
                            recipe, available_ingredients
                        ),
                    }
                )

        # Sort by match score and return top recipes
        scored_recipes.sort(key=lambda x: x["match_score"], reverse=True)
        return scored_recipes[:num_recipes]

    def _filter_by_dietary_restrictions(self, user_preferences):
        """Filter recipes based on dietary restrictions"""
        if not user_preferences or self.recipes_data is None:
            return self.recipes_data

        filtered_df = self.recipes_data.copy()

        # Filter by diet type
        diet_type = user_preferences.get("diet_type", "").lower()
        if diet_type in ["vegetarian", "vegan"]:
            # For now, we'll use keyword matching in ingredients
            # In a more sophisticated implementation, we'd have a proper categorization
            if diet_type == "vegetarian":
                # Exclude recipes with meat ingredients
                meat_keywords = [
                    "chicken",
                    "beef",
                    "pork",
                    "lamb",
                    "fish",
                    "seafood",
                    "meat",
                ]
                filtered_df = filtered_df[
                    ~filtered_df["RecipeIngredientParts"].apply(
                        lambda ings: any(
                            meat in " ".join(ings).lower()
                            for meat in meat_keywords
                        )
                    )
                ]

        # Filter by allergies
        allergies = user_preferences.get("allergies", [])
        if allergies:
            allergy_keywords = {
                "nuts": [
                    "nut",
                    "almond",
                    "walnut",
                    "pecan",
                    "cashew",
                    "peanut",
                ],
                "dairy": ["milk", "cheese", "cream", "butter", "yogurt"],
                "gluten": ["wheat", "flour", "bread", "pasta"],
                "eggs": ["egg"],
            }

            for allergy in allergies:
                if allergy.lower() in allergy_keywords:
                    keywords = allergy_keywords[allergy.lower()]
                    filtered_df = filtered_df[
                        ~filtered_df["RecipeIngredientParts"].apply(
                            lambda ings: any(
                                keyword in " ".join(ings).lower()
                                for keyword in keywords
                            )
                        )
                    ]

        # Filter by health condition
        health_condition = user_preferences.get("health_condition", "").lower()
        if health_condition:
            filtered_df = self._apply_health_condition_filters(filtered_df, health_condition)

        return filtered_df

    def _apply_health_condition_filters(self, df, health_condition):
        """Apply filters based on health condition"""
        if health_condition == "diabetes":
            # Low carb, high fiber foods
            return df[df["CarbohydrateContent"] < 30]
        elif health_condition == "hypertension":
            # Low sodium foods (assuming low salt in ingredients)
            return df[
                ~df["RecipeIngredientParts"].str.join(" ").str.contains(
                    "salt|soy sauce|sodium", case=False
                )
            ]
        elif health_condition == "thyroid":
            # Foods rich in iodine and selenium
            return df[
                df["RecipeIngredientParts"].str.join(" ").str.contains(
                    "seafood|fish|nuts|seeds", case=False
                )
            ]
        elif health_condition == "pcos":
            # Low glycemic index foods
            return df[df["CarbohydrateContent"] < 25]
        elif health_condition == "high_cholesterol":
            # Low fat, high fiber foods
            return df[(df["FatContent"] < 15) & (df["FiberContent"] > 5)]
        return df

    def _calculate_match_score(
        self, recipe, available_ingredients, skill_level
    ):
        """Calculate how well a recipe matches available ingredients and skill level"""
        recipe_ingredients = recipe.get("RecipeIngredientParts", [])
        if not isinstance(recipe_ingredients, list):
            return 0

        # Ingredient matching score
        matching_ingredients = 0
        total_ingredients = len(recipe_ingredients)

        for ingredient in recipe_ingredients:
            ingredient = ingredient.lower().strip()
            # Check for partial matches
            if any(
                avail_ing in ingredient or ingredient in avail_ing
                for avail_ing in available_ingredients
            ):
                matching_ingredients += 1

        ingredient_score = (
            matching_ingredients / max(total_ingredients, 1) * 70
        )  # Max 70 points

        # Skill level matching
        complexity_score = recipe.get("complexity_score", 5)

        if skill_level == "beginner" and complexity_score <= 3:
            skill_score = 30
        elif skill_level == "intermediate" and complexity_score <= 6:
            skill_score = 30
        elif skill_level == "advanced":
            skill_score = 30
        else:
            skill_score = 15  # Partial match

        return ingredient_score + skill_score

    def _get_missing_ingredients(self, recipe, available_ingredients):
        """Get list of ingredients not available to the user"""
        recipe_ingredients = recipe.get("RecipeIngredientParts", [])
        if not isinstance(recipe_ingredients, list):
            return []

        missing = []
        for ingredient in recipe_ingredients:
            ingredient = ingredient.lower().strip()
            if not any(
                avail_ing in ingredient or ingredient in avail_ing
                for avail_ing in available_ingredients
            ):
                missing.append(ingredient)

        return missing

    def get_recipe_details(self, recipe_name):
        """Get detailed information for a specific recipe"""
        if self.recipes_data is None:
            return None

        # Find recipe by name
        recipe = self.recipes_data[
            self.recipes_data["Name"].str.lower() == recipe_name.lower()
        ]
        if recipe.empty:
            return None

        recipe = recipe.iloc[0]

        return {
            "name": recipe.get("Name", "Unknown"),
            "description": recipe.get("Description", ""),
            "ingredients": recipe.get("RecipeIngredientParts", []),
            "instructions": recipe.get("RecipeInstructions", []),
            "prep_time": recipe.get("PrepTime", 0),
            "carbs": recipe.get("CarbohydrateContent", 0),
            "fat": recipe.get("FatContent", 0),
            "complexity": recipe.get("complexity_score", 5),
            "category": recipe.get("RecipeCategory", "Unknown"),
        }


# =============================================================================
# MAIN SMARTBITE MODELS CLASS
# =============================================================================


class SmartBiteModels:
    """
    Main class that combines all SmartBite AI models.
    """

    def __init__(self):
        self.predictor = UserProgressPredictor()
        self.recipe_generator = RecipeGenerator()
        self.recommender = NutritionRecommender(self.recipe_generator)
        self.hydration_recommender = HydrationRecommender()
        self.classifier = None  # Will be initialized when needed

    def initialize_all_models(self):
        """Initialize and train all models"""
        print("=== Initializing SmartBite AI Models ===")

        # Initialize progress predictor
        print("\n1. Training User Progress Predictor...")
        self.predictor.train_models()

        # Initialize nutrition recommender
        print("\n2. Training Nutrition Recommender...")
        if self.recommender.load_and_preprocess_data():
            self.recommender.train_models()
            print("   [OK] Nutrition recommender ready")
        else:
            print("   [WARN] Using fallback nutrition recommendations")

        # Initialize recipe generator
        print("\n3. Loading Recipe Generator...")
        if self.recipe_generator.load_recipes_data():
            print("   [OK] Recipe generator ready")
        else:
            print("   [WARN] Using fallback recipe recommendations")

        print("\n=== All models initialized successfully! ===")
        return True

    def load_all_models(self):
        """Load pre-trained models"""
        # For now, just initialize new models
        return self.initialize_all_models()

    def predict_progress(self, user_data):
        """Predict user progress"""
        return self.predictor.predict_user_progress(user_data)

    def get_recommendations(self, preferences):
        """Get nutrition recommendations"""
        return self.recommender.get_recommendations(preferences)

    def get_health_insights(self, user_data):
        """Get health insights"""
        return self.predictor.get_health_insights(user_data)

    def generate_recipes(
        self,
        available_ingredients,
        user_preferences=None,
        skill_level="intermediate",
        num_recipes=5,
    ):
        """Generate recipe suggestions"""
        return self.recipe_generator.generate_recipes(
            available_ingredients, user_preferences, skill_level, num_recipes
        )

    def get_recipe_details(self, recipe_name):
        """Get detailed recipe information"""
        return self.recipe_generator.get_recipe_details(recipe_name)

    def get_recipe_recommendations(self, available_ingredients, preferences):
        """Get recipe recommendations with dietary/health filtering"""
        return self.recipe_generator.generate_recipes(
            available_ingredients,
            preferences
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def transform_meal_item(item):
    """Transform meal item for template compatibility"""
    return {
        "Shrt_Desc": item.get("name", "Unknown"),
        "Energ_Kcal": item.get("calories", "-"),
        "Protein_(g)": item.get("protein_grams", "-"),
        "Carbohydrt_(g)": item.get("carb_grams", "-"),
        "Lipid_Tot_(g)": item.get("fat_grams", "-"),
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("SmartBite AI Models - Standalone Test")
    print("=" * 50)

    # Test the models
    smartbite = SmartBiteModels()
    smartbite.initialize_all_models()

    # Test recommendations
    test_preferences = {"diet_type": "vegetarian", "calorie_goal": 2000}

    recommendations = smartbite.get_recommendations(test_preferences)
    print(
        f"\n[OK] Generated recommendations for {test_preferences['diet_type']} diet"
    )
    print(
        f"[STATS] Daily meals: {len(recommendations['daily_meals']['breakfast'])} breakfast items"
    )

    print("\n" + "=" * 50)
    print("SmartBite AI Models - Ready!")
