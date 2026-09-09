import math
import pandas as pd
import numpy as np
import pytest

from models.smartbite_models import (
    UserProgressPredictor,
    NutritionRecommender,
    HydrationRecommender,
    RecipeGenerator,
)


# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def sample_user_df():
    upp = UserProgressPredictor()
    df = upp.generate_synthetic_data(n_users=200)
    return upp, df


@pytest.fixture
def recipes_csv_path(tmp_path):
    # Minimal recipes CSV with required columns
    csv_content = (
        "Name,RecipeIngredientParts,RecipeInstructions,Calories,FatContent,SaturatedFatContent,CholesterolContent,SodiumContent,CarbohydrateContent,FiberContent,SugarContent,ProteinContent,PrepTime,CookTime,Description\n"
        "Veggie Omelette,\"egg, bell pepper, onion\",\"beat eggs\",250,15,5,200,300,5,1,2,18,PT5M,PT5M,\"easy omelette\"\n"
        "Grilled Chicken Salad,\"chicken breast, lettuce, tomato\",\"grill chicken\",400,10,3,70,300,10,3,4,30,PT10M,PT10M,\"grilled and healthy\"\n"
        "Tofu Stir Fry,\"tofu, broccoli, soy sauce\",\"stir fry\",350,12,2,0,200,20,6,6,18,PT10M,PT10M,\"quick stir fry\"\n"
        "Oatmeal Bowl,\"oats, milk, banana\",\"cook oats\",300,7,2,0,150,55,6,12,8,PT5M,PT5M,\"breakfast oats\"\n"
    )
    path = tmp_path / "recipes_dataset.csv"
    path.write_text(csv_content, encoding="utf-8")
    return str(path)


@pytest.fixture
def indian_food_csv_path(tmp_path):
    content = (
        "name,ingredients,diet,course,flavor_profile,state,region,prep_time,cook_time\n"
        "Upma,\"semolina, onion, chili\",vegetarian,breakfast,spicy,Karnataka,South,10,10\n"
        "Chicken Curry,\"chicken, chili, onion\",non vegetarian,dinner,spicy,Maharashtra,West,20,30\n"
        "Veg Pulao,\"rice, peas, carrot\",vegetarian,lunch,spicy,Delhi,North,15,20\n"
    )
    path = tmp_path / "indian_food.csv"
    path.write_text(content, encoding="utf-8")
    return str(path)


# -----------------------------
# UserProgressPredictor tests
# -----------------------------

def test_train_and_predict_progress(sample_user_df):
    upp, df = sample_user_df

    X_scaled, processed = upp.prepare_features(df)
    assert X_scaled.shape[0] == len(df)

    trained = upp.train_models(processed)
    assert trained is True
    assert set(upp.models.keys()) == {
        "weight_loss_4weeks",
        "fitness_score_improvement",
        "goal_achievement_rate",
    }

    user_row = processed.iloc[0].to_dict()
    # Keep only expected fields consumed by prepare_features
    user_data = {k: user_row.get(k) for k in [
        "age","weight","height","gender","activity_level","workout_frequency",
        "avg_workout_duration","calories_burned","diet_compliance","sleep_hours",
        "stress_level","motivation_level","goal_type"
    ]}

    preds = upp.predict_user_progress(user_data)
    assert preds is not None
    # Ensure non-negative outputs as enforced by implementation
    assert preds["weight_loss_4weeks"] >= 0
    assert preds["fitness_score_improvement"] >= 0
    assert 0 <= preds["goal_achievement_rate"]


def test_get_health_insights_structure(sample_user_df):
    upp, df = sample_user_df
    upp.train_models(df)

    user_row = df.iloc[0].to_dict()
    user_data = {k: user_row.get(k) for k in [
        "age","weight","height","gender","activity_level","workout_frequency",
        "avg_workout_duration","calories_burned","diet_compliance","sleep_hours",
        "stress_level","motivation_level","goal_type"
    ]}
    user_data["healthData"] = {"weeklyCalories": [2200, 2500, 2400], "goal": 2500}

    insights = upp.get_health_insights(user_data)
    assert set(insights.keys()) == {"calorie_analysis", "predictions", "nutrition_score", "activity_level", "recommendations"}
    assert "average_daily" in insights["calorie_analysis"]
    assert isinstance(insights["recommendations"], list)


# -----------------------------
# NutritionRecommender tests
# -----------------------------

def test_nutrition_recommender_with_recipes(recipes_csv_path):
    rg = RecipeGenerator()
    # reload with temp dataset to avoid reading project file
    assert rg.load_recipes_data(recipes_csv_path) is True

    nr = NutritionRecommender(recipe_generator=rg)
    # Force a simple meal plan with health condition filters and calorie targets
    plan = nr._generate_meal_plan(diet_type="vegetarian", total_calories=2100, meals_per_day=3, health_condition="hypertension")

    # Expect keys and non-empty suggestions
    assert set(plan.keys()) == {"breakfast", "lunch", "dinner"}
    assert all(isinstance(plan[k], list) for k in plan)
    # Items should have required fields
    for k in plan:
        if plan[k]:
            item = plan[k][0]
            assert {"name", "calories", "protein_grams", "carb_grams", "fat_grams"}.issubset(item.keys())


def test_nutrition_recommender_food_fallback(indian_food_csv_path, monkeypatch):
    nr = NutritionRecommender()
    assert nr.load_and_preprocess_data(indian_food_csv_path) is True
    assert nr.train_models() is True

    # No recipe generator provided -> food-based path exercised
    result = nr.get_recommendations({"diet_type": "vegetarian", "calorie_goal": 1800, "health_condition": "diabetes"})
    assert set(result.keys()) == {"daily_meals", "weekly_plan", "shopping_list"}
    plan = result["daily_meals"]
    assert set(plan.keys()) == {"breakfast", "lunch", "dinner"}


# -----------------------------
# HydrationRecommender tests
# -----------------------------

def test_hydration_calculation_and_schedule():
    hr = HydrationRecommender()
    profile = {
        "age": 28,
        "weight": 80,
        "activity_level": "active",
        "climate": "warm",
        "health_conditions": ["diabetes"],
        "gender": "male",
    }
    goal = hr.calculate_daily_water_goal(profile)
    assert 1500 <= goal["daily_goal_ml"] <= 4000
    assert goal["daily_goal_ounces"] > 0
    assert goal["daily_goal_glasses"] > 0

    schedule = hr.get_hydration_schedule(goal["daily_goal_ml"], wake_time="06:00", sleep_time="22:00")
    # Should distribute across awake hours
    assert len(schedule) > 0
    assert sum(item["recommended_intake_ml"] for item in schedule) <= goal["daily_goal_ml"]

    tips = hr.get_hydration_tips(profile, current_intake_ml=500)
    assert 0 <= tips["progress_percentage"] <= 100
    assert tips["remaining_ml"] >= 0
    assert isinstance(tips["tips"], list) and len(tips["tips"]) > 0
