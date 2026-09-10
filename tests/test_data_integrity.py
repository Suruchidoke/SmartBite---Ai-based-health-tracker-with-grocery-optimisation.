import pytest
from app import create_app, db_manager, get_or_create_user_state, GroceryService, NutritionService


@pytest.fixture
def app():
    app = create_app("testing")
    with app.app_context():
        db_manager.users.delete_many({})
        db_manager.health_data.delete_many({})
        db_manager.grocery_data.delete_many({})
        db_manager.activity_log.delete_many({})
        db_manager.meal_planner.delete_many({})
        yield app


@pytest.fixture
def client(app):
    return app.test_client()


def test_user_state_persistence(app):
    """Ensure user data is correctly stored and fetched from MongoDB without in-memory USERS."""
    with app.app_context():
        user_id = "test_persistence@example.com"
        state = get_or_create_user_state(user_id)
        assert state["health"]["userId"] == user_id
        assert state["grocery"]["userId"] == user_id
        assert state["health"]["targetCalories"] == 2000

        # Verify documents exist in MongoDB
        health_doc = db_manager.health_data.find_one({"userId": user_id})
        assert health_doc is not None
        assert health_doc["calories"] == 0


def test_pantry_and_shopping_operations(app):
    """Test persistent pantry addition, removal, and transfer to shopping list."""
    with app.app_context():
        user_id = "chef@example.com"

        # 1. Add pantry items
        pantry = GroceryService.add_pantry_item(user_id, "Rolled Oats (500g)", "Breakfast")
        assert len(pantry) == 1
        assert pantry[0]["name"] == "Rolled Oats"

        # 2. Add shopping items
        shopping = GroceryService.add_shopping_item(user_id, "Fresh Spinach", "Produce", 30.0)
        assert len(shopping) == 1
        assert shopping[0]["name"] == "Fresh Spinach"

        # 3. Move item from shopping to pantry
        updated = GroceryService.move_item(user_id, "Fresh Spinach", "shopping", "pantry")
        assert any(p["name"] == "Fresh Spinach" for p in updated["pantry"])
        assert not any(s["name"] == "Fresh Spinach" for s in updated["shoppingList"])

        # 4. Remove item from pantry
        pantry_after = GroceryService.remove_pantry_item(user_id, "Rolled Oats")
        assert not any(p["name"] == "Rolled Oats" for p in pantry_after)


def test_diet_to_grocery_zero_waste_consolidation(app):
    """
    Verify meal plan ingredients are extracted, normalized,
    and items already in pantry are auto-deducted from the shopping list.
    """
    with app.app_context():
        user_id = "meal_prepper@example.com"

        # Add Olive Oil to pantry
        GroceryService.add_pantry_item(user_id, "Olive Oil", "Oils")

        sample_meal_plan = {
            "Monday": {
                "meals": {
                    "dinner": [
                        {
                            "name": "Mediterranean Salad",
                            "ingredients": "Cucumber, Tomatoes, 2 tbsp Olive Oil, Feta Cheese"
                        }
                    ]
                }
            }
        }

        result = GroceryService.consolidate_diet_to_grocery(user_id, sample_meal_plan)

        # Olive oil should be deducted from pantry
        assert result["deducted_from_pantry_count"] >= 1
        assert any("Olive Oil" in d for d in result["deducted_items"])

        # Cucumber, Tomatoes, Feta Cheese should be added to shopping list
        grocery_state = GroceryService.get_user_grocery_data(user_id)
        shopping_names = [s["name"] for s in grocery_state["shoppingList"]]
        assert "Cucumber" in shopping_names
        assert "Tomatoes" in shopping_names
        assert "Feta Cheese" in shopping_names


def test_mifflin_st_jeor_calculation():
    """Verify clinical Mifflin-St Jeor engine and safety deficit bounds."""
    # Male, 80kg, 180cm, 30yo, Moderate activity, Weight loss
    res_male = NutritionService.calculate_bmr_tdee(
        weight_kg=80, height_cm=180, age=30, gender="Male", activity_level="Moderate", goal="weight_loss"
    )
    # BMR = 10*80 + 6.25*180 - 5*30 + 5 = 800 + 1125 - 150 + 5 = 1780
    assert res_male["bmr"] == 1780
    # TDEE = 1780 * 1.55 = 2759
    assert res_male["tdee"] == 2759
    # Deficit target = 2759 - 500 = 2259
    assert res_male["target_calories"] == 2259
    assert res_male["macros"]["protein"] > 0
    assert res_male["warning"] is None

    # Extreme deficit protection (clamped to safety floor)
    res_deficit = NutritionService.calculate_bmr_tdee(
        weight_kg=45, height_cm=150, age=40, gender="Female", activity_level="Sedentary", goal="weight_loss"
    )
    assert res_deficit["target_calories"] >= 1200
    assert res_deficit["warning"] is not None
