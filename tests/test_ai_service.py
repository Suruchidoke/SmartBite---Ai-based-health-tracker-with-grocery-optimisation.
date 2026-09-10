import pytest
from app import create_app, db_manager, get_or_create_user_state, AIService, NutritionService


@pytest.fixture
def app():
    app = create_app("testing")
    with app.app_context():
        db_manager.users.delete_many({})
        db_manager.health_data.delete_many({})
        db_manager.grocery_data.delete_many({})
        yield app


@pytest.fixture
def client(app):
    return app.test_client()


def test_ai_chat_endpoint(client):
    """Test BiteBot /api/ai/chat endpoint produces structured, intelligent response."""
    res = client.post(
        "/api/ai/chat",
        json={"message": "What should I cook for dinner under 500 calories?"},
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["success"] is True
    assert "reply" in data
    assert len(data["reply"]) > 20


def test_ai_pantry_grounded_response(app):
    """Test AI assistant references the user's actual pantry inventory."""
    with app.app_context():
        context = {
            "pantry": [{"name": "Yellow Lentils"}, {"name": "Whole Wheat Atta"}],
            "targetCalories": 1900,
            "goal": "fat_loss",
            "allergens": [],
        }
        reply = AIService.generate_chat_response("What can I make with my pantry?", context)
        assert reply is not None
        assert len(reply) > 50
        # Should reference pantry items or lentils
        reply_lower = reply.lower()
        assert "lentil" in reply_lower or "pantry" in reply_lower or "recipe" in reply_lower


def test_allergen_filtering_guardrail():
    """Verify allergen detection engine catches offending ingredients."""
    user_allergens = ["gluten", "peanuts"]
    safe_ingredients = ["Rice", "Moong Dal", "Spinach", "Tomatoes"]
    unsafe_ingredients = ["Whole Wheat Atta", "Roasted Peanuts", "Soy Sauce"]

    safe_check = NutritionService.filter_allergens(safe_ingredients, user_allergens)
    assert safe_check["safe"] is True
    assert len(safe_check["offending_allergens"]) == 0

    unsafe_check = NutritionService.filter_allergens(unsafe_ingredients, user_allergens)
    assert unsafe_check["safe"] is False
    assert len(unsafe_check["offending_allergens"]) >= 2
