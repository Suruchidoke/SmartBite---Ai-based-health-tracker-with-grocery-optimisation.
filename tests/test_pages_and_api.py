import pytest
from app import create_app, db_manager, GroceryService


@pytest.fixture
def app():
    app = create_app("testing")
    with app.app_context():
        yield app


@pytest.fixture
def client(app):
    return app.test_client()


def test_fitness_games_and_quizzes_routes(client):
    """Verify fitness games hub and quiz routes."""
    # Unauthenticated redirect to login
    res = client.get("/fitness_games")
    assert res.status_code == 302
    assert "/login" in res.headers["Location"]

    # Authenticated session
    with client.session_transaction() as sess:
        sess["username"] = "testuser@smartbite.com"

    res = client.get("/fitness_games")
    assert res.status_code == 200
    assert b"Fitness Games" in res.data
    assert b"Yoga Pose Quiz" in res.data
    assert b"Nutrition Label Quiz" in res.data

    res_yoga = client.get("/yoga_pose_quiz")
    assert res_yoga.status_code == 200

    res_quiz = client.get("/nutrition_label_quiz")
    assert res_quiz.status_code == 200

    res_plank = client.get("/plank_timer")
    assert res_plank.status_code == 200


def test_diet_plan_page_and_push_to_grocery(client):
    """Verify diet plan page renders without Jinja errors and push_to_grocery works."""
    uid = "cheftest@smartbite.com"
    with client.session_transaction() as sess:
        sess["username"] = uid

    res = client.get("/diet_plan")
    assert res.status_code == 200
    assert b"Personalized Diet" in res.data
    assert b"Push to Grocery" in res.data

    # Test push to grocery
    res_push = client.post("/api/diet_plan/push_to_grocery", json={})
    assert res_push.status_code == 200
    data = res_push.get_json()
    assert data["success"] is True
    assert "count" in data
    assert "message" in data


def test_add_custom_recipe_and_grocery_page(client):
    """Verify custom recipe addition and grocery optimizer view."""
    uid = "cheftest@smartbite.com"
    with client.session_transaction() as sess:
        sess["username"] = uid

    recipe_data = {
        "name": "Superfood Protein Bowl",
        "calories": 420,
        "protein": 30,
        "carbs": 45,
        "fats": 12,
        "ingredients": ["Quinoa", "Edamame", "Avocado", "Tofu"],
        "instructions": ["Cook quinoa", "Slice tofu", "Assemble bowl"]
    }
    res_add = client.post(f"/api/diet_plan/{uid}/add_custom_recipe", json=recipe_data)
    assert res_add.status_code == 200
    assert res_add.get_json()["success"] is True

    # Check grocery optimizer page
    res_groc = client.get("/grocery")
    assert res_groc.status_code == 200
    assert b"Grocery" in res_groc.data
