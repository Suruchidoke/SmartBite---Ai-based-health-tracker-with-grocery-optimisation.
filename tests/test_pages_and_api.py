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


def test_dashboard_ai_recipe_generator_and_helpers(client):
    """Verify AI recipe generator on dashboard, activity logging, and calorie reset."""
    uid = "cheftest@smartbite.com"
    with client.session_transaction() as sess:
        sess["username"] = uid

    # 1. AI Recipe Generator on Dashboard
    res_gen = client.post(f"/api/recipes/generate/{uid}", json={"skill_level": "Beginner", "num_recipes": 3})
    assert res_gen.status_code == 200
    data = res_gen.get_json()
    assert data["success"] is True
    assert len(data["recipes"]) == 3
    assert "name" in data["recipes"][0]
    assert "cook_time" in data["recipes"][0]

    # 2. Activity logger
    res_act = client.post(f"/api/activity/{uid}/add", json={"activity": "Completed 30 min morning jog"})
    assert res_act.status_code == 200
    assert res_act.get_json()["success"] is True

    # 3. Calorie reset
    res_reset = client.post(f"/api/health/{uid}/reset_calories")
    assert res_reset.status_code == 200
    assert res_reset.get_json()["calories"] == 0

    # 4. Product detail JSON API
    res_prod = client.get("/api/grocery/product/p1")
    assert res_prod.status_code == 200
    prod_data = res_prod.get_json()
    assert prod_data["name"] == "Yellow Lentils (Moong Dal)"
    assert "nutrition" in prod_data

