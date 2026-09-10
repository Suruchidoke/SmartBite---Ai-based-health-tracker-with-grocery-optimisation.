import pytest
from app import create_app, db_manager, AuthService


@pytest.fixture
def app():
    app = create_app("testing")
    with app.app_context():
        # Clear collections in testing mock db
        db_manager.users.delete_many({})
        db_manager.health_data.delete_many({})
        db_manager.grocery_data.delete_many({})
        yield app


@pytest.fixture
def client(app):
    return app.test_client()


def test_password_hashing():
    """Verify password is encrypted with salt/hash and never stored plain text."""
    plain = "SuperSecureP@ss123"
    hashed = AuthService.hash_password(plain)
    assert hashed != plain
    assert AuthService.check_password(hashed, plain) is True
    assert AuthService.check_password(hashed, "WrongPassword") is False


def test_user_registration_and_login(client):
    """Test user signup and subsequent login session creation."""
    # 1. Sign up
    res = client.post(
        "/signup",
        data={
            "email": "alice@example.com",
            "password": "Password123",
            "confirm_password": "Password123",
        },
        follow_redirects=True,
    )
    assert res.status_code == 200

    # Verify user exists in MongoDB
    user = db_manager.users.find_one({"email": "alice@example.com"})
    assert user is not None
    assert user["password"] != "Password123"

    # 2. Login with wrong password
    bad_login = client.post(
        "/login",
        data={"email": "alice@example.com", "password": "WrongPassword"},
        follow_redirects=True,
    )
    assert bad_login.status_code == 200

    # 3. Login with correct password
    good_login = client.post(
        "/login",
        data={"email": "alice@example.com", "password": "Password123"},
        follow_redirects=True,
    )
    assert good_login.status_code == 200
    with client.session_transaction() as sess:
        assert sess.get("username") == "alice@example.com"


def test_idor_protection_dashboard(client):
    """
    CRITICAL SECURITY TEST:
    Verify User A is blocked (403 Forbidden) from accessing User B's dashboard.
    """
    # Create User A and User B
    client.post(
        "/signup",
        data={"email": "usera@example.com", "password": "Password123", "confirm_password": "Password123"},
        follow_redirects=True,
    )
    client.post(
        "/signup",
        data={"email": "userb@example.com", "password": "Password123", "confirm_password": "Password123"},
        follow_redirects=True,
    )

    # Log in as User A
    client.post(
        "/login",
        data={"email": "usera@example.com", "password": "Password123"},
        follow_redirects=True,
    )

    # User A accesses User A's own dashboard API -> 200 OK
    res_own = client.get("/api/dashboard/usera@example.com")
    assert res_own.status_code == 200
    data_own = res_own.get_json()
    assert data_own["userProfile"]["name"] is not None

    # User A attempts to access User B's dashboard API -> 403 FORBIDDEN (IDOR prevented!)
    res_idor = client.get("/api/dashboard/userb@example.com")
    assert res_idor.status_code == 403
    err_data = res_idor.get_json()
    assert err_data["code"] == "IDOR_PREVENTED"


def test_password_reset_token_flow(app):
    """Test cryptographic password reset token generation, verification, and expiration."""
    with app.app_context():
        email = "reset_test@example.com"
        token = AuthService.generate_reset_token(email)
        assert token is not None

        # Verify valid token
        verified = AuthService.verify_reset_token(token)
        assert verified == email

        # Bad token verification
        invalid = AuthService.verify_reset_token("corrupted.token.value")
        assert invalid is None

        # Expired token verification
        expired = AuthService.verify_reset_token(token, max_age_seconds=-1)
        assert expired is None
