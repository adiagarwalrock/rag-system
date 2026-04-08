import pytest
from fastapi import HTTPException

from app.core.config import settings
from app.core.dependencies import get_current_active_user, get_current_user
from app.core.security import create_access_token
from app.db.models import User


def test_get_current_user_returns_stub_when_auth_disabled(db_session, monkeypatch):
    monkeypatch.setattr(settings, "AUTH_ENABLED", False)

    user = get_current_user(token=None, db=db_session)

    assert user.id == "dev-user"
    assert user.email == "dev@localhost"


def test_get_current_user_raises_for_invalid_token_when_auth_enabled(
    db_session, monkeypatch
):
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)

    with pytest.raises(HTTPException) as exc:
        get_current_user(token="not-a-valid-token", db=db_session)

    assert exc.value.status_code == 401
    assert exc.value.detail == "Could not validate credentials"


def test_get_current_user_returns_db_user_for_valid_token(db_session, monkeypatch):
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)

    user = User(
        id="auth-user-1",
        email="auth-user@example.com",
        password_hash="hash",
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()

    token = create_access_token({"sub": user.id})
    current_user = get_current_user(token=token, db=db_session)

    assert current_user.id == user.id


def test_get_current_active_user_rejects_inactive_users_when_auth_enabled(monkeypatch):
    monkeypatch.setattr(settings, "AUTH_ENABLED", True)

    inactive_user = User(
        id="inactive-user",
        email="inactive@example.com",
        password_hash="hash",
        is_active=False,
    )

    with pytest.raises(HTTPException) as exc:
        get_current_active_user(current_user=inactive_user)

    assert exc.value.status_code == 400
    assert exc.value.detail == "Inactive user"
