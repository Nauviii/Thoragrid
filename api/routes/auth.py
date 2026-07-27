"""POST /auth/token — authenticate and issue a JWT.
POST /auth/resume — exchange an opaque browser session key for the token it stands for.
DELETE /auth/session — revoke an opaque browser session key on sign-out."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session as DBSession

from api.middleware.auth import create_access_token, verify_password
from api.schemas.requests import ResumeSessionRequest
from api.schemas.responses import TokenResponse
from scripts.db_session import get_db
from scripts.db_models import User, Session as UserSession
from core.memory.frontend_session import (
    create_session_key, resolve_session_key, revoke_session_key,
)

router = APIRouter()


@router.post("/auth/token", response_model=TokenResponse)
def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[DBSession, Depends(get_db)],
) -> TokenResponse:
    """Verify credentials, open a new Session row, and return a JWT access token."""
    user = db.query(User).filter_by(username=form_data.username).first()
    if user is None or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect username or password")

    session = UserSession(user_id=user.id, role=user.role)
    db.add(session)
    db.commit()
    db.refresh(session)

    token = create_access_token(user.id, user.role, session.id)
    session_key = create_session_key(token, user.username, user.role)
    return TokenResponse(
        access_token=token, token_type="bearer", role=user.role,
        username=user.username, session_key=session_key,
    )


@router.post("/auth/resume", response_model=TokenResponse)
def resume(body: ResumeSessionRequest) -> TokenResponse:
    """Resolve an opaque browser session key back to its token, for resuming after a page reload."""
    stored = resolve_session_key(body.session_key)
    if stored is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Session expired or revoked")

    return TokenResponse(
        access_token=stored["token"], token_type="bearer", role=stored["role"],
        username=stored["username"], session_key=body.session_key,
    )


@router.delete("/auth/session", status_code=status.HTTP_204_NO_CONTENT)
def end_browser_session(body: ResumeSessionRequest) -> None:
    """Revoke an opaque browser session key so a copied cookie stops working immediately."""
    revoke_session_key(body.session_key)