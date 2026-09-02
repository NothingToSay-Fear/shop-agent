"""本地账号注册、登录、退出和当前用户接口。"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import AuthToken, User
from app.schemas.auth import AuthSessionRead, LoginRequest, RegisterRequest, UserRead
from app.services.authentication import (
    create_access_token,
    get_current_user,
    hash_password,
    token_digest,
    verify_password,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

router = APIRouter(prefix="/api/auth", tags=["auth"])
_token_scheme = HTTPBearer(auto_error=False)


@router.post("/register", response_model=AuthSessionRead, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest, session: AsyncSession = Depends(get_session)) -> AuthSessionRead:
    """创建账号并立即签发登录令牌。"""
    exists = await session.scalar(select(User.id).where(User.username == payload.username))
    if exists is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="该账号已被注册")
    user = User(
        username=payload.username,
        display_name=payload.display_name,
        password_hash=hash_password(payload.password),
    )
    session.add(user)
    await session.flush()
    access_token = await create_access_token(session, user)
    return AuthSessionRead(access_token=access_token, user=user)


@router.post("/login", response_model=AuthSessionRead)
async def login(payload: LoginRequest, session: AsyncSession = Depends(get_session)) -> AuthSessionRead:
    """校验账号密码后返回新的可撤销令牌。"""
    user = await session.scalar(select(User).where(User.username == payload.username))
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="账号或密码错误")
    access_token = await create_access_token(session, user)
    return AuthSessionRead(access_token=access_token, user=user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    credentials: HTTPAuthorizationCredentials | None = Depends(_token_scheme),
    _: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> None:
    """撤销当前令牌；同一账号的其他登录设备不受影响。"""
    if credentials is not None:
        token = await session.scalar(
            select(AuthToken).where(AuthToken.token_hash == token_digest(credentials.credentials))
        )
        if token is not None:
            await session.delete(token)
            await session.commit()


@router.get("/me", response_model=UserRead)
async def current_user(user: User = Depends(get_current_user)) -> User:
    """返回当前令牌对应的用户公开信息。"""
    return user
