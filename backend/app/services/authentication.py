"""本地账号认证：密码散列、可撤销令牌和当前用户依赖。"""

import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_session
from app.models import AuthToken, User

LEGACY_MIGRATION_USERNAME = "legacy_migration_owner"

_PASSWORD_ALGORITHM = "scrypt"
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_TOKEN_SCHEME = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    """使用带随机盐的 scrypt 保存密码，不依赖额外认证库。"""
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=64
    )
    return "$".join(
        (
            _PASSWORD_ALGORITHM,
            str(_SCRYPT_N),
            str(_SCRYPT_R),
            str(_SCRYPT_P),
            base64.urlsafe_b64encode(salt).decode("ascii"),
            base64.urlsafe_b64encode(digest).decode("ascii"),
        )
    )


def verify_password(password: str, encoded: str) -> bool:
    """以恒定时间比较校验密码；对历史或无效散列统一返回失败。"""
    try:
        algorithm, n, r, p, encoded_salt, encoded_digest = encoded.split("$")
        if algorithm != _PASSWORD_ALGORITHM:
            return False
        salt = base64.urlsafe_b64decode(encoded_salt.encode("ascii"))
        expected_digest = base64.urlsafe_b64decode(encoded_digest.encode("ascii"))
        actual_digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected_digest),
        )
    except (ValueError, TypeError, UnicodeError):
        return False
    return hmac.compare_digest(actual_digest, expected_digest)


def token_digest(token: str) -> str:
    """生成令牌的固定长度摘要，供数据库查询与撤销使用。"""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


async def create_access_token(session: AsyncSession, user: User) -> str:
    """创建有限期登录令牌；原始值仅在本次响应中返回给浏览器。"""
    raw_token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(days=get_settings().auth_token_ttl_days)
    session.add(AuthToken(user_id=user.id, token_hash=token_digest(raw_token), expires_at=expires_at))
    await session.commit()
    return raw_token


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_TOKEN_SCHEME),
    session: AsyncSession = Depends(get_session),
) -> User:
    """从 Bearer 令牌解析当前用户，过期或撤销时统一拒绝访问。"""
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="登录状态无效或已过期",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise unauthorized
    token = await session.scalar(
        select(AuthToken).where(AuthToken.token_hash == token_digest(credentials.credentials))
    )
    if token is None or token.expires_at <= datetime.now(timezone.utc):
        raise unauthorized
    user = await session.get(User, token.user_id)
    if user is None:
        raise unauthorized
    return user

