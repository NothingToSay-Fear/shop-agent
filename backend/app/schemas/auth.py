"""账号注册、登录与当前登录用户的接口数据结构。"""

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class RegisterRequest(BaseModel):
    """创建本地运营账号所需的字段。"""

    username: str = Field(min_length=3, max_length=50)
    display_name: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized.replace("_", "").replace("-", "").isalnum() or not normalized.isascii():
            raise ValueError("账号只能使用 3 至 50 位英文字母、数字、下划线或短横线")
        return normalized

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("显示名称不能为空")
        return normalized


class LoginRequest(BaseModel):
    """登录时使用的账号与密码。"""

    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("username")
    @classmethod
    def normalize_username(cls, value: str) -> str:
        return value.strip().lower()


class UserRead(BaseModel):
    """可返回给浏览器的用户公开信息。"""

    id: str
    username: str
    display_name: str
    is_admin: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class AuthSessionRead(BaseModel):
    """登录或注册成功后返回的访问令牌和用户信息。"""

    access_token: str
    token_type: str = "bearer"
    user: UserRead
