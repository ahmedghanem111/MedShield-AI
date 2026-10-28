
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    TEMP_FILE_ENCRYPTION_KEY: str | None = None
    REDIS_URL: str = "redis://127.0.0.1:6379/0"

    STAGING_DIR: str = "storage/staging"
    STAGING_TTL_HOURS: int = 2

    class Config:
        env_file = ".env"


settings = Settings()