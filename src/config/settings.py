"""Configuration management for MCPulse application."""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # LLM API Keys
    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, alias="ANTHROPIC_API_KEY")
    
    # MongoDB Configuration
    mongodb_uri: Optional[str] = Field("mongodb://localhost:27017", alias="MONGODB_URI")
    mongodb_database: str = Field("mcpulse", alias="MONGODB_DATABASE")
    mongodb_collection: str = Field("chat_history", alias="MONGODB_COLLECTION")
    
    # Application Settings
    default_model: str = Field("gpt-4-turbo-preview", alias="DEFAULT_MODEL")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    gradio_server_port: int = Field(7860, alias="GRADIO_SERVER_PORT")
    gradio_server_name: str = Field("0.0.0.0", alias="GRADIO_SERVER_NAME")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
