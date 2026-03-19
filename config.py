"""
Application configuration using Pydantic Settings.
Loads from environment variables and .env file.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    # LLM
    GROQ_API_KEY: str = Field(default="", env="GROQ_API_KEY")
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile", env="GROQ_MODEL")

    # MongoDB
    MONGODB_URI: str = Field(default="mongodb://localhost:27017", env="MONGODB_URI")
    MONGODB_DB_NAME: str = Field(default="ai_interview_db", env="MONGODB_DB_NAME")

    # App
    APP_ENV: str = Field(default="development", env="APP_ENV")
    SECRET_KEY: str = Field(default="change-me-in-production", env="SECRET_KEY")
    MAX_QUESTIONS: int = Field(default=8, env="MAX_QUESTIONS")
    FOLLOW_UP_QUESTIONS: int = Field(default=2, env="FOLLOW_UP_QUESTIONS")

    # Embeddings
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    FAISS_INDEX_PATH: str = Field(default="./faiss_index", env="FAISS_INDEX_PATH")

    # API
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()