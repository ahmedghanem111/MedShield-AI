from fastapi import FastAPI
from app.db.database import Base, engine
from app.db import base
from app.api.auth_api import router as auth_router
from app.middleware.security_headers import SecurityHeadersMiddleware

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="MedShield AI API",
    description="Medical Deepfake & Tampered Report Detection API",
    version="1.0.0",
)
app.add_middleware(SecurityHeadersMiddleware)

app.include_router(auth_router)


@app.get("/")
def root():
    return {
        "message": "MedShield AI API is running"
    }
