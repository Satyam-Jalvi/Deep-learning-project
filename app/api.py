# app/api.py

from fastapi import FastAPI
from app.router import router

app = FastAPI(title="Medical Diagnosis API")

# include routes
app.include_router(router)