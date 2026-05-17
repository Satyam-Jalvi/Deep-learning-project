from fastapi import FastAPI
from app.router import router

app = FastAPI(
    title="Medical Diagnosis API"
)

app.include_router(router)

@app.get("/")
def home():
    return {"message": "API Running Successfully"}