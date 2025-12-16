from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, field_validator
from typing import List
from pipeline.pipeline import analyze_articles

app = FastAPI(title="PubMed NLP Analyzer")

# 🔓 Middleware para liberar CORS completamente (qualquer origem)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # permite qualquer origem
    allow_credentials=True,
    allow_methods=["*"],  # permite todos os métodos HTTP (GET, POST, etc.)
    allow_headers=["*"],  # permite todos os cabeçalhos
)

class PubMedRequest(BaseModel):
    email: EmailStr
    article_ids: List[str]

    @field_validator("article_ids")
    @classmethod
    def validate_article_ids(cls, v):
        if not v:
            raise ValueError("A lista de IDs de artigos não pode estar vazia.")
        invalid_ids = [i for i in v if not i.isdigit()]
        if invalid_ids:
            raise ValueError(f"Os seguintes IDs são inválidos: {invalid_ids}")
        return v


@app.post("/analyze")
async def analyze(request: PubMedRequest):
    try:
        result = analyze_articles(request.article_ids, request.email)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
