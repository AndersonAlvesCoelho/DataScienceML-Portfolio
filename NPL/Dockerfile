FROM python:3.12-slim

# Diretório de trabalho
WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia o requirements e instala dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Instala bibliotecas adicionais necessárias para o NLP
RUN pip install --no-cache-dir biopython wordcloud scikit-learn spacy sumy nltk

# Baixa o modelo do spaCy
RUN python -m spacy download en_core_web_sm

# Baixa os dados do NLTK (punkt e punkt_tab)
RUN python -m nltk.downloader punkt punkt_tab

# Copia o código da API
COPY . .

# Expõe a porta do FastAPI
EXPOSE 8000

# Comando para iniciar a API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
