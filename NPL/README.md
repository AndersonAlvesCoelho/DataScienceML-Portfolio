# API PubMed NLP Analyzer
![Python](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1-green)
![Docker](https://img.shields.io/badge/docker-enabled-blue)

### Descrição do Projeto

Este projeto consiste em uma **API REST** que realiza **análise de artigos científicos da PubMed** utilizando técnicas de **Processamento de Linguagem Natural (NLP)**. O objetivo é extrair informações relevantes de artigos, como **resumos, abstracts, sumários** e gerar análises automatizadas para facilitar pesquisas e estudos científicos.

A API foi desenvolvida em **Python** utilizando **FastAPI**, tornando-a rápida, escalável e de fácil integração com outros sistemas.

---

### Tecnologias e Bibliotecas Utilizadas

### Backend e API
- **Python 3.12**
- **FastAPI**: framework principal da API REST
- **Uvicorn**: servidor ASGI para rodar a API

### NLP (Processamento de Linguagem Natural)
Para o tratamento de texto e análise dos artigos, foram utilizadas as seguintes bibliotecas:
- **spaCy**: tokenização, parsing e lematização de texto
- **en_core_web_sm**: modelo de linguagem da spaCy para inglês
- **NLTK**: análise de texto, stopwords e pré-processamento
- **Sumy**: sumarização automática de textos
- **Biopython**: processamento de dados biomédicos
- **WordCloud**: geração de nuvens de palavras
- **Scikit-learn**: análise estatística e pré-processamento

### Outras Bibliotecas
- **Pandas / Numpy**: manipulação e análise de dados
- **Requests**: consumo de APIs externas
- **Email-validator**: validação de emails enviados à API

---

## API PubMed

A API acessa artigos da **PubMed** a partir de seus IDs e realiza:

- Extração de **abstract** e **summary**
- Análise de palavras-chave
- Geração de relatórios resumidos
- Possibilidade de envio do resultado para email informado

### Endpoint Principal

POST /analyze


**Body JSON esperado:**

```json
{
  "email": "usuario@example.com",
  "article_ids": ["12345678", "87654321"]
}
```

Exemplo de resposta:

```json
{
  "12345678": {
    "title": "Exemplo de título",
    "abstract": "Resumo do artigo...",
    "summary": "Sumário gerado pelo NLP"
  }
}
```
### Validações realizadas:
- O campo article_ids não pode estar vazio
- Cada ID deve ser numérico
- O email deve ser válido


### Rodando o Projeto
#### Pré-requisitos
- Docker
- Docker Compose
- Git

### Passo a Passo
1. Clonar o repositório
```bash
git clone https://github.com/seu-usuario/api-pubmed-nlp.git
cd api-pubmed-nlp
```
2. Construir a imagem Docker
```bash
docker compose build
```
3. Subir o container
```bash
docker compose up -d
```
4. Acessar a API
- A API estará disponível em http://<SEU_SERVIDOR>:8000
- A documentação interativa do FastAPI pode ser acessada em http://<SEU_SERVIDOR>:8000/docs
