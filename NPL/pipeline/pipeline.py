from pipeline.data_fetch import fetch_pubmed_articles
from pipeline.preprocessing import preprocess_abstracts
from pipeline.summarizer import summarize_abstracts, summarize_general

def analyze_articles(pubmed_ids: list[str], email: str) -> dict:
    """
    Pipeline principal — busca artigos, processa textos e retorna resumos.
    """
    # 1. Buscar artigos no PubMed com e-mail do usuário
    articles = fetch_pubmed_articles(pubmed_ids, email)
    if not articles:
        return {"error": "Nenhum artigo encontrado."}

    # 2. Pré-processamento
    processed = preprocess_abstracts(articles)

    # 3. Resumo individual
    summarized = summarize_abstracts(processed)

    # 4. Resumo geral
    general_summary = summarize_general([a["abstract"] for a in summarized])

    # 5. Retorno final, incluindo IDs
    return {
        "articles": summarized,
        "general_summary": general_summary
    }
