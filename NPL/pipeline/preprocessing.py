# pipeline/preprocessing.py
import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text: str) -> str:
    doc = nlp(text.lower())
    return " ".join([
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and len(token.text) > 3
    ])

def preprocess_abstracts(articles: list[dict]) -> list[dict]:
    for a in articles:
        a["processed"] = preprocess_text(a["abstract"])
    return articles
