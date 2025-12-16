from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer

LANGUAGE = "english"

def generate_summary(text: str, sentences: int = 5) -> str:
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    summarizer = Summarizer()
    summary = summarizer(parser.document, sentences)
    return " ".join([str(sentence) for sentence in summary])

def summarize_abstracts(articles: list[dict]) -> list[dict]:
    for a in articles:
        a["summary"] = generate_summary(a["abstract"])
    return articles

def summarize_general(abstracts: list[str]) -> str:
    combined = " ".join(abstracts)
    return generate_summary(combined, sentences=15)
