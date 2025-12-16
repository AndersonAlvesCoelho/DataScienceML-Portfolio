from Bio import Entrez

def fetch_pubmed_articles(id_list, email):
    """
    Busca artigos no PubMed usando IDs.
    Retorna uma lista de dicionários com id, título e abstract.
    """
    Entrez.email = email  # define o email para a requisição

    handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
    records = handle.read()
    handle.close()

    articles = []
    titles, abstracts = [], []
    current_abstract = ""
    in_abstract = False

    for line in records.strip().split('\n'):
        if line.startswith('PMID- '):
            pmid = line.replace('PMID- ', '').strip()
        elif line.startswith('TI  - '):
            if in_abstract and current_abstract:
                abstracts.append(current_abstract.strip())
                current_abstract = ""
            titles.append(line[6:])
            in_abstract = False
        elif line.startswith('AB  - '):
            current_abstract += line[6:]
            in_abstract = True
        elif line.startswith('    ') and in_abstract:
            current_abstract += " " + line.strip()

    if in_abstract and current_abstract:
        abstracts.append(current_abstract.strip())

    # Garante correspondência id ↔ título/abstract
    for idx, (title, abstract) in enumerate(zip(titles, abstracts)):
        article_id = id_list[idx] if idx < len(id_list) else None
        articles.append({
            "id": article_id,
            "title": title,
            "abstract": abstract
        })

    return articles
