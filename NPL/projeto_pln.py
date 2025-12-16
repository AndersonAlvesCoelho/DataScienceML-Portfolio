# Importação geral
import spacy
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
import pandas as pd
from Bio import Entrez
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer

# Configurações
pd.set_option('display.max_colwidth', 200)
sns.set_style('whitegrid')

# Coleta de dados sobre Câncer de Pele, utilizando "Entrez"
Entrez.email = "gustavo.martinss@sempreceub.com"

# Usamos o termo em inglês "skin cancer" para obter mais resultados, pois é o idioma principal do PubMed.
search_term = "skin cancer"
num_articles = 500  # Ajustamos para coleta de 500 artigos.

# Busca os IDs dos artigos
print(f"Buscando {num_articles} artigos sobre '{search_term}'...")
handle = Entrez.esearch(db="pubmed", term=search_term, retmax=num_articles, sort="relevance")
record = Entrez.read(handle)
handle.close()
id_list = record["IdList"]

# Busca os detalhes dos artigos usando os IDs
handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
records = handle.read()
handle.close()

# Extrai as informações e cria um DataFrame
titles = []
abstracts = []

# Loop para extrair Título (TI) e Resumo (AB)
current_abstract = ""
in_abstract = False
for line in records.strip().split('\n'):
    if line.startswith('TI  - '):
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

# Adiciona o último resumo, se houver
if in_abstract and current_abstract:
    abstracts.append(current_abstract.strip())

# Garante que temos a mesma quantidade de títulos e abstracts
min_len = min(len(titles), len(abstracts))
df = pd.DataFrame({
    'title': titles[:min_len],
    'abstract': abstracts[:min_len]
})

print(f"Foram coletados {len(df)} artigos.")
df.head()

# Pré-processamento
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Processa o texto com o spaCy
    doc = nlp(text.lower())

    # Gera uma lista de tokens lematizados, sem stop words e pontuação
    result = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct and len(token.text) > 3
    ]

    return " ".join(result)

# Aplica a função na coluna de abstracts
df['processed_abstract'] = df['abstract'].apply(preprocess_text)

print("Pré-processamento concluído.")
df[['abstract', 'processed_abstract']].head()

# Frequência de Palavras
# Junta todos os textos processados em um só
all_text = " ".join(df['processed_abstract'])
all_words = all_text.split()

# Conta as palavras
word_counts = Counter(all_words)

# Cria um DataFrame com as 20 palavras mais comuns
common_words_df = pd.DataFrame(word_counts.most_common(20), columns=['word', 'count'])

# Plota o gráfico
plt.figure(figsize=(12, 8))
sns.barplot(x='count', y='word', data=common_words_df, palette='viridis')
plt.title(f'20 Palavras Mais Comuns em Artigos sobre "{search_term}"')
plt.xlabel('Frequência')
plt.ylabel('Palavra')
plt.show()

# Nuvem de Palavras
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title(f'Nuvem de Palavras para "{search_term}"')
plt.show()

# Vetorização com TF-IDF

vectorizer = TfidfVectorizer(max_features=1000) # Limitamos para as 1000 palavras mais importantes
tfidf_matrix = vectorizer.fit_transform(df['processed_abstract'])

# Encontrando o número ideal de clusters (Método do Cotovelo)

wcss = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(tfidf_matrix)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(2, 11), wcss, marker='o')
plt.title('Método do Cotovelo (Elbow Method)')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.show()

# Aplicando K-Means e Analisando os Clusters
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(tfidf_matrix)

# Adiciona o rótulo do cluster ao DataFrame
df['cluster'] = kmeans.labels_

# Mostra os 10 termos mais importantes por cluster
print("Termos mais importantes por cluster:")
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
for i in range(num_clusters):
    print(f"Cluster {i}: ", end="")
    for ind in order_centroids[i, :10]:
        print(f"{terms[ind]} ", end="")
    print("\n")

# Reduz a dimensionalidade da matriz TF-IDF para 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_results = tsne.fit_transform(tfidf_matrix.toarray())

# Adiciona os resultados ao DataFrame
df['tsne_1'] = tsne_results[:,0]
df['tsne_2'] = tsne_results[:,1]

# Plota o gráfico de dispersão
plt.figure(figsize=(14, 10))
sns.scatterplot(
    x="tsne_1", y="tsne_2",
    hue="cluster",
    palette=sns.color_palette("hls", num_clusters),
    data=df,
    legend="full",
    alpha=0.8
)
plt.title('Visualização dos Clusters de Artigos com t-SNE')
plt.xlabel('Componente t-SNE 1')
plt.ylabel('Componente t-SNE 2')
plt.show()

# Modelagem de Tópicos com LDA
# LDA funciona melhor com contagem de palavras, não TF-IDF
count_vectorizer = CountVectorizer(max_features=1000, stop_words='english')
count_matrix = count_vectorizer.fit_transform(df['processed_abstract'])

# Define o número de tópicos e aplica o LDA
num_topics = 5
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(count_matrix)

# Função para mostrar os tópicos
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Tópico {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

# Mostra os 10 termos mais importantes por tópico
print("\nTópicos encontrados com LDA:")
display_topics(lda, count_vectorizer.get_feature_names_out(), 10)

# Criando a função para gerar resumos

# Define o idioma para o tokenizador
LANGUAGE = "english"

def generate_summary(text, num_sentences=10):
    """
    Gera um resumo extrativo de um texto usando o algoritmo LSA.
    """
    # Inicia o parser a partir do texto original
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))

    # Inicia o sumarizador LSA
    summarizer = Summarizer()

    # Gera o resumo com o número de sentenças desejado
    summary = summarizer(parser.document, num_sentences)

    # Junta as sentenças selecionadas para formar o resumo final
    short_summary = " ".join([str(sentence) for sentence in summary])

    return short_summary

# Exemplo de teste com o primeiro artigo do nosso DataFrame
exemplo_resumo = generate_summary(df['abstract'].iloc[0])

print("EXEMPLO DE RESUMO GERADO:")
print(exemplo_resumo)

# Gerando resumos para todos os artigos e visualizando

# Aplica a função de resumo à coluna 'abstract'
print("Gerando resumos para todos os artigos...")
df['summary'] = df['abstract'].apply(generate_summary)
print("Resumos gerados com sucesso!")

# Mostra o resultado lado a lado para comparação
# A coluna 'abstract' tem o texto original, e 'summary' tem o novo resumo curto
df[['abstract', 'summary']].head()

# Imput de busca

# Verifica se o DataFrame 'df' existe para iniciar a ferramenta
if 'df' in locals() and 'title' in df.columns:
    print(" Ferramenta de Busca ")

    # Loop infinito para permitir várias buscas
    while True:
        print("\nDigite uma palavra-chave do título do artigo que deseja buscar.")
        user_input = input("Para sair, digite 'sair': ")

        if user_input.lower() == 'sair':
            print("Sessão encerrada.")
            break

        # Busca por artigos cujo título contém o texto inserido
        search_results = df[df['title'].str.contains(user_input, case=False, na=False)]

        # Lida com os resultados da busca
        if len(search_results) == 0:
            print(f"\n Nenhum artigo encontrado com o termo '{user_input}'. Tente novamente.")
            continue

        elif len(search_results) == 1:
            selected_article = search_results.iloc[0]
            print(f"\n Artigo encontrado! ")

        else:
            print(f"\n Foram encontrados {len(search_results)} artigos. Por favor, escolha um:")
            for i, title in enumerate(search_results['title']):
                print(f"  [{i}] - {title[:100]}...")

            try:
                choice = int(input("Digite o número do artigo: "))
                if 0 <= choice < len(search_results):
                    selected_article = search_results.iloc[choice]
                    print(f"\n Artigo selecionado! ")
                else:
                    print("Escolha inválida. Tente novamente.")
                    continue
            except ValueError:
                print("Entrada inválida. Por favor, digite um número. Tente novamente.")
                continue

        # --- APENAS VISUALIZAÇÃO ---
        if 'selected_article' in locals() and selected_article is not None:
            # Mostra as informações do artigo selecionado
            print(f"\nTÍTULO: {selected_article['title']}\n")
            print(f"CLUSTER DO TEMA: {selected_article['cluster']}\n")
            print("--- RESUMO GERADO ---")
            print(f"{selected_article['summary']}\n")

            # Reseta a variável para a próxima busca
            selected_article = None

else:
    print("ERRO: O DataFrame 'df' não foi encontrado ou não contém a coluna 'title'.")
    print("Por favor, execute as células anteriores do notebook primeiro.")