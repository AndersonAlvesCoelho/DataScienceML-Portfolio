# 🌿 Inteligência Botânica: Bioacústica, Espécies e Doenças

Este repositório apresenta um pipeline de Deep Learning unificado que integra três frentes de análise botânica em uma API funcional, capaz de identificar espécies, diagnosticar patologias e monitorar o estresse vegetal por sinais sonoros.

## 📁 Estrutura do Projeto
```
projeto_final/
├── plant_sounds.ipynb      # Classificação de estresse via áudio ultrassônico
├── plant_species.ipynb     # Identificação de 47 espécies botânicas via Transfer Learning
├── plant_disease.ipynb     # Diagnóstico de 15 tipos de doenças foliares via CNN customizada
├── api.py                  # Servidor FastAPI que gerencia o pipeline de inferência multiclasse
├── processed/              # Modelos treinados (.keras) e artefatos de normalização (.pkl)
└── datasets/               # Datasets (não inclusos - ver instruções abaixo)
```

## 🧠 Modelos e Metodologias

### 1. Bioacústica (Sons de Plantas)

- **Descrição**: Identifica estados de estresse (seco/cortado) baseando-se em emissões ultrassônicas
- **Técnica**: Conversão de áudio para Mel-Espectrograma e classificação via CNN 2D
- **Classes**: `Tomato Dry`, `Tomato Cut`, `Tobacco Dry`, `Tobacco Cut` e `Empty Pot`

### 2. Classificação de Espécies (47 Classes)

- **Descrição**: Identificação de 47 categorias de plantas domésticas
- **Técnica**: Transfer Learning (VGG16) com Fine-Tuning do bloco final
- **Diferencial**: Alta precisão em distinguir padrões complexos de venação foliar

### 3. Diagnóstico de Doenças (15 Classes)

- **Descrição**: Detecção de patologias em Tomate, Batata e Pimentão
- **Técnica**: CNN profunda treinada com Data Augmentation para lidar com desbalanceamento de dados

## 💾 Configuração dos Datasets

Devido ao tamanho, os datasets não estão inclusos no repositório. Para rodar os notebooks, baixe os arquivos nos links abaixo e salve-os na pasta `projeto_final/datasets/` renomeados conforme indicado:

| Dataset | Link para Download | Nomear Arquivo como |
|---------|-------------------|---------------------|
| Sons (Dryad) | [Download aqui](https://datadryad.org/stash/dataset/doi:10.5061/dryad.qv9s4mwh8) | `PlantSounds.zip` |
| Espécies (Kaggle) | [Acessar Kaggle](https://www.kaggle.com/datasets/kacpergregorowicz/house-plant-species) | `HousePlantSpecies.zip` |
| Doenças (Kaggle) | [Acessar Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) | `PlantVillageDataset.zip` |

## 🛠️ Como Executar a API

### 1. Preparação

Certifique-se de ter o Anaconda instalado e o ambiente de Machine Learning ativo:
```bash
conda activate seu_ambiente
pip install fastapi uvicorn tensorflow librosa joblib python-multipart pillow
```

### 2. Rodando o Servidor
```bash
uvicorn api:app --reload
```

### 3. Testando (Swagger UI)

Acesse: `http://127.0.0.1:8000/docs`

## 🧪 Instruções de Teste

- **ID 1 (Sons)**: Envie um arquivo `.wav`
- **ID 2 (Espécies)**: Envie uma imagem da planta (Redimensionada para 224x224 internamente)
- **ID 3 (Doenças)**: Envie uma imagem da folha (Redimensionada para 256x256 internamente)

---

## 3. Testando a API

Você pode testar a inferência dos modelos de duas maneiras:

### Opção A: Interface Visual (Swagger UI)

1. Acesse: `http://127.0.0.1:8000/docs`
2. Clique no endpoint `POST /predict` e selecione "Try it out".
3. No campo `model_id`, insira o número do modelo desejado:
   * `1` (Bioacústica): Aceita apenas áudio `.wav`.
   * `2` (Espécies): Aceita imagens `.jpg`, `.jpeg` ou `.png`.
   * `3` (Doenças): Aceita imagens `.jpg`, `.jpeg` ou `.png`.
4. No campo `file`, faça o upload do arquivo de teste.
5. Clique em "Execute" e verifique a resposta JSON no final da página.

### Opção B: Clientes de API (Postman, Insomnia ou cURL)

Se preferir usar ferramentas externas para automação ou testes de integração, configure a requisição da seguinte forma:

* **Método**: `POST`
* **URL**: `http://127.0.0.1:8000/predict`
* **Body**: Selecione o formato `form-data` (multipart/form-data).
* **Parâmetros**:
   * `model_id`: (Valor `1`, `2` ou `3`)
   * `file`: (Selecione o arquivo do seu computador)

#### Exemplo via Linha de Comando (cURL):
```bash

## 📊 Requisitos

- Python 3.8+
- TensorFlow 2.x
- FastAPI
- Librosa
- Scikit-learn
- Pillow

## 📝 Licença

Este projeto está sob licença MIT. Consulte o arquivo LICENSE para mais detalhes.

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.