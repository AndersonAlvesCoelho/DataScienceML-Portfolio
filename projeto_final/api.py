import os
import shutil
import numpy as np
import librosa
import joblib
import json
from typing import Annotated
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = FastAPI(title="Plant Intelligence")

# CONFIGURAÇÃO DE CAMINHOS
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(CURRENT_DIR, "processed")

# Mapeamentos baseados nos seus modelos
AUDIO_LABELS = ["Tomato Dry", "Tomato Cut", "Tobacco Dry", "Tobacco Cut", "Empty Pot"]
DISEASE_LABELS = [
    "Pepper Bell Bacterial Spot", "Pepper Bell Healthy", "Potato Early Blight",
    "Potato Late Blight", "Potato Healthy", "Tomato Bacterial Spot",
    "Tomato Early Blight", "Tomato Late Blight", "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot", "Tomato Spider Mites", "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus", "Tomato Mosaic Virus", "Tomato Healthy"
]

MODELS_CONFIG = {
    "1": {"name": "Bioacústica (Sons)", "file": "audio_model.keras", "category": "audio", "labels": AUDIO_LABELS},
    "2": {"name": "Espécies (47 classes)", "file": "model_plant_species.keras", "category": "image", "size": 224, "labels": None},
    "3": {"name": "Doenças (15 classes)", "file": "plant_disease_cnn_model.keras", "category": "image", "size": 256, "labels": DISEASE_LABELS}
}

# FUNÇÕES DE PROCESSAMENTO
def process_audio(file_path):
    scaler = joblib.load(os.path.join(BASE_PATH, 'audio_scaler.pkl'))
    y, sr = librosa.load(file_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    if mel_db.shape[1] > 64: mel_db = mel_db[:, :64]
    else: mel_db = np.pad(mel_db, ((0, 0), (0, 64 - mel_db.shape[1])), mode='constant')
    features = scaler.transform(mel_db.flatten().reshape(1, -1))
    return features.reshape(1, 128, 64, 1)

def process_image(file_path, size):
    img = image.load_img(file_path, target_size=(size, size))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ROTA
@app.post("/predict", summary="Realiza a predição baseada no modelo selecionado")
async def predict_pipeline(
    model_id: Annotated[str, Form(
        description="ID do modelo: 1=Sons (Bioacústica), 2=Espécies (47 plantas), 3=Doenças (15 patologias)",
        examples=["1", "2", "3"]
    )],
    file: Annotated[UploadFile, File(description="Arquivo de entrada (Áudio .wav para ID 1 ou Imagem .jpg/.png para IDs 2 e 3)")]
):
    """
    Endpoint principal que unifica os três projetos de Deep Learning.
    - **Model ID 1**: Recebe áudio e retorna o estado de estresse da planta.
    - **Model ID 2**: Recebe imagem e retorna a espécie botânica.
    - **Model ID 3**: Recebe imagem e retorna a saúde/doença da planta.
    """
    if model_id not in MODELS_CONFIG:
        raise HTTPException(status_code=400, detail="ID inválido. Use 1, 2 ou 3.")
    
    config = MODELS_CONFIG[model_id]
    
    # Validação de Segurança por Categoria
    if config["category"] == "audio" and not file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=415, 
            detail=f"Arquivo inválido para '{config['name']}'. Esperado: Áudio (.wav). Recebido: {file.content_type}."
        )

    if config["category"] == "image" and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415, 
            detail=f"Arquivo inválido para '{config['name']}'. Esperado: Imagem (.jpg, .png). Recebido: {file.content_type}."
        )

    temp_file = os.path.join(CURRENT_DIR, f"temp_{file.filename}")
    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        model = load_model(os.path.join(BASE_PATH, config["file"]))
        
        if config["category"] == "audio":
            input_data = process_audio(temp_file)
        else:
            input_data = process_image(temp_file, config["size"])

        prediction = model.predict(input_data)
        idx = int(np.argmax(prediction))
        
        # Recuperação do Nome da Classe
        if model_id == "2":
            with open(os.path.join(BASE_PATH, "class_indices.json"), 'r', encoding='utf-8') as f:
                mapping = json.load(f)
                res_label = {v: k for k, v in mapping.items()}[idx]
        else:
            res_label = config["labels"][idx]

        return {
            "projeto": config["name"],
            "identificacao": res_label,
            "confianca": f"{float(np.max(prediction)):.2%}",
            "status_analise": "concluída"
        }
    finally:
        if os.path.exists(temp_file): os.remove(temp_file)