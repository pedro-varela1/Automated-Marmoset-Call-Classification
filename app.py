from flask import Flask, request, send_file, jsonify, render_template
import torch
from werkzeug.utils import secure_filename
import os
from inceptionResnetV1 import InceptionResnetV2
import zipfile
from io import BytesIO
import numpy as np
from preprocessing import DataPreparation
from dataloader import PredictionDataset
import pandas as pd
from torch.utils.data import DataLoader
import shutil


app = Flask(__name__)

dp = DataPreparation()  # Initialize the DataPreparation class

# Configurações
UPLOAD_FOLDER = 'uploads'
ALLOWED_AUDIO_EXTENSIONS = {'wav'}
ALLOWED_CSV_EXTENSIONS = {'csv'}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = "./model.pth"
CLASS_NAMES = ['p', 'e', 'l', 'c','w', 's','a', 'r',
               'k','g', 'y', 'n','h', 'm', 'v','o', 'z']
NUM_CLASSES = len(CLASS_NAMES)
BATCH_SIZE = 128
NUM_WORKERS = 0

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_audio_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

def allowed_csv_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_CSV_EXTENSIONS

def load_model(checkpoint_path, model, device):
    """
    Carrega o modelo treinado a partir do checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Handle both cases: full model save and state_dict save
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)
    return model.eval()

def preprocess_audio(audio_path, time_segments):
    """
    Função que será implementada para processar o áudio e gerar espectrogramas
    
    Args:
        audio_path: Caminho para o arquivo de áudio
        time_segments: DataFrame com os tempos de início e fim
    
    Returns:
        List de caminhos para as imagens dos espectrogramas gerados
    """
    pass

@torch.no_grad()
def classify_spectrograms(model, temp_spec_folder, device):
    """
    Classifica os espectrogramas gerados

    Args:
        model: Modelo treinado
        temp_spec_folder: Pasta com os espectrogramas
        device: Dispositivo onde o modelo está
    
    Returns:
        Dicionário com os índices sendo o caminho da imagem  e os valores sendo um dicionário com a predição e a confiança
    """
    dataset = PredictionDataset(temp_spec_folder)
    classification_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    softmax = torch.nn.Softmax(dim=1)

    # Gerar as predições
    predictions_index = {}
    for inputs, batch_paths in classification_loader:
        inputs = inputs.to(device)

        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        probs = softmax(outputs)

        for i, path in enumerate(batch_paths):
            predictions_index[path] = {
                'prediction': CLASS_NAMES[predictions[i].item()],
                'confidence': probs[i][predictions[i]].item()
            }

    return predictions_index


def update_csv(csv_file, predictions, paths_index):
    """
    Atualiza o arquivo CSV com as predições
    
    Args:
        csv_file: Caminho para o arquivo CSV
        predictions: Dicionário com as predições
        paths_index: Dicionário com os índices sendo o índice da linha e os valores sendo um dict com image_path
    """
    df = pd.read_csv(csv_file)
    
    # Adicionar colunas de predição se não existirem
    if 'prediction' not in df.columns:
        df['prediction'] = ""
    if 'confidence' not in df.columns:
        df['confidence'] = ""

    for index, row in df.iterrows():
        # Inicializar como vazio
        df.at[index, 'prediction'] = ""
        df.at[index, 'confidence'] = ""
        
        # Se temos o índice no paths_index
        if str(index) in paths_index:
            image_path = paths_index[str(index)]['image_path']
            if image_path and image_path in predictions:
                df.at[index, 'prediction'] = str(predictions[image_path]['prediction'])
                df.at[index, 'confidence'] = str(predictions[image_path]['confidence'])
    
    df.to_csv(csv_file, index=False)

@app.route('/classify', methods=['POST'])
def classify_vocalizations():
    # Check if CSV file and audio files are present in request
    if 'csv' not in request.files:
        return jsonify({'error': 'Missing CSV file'}), 400
    
    if 'audios' not in request.files:
        return jsonify({'error': 'Missing audio files'}), 400
    
    csv_file = request.files['csv']
    audio_files = request.files.getlist('audios')
    
    # Check if CSV is valid
    if not allowed_csv_file(csv_file.filename):
        return jsonify({'error': 'Invalid CSV file format'}), 400
    
    # Create temporary paths
    csv_path = os.path.join(UPLOAD_FOLDER, secure_filename(csv_file.filename))
    audio_folder = os.path.join(UPLOAD_FOLDER, 'audio_files')
    os.makedirs(audio_folder, exist_ok=True)
    
    # Save CSV file
    csv_file.save(csv_path)
    
    # Save all audio files
    saved_audio_files = {}
    for audio_file in audio_files:
        if audio_file and allowed_audio_file(audio_file.filename):
            filename = secure_filename(audio_file.filename)
            audio_path = os.path.join(audio_folder, filename)
            audio_file.save(audio_path)
            saved_audio_files[filename] = audio_path
    
    if not saved_audio_files:
        return jsonify({'error': 'No valid audio files uploaded'}), 400
    
    try:
        # Read CSV to get audio file names
        df = pd.read_csv(csv_path)
        
        if 'audio_path' not in df.columns:
            return jsonify({'error': 'CSV must contain "audio_path" column'}), 400
        
        # Create temporary folder for spectrograms
        temp_spec_folder = os.path.join(UPLOAD_FOLDER, 'spectrograms')
        os.makedirs(temp_spec_folder, exist_ok=True)
        
        # Process each unique audio file mentioned in CSV
        all_paths_index = {}
        unique_audio_files = df['audio_path'].unique()
        
        print(f"Processing {len(unique_audio_files)} unique audio files...")
        
        for audio_filename in unique_audio_files:
            # Check if this audio file was uploaded
            if audio_filename not in saved_audio_files:
                print(f"Warning: Audio file '{audio_filename}' not found in uploaded files. Skipping...")
                continue
            
            audio_path = saved_audio_files[audio_filename]
            
            # Create a temporary CSV for this specific audio file
            temp_csv_path = os.path.join(UPLOAD_FOLDER, f'temp_{audio_filename}.csv')
            audio_df = df[df['audio_path'] == audio_filename].copy()
            
            # Reset index but keep original index for later mapping
            original_indices = audio_df.index.tolist()
            audio_df.reset_index(drop=True, inplace=True)
            audio_df.to_csv(temp_csv_path, index=False)
            
            try:
                # Generate spectrograms for this audio file
                paths_index = dp.transform_data(temp_csv_path, audio_path, temp_spec_folder)
                
                # Map back to original indices
                for new_idx, orig_idx in enumerate(original_indices):
                    if str(new_idx) in paths_index:
                        all_paths_index[str(orig_idx)] = paths_index[str(new_idx)]
                
            except Exception as e:
                print(f"Error processing {audio_filename}: {e}")
            finally:
                # Clean up temporary CSV
                if os.path.exists(temp_csv_path):
                    os.remove(temp_csv_path)
        
        # Load and prepare model
        print("Loading model...")
        try:
            model = InceptionResnetV2(device=DEVICE, classify=True, num_classes=NUM_CLASSES)
            model = load_model(CHECKPOINT_PATH, model, DEVICE)
            model = model.to(DEVICE)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Get predictions for all spectrograms
        print("Classifying spectrograms...")
        try:
            predictions = classify_spectrograms(model, temp_spec_folder, DEVICE)
        except Exception as e:
            print(f"Error classifying spectrograms: {e}")
            raise
        
        # Update CSV file with predictions
        print("Updating CSV with predictions...")
        try:
            update_csv(csv_path, predictions, all_paths_index)
        except Exception as e:
            print(f"Error updating CSV file: {e}")
            raise
        
        # Create ZIP file with results
        print("Creating ZIP file...")
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            zf.write(csv_path, os.path.basename(csv_path))
            # Add all images in spectrogram folder to zip file
            for root, _, files in os.walk(temp_spec_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_spec_folder)
                    zf.write(file_path, os.path.join('spectrograms', arcname))
        memory_file.seek(0)
        
        # Clean up temporary files
        print("Cleaning up...")
        if os.path.exists(csv_path):
            os.remove(csv_path)
        if os.path.exists(audio_folder):
            shutil.rmtree(audio_folder)
        if os.path.exists(temp_spec_folder):
            shutil.rmtree(temp_spec_folder)
        
        print("Done!")
        return send_file(
            memory_file,
            as_attachment=True,
            download_name='predictions.zip'
        )
    
    except Exception as e:
        # Clean up in case of error
        if os.path.exists(csv_path):
            os.remove(csv_path)
        if os.path.exists(audio_folder):
            shutil.rmtree(audio_folder)
        if os.path.exists(temp_spec_folder):
            shutil.rmtree(temp_spec_folder)
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,
            debug=True)