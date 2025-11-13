# Automated Source Classification of Marmoset Vocalizations

## Overview

This application uses deep learning to automatically classify marmoset vocalizations from audio recordings. It processes multiple audio files, generates spectrograms, and provides predictions for different call types.

## Supported Call Types

The model can classify the following call types:
- **p, e, l, c, w, s, a, r, k, g, y, n, h, m, v, o, z**

## App

### Docker Usage

Clone the repository:
```bash
git clone https://github.com/pedro-varela1/Automated-Source-Classification-of-Marmoset-Vocalizations.git
```

```bash
cd Automated-Source-Classification-of-Marmoset-Vocalizations
```

Download the model in the repository (https://drive.google.com/file/d/1GnPa4WnJrkCIepKd_kABOIqoNZ-AYzYZ/view?usp=sharing).

Build the container by replacing it with the ```tag```, ```container name``` and ```version``` you want:
```bash
docker build -t <tag>/<container_name>:<version> .
```

Finally, run the app:
```bash
docker container run -p 5000:5000 <tag>/<container_name>:<version>
```

In your browser, go to _http://127.0.0.1:5000/_.

### App UI

![App UI](<app_ui.png>)

## Usage

### Data Entry

The application accepts **multiple audio files** and a single CSV annotation file.

#### Required Files:

1. **Audio Files** (`.wav` format)
   - Multiple audio files can be uploaded simultaneously
   - Files should be in WAV format containing marmoset calls

2. **CSV Annotation File**
   - Must contain the following columns:
     - `audio_path`: Name of the audio file (e.g., `20220302_Blue2C_volta0.wav`)
     - `onset_s`: Start time of the call in seconds
     - `offset_s`: End time of the call in seconds
     - `label`: Label indicating the type of call (optional, can be any of: p, e, l, c, w, s, a, r, k, g, y, n, h, m, v, o, z)

#### Example CSV Format:

```csv
audio_path,onset_s,offset_s,label
20220302_Blue2C_volta0.wav,1.5,2.3,p
20220302_Blue2C_volta0.wav,5.2,6.1,e
20220303_Blue3A_ida1.wav,3.4,4.2,p
20220303_Blue3A_ida1.wav,7.8,8.5,l
```

**Important Notes:**
- The `audio_path` column must contain the exact filename of the uploaded audio files
- If an audio file mentioned in the CSV is not uploaded, those rows will have empty `prediction` and `confidence` fields
- All call types listed in the CSV will be processed (not limited to 'p' as in previous versions)

### Data Output

The application returns a ZIP file containing:

1. **Updated CSV file** with two additional columns:
   - `prediction`: Predicted call type (p, e, l, c, w, s, a, r, k, g, y, n, h, m, v, o, z)
   - `confidence`: Confidence score of the prediction (0.0 to 1.0)

2. **Spectrograms folder** with PNG images of all processed calls
   - Named as: `{audio_filename}_{start_time}_{end_time}.png`

#### Example Output CSV:

```csv
audio_path,onset_s,offset_s,label,prediction,confidence
20220302_Blue2C_volta0.wav,1.5,2.3,p,p,0.95
20220302_Blue2C_volta0.wav,5.2,6.1,e,e,0.87
20220303_Blue3A_ida1.wav,3.4,4.2,p,p,0.92
audio_missing.wav,7.8,8.5,l,,
```

*Note: The last row has empty prediction/confidence because the audio file was not found.*

## Using the API

### 1. Web Interface

1. Navigate to `http://localhost:5000`
2. Select your CSV annotation file
3. Select multiple audio files (use Ctrl+Click or Shift+Click to select multiple)
4. Click "Classify Vocalizations"
5. Wait for processing to complete
6. Download the `predictions.zip` file automatically

### 2. Python API

```python
import requests

url = 'http://localhost:5000/classify'

# Prepare the files
files = {
    'csv': open('annotations.csv', 'rb'),
    'audios': [
        open('20220302_Blue2C_volta0.wav', 'rb'),
        open('20220303_Blue3A_ida1.wav', 'rb'),
        # Add more audio files as needed
    ]
}

# Make the request
response = requests.post(url, files=files)

# Save the result
if response.status_code == 200:
    with open('predictions.zip', 'wb') as f:
        f.write(response.content)
    print("Classification complete! Results saved to predictions.zip")
else:
    print(f"Error: {response.json()['error']}")
```

### 3. cURL

```bash
curl -X POST http://localhost:5000/classify \
  -F "csv=@annotations.csv" \
  -F "audios=@audio1.wav" \
  -F "audios=@audio2.wav" \
  -F "audios=@audio3.wav" \
  -o predictions.zip
```

## Training Your Own Model

### Data Preprocessing

Process raw audio files and annotations into training/test datasets:

```bash
python preprocessing_train.py \
  --input_folder "path/to/audio/folder" \
  --output_path "path/to/output" \
  --labels p e l c w s a r k g y n h m v o z \
  --train_ratio 0.85 \
  --freq_cutoff_up 18000 \
  --freq_cutoff_down 750 \
  --min_duration 100 \
  --max_duration 1600
```

**Input Structure:**
```
input_folder/
├── audio1.wav
├── audio1.wav.csv
├── audio2.wav
├── audio2.wav.csv
└── ...
```

**Output Structure:**
```
output_path/
├── train/
│   ├── p/
│   ├── e/
│   ├── l/
│   └── ...
└── test/
    ├── p/
    ├── e/
    ├── l/
    └── ...
```

### Training the Model

Train the classification model:

```bash
python train.py \
  --train_dir "./data/train" \
  --test_dir "./data/test" \
  --batch_size 128 \
  --num_epochs 40 \
  --learning_rate 0.0001 \
  --model v2
```

**Parameters:**
- `--train_dir`: Path to training data directory
- `--test_dir`: Path to test data directory
- `--batch_size`: Training batch size (default: 128)
- `--num_epochs`: Number of training epochs (default: 40)
- `--learning_rate`: Learning rate (default: 0.0001)
- `--weight_decay`: L2 regularization (default: 1e-4)
- `--model`: Model version - 'v1' or 'v2' (default: v2)
- `--save_dir`: Directory to save checkpoints (default: checkpoints)
- `--seed`: Random seed for reproducibility (default: 43)

The number of classes is automatically detected from the subdirectories in the training folder.

### Visualizing Results

Generate t-SNE visualization of the learned embeddings:

```bash
python plot_visualization2d.py
```

This will create `embeddings_tsne.png` showing the clustering of different call types in the embedding space.

## Requirements

- Python 3.8+
- PyTorch
- librosa
- Flask
- pandas
- numpy
- scikit-learn
- matplotlib
- pydub
- tqdm

See `requirements.txt` for complete dependencies.

## License

[Add your license information here]

## Citation

If you use this work, please cite:

[Add citation information here]