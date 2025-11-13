import numpy as np
import pandas as pd
import os
import librosa
import matplotlib
import matplotlib.pyplot as plt
import io
from pydub import AudioSegment
import glob
from tqdm import tqdm
from scipy.signal import butter, lfilter
import urllib.parse
import base64
matplotlib.use('Agg')

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

class DataPreparation:
    def __init__(self, background_noise_path="background_noise",
                 freq_cutoff_up=18000, freq_cutoff_down=750,
                 sr=48000, n_fft=1024, hop_length=256,
                 min_duration=300, max_duration=2000):
        self.background_noise_path = background_noise_path
        self.freq_cutoff_up = freq_cutoff_up
        self.freq_cutoff_down = freq_cutoff_down
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def _get_background_noise(self):
        # Get a random file of background noise.
        wav_files = glob.glob(os.path.join(self.background_noise_path, "*.wav"))
        bgn_file = np.random.choice(wav_files)
        # Get random slice of background noise
        bn = AudioSegment.from_wav(bgn_file)
        bn = bn.set_frame_rate(self.sr)
        # The slice has size equal to the max_duration
        start_time = np.random.randint(0, len(bn)-self.max_duration)
        end_time = start_time + self.max_duration
        bn_slice = bn[start_time:end_time]
        return bn_slice

    def _load_audio(self, audio):
        # Load the audio, resample and add background noise.
        # Return audio in librosa format.
        audio = audio.set_frame_rate(self.sr)   # Resample
        duration = len(audio)   # Duration of the audio
        if duration < self.min_duration or duration > self.max_duration:
            raise Exception(
                f"Audio duration {duration} is not in the range [{self.min_duration}, {self.max_duration}]"
                )
        # Mix with the background noise
        background_noise = self._get_background_noise()
        audio = background_noise.overlay(audio,
                                         # Make shure that the cut audio is
                                         # in the middle of the background noise
                                         position=(self.max_duration-duration)//2)
        audio_stream = io.BytesIO()
        audio.export(audio_stream, format='wav')
        audio_stream.seek(0)
        y, _ = librosa.load(audio_stream, sr=self.sr)
        audio_stream.close()
        return y
    
    def _filter_audio(self, y):
        # Filter the audio
        y = butter_bandpass_filter(y, self.freq_cutoff_down, self.freq_cutoff_up, self.sr)
        return y
    
    def _create_spec(self, output_path):
        mydpi = 100
        x = self._filter_audio(self.y)  # Bandpass filter
        xdb = librosa.amplitude_to_db(
            abs(librosa.stft(x, hop_length=self.hop_length)), ref=np.max
            )
        plt.figure(figsize=(227/mydpi, 227/mydpi), dpi=mydpi)
        plt.axis('off')
        librosa.display.specshow(xdb, sr=self.sr, x_axis='time', y_axis='log', cmap='gray')
        plt.ylim([self.freq_cutoff_down, self.freq_cutoff_up])  # Limit the frequency range
        plt.savefig(output_path, dpi=mydpi, bbox_inches='tight', pad_inches=0)
        plt.close()

    def _decode_image(sekf, str_base64):
        base64_encoded = urllib.parse.unquote(str_base64)
        image_data = base64.b64decode(base64_encoded)
        bytes_image = io.BytesIO(image_data)
        return bytes_image
    
    def _encode_image(self, image_path):
        # Encode the image in base64 with the urllib.parse.quote function
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        return urllib.parse.quote(encoded_image)

    def transform_data(self, annotation_file_path, audio_path, output_path):
        """
        Get phee calls and transform them into spectrograms.

        Args:
            annotation_file_path (str): Path to the annotation file.
            audio_path (str): Path to the audio file.
            output_path (str): Path to the output directory.
        """
        audio = AudioSegment.from_wav(audio_path)
        audio_file_name = os.path.splitext(os.path.basename(audio_path))[0]
        annotation_df = pd.read_csv(annotation_file_path)
        paths_index = {}
        for index, annotation_row in tqdm(annotation_df.iterrows(), desc=f"Preprocessing calls", leave=False):
            if annotation_row['label'] == 'p':
                start_time = int(annotation_row['onset_s']*1000)    # Start time in ms
                end_time = int(annotation_row['offset_s']*1000)     # End time in ms
                if end_time <= start_time:
                    print()
                    print(f"Skipping invalid segment: start_time={start_time}, end_time={end_time}")
                    continue
                output_file_path = os.path.join(output_path,
                                                f"{audio_file_name}_{str(start_time)}_{str(end_time)}.png") # Output file path
                # Process the vocalization
                cut_audio = audio[start_time:end_time]  # Cut the audio
                try:
                    self.y = self._load_audio(cut_audio)    # Load the audio with background noise
                    if self.y is None or len(self.y) == 0:
                        continue
                    self._create_spec(output_file_path)
                except Exception as e:
                    print()
                    print(f"Error processing {output_file_path}: {e}")
                    continue
                paths_index[str(index)] = {
                                        "image_path" : output_file_path,
                                        "base64_image" : self._encode_image(output_file_path)
                                    }
                                      

        return paths_index
            