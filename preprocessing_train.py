import numpy as np
import pandas as pd
import os
import librosa # type: ignore
import matplotlib
import matplotlib.pyplot as plt
import io
from pydub import AudioSegment # type: ignore
import glob
from tqdm import tqdm
from scipy.signal import butter, lfilter
import argparse
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

    def transform_data(self, input_folder, output_path, labels, train_ratio=0.85):
        """
        Get calls by labels and transform them into spectrograms.
        Organize into train/test folders with subfolders for each label.

        Args:
            input_folder (str): Path to the folder containing audio files and their annotations.
            output_path (str): Path to the output directory.
            labels (list): List of labels to process (e.g., ['p', 't', 'ts']).
            train_ratio (float): Ratio of data to use for training (default: 0.85).
        """
        # Find all audio files in the input folder
        audio_files = glob.glob(os.path.join(input_folder, "*.wav"))
        
        if not audio_files:
            print(f"No audio files found in {input_folder}")
            return
        
        print(f"Found {len(audio_files)} audio file(s)")
        
        # Create output directories
        train_path = os.path.join(output_path, 'train')
        test_path = os.path.join(output_path, 'test')
        
        # Collect all annotations by label from all files
        label_annotations = {label: [] for label in labels}
        
        # Process each audio file
        for audio_path in audio_files:
            audio_file_name = os.path.basename(audio_path)
            annotation_file_path = audio_path + ".csv"
            
            # Check if annotation file exists
            if not os.path.exists(annotation_file_path):
                print(f"Warning: Annotation file not found for {audio_file_name}, skipping...")
                continue
            
            print(f"Processing {audio_file_name}...")
            
            try:
                audio = AudioSegment.from_wav(audio_path)
                annotation_df = pd.read_csv(annotation_file_path)
                audio_name = os.path.splitext(audio_file_name)[0]
                
                # Collect annotations from this file
                for index, annotation_row in annotation_df.iterrows():
                    if annotation_row['label'] in labels:
                        label_annotations[annotation_row['label']].append({
                            'annotation_row': annotation_row,
                            'audio': audio,
                            'audio_name': audio_name
                        })
            except Exception as e:
                print(f"Error loading {audio_file_name}: {e}")
                continue
        
        # Process each label
        print("\nSplitting data into train/test sets...")
        for label in labels:
            annotations = label_annotations[label]
            if not annotations:
                print(f"No annotations found for label '{label}'")
                continue
            
            # Create label directories
            label_train_path = os.path.join(train_path, label)
            label_test_path = os.path.join(test_path, label)
            os.makedirs(label_train_path, exist_ok=True)
            os.makedirs(label_test_path, exist_ok=True)
            
            # Shuffle and split annotations
            np.random.shuffle(annotations)
            split_index = int(len(annotations) * train_ratio)
            train_annotations = annotations[:split_index]
            test_annotations = annotations[split_index:]
            
            # Process training data
            for item in tqdm(train_annotations, desc=f"Processing label '{label}' - train", leave=False):
                self._process_annotation(
                    item['annotation_row'], 
                    item['audio'], 
                    item['audio_name'], 
                    label_train_path
                )
            
            # Process test data
            for item in tqdm(test_annotations, desc=f"Processing label '{label}' - test", leave=False):
                self._process_annotation(
                    item['annotation_row'], 
                    item['audio'], 
                    item['audio_name'], 
                    label_test_path
                )
            
            print(f"Label '{label}': {len(train_annotations)} train, {len(test_annotations)} test")
    
    def _process_annotation(self, annotation_row, audio, audio_file_name, output_dir):
        """
        Process a single annotation and save the spectrogram.
        
        Args:
            annotation_row: The annotation row from the dataframe.
            audio: The AudioSegment object.
            audio_file_name: The name of the audio file.
            output_dir: The directory to save the spectrogram.
        """
        start_time = int(annotation_row['onset_s']*1000)    # Start time in ms
        end_time = int(annotation_row['offset_s']*1000)     # End time in ms
        
        if end_time <= start_time:
            print()
            print(f"Skipping invalid segment: start_time={start_time}, end_time={end_time}")
            return
        
        output_file_path = os.path.join(output_dir,
                                        f"{audio_file_name}_{str(start_time)}_{str(end_time)}.png")
        
        # Process the vocalization
        cut_audio = audio[start_time:end_time]  # Cut the audio
        try:
            self.y = self._load_audio(cut_audio)    # Load the audio with background noise
            if self.y is None or len(self.y) == 0:
                return
            self._create_spec(output_file_path)
        except Exception as e:
            print()
            print(f"Error processing {output_file_path}: {e}")
            return


def main():
    parser = argparse.ArgumentParser(description='Transform marmoset calls into spectrograms')
    
    # Required arguments
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Path to the folder containing audio files (.wav) and their annotation files (.wav.csv)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to the output directory (train/test folders will be created here)')
    
    # Labels
    parser.add_argument('--labels', type=str, nargs='+', default=['p', 'e', 'l', 
                                                                  'c','w', 's',
                                                                  'a', 'r','k',
                                                                  'g', 'y', 'n',
                                                                  'h', 'm', 'v',
                                                                  'o', 'z'],
                        help='List of labels to process (e.g., p t ts). Default: p e l c w s a r k g y n h m v o z')
    
    # Spectrogram parameters
    parser.add_argument('--freq_cutoff_up', type=int, default=18000,
                        help='Upper frequency cutoff in Hz (default: 18000)')
    parser.add_argument('--freq_cutoff_down', type=int, default=750,
                        help='Lower frequency cutoff in Hz (default: 750)')
    parser.add_argument('--sr', type=int, default=48000,
                        help='Sample rate in Hz (default: 48000)')
    parser.add_argument('--n_fft', type=int, default=1024,
                        help='FFT window size (default: 1024)')
    parser.add_argument('--hop_length', type=int, default=256,
                        help='Hop length for STFT (default: 256)')
    parser.add_argument('--min_duration', type=int, default=100,
                        help='Minimum call duration in ms (default: 100)')
    parser.add_argument('--max_duration', type=int, default=1600,
                        help='Maximum call duration in ms (default: 1600)')
    
    # Train/test split
    parser.add_argument('--train_ratio', type=float, default=0.85,
                        help='Ratio of data for training (default: 0.85)')
    
    # Background noise
    parser.add_argument('--background_noise_path', type=str, default='background_noise',
                        help='Path to background noise directory (default: background_noise)')
    
    args = parser.parse_args()
    
    # Create DataPreparation instance
    data_prep = DataPreparation(
        background_noise_path=args.background_noise_path,
        freq_cutoff_up=args.freq_cutoff_up,
        freq_cutoff_down=args.freq_cutoff_down,
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        min_duration=args.min_duration,
        max_duration=args.max_duration
    )
    
    # Process the data
    print(f"Input folder: {args.input_folder}")
    print(f"Output directory: {args.output_path}")
    print(f"Labels: {args.labels}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Frequency range: {args.freq_cutoff_down} - {args.freq_cutoff_up} Hz")
    print(f"Duration range: {args.min_duration} - {args.max_duration} ms")
    print()
    
    data_prep.transform_data(
        input_folder=args.input_folder,
        output_path=args.output_path,
        labels=args.labels,
        train_ratio=args.train_ratio
    )
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
            