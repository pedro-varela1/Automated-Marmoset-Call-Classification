import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from inceptionResnetV1 import InceptionResnetV2
from tqdm import tqdm
import torch.nn.functional as F

class TestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Dataset for loading all test images with their respective labels
        
        Args:
            data_dir: Directory containing subdirectories for each call type
            transform: Optional transform to be applied on images
        """
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_to_id = {}
        
        # Walk through all subdirectories (each representing a call type)
        for label_idx, call_type_dir in enumerate(sorted(os.listdir(data_dir))):
            call_type_path = os.path.join(data_dir, call_type_dir)
            if os.path.isdir(call_type_path):
                self.label_to_id[call_type_dir] = label_idx
                for img_name in os.listdir(call_type_path):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(call_type_path, img_name)
                        self.images.append(img_path)
                        self.labels.append(label_idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path

def get_best_images(model, dataloader, device):
    """Extract embeddings for all images"""
    model.eval()
    most_confident_images = {}

    with torch.no_grad():
        for images, batch_labels, batch_paths in tqdm(dataloader, desc="Extracting best images"):
            images = images.to(device)

            logits = model.forward(images)
            predicted_labels = torch.argmax(logits, dim=1)
            confidences, _ = torch.max(F.softmax(logits, dim=1), dim=1)
            for i in range(len(predicted_labels)):
                if predicted_labels[i] == batch_labels[i]:
                    label = predicted_labels[i].item()
                    confidence = confidences[i].item()
                    path = batch_paths[i]
                    if label not in most_confident_images or confidence >= most_confident_images.get(label, {'confidence': -1})['confidence']:
                        most_confident_images[label] = {
                            'path': path,
                            'confidence': confidence
                        }
    
    return most_confident_images

def get_embeddings(model, dataloader, device):
    """Extract embeddings for all images"""
    model.eval()
    embeddings = []
    labels = []
    paths = []
    
    with torch.no_grad():
        for images, batch_labels, batch_paths in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)

            ##### Get features from the model
            features = model.forward_once(images)
            features = F.normalize(features, p=2, dim=1)  # Normalize for cosine similarity

            # Store results
            embeddings.append(features.cpu().numpy())
            labels.extend(batch_labels.numpy())
            paths.extend(batch_paths)
    
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    return embeddings, labels, paths

def plot_tsne(embeddings, labels, output_path, label_to_id=None):
    """Plot t-SNE visualization of the embeddings with distinct colors"""
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, init='random', perplexity=40)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Generate distinct colors
    n_classes = len(np.unique(labels))
    
    # Define colors and markers for the new classes
    # Using a color palette and marker variety for better distinction
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
              '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
              '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
              '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h',
               'H', '+', 'x', 'd', '|', '_', '.', ',', '1', '2']
    
    # Plot each class separately with different colors and markers
    for i, label in enumerate(np.unique(labels)):
        mask = labels == label
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        class_name = list(label_to_id.keys())[list(label_to_id.values()).index(label)] if label_to_id else f"Class {label}"
        
        plt.scatter(embeddings_2d[mask, 0], 
                   embeddings_2d[mask, 1],
                   c=[color],
                   marker=marker,
                   s=100,  # Increased marker size
                   alpha=0.7,
                   label=class_name)
    
    # Customize plot
    plt.title('t-SNE visualization of embeddings', fontsize=14, pad=20, fontweight='bold')
    plt.xlabel('t-SNE dimension 1', fontsize=12, fontweight='bold')
    plt.ylabel('t-SNE dimension 2', fontsize=12, fontweight='bold')
    
    # Add legend with multiple columns
    ncols = min(3, (n_classes + 2) // 3)
    legend = plt.legend(bbox_to_anchor=(1.05, 1), 
                        loc='upper left', 
                        borderaxespad=0.,
                        fontsize=10,
                        ncol=ncols,
                        title='Call Types')
    legend.get_title().set_fontweight('bold')
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add a border around the plot
    plt.box(True)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()

    # Create a second plot with larger figure for better legend visibility
    fig_height = max(6, n_classes // 3)
    plt.figure(figsize=(12, fig_height))
    plt.axis('off')
    
    # Create legend only plot
    for i, label in enumerate(np.unique(labels)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        class_name = list(label_to_id.keys())[list(label_to_id.values()).index(label)] if label_to_id else f"Class {label}"
        
        plt.scatter([], [], 
                   c=[color],
                   marker=marker,
                   s=100,
                   label=class_name)
    
    legend = plt.legend(bbox_to_anchor=(0.5, 0.5),
              loc='center',
              fontsize=12,
              ncol=min(3, n_classes),
              title='Call Types')
    legend.get_title().set_fontweight('bold')
    
    plt.savefig('legend_tsne.png', bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()

def main():
    # Parameters
    TEST_DATA_DIR = "./data/test"  # Directory with subdirectories for each call type
    MODEL_PATH = "model.pth"  # Path to trained model
    BATCH_SIZE = 128
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Class names (call types)
    # CLASS_NAMES = ['p', 'e', 'l', 'c', 'w', 's', 'a', 'r', 'k', 'g', 'y', 'n', 'h', 'm', 'v', 'o', 'z']
    
    # Define transforms
    transform = transforms.Compose([transforms.Resize((160, 160)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5], std=[0.5])])
    
    # Create dataset and dataloader
    dataset = TestDataset(TEST_DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Determine number of classes from dataset
    num_classes = len(dataset.label_to_id)
    print(f"Detected {num_classes} call types: {list(dataset.label_to_id.keys())}")
    
    # Load model
    model = InceptionResnetV2(device=DEVICE, classify=True, num_classes=num_classes)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
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
    
    # Get embeddings
    print("Extracting embeddings...")
    embeddings, labels, paths = get_embeddings(model, dataloader, DEVICE)

    # Get most confident images (optional, commented out)
    """print("Extracting most confident images...")
    most_confident_images = get_best_images(model, dataloader, DEVICE)
    for label, info in most_confident_images.items():
        call_type = [k for k, v in dataset.label_to_id.items() if v == label][0]
        print(f"{call_type} (Class {label}): {info['path']} (Confidence: {info['confidence']:.4f})")"""
    
    # Plot t-SNE
    print("Creating t-SNE visualization...")
    plot_tsne(embeddings, labels, 'embeddings_tsne.png', dataset.label_to_id)
    
    print("Done! Visualization saved as 'embeddings_tsne.png'")
    
    # Print some statistics
    unique_labels = np.unique(labels)
    print(f"\nNumber of unique call types: {len(unique_labels)}")
    print(f"Total number of images: {len(labels)}")
    print("\nImages per call type:")
    for label in unique_labels:
        call_type = [k for k, v in dataset.label_to_id.items() if v == label][0]
        count = np.sum(labels == label)
        print(f"{call_type}: {count} images")

if __name__ == "__main__":
    main()