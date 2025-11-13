import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import torchvision.datasets as datasets
from dataloader import create_dataloader
from inceptionResnetV1 import InceptionResnetV2

def load_model(checkpoint_path, model, device):
    """
    Carrega o modelo treinado a partir do checkpoint
    
    Args:
        checkpoint_path: Caminho para o arquivo do checkpoint
        model: Instância do modelo
        device: Dispositivo onde o modelo será carregado
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model

@torch.no_grad()
def get_predictions(model, test_loader, device):
    """
    Obtém as predições do modelo para todo o conjunto de teste
    
    Args:
        model: Modelo treinado
        test_loader: DataLoader do conjunto de teste
        device: Dispositivo onde o modelo está
    
    Returns:
        true_labels: Lista com os rótulos verdadeiros
        pred_labels: Lista com as predições do modelo
    """
    model.eval()
    true_labels = []
    pred_labels = []
    
    for inputs, labels in tqdm(test_loader, desc='Gerando predições'):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(predictions.cpu().numpy())
    
    return true_labels, pred_labels

def plot_confusion_matrix(true_labels, pred_labels, class_names):
    """
    Plota a matriz de confusão
    
    Args:
        true_labels: Rótulos verdadeiros
        pred_labels: Predições do modelo
        class_names: Lista com os nomes das classes
    """
    # Calcula a matriz de confusão
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Normaliza a matriz
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Configuração do plot
    plt.figure(figsize=(15, 10))
    sns.heatmap(cm_normalized, 
                annot=True, 
                fmt='.2%', 
                cmap='viridis',
                xticklabels=class_names,
                yticklabels=class_names,
                annot_kws={"fontweight": "bold"})
    
    # Rotaciona os labels para melhor visualização
    plt.xticks(rotation=45, ha='right', fontweight='bold')
    plt.yticks(rotation=0, fontweight='bold')
    
    plt.title('Normalized Confusion Matrix', fontweight='bold')
    plt.xlabel('Predicted class', fontweight='bold')
    plt.ylabel("True class", fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Configurações
    BATCH_SIZE = 128
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Diretórios
    TEST_DATA_DIR = "./data/test"
    CHECKPOINT_PATH = "./model.pth"
    
    # Obtém os nomes das classes diretamente do diretório de teste
    dataset = datasets.ImageFolder(root=TEST_DATA_DIR)
    CLASS_NAMES = dataset.classes
    num_classes = len(CLASS_NAMES)
    
    print(f"Classes encontradas: {CLASS_NAMES}")
    print(f"Número total de classes: {num_classes}")
    
    # Cria o dataloader de teste
    _, test_loader = create_dataloader(
        TRAIN_DATA_DIR=TEST_DATA_DIR,  # não precisamos do train_loader
        TEST_DATA_DIR=TEST_DATA_DIR,
        batch_size=BATCH_SIZE
    )
    
    # Cria e carrega o modelo
    model = InceptionResnetV2(device=DEVICE, classify=True, num_classes=num_classes)
    model = load_model(CHECKPOINT_PATH, model, DEVICE)
    model = model.to(DEVICE)
    
    # Obtém as predições
    true_labels, pred_labels = get_predictions(model, test_loader, DEVICE)
    
    # Plota a matriz de confusão
    plot_confusion_matrix(true_labels, pred_labels, CLASS_NAMES)
    
    # Calcula e exibe métricas por classe
    cm = confusion_matrix(true_labels, pred_labels)
    print("\nMétricas por classe:")
    for i, class_name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{class_name}:")
        print(f"Precisão: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Amostras totais: {np.sum(cm[i, :])}")

if __name__ == "__main__":
    # Define seeds para reprodutibilidade
    seed = 43
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    main()