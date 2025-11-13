import torch
import torch.nn as nn
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
def get_predictions_and_probabilities(model, test_loader, device, num_classes):
    """
    Obtém as predições e probabilidades do modelo para todo o conjunto de teste
    
    Args:
        model: Modelo treinado
        test_loader: DataLoader do conjunto de teste
        device: Dispositivo onde o modelo está
        num_classes: Número de classes
    
    Returns:
        true_labels: Lista com os rótulos verdadeiros
        probabilities: Array com as probabilidades para cada classe
    """
    model.eval()
    softmax = nn.Softmax(dim=1)
    true_labels = []
    all_probabilities = []
    
    for inputs, labels in tqdm(test_loader, desc='Gerando predições e probabilidades'):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        probs = softmax(outputs)
        
        true_labels.extend(labels.cpu().numpy())
        all_probabilities.extend(probs.cpu().numpy())
    
    return np.array(true_labels), np.array(all_probabilities)

def calculate_probability_matrix(true_labels, probabilities, num_classes):
    """
    Calcula a matriz de probabilidades médias
    
    Args:
        true_labels: Array com os rótulos verdadeiros
        probabilities: Array com as probabilidades para cada classe
        num_classes: Número de classes
    
    Returns:
        prob_matrix: Matriz com as probabilidades médias
    """
    prob_matrix = np.zeros((num_classes, num_classes))
    
    for true_class in range(num_classes):
        # Seleciona todas as amostras que pertencem à classe verdadeira atual
        mask = true_labels == true_class
        if not np.any(mask):
            continue
            
        # Calcula a média das probabilidades para cada classe predita
        class_probs = probabilities[mask]
        prob_matrix[true_class,:] = np.mean(class_probs, axis=0)
    
    return prob_matrix

def plot_probability_matrix(prob_matrix, class_names):
    """
    Plota a matriz de probabilidades médias
    
    Args:
        prob_matrix: Matriz com as probabilidades médias
        class_names: Lista com os nomes das classes
    """
    plt.figure(figsize=(15, 10))
    sns.heatmap(prob_matrix, 
                annot=True, 
                fmt='.2%', 
                cmap='viridis',
                xticklabels=class_names,
                yticklabels=class_names,
                annot_kws={"fontweight": "bold"})
    
    plt.xticks(rotation=45, ha='right', fontweight='bold')
    plt.yticks(rotation=0, fontweight='bold')
    
    plt.title('Mean Probability Matrix', fontweight='bold')
    plt.xlabel('Mean predicted class', fontweight='bold')
    plt.ylabel('True class', fontweight='bold')
    plt.tight_layout()
    plt.savefig('probability_matrix.png', bbox_inches='tight', dpi=300)
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
    
    # Obtém as predições e probabilidades
    true_labels, probabilities = get_predictions_and_probabilities(model, test_loader, DEVICE, num_classes)
    
    # Calcula e plota a matriz de probabilidades
    prob_matrix = calculate_probability_matrix(true_labels, probabilities, num_classes)
    plot_probability_matrix(prob_matrix, CLASS_NAMES)
    
    # Imprime as probabilidades médias por classe
    print("\nProbabilidades médias por classe:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"\n{class_name}:")
        print("Quando é a classe verdadeira:")
        for j, pred_class in enumerate(CLASS_NAMES):
            print(f"- Probabilidade média de ser {pred_class}: {prob_matrix[j, i]:.4f}")

if __name__ == "__main__":
    # Define seeds para reprodutibilidade
    seed = 43
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    main()