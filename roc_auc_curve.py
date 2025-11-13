import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.datasets as datasets
from dataloader import create_dataloader
from inceptionResnetV1 import InceptionResnetV1

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
def get_predictions_proba(model, test_loader, device):
    """
    Obtém as probabilidades preditas pelo modelo para todo o conjunto de teste
    
    Args:
        model: Modelo treinado
        test_loader: DataLoader do conjunto de teste
        device: Dispositivo onde o modelo está
    
    Returns:
        true_labels: Array com os rótulos verdadeiros
        pred_probs: Array com as probabilidades preditas para cada classe
    """
    model.eval()
    all_labels = []
    all_probs = []
    
    for inputs, labels in tqdm(test_loader, desc='Gerando predições'):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Obtém as probabilidades usando softmax
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_probs)

def plot_roc_curves(true_labels, pred_probs, class_names):
    """
    Plota as curvas ROC para cada classe e calcula as AUCs
    
    Args:
        true_labels: Array com os rótulos verdadeiros
        pred_probs: Array com as probabilidades preditas
        class_names: Lista com os nomes das classes
    """
    n_classes = len(class_names)
    
    # Prepara os rótulos no formato one-hot
    true_labels_onehot = np.eye(n_classes)[true_labels]
    
    # Configura o plot
    plt.figure(figsize=(15, 10))
    
    # Cores para as curvas
    colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
    
    # Lista para armazenar as AUCs
    aucs = []
    
    # Plota a curva ROC para cada classe
    for i, (color, class_name) in enumerate(zip(colors, class_names)):
        fpr, tpr, _ = roc_curve(true_labels_onehot[:, i], pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        plt.plot(fpr, tpr, color=color, lw=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    # Calcula e plota a curva ROC média (macro)
    all_fpr = np.unique(np.concatenate([roc_curve(true_labels_onehot[:, i], 
                        pred_probs[:, i])[0] for i in range(n_classes)]))
    
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(true_labels_onehot[:, i], pred_probs[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    
    mean_tpr /= n_classes
    macro_roc_auc = auc(all_fpr, mean_tpr)
    
    plt.plot(all_fpr, mean_tpr, color='red', linestyle=':', lw=4,
            label=f'Média macro-ROC (AUC = {macro_roc_auc:.3f})')
    
    # Configura o gráfico
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curvas ROC por Classe')
    plt.legend(loc="lower right", bbox_to_anchor=(1.7, 0))
    
    # Salva o gráfico
    plt.tight_layout()
    plt.savefig('roc_curves.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    return aucs, macro_roc_auc

def main():
    # Configurações
    BATCH_SIZE = 128
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Diretórios
    TEST_DATA_DIR = "./data/test"
    CHECKPOINT_PATH = "./model.pth"
    
    # Obtém os nomes das classes do diretório de teste
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
    model = InceptionResnetV1(device=DEVICE, classify=True, num_classes=num_classes)
    model = load_model(CHECKPOINT_PATH, model, DEVICE)
    model = model.to(DEVICE)
    
    # Obtém as predições com probabilidades
    true_labels, pred_probs = get_predictions_proba(model, test_loader, DEVICE)
    
    # Plota as curvas ROC e obtém as AUCs
    aucs, macro_auc = plot_roc_curves(true_labels, pred_probs, CLASS_NAMES)
    
    # Imprime as métricas
    print("\nResultados da análise ROC-AUC:")
    print(f"AUC Macro média: {macro_auc:.4f}")
    print("\nAUC por classe:")
    for class_name, auc_score in zip(CLASS_NAMES, aucs):
        print(f"{class_name}: {auc_score:.4f}")

if __name__ == "__main__":
    # Define seeds para reprodutibilidade
    seed = 43
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    main()