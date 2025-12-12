import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import glob 

class SimpleImageFolder(Dataset):
    """
    Carrega imagens diretamente de uma pasta, sem esperar subpastas de classes.
    """
    def __init__(self, path, transform, limit=np.inf):
        super().__init__()
        
        # Encontra todas as imagens na pasta usando glob
        # Isto lida com .png, .jpg, .jpeg, etc.
        self.image_paths = sorted(glob.glob(os.path.join(path, '*.*[gG|pG|eG]'))) # Padrão para .jpg, .png, .jpeg

        if limit != np.inf:
            self.image_paths = self.image_paths[:min(len(self.image_paths), int(limit))]
        # Se o limite for np.inf, simplesmente usamos a lista completa.
        
        self.limit = limit 
        self.transform = transform
        
        if len(self.image_paths) == 0:
             raise FileNotFoundError(f"Nenhuma imagem encontrada em {path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # Abre a imagem e garante que está em modo RGB
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Erro ao carregar a imagem {img_path}: {e}")
            # Retorna uma imagem de placeholder se falhar
            image = Image.new('RGB', (360, 360), (0, 0, 0)) 

        if self.transform:
            image = self.transform(image)
        
        return image, 0

class DataLoader(torch.utils.data.DataLoader):
    """
    Simplifica a criação a partir de uma pasta de imagens, com transformações padrão já incluídas.
    """
    def __init__(self, path, limit=np.inf, shuffle=True,
                 num_workers=8, batch_size=4, image_size=360, *args, **kwargs):

        # Define a sequência de transformações
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, pad_if_needed=True),
            transforms.ToTensor(), # Converte para [0, 1]
        ])
        
        # Usa a nova classe SimpleImageFolder em vez da ImageFolder antiga
        dataset = SimpleImageFolder(path, transform, limit)
        
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            *args, **kwargs
        )