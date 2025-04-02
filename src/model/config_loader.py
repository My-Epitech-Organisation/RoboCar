import os
import yaml

def load_train_config():
    """
    Charge la configuration d'entrainement depuis le fichier YAML.
    
    Returns:
        dict: La configuration chargée
    """
    # Remonte un niveau supplémentaire pour atteindre la racine du projet
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                              'config', 'train_config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config
