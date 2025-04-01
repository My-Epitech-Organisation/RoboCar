import torch
import ast
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# Fonction pour choisir un fichier d'entraînement
def choisir_fichier_entraînement():
    fichiers = [f for f in os.listdir('./data') if f.endswith('.csv')]
    print("Choisissez un fichier d'entraînement parmi les suivants :")
    for i, fichier in enumerate(fichiers, start=1):
        print(f"{i}. {fichier}")
    
    choix = int(input("Entrez le numéro du fichier : ")) - 1
    fichier_choisi = fichiers[choix]
    return os.path.join('./data', fichier_choisi)

# Charger et préparer les données
fichier_entraînement = choisir_fichier_entraînement()
data = pd.read_csv(fichier_entraînement)

data['raycasts'] = data['raycasts'].apply(ast.literal_eval)

for i in range(10):
    data['raycast_{}'.format(i)] = data['raycasts'].apply(lambda x: x[i] if isinstance(x, list) else None)

features = data[['steering_input', 'acceleration_input'] + [f'raycast_{i}' for i in range(10)] + ['position_x', 'position_y', 'position_z']].values
target = data[['speed', 'steering']].values

# Normaliser les données (très important pour les réseaux neuronaux)
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Convertir en tensors
features_tensor = torch.tensor(features, dtype=torch.float32)
target_tensor = torch.tensor(target, dtype=torch.float32)

# Diviser en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(features_tensor, target_tensor, test_size=0.2, random_state=42)

# Définir le modèle de régression multivariée
class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Couche cachée avec 64 neurones
        self.fc2 = nn.Linear(64, 32)          # Couche cachée avec 32 neurones
        self.fc3 = nn.Linear(32, output_size) # Couche de sortie avec 2 neurones (pour speed et steering)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Fonction d'activation ReLU
        x = torch.relu(self.fc2(x))  # Fonction d'activation ReLU
        x = self.fc3(x)              # Sortie
        return x

# Définir le modèle, la fonction de perte et l'optimiseur
input_size = X_train.shape[1]  # Nombre de features (entrées)
output_size = y_train.shape[1] # Nombre de cibles (2: speed et steering)

model = RegressionModel(input_size, output_size)
criterion = nn.MSELoss()  # Erreur quadratique moyenne
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraîner le modèle
epochs = 100
for epoch in range(epochs):
    # Entraînement
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Afficher la perte à chaque 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Sauvegarder le modèle après l'entraînement
torch.save(model.state_dict(), 'model_checkpoint.pth')
print("Modèle sauvegardé sous 'model_checkpoint.pth'")

# Évaluer le modèle
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    test_loss = criterion(y_pred, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

    # Comparer les prédictions avec les valeurs réelles
    print("Prédictions :\n", y_pred[:5])
    print("Réel :\n", y_test[:5])

# Pour charger un modèle sauvegardé, utilise :
# model.load_state_dict(torch.load('model_checkpoint.pth'))
# model.eval()
