Epoch 12/100, Train Loss: 0.1866, Val Loss: 0.1866, Steering MAE: 0.0988, Accel MAE: 0.2585 explique moi tout ces termes en francais

GitHub Copilot
Explication des Termes de Formation de l'IA en Français
Epoch 12/100
Époque 12/100 : Une époque représente un cycle complet d'entraînement où le modèle a vu l'ensemble des données une fois. Ici, nous sommes à la 12ème époque sur un total prévu de 100 épocques d'entraînement.

Train Loss: 0.1866
Perte d'Entraînement: 0.1866 : C'est la mesure d'erreur moyenne sur les données d'entraînement. Elle indique l'écart entre les prédictions du modèle et les valeurs réelles pendant la phase d'apprentissage. Plus cette valeur est basse, mieux le modèle reproduit les données d'entraînement.

Val Loss: 0.1866
Perte de Validation: 0.1866 : C'est la mesure d'erreur sur les données de validation (données que le modèle n'a pas vues pendant l'entraînement). Elle permet de vérifier si le modèle généralise bien. Dans ce cas, la perte de validation est identique à la perte d'entraînement, ce qui suggère que le modèle ne souffre pas de surapprentissage.

Steering MAE: 0.0988
EAM Direction: 0.0988 : L'Erreur Absolue Moyenne pour les prédictions d'angle de direction. Cette valeur représente l'écart moyen (en valeur absolue) entre l'angle de direction prédit par le modèle et l'angle réel. Une valeur de 0.0988 signifie que le modèle se trompe en moyenne d'environ 0,1 unité pour les prédictions de direction.

Accel MAE: 0.2585
EAM Accélération: 0.2585 : L'Erreur Absolue Moyenne pour les prédictions d'accélération. Cette valeur plus élevée (0,26) indique que le modèle est moins précis pour prédire l'accélération que pour prédire la direction.

Ces métriques permettent de suivre l'évolution de l'apprentissage et d'évaluer les performances du modèle au fil des époques d'entraînement.