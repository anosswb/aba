# Système de Détection des Caries Dentaires
## Documentation Technique

### Aperçu du Projet
Ce système utilise une caméra ESP32-CAM et TensorFlow Lite pour détecter les caries dentaires en temps réel. Le système capture des images dentaires et les analyse à l'aide d'un modèle de réseau neuronal préentraîné.

### Configuration Matérielle Requise
- Module ESP32-CAM AI-Thinker
- Adaptateur USB-TTL pour la programmation
- Alimentation 5V stable
- LED flash intégrée pour l'éclairage

### Spécifications Techniques
- Résolution d'image : 96x96 pixels
- Format de couleur : RGB565
- Taille mémoire tensor : 150Ko
- Seuil de détection : 0.5

### Installation et Configuration
1. **Configuration de l'Arduino IDE**
   - Sélectionner la carte : "AI Thinker ESP32-CAM"
   - Schéma de partition : "Huge APP (3MB No OTA/1MB SPIFFS)"
   - Vitesse de téléversement : 115200

2. **Connexions Matérielles**
### Utilisation
Le système répond aux commandes série suivantes :
- 'd' : Démarrer une détection
- 'f' : Activer/désactiver le flash
- 'm' : Afficher les statistiques mémoire

### Résolution des Problèmes Courants
1. **Erreur de Démarrage**
- Vérifier l'alimentation 5V stable
- Réinitialiser le module
- Vérifier les connexions série

2. **Erreurs Mémoire**
- Le système nécessite minimum 150Ko de mémoire libre
- Surveillance via la commande 'm'

3. **Problèmes de Capture**
- Vérifier l'éclairage
- Assurer une distance appropriée
- Nettoyer la lentille de la caméra

### Améliorations Apportées
1. Optimisation de la gestion m
