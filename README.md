# Projet de Détection des Caries Dentaires (Dental Caries Detection Project)
## ESP32-CAM avec TensorFlow Lite

### État Actuel du Projet
Le projet est actuellement en phase de développement avec des problèmes spécifiques à résoudre, notamment:
- Erreur DEQUANTIZE lors de l'initialisation du modèle TensorFlow
- Gestion de la mémoire sur ESP32-CAM
- Optimisation du modèle pour les contraintes matérielles

### Configuration Matérielle
- ESP32-CAM AI-Thinker
- Module caméra OV2640
- Adaptateur FTDI/USB-TTL pour la programmation

### Dépendances Logicielles
1. **Bibliothèques Arduino**:
   - ESP32 board package
   - esp_camera.h
   - Arduino.h

2. **TensorFlow Lite**:
   ```cpp
   #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
   #include "tensorflow/lite/micro/micro_interpreter.h"
   #include "tensorflow/lite/micro/micro_log.h"
   #include "tensorflow/lite/micro/system_setup.h"
   #include "tensorflow/lite/schema/schema_generated.h"
   ```

### Structure du Projet
```
/aba (Repo)
├── model_esp32.h       # Modèle converti en header file
└── main.cpp           # Code principal ESP32
```

### Configuration de l'Image
- Résolution: 96x96 pixels
- Format: RGB565
- Canaux: 3 (RGB)
- Taille du Tensor Arena: 100KB

### Points d'Attention pour les Ingénieurs
1. **Problèmes Actuels**:
   - Erreur: `Didn't find op for builtin opcode 'DEQUANTIZE'`
   - Contraintes de mémoire avec ESP32-CAM
   - Possible problème de compatibilité du modèle

2. **Aide Requise**:
   - Optimisation du modèle TensorFlow Lite
   - Implémentation correcte de l'opérateur DEQUANTIZE
   - Gestion efficace de la mémoire
   - Test et validation du système de détection

### Configuration Arduino IDE
1. **Paramètres de la Carte**:
   - Board: "AI Thinker ESP32-CAM"
   - Partition Scheme: "Huge APP (3MB No OTA/1MB SPIFFS)"
   - Upload Speed: 115200

2. **Connexions pour Programmation**:
   ```
   ESP32-CAM      FTDI
   GND       -->   GND
   5V        -->   5V
   U0R (RX)  -->   TX
   U0T (TX)  -->   RX
   ```

### Commandes Disponibles (via Moniteur Série)
- 'd' : Lancer la détection
- 'f' : Contrôler le flash
- 'm' : Afficher les statistiques mémoire

### Plan d'Action Proposé
1. **Court Terme**:
   - Corriger l'erreur DEQUANTIZE
   - Optimiser la gestion de la mémoire
   - Valider le fonctionnement de base

2. **Moyen Terme**:
   - Améliorer la précision du modèle
   - Optimiser les performances
   - Ajouter des fonctionnalités de débogage

3. **Long Terme**:
   - Interface utilisateur
   - Stockage des résultats
   - Documentation complète

### Demande d'Aide Spécifique
Nous recherchons de l'aide pour:
1. Résolution du problème DEQUANTIZE dans TensorFlow Lite
2. Optimisation du modèle pour ESP32-CAM
3. Amélioration de la gestion mémoire
4. Test et validation du système

### Contact
- GitHub: [anosswb/aba](https://github.com/anosswb/aba)

### Notes Importantes
- Le modèle actuel nécessite une révision pour la compatibilité TensorFlow Lite
- La gestion de la mémoire est critique pour la stabilité
- Le système nécessite une validation approfondie

### Prochaines Étapes Suggérées
1. Réviser l'implémentation du modèle TensorFlow
2. Ajouter support pour DEQUANTIZE
3. Optimiser l'utilisation de la mémoire
4. Améliorer la robustesse du système
