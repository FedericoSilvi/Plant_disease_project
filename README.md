# 🍅 Tomato Leaf Disease Classification

Progetto di classificazione multiclasse delle malattie della pianta di pomodoro da immagini delle foglie.

## 👨‍👩‍👧‍👦 Autori
- Marco Caruso  
- Silvia Giannetti  
- Giacomo Alberto Napolitano   
- Federico Silvi   


---

## 🎯 Obiettivo
Realizzare un classificatore multiclasse per identificare malattie delle foglie di pomodoro tramite immagini, sfruttando tecniche di Image Processing, Feature Extraction e Machine Learning.

---

## 📁 Dataset
- **Fonte**: [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Classi selezionate**: 7 malattie + 1 classe di foglia sana
- **Formato**: immagini a colori 256×256
- **Numero immagini**: 160 per classe

---

## 🖼️ Image Processing
Segmentazione foglie tramite tecniche graph-based:
- Approccio con **GraphCut** con semi di foreground/background generati automaticamente
- Post-processing per migliorare la qualità della maschera
- Alcune classi sottoposte a trattamento aggiuntivo per rimuovere l’ombra

---

## 📊 Feature Extraction
Feature derivate da immagini segmentate:
- **LBP** (Local Binary Patterns)
- **GLCM** (Gray-Level Co-occurrence Matrix)
- **Gabor filter** (manuale e tramite **AlexNet**)

Dataset generati con:
- Immagini originali e filtrate (Non-Local Means)
- Combinazioni multiple di feature (D0–D4)

---

## 🤖 Machine Learning
### Supervised
- Modelli: KNN, Random Forest, SVM, XGBoost
- Pipeline: Normalizzazione + PCA/KBest + Grid Search con CV
- **Miglior modello**: SVM/XGBoost con accuracy fino al **97.42%** con majority voting

### Unsupervised
- Modelli: K-Means, GMM
- Metrica: Silhouette Score ≈ **0.47**

---

## 🔬 Risultati principali

| Segmentazione       | Metodo        | Accuracy   |
|---------------------|---------------|------------|
| Foglia segmentata   | Ensemble      | **97.42%** |
| No segmentazione    | Ensemble      | 96.72%     |
| Solo sfondo         | Ensemble      | 93.05%     |

---

## 📌 Conclusioni
- La segmentazione è cruciale per evitare che le feature siano influenzate dallo sfondo.
- I metodi supervised superano nettamente quelli unsupervised.
- Il modello finale ensemble raggiunge prestazioni comparabili alla letteratura di settore.

---

## 🔗 Riferimenti
- [PlantVillage Dataset su Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- [PubMed Study Reference](https://pmc.ncbi.nlm.nih.gov/articles/PMC11790621/)
