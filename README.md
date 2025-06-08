# 🍅 Tomato Leaf Disease Classification

Multiclass classification project for tomato plant leaf diseases using leaf images.

## 👨‍👩‍👧‍👦 Authors
- Marco Caruso   
- Silvia Giannetti   
- Giacomo Alberto Napolitano 
- Federico Silvi   


---

## 🎯 Objective
The goal is to develop a multiclass classifier to identify tomato leaf diseases from images using Image Processing, Feature Extraction, and Machine Learning techniques.

---

## 📁 Dataset
- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Selected classes**: 7 disease classes + 1 healthy class
- **Format**: Color images, 256×256 resolution
- **Images per class**: 160 (balanced dataset)

---

## 🖼️ Image Processing
Leaf segmentation was performed using a graph-based approach:
- Initial attempt with **GrabCut** was partially effective
- Final solution: **GraphCut** using auto-generated foreground/background seeds
- Post-processing to clean up artifacts
- Shadow removal for specific classes (e.g., Spider Mites, Yellow Curl Virus)

---

## 📊 Feature Extraction
Extracted features from segmented images:
- **LBP** (Local Binary Patterns)
- **GLCM** (Gray-Level Co-occurrence Matrix)
- **Gabor Filters** (manual + via **AlexNet**)

Generated multiple datasets (D0–D4) using:
- Raw and denoised images (Non-Local Means)
- Combined and incremental feature sets

---

## 🤖 Machine Learning
### Supervised
- Models: KNN, Random Forest, SVM, XGBoost
- Pipeline: Normalization + PCA/KBest + Grid Search with cross-validation
- **Best performance**: SVM/XGBoost with up to **97.42% accuracy** using majority voting ensemble

### Unsupervised
- Models: K-Means, GMM
- Metric: Silhouette Score ≈ **0.47**

---

## 🔬 Key Results

| Segmentation Type    | Method    | Accuracy   |
|----------------------|-----------|------------|
| Full leaf segmented  | Ensemble  | **97.42%** |
| No segmentation      | Ensemble  | 96.72%     |
| Background segmented | Ensemble  | 93.05%     |

---

## 📌 Conclusions
- Segmentation is essential to prevent models from learning spurious correlations from the background.
- Supervised methods significantly outperform unsupervised ones.
- The final ensemble model achieved results comparable to state-of-the-art benchmarks in the literature.

---

## 🔗 References
- [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- [Benchmark Study (PubMed)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11790621/)
