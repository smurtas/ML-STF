# Image Similarity Retrieval — Machine Learning Competition

## Overview

This project was developed for a machine learning competition focused on **image similarity retrieval**. Given a **query image**, the task was to retrieve the top-k most visually similar images from a separate **gallery** set. This problem is relevant to applications like content-based image search, recommendation engines, and face recognition.

The competition required balancing **accuracy**, **speed**, and **resource constraints**, and participants were only allowed to use the provided training/test sets.

---

## 🗂️ Dataset Structure

├── train/ # Labeled images (for fine-tuning) |── test/


---

## 🧩 Task Pipeline

1. **Feature Extraction**  
   Using pretrained or fine-tuned models to extract embeddings from both query and gallery images.

2. **Normalization & Similarity Computation**  
   Features are normalized and cosine similarity is used to compute closeness between vectors.

3. **Top-K Retrieval**  
   For each query, retrieve the top-K gallery images with the highest similarity.

---

## 📚 Models Used

### ✅ Deployed in Competition

| Model                | Description                                                                 | Accuracy |
|----------------------|-----------------------------------------------------------------------------|----------|
| **CLIP (Zero-Shot)** | Large-scale vision-language pretrained model. No fine-tuning.               | 345.57   |
| **Siamese Networks** | Fine-tuned contrastive model for visual similarity.                         | 46.52    |
| **HuggingFace ViT**  | Transformer-based vision model using `ViT` embeddings.                      | 43.37    |
| **ResNet18**         | Lightweight CNN used as feature extractor.                                  | 30.72    |
| **ResNet50 (Pretrained)** | Deeper CNN used without final classification head.                     | 30.58    |
| **FAISS**            | Indexing library for efficient similarity search.                           | 22.75    |

> Note: Accuracy is based on the competition’s top-k metric. Some promising models were not deployed in time due to training overhead.

### 🚫 Not Deployed in Time (ND)

- **ResNet50 (Fine-Tuned)** – retrained with task-specific labels.
- **CLIP (Fine-Tuned)** – fine-tuned vision-language alignment.
- **Triplet Network** – trained using anchor-positive-negative triplets.
- **ArcFace** – angular margin-based classifier.
- **SimCLR** – self-supervised contrastive learning.
- **Timm Models** – baseline embeddings via `timm` pretrained models.

---

## 🔬 Key Insights

- **CLIP (Zero-Shot)** delivered the best performance without any additional training, showcasing the power of large-scale multimodal pretraining.
- **Siamese Networks** and **ViT models** offered a strong trade-off between training time and accuracy.
- **Triplet, ArcFace, and SimCLR** models, while theoretically robust, could not be trained and deployed within the strict 2-hour competition window.

---

## 📈 Evaluation

The models were evaluated using **top-k accuracy**, defined as:

> The proportion of query images where the correct match appears in the top-K retrieved results.

Visual inspection was also used to qualitatively assess the retrieval consistency.

---

## 📎 Technologies Used

- **PyTorch** — Model training and inference
- **Torchvision / Timm** — CNN and ViT backbones
- **Transformers (HuggingFace)** — For ViT and CLIP models
- **FAISS** — Efficient similarity indexing
- **Matplotlib / PIL** — Image visualization

---

## 👥 Contributors

| Name             | Models Implemented                                          | Report | Repository |
|------------------|-------------------------------------------------------------|--------|------------|
| Tommaso Grotto   | CLIP, HuggingFace ViT, ResNet50, Triplet, ArcFace, SimCLR   | 70%    | 30%        |
| Stefano Murtas   | ResNet18, FAISS, Timm, Siamese Networks                     | 30%    | 70%        |

---

## 📝 References

- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- Radford et al., *Learning Transferable Visual Models from Natural Language Supervision*, ICML 2021.
- Schroff et al., *FaceNet: A Unified Embedding for Face Recognition*, CVPR 2015.
- Deng et al., *ArcFace: Additive Angular Margin Loss*, CVPR 2019.
- Chen et al., *SimCLR: A Simple Framework for Contrastive Learning*, ICML 2020.
- Koch et al., *Siamese Networks for One-shot Learning*, ICML Workshop 2015.
- Johnson et al., *FAISS: Billion-scale Similarity Search*, IEEE Big Data 2019.

---

## 💾 Repository

All code and submission files are available at:  
👉 [GitHub Repository](https://github.com/smurtas/ML-STF.git)


