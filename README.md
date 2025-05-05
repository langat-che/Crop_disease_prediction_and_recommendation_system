# Crop Disease Prediction AI
## Overview
This project focuses on developing an AI-driven system to detect and classify crop diseases from leaf images, specifically targeting maize, tomato, and onion. Utilizing a fine-tuned InceptionV3 and Custom CNN deep learning model, the system not only identifies diseases but also provides tailored treatment and prevention recommendations, aiming to assist farmers and agricultural professionals in early disease detection and management.

---
## Business Understanding
Agriculture is a cornerstone of economies worldwide, yet crop diseases significantly impact yield and farmer livelihoods. Early detection and management of these diseases are crucial. This project aims to leverage AI to assist farmers and agricultural professionals in identifying crop diseases promptly, thereby reducing losses and improving food security.

---
## Objectives
Main Objective
- To develop an AI-powered system that accurately detects crop diseases and pests from images.

Specific Objectives
- To develop an AI-based model that accurately identifies pests or diseases in maize, tomato, and onion using leaf images.
- To integrate a recommendation engine that provides specific treatment and prevention strategies based on the identified disease.
- To design a user-friendly interface that allows farmers and agricultural professionals to easily upload images and receive immediate feedback.
-  To deploy the system in a scalable manner, facilitating integration with mobile and web applications for broader accessibility.
-  
---
  ### Features
- Multi-Crop Support: Detects diseases across maize, tomato, and onion crops.
- Deep Learning Model: Employs a fine-tuned InceptionV3 and Custom CNN model architecture for accurate classification.
- Data Augmentation: Enhances model robustness through extensive image augmentation techniques.
- Recommendation Engine: Offers specific treatment and prevention advice based on the identified disease.
- User-Friendly Interface: Designed for easy integration into web or mobile applications for real-time predictions.

---
### Dataset
Source: Mendeley TOM2024 Dataset
- 25,844 raw images
- 12,227 labeled images
- 3 Crops: Tomato, Onion, Maize
- 30 disease categories
- The dataset supports sustainable agriculture and early detection of plant health issues.

---
### 🧪 Modeling
To classify crop diseases from images, we experimented with two primary convolutional models:
| Model           | Description                                                              |
| --------------- | ------------------------------------------------------------------------ |
| **Custom CNN**  | A manually designed CNN with multiple convolutional and pooling layers   |
| **InceptionV3** | A pre-trained, deep architecture known for handling multi-scale features |

---
### 🏗️ Steps Involved
1. Data Preprocessing
- Image resizing (typically to 299x299 for InceptionV3)
- Normalization and augmentation (horizontal flip, rotation, zoom)
2. Model Training
- Custom CNN trained from scratch
- InceptionV3 used transfer learning with ImageNet weights
- Fine-tuning applied to top layers of InceptionV3
- Early stopping and learning rate decay applied
3. Evaluation
- Dataset split: 80% training, 20% validation
- Evaluated using Accuracy, Precision, Recall, and F1 Score
- Confusion matrix plotted for class-wise performance

----
### 🏆 Performance Summary
| Model       | Accuracy  | F1 Score   | Remarks                               |
| ----------- | --------- | ---------- | ------------------------------------- |
| Custom CNN  | \~85%     | \~0.83     | Lightweight, but less accurate        |
| InceptionV3 | **\~91%** | **\~0.89** | Best performer, strong generalization |

----
### Visualizations
The notebooks/ directory contains Jupyter notebooks with:
- Training and validation accuracy/loss plots.
- Confusion matrices.
- Sample predictions with corresponding recommendations.

### Recommendation Engine
Upon predicting the disease class, the system provides:
- Treatment Suggestions: Recommended pesticides or fungicides.
- Preventive Measures: Best practices to prevent disease recurrence.
- Agronomic Tips: Additional advice tailored to the specific crop and disease.

### Model Architecture
Base Model: InceptionV3 pre-trained on ImageNet.
Fine-Tuning: Unfrozen top layers for domain-specific feature learning.
Additional Layers:
- Global Average Pooling
- Dropout (0.5)
- Dense layer with softmax activation for classification.

## Conclusion
This AI-powered system demonstrates the potential of deep learning in agriculture, offering a practical solution for early detection and management of crop diseases. By integrating advanced image classification with actionable recommendations, it serves as a valuable tool for farmers and agricultural stakeholders.

## Recommendations
- Data Expansion: Incorporate more diverse datasets to improve model generalization.
- Mobile Integration: Develop a mobile application for on-field disease detection.
- Real-Time Updates: Implement a system for updating the model with new data to adapt to emerging diseases.

---
## 📂 Repository Contents

## 🤝 Acknowledgements
Thanks to all contributors and stakeholders for their input and support in making this project successful.

---
###  Contact
For questions or suggestions:

Email:
- langatchebetbev@gmail.com
- oyakapeliamase@gmail.com
- felixmwendwa014@gmail.com
- kelvinnyawira2022@gmail.com
- adnanahmedmohamud1@gmail.com
  
Linkedln:
- https://www.linkedin.com/in/beverlyne-l-7926041a2
- https://www.linkedin.com/in/amase-oyakapeli-7848a8343/
- https://www.linkedin.com/in/felix-mwendwa-3b78a2238/
- https://www.linkedin.com/in/kelvin-nyawira
- https://www.linkedin.com/in/adnan-mohamud-4567942b7
