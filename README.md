# Crop Disease Prediction AI
## Overview
This project focuses on developing an AI-driven system to detect and classify crop diseases from leaf images, specifically targeting maize, tomato, and onion. Utilizing a fine-tuned ResNet50 deep learning model, the system not only identifies diseases but also provides tailored treatment and prevention recommendations, aiming to assist farmers and agricultural professionals in early disease detection and management.

## Business Understanding
Agriculture is a cornerstone of economies worldwide, yet crop diseases significantly impact yield and farmer livelihoods. Early detection and management of these diseases are crucial. This project aims to leverage AI to assist farmers and agricultural professionals in identifying crop diseases promptly, thereby reducing losses and improving food security.

## Objectives
- Disease Detection: Develop an AI model capable of accurately identifying diseases in maize, tomato, and onion crops through leaf imagery.​
- User Accessibility: Create an intuitive interface that allows users to upload images and receive immediate diagnoses and recommendations.​
- Scalability: Ensure the system can be expanded to include additional crops and diseases in the future.​
- Integration: Facilitate integration with existing agricultural platforms and tools to maximize reach and utility.

 ### Table of Contents
- Features
- Dataset
- Installation
- Usage
- Model Performance
- Visualizations
- Recommendation Engine
- Model Architecture
- Conclusion
- Recommendations
- License
- Contributing
- Contact

  ### Features
- Multi-Crop Support: Detects diseases across maize, tomato, and onion crops.
- Deep Learning Model: Employs a fine-tuned ResNet50 architecture for accurate classification.
- Data Augmentation: Enhances model robustness through extensive image augmentation techniques.
- Recommendation Engine: Offers specific treatment and prevention advice based on the identified disease.
- User-Friendly Interface: Designed for easy integration into web or mobile applications for real-time predictions.

### Dataset
The model is trained on the TOM2024 Category B dataset, which comprises thousands of labeled images representing various diseases affecting maize, tomato, and onion crops.

### Model Performance
Validation Accuracy: Achieved up to 85% accuracy on the validation set after fine-tuning.
Confusion Matrix: Detailed confusion matrices are available in the results/ directory, showcasing the model's performance across different classes.

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
Base Model: ResNet50 pre-trained on ImageNet.
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

###  Contact
For questions or suggestions:
Email: your.email@example.com
Linkedln:
