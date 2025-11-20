# Road Segmentation with U-Net + ONNX Export for QGIS

This repository contains two Jupyter notebooks demonstrating a complete workflow for training a U-Net model for road segmentation, augmenting the dataset, and exporting the trained model to ONNX format for integration with GIS tools such as **QGIS**.

## 📂 Notebooks Included

### 1. **Road Segmentation Training with U-Net**
**File:** `code-of-augmentation-unet-road-segmentation.ipynb`

This notebook walks through:
- Preparing the dataset for road segmentation  
- Applying data augmentation techniques  
- Building and training a U-Net model using TensorFlow/Keras  
- Visualizing training results and segmentation masks  
- Saving the trained model for later use  

It provides a full deep-learning training pipeline suitable for segmentation tasks on overhead or satellite imagery.

---

### 2. **Exporting Trained Model to ONNX for QGIS**
**File:** `Code to get onnx file to test on qgis.ipynb`

This notebook focuses on:
- Loading the trained U-Net model  
- Converting it into **ONNX format** using `tf2onnx`  
- Preparing the exported model so it can be loaded and tested in **QGIS Deep Learning Tools**  
- Validating the exported ONNX model  

This is useful for deploying machine learning models into GIS workflows for real-world mapping and geospatial analysis.

---

## 🚀 Workflow Overview

1. **Train the segmentation model**  
   - Use augmented satellite/overhead imagery  
   - Train a U-Net for pixel-wise road segmentation  

2. **Export trained model to ONNX**  
   - Convert `.h5` or SavedModel format to `.onnx`  
   - Ensure compatibility with QGIS processing tools  

3. **Use in QGIS**  
   - Load the ONNX model for inference  
   - Run segmentation directly on geospatial data  

---

## 📦 Dependencies

The notebooks rely on the following key libraries:

- Python 3.x  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Matplotlib  
- scikit-learn  
- `tf2onnx`  
- OpenCV (optional)  

Refer to the notebook cells for exact versions used during experimentation.

---

## 🗂️ Repository Structure

