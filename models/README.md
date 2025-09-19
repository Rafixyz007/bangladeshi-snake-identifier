# 🐍 Bangladeshi Snake Recognizer – Trained Models

This repository provides the **trained machine learning models** for recognizing 17 common snake species in Bangladesh. The models were trained using **FastAI** on **Google Colab with GPU**.

---

## 📂 Available Models

1. **ResNet34 Model**  
   - Google Drive Link: [Download](https://drive.google.com/file/d/1qhSFOZ2zT1GWu9YSY_K87iUnxr9ZmjxZ/view?usp=sharing)  
   - Description: Trained with ResNet34 architecture, moderate accuracy (~65%).  

2. **EfficientNet-B0 Model**  
   - Google Drive Link: [Download](https://drive.google.com/file/d/1iqikuHYgqXE4t6JcgntpIl7W1UqWzIBf/view?usp=sharing)  
   - Description: Trained with EfficientNet-B0, better accuracy (~76%).  

3. **ResNet50 Model (Final Deployed Model)**  
   - Google Drive Link: [Download](https://drive.google.com/file/d/17h2LQUohd4LCsYnwmXtQeRqWi5vg_kIL/view?usp=sharing)  
   - Description: Trained with ResNet50, highest accuracy (~87–88%), used in the Hugging Face deployment.

---

## 🌐 Deployment

- The **ResNet50 model** is deployed on **Hugging Face Spaces**:  
  [Bangladeshi Snake Recognizer](https://huggingface.co/spaces/Rafix007/Bangladeshi-Snake-Recognizer)  

Users can upload snake images and get predictions in **English and Bangla**.  

---

## ⚡ Notes

- All models are exported using FastAI’s `.pkl` format.  
- Use `load_learner` in FastAI to load and predict images locally.
