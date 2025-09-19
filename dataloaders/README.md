# 🐍 Bangladeshi Snake Recognizer – DataLoaders

This repository contains the **FastAI DataLoaders** used for training and inference of the Bangladeshi Snake Recognizer models. Each DataLoader corresponds to a specific trained model and contains the prepared dataset for that model.

---

## 📂 Available DataLoaders

1. **DataLoader 1 (ResNet34 Model)**  
   - Google Drive Link: [Download](https://drive.google.com/file/d/1WvdyWGmjT6erqy_wpx-9J7P0MfE6Zx40/view?usp=sharing)  
   - Description: Prepared DataLoader for the ResNet34 model. Contains preprocessed images and labels used during training.

2. **DataLoader 2 (EfficientNet-B0 Model)**  
   - Google Drive Link: [Download](https://drive.google.com/file/d/1SwUYe8VN_Lzjv0bNnlLnbMwC4L9DwtDo/view?usp=sharing)  
   - Description: Prepared DataLoader for the EfficientNet-B0 model. Optimized for this architecture and training split.

3. **DataLoader 3 (ResNet50 Model – Final Deployed)**  
   - Google Drive Link: [Download](https://drive.google.com/file/d/1KeVUPGioATV9mg4vsf7aYz88VnHC2-8O/view?usp=sharing)  
   - Description: Prepared DataLoader for the ResNet50 model. Used for the final deployment on Hugging Face Spaces.

---

## ⚡ Notes

- Each DataLoader is saved using **FastAI’s `.pkl` format**.  
- Load using `load_data` or `load_learner` depending on your workflow.  
- Useful for replicating training or making predictions locally.
