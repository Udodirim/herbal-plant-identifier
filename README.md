# Herbal-Plant-Identifier
Herbal Plant Identifier using VGG16 CNN - A computer vision model for identifying 37 medicinal plants.

<img width="3356" height="1802" alt="image" src="https://github.com/user-attachments/assets/7489d08a-9ccb-47e3-a929-8faf06c6e299" />

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Plant Classes** | 37 species |
| **Test Accuracy** | 88.06% |
| **Train Accuracy** | 90.51% |
| **Dataset Size** | 3,734 images |
| **Architecture** | VGG16 Transfer Learning |
| **Input Size** | 224×224 pixels |

---

## 🎯 Key Features

✅ **Upload Images** - Drag & drop plant photos for instant identification  
✅ **Camera Capture** - Take photos directly in the app  
✅ **Batch Analysis** - Identify multiple plants at once  
✅ **Confidence Scores** - See prediction confidence with visual gauges  
✅ **Top 3 Predictions** - View alternative predictions ranked by confidence  
✅ **Beautiful UI** - Clean, intuitive Streamlit interface  


---

## 📖 View the Complete ML Pipeline

**For technical details, see the Jupyter notebook:**
```bash
jupyter notebook Herbal_Plant_Identifier.ipynb
```

The notebook includes:
- ✅ Complete data loading pipeline (3,734 images)
- ✅ VGG16 model architecture & training
- ✅ Data visualization (sample images, class distribution)
- ✅ Model evaluation (confusion matrix, classification report)
- ✅ Real-world predictions on test images
- ✅ Proof that model achieves 88% accuracy on unseen data

**Start here to understand the full ML implementation!**

---

## 📁 Project Structure
```
herbal-plant-identifier/
├── streamlit_app.py                    # Interactive web application
├── Herbal_Plant_Identifier.ipynb       # Complete ML pipeline
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
└── herbal_plant_model/
    ├── herbal_plant_model.h5           # Trained model (auto-downloaded)
    ├── model_metadata.json             # Class mappings & metadata
    └── training_history.json           # Training progress per epoch
```

---

## 🌱 Supported Plants (37 Species)

**Medicinal Herbs:**
Aloe vera, Mint, Basil, Neem, Ginger, Garlic, Moringa, Ficus, Hibiscus, Jasmine, Lemon, Mango, Tulsi, Curry leaves, Sandalwood, and 22+ more

---

## 📊 How It Works

### **Data Processing Pipeline**
1. Load 3,734 plant images (37 classes)
2. Split: 70% training, 15% validation, 15% testing
3. Augmentation: Rotation, shift, zoom, flip, brightness
4. Normalize pixels to 0-1 range
5. Resize all images to 224×224

### **Model Architecture**
- **Base Model:** VGG16 (pretrained on ImageNet)
- **Custom Head:**
  - Flatten → Dense(512) + Dropout(0.5)
  - Dense(256) + Dropout(0.3)
  - Dense(37) with softmax activation
- **Training:** 50 epochs with early stopping

### **Evaluation**
- Confusion matrix (37×37)
- Per-class precision, recall, F1-score
- Real-world test predictions with confidence scores

---

## 🔧 Technologies Used

- **TensorFlow/Keras** - Deep learning framework
- **VGG16** - Pre-trained CNN model
- **Streamlit** - Web application interface
- **NumPy/Pandas** - Data processing
- **Matplotlib/Seaborn** - Visualizations
- **scikit-learn** - Metrics & evaluation

---

## 📚 Project Files Explained

### **streamlit_app.py**
Interactive web app with:
- Image upload interface
- Camera capture
- Batch processing
- Confidence visualization
- Real-time predictions

### **Herbal_Plant_Identifier.ipynb**
Complete ML pipeline showing:
- Cell 1-3: Setup & configuration
- Cell 4-7: Data loading & exploration
- Cell 8-10: Model building & training
- Cell 11-15: Evaluation & analysis
- Cell 16-18: Results & export

**This notebook is your proof that the model works!**

---

## ⚙️ Model Details

**Why VGG16?**
- Proven architecture for image classification
- Transfer learning from ImageNet (1.2M images)
- Reduces training time significantly
- Achieves high accuracy with limited data

**Training Strategy:**
- Frozen base model (ImageNet weights)
- Custom classification head (37 classes)
- Data augmentation to prevent overfitting
- Early stopping (patience=5)
- Learning rate scheduling

**Performance:**
- Train: 90.51%
- Test: 88.06%
- Handles real-world plant images well

---

## 🎓 What You'll Learn

By exploring this project, you'll understand:
- ✅ Transfer learning with VGG16
- ✅ Image preprocessing & augmentation
- ✅ Model training with callbacks
- ✅ Evaluation metrics (confusion matrix, F1-score)
- ✅ Web app deployment with Streamlit
- ✅ End-to-end ML pipeline

---

## 🚀 Deployment

### **Local Testing**
```bash
streamlit run streamlit_app.py
```

**For Technical Understanding:**
1. Open the Jupyter notebook
2. Run cells sequentially
3. See model training, evaluation, real-world predictions
4. Understand the complete ML pipeline


---

## 📧 udynwosu@gmail.com Questions?

Check the Jupyter notebook for complete implementation details and proof of model accuracy on real-world data!

---

**Built with ❤️ using Python, TensorFlow, and Streamlit**

*Project demonstrates complete ML pipeline from data loading to deployment*
