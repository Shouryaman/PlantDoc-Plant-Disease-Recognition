# 🌿 PlantDoc - Plant Disease Recognition System

An AI-powered web application built with Streamlit and TensorFlow to quickly and accurately identify plant diseases from images. 

## 📝 Overview
PlantDoc's mission is to help farmers, gardeners, and agronomists identify plant diseases efficiently. By simply uploading an image of a plant leaf, the system uses a custom-built Convolutional Neural Network (CNN) to analyze it and detect any signs of diseases across 38 distinct crop/disease categories.

## ✨ Features
- **User-Friendly Interface**: Simple and intuitive design powered by Streamlit.
- **High Accuracy**: Uses a Deep Learning model built with TensorFlow/Keras to achieve high confidence in disease detection.
- **Fast Prediction**: Near-instant results allowing for quick decision-making.
- **Wide Range of Classes**: Capable of detecting 38 different categories (from Apple Scab to Tomato Yellow Leaf Curl Virus).

## 🚀 How It Works
1. **Upload Image**: Navigate to the **Disease Recognition** page from the sidebar and upload an image of a plant leaf.
2. **Analysis**: Click the "Predict" button. The system will preprocess the image and feed it into the CNN model.
3. **Results**: View the predicted disease classification on the screen instantly.

## 🛠️ Tech Stack
- **Python 3.x**
- **Streamlit**: Web application frontend framework.
- **TensorFlow & Keras**: Deep Learning model inference and prediction.
- **NumPy & Pandas**: Data manipulation.
- **Scikit-learn, Matplotlib, Seaborn**: Data analysis and visualization.

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shouryaman/PlantDoc-Plant-Disease-Recognition.git
   cd PlantDoc-Plant-Disease-Recognition
   ```

2. **Create a Virtual Environment (Optional but recommended):**
   ```bash
   python -m venv .venv
   # On Windows: 
   .venv\Scripts\activate
   # On macOS/Linux: 
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   ```bash
   streamlit run main.py
   ```

## 📊 Dataset Detail
The dataset used to train this model consists of roughly 87,000 RGB images of healthy and diseased crop leaves categorized into 38 different classes.
* It is split approximately into 80% training data and 20% validation data.
* A small batch of 33 test images is used exclusively to evaluate isolated performance.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request if you'd like to improve the recognition system or accuracy.