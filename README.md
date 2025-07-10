# twitter-sentiment-analysis
LSTM-based deep learning model for classifying tweet sentiment using NLP techniques and a Kaggle dataset.

# 🧠 Twitter Sentiment Analysis using Deep Learning

This project focuses on analyzing the **sentiment of tweets** using a deep learning model (LSTM). It classifies tweets into **positive**, **neutral**, or **negative** sentiment categories. The project was developed using **Python**, **TensorFlow**, and **Google Colab**, and uses a **Kaggle dataset** for training and evaluation.

---

## 📌 Features

- Sentiment classification of tweets (positive, neutral, negative)
- Natural Language Processing (NLP) pipeline
- Deep learning model using LSTM layers
- Visualization of accuracy, loss, and confusion matrix
- Built and tested in Google Colab

---

## 🔧 Technologies Used

- Python 3
- Google Colab
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- Kaggle dataset

---

## 🗂 Dataset

The dataset was obtained from [Kaggle]([https://www.kaggle.com/](https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset)) and contains thousands of labeled tweets for sentiment classification.

> *Note: Due to licensing, the dataset is not uploaded directly. You may download the dataset from the Kaggle link and upload it to your Colab environment.*

---

## ⚙️ How to Run

1. Open the `Twitter_Sentiment_Analysis.ipynb` notebook in Google Colab.
2. Upload the dataset (CSV file) to your Colab environment.
3. Run the cells in order:
   - Data cleaning and preprocessing
   - Tokenization and padding
   - Model creation (LSTM)
   - Training and evaluation
4. View accuracy graphs and confusion matrix to analyze performance.

---


## 👨‍💻 Contributors

- **Atienza, Kurt Cydrick A.**
- Agoncillo, Royet L.
- Robles, Daniella L.
- Sanchez, Kylha A.

---

## 📄 License

This project is for academic purposes only. Dataset belongs to its original creators on Kaggle.

---

## 🚀 Future Improvements

- Deploying the model as a web app using Flask or Streamlit
- Use BERT or transformer-based models for higher accuracy
- Train on real-time tweets using Twitter API

- Regularization Techniques: Adding dropout layers or L2 regularization to prevent overfitting.
- Data Augmentation: Applying transformations like rotation, flipping, and zooming to increase dataset diversity.
- Hyperparameter Tuning: Experimenting with different learning rates, batch sizes, and optimizers.
- Transfer Learning: Leveraging pre-trained models such as VGG16 or ResNet, which are already trained on large datasets and can provide a better starting point for classification.
- Class Balancing: If dataset imbalance is an issue, oversampling minority classes or undersampling majority classes could help.

