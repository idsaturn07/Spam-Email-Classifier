# 📧 Email Spam Classifier

A simple yet powerful machine learning tool to classify SMS messages as **spam** or **ham** (not spam).  
Built using **Multinomial Naive Bayes** and **TF-IDF vectorization**, this project achieves over **97% accuracy** on real-world data.

---

## 🚀 Features

- 🔍 Detects and classifies messages as **spam** or **ham**
- 💾 Automatically saves and loads model/vectorizer using `joblib`
- 📉 Displays accuracy, precision, recall, F1-score, and confusion matrix
- 🧪 Allows testing of sample and custom messages via CLI
- 🔁 Retrain the model anytime using an interactive menu
- 🧼 Text is cleaned and normalized before training for better accuracy

---

## 🛠️ Tech Stack

| Component        | Tool/Library             |
|------------------|--------------------------|
| Language         | Python 3.10+             |
| ML Algorithm     | Multinomial Naive Bayes  |
| Vectorizer       | TfidfVectorizer          |
| ML Library       | scikit-learn             |
| Data Handling    | pandas, numpy            |
| Model Storage    | joblib                   |
| Dataset Source   | UCI SMS Spam Collection  |

---

## 📥 Installation & Setup

```bash
# 1. Clone the repository and enter the project directory
git clone https://github.com/idsaturn07/Spam-Email-Classifier.git
cd email-spam-classifier

# 2. Create and activate virtual environment (recommended)
python -m venv venv
.env\Scriptsctivate       # On Windows
source venv/bin/activate     # On Mac/Linux

# 3. Install required dependencies
pip install scikit-learn pandas numpy joblib

# 4. First-time setup:
#    Uncomment the train() line inside main() of spam_detector.py and run:
python spam_detector.py

# 5. For subsequent runs:
#    Keep the train() line commented. Just run the script:
python spam_detector.py
```

---

## ▶️ Usage Guide

When you run the script, you'll be presented with an interactive menu:

```text
1. Test example emails
2. Enter custom email text
3. Retrain model
4. Exit
```

- Select `1` to test pre-defined example messages
- Select `2` to input your own email text
- Select `3` to retrain the model from scratch
- Select `4` to exit the program

---

## 🧪 Example Output

```text
📨 Message: Claim your $1000 prize now!
🔮 Prediction: spam
📊 Confidence: 96.7%
Probabilities: ham 3.3%, spam 96.7%

📨 Message: Hi John, just checking in about tomorrow's meeting
🔮 Prediction: ham
📊 Confidence: 99.1%
```

---

## 📁 Project Structure

```
email-spam-classifier/
├── spam_detector.py         # Main Python script with menu
├── spam_classifier.pkl        # Trained model (auto-saved)
├── tfidf_vectorizer.pkl       # Saved vectorizer (auto-saved)
├── sms_dataset.tsv            # Dataset file (auto-downloaded on first run)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 📚 Dataset & References

- 📥 Dataset: [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- 📦 Source: Auto-downloaded from [JustMarkham's GitHub](https://github.com/justmarkham/pycon-2016-tutorial)
- 🤖 Model: Multinomial Naive Bayes with TF-IDF vectorization
- 🧠 Dependencies: `scikit-learn`, `pandas`, `numpy`, `joblib`

---

## 🔮 Future Roadmap

- 🖥️ Build a web UI using Streamlit or Flask
- 📤 Export predictions and logs to CSV
- 🔐 Add email sanitization and phishing protection
- 🌐 Add multilingual spam detection
- 📊 Integrate advanced features like link/attachment scanning

---

## 📜 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and share it for educational or commercial purposes.

---
