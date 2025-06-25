import pandas as pd
import re
import joblib
from pathlib import Path
import urllib.request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class SpamClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.model_file = 'spam_classifier.pkl'
        self.vectorizer_file = 'tfidf_vectorizer.pkl'
        self.data_file = 'sms_dataset.tsv'

    def _download_data(self):
        """Download the dataset if it doesn't exist locally"""
        if not Path(self.data_file).exists():
            url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
            urllib.request.urlretrieve(url, self.data_file)
            print("📥 Dataset downloaded!")

    def _clean_text(self, text):
        """Lowercase and remove punctuation and digits from the message"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        return text

    def load_data(self):
        """Load and clean the SMS dataset"""
        self._download_data()
        df = pd.read_csv(self.data_file, sep='\t', header=None, names=['label', 'message'], encoding='utf-8')
        df['message'] = df['message'].apply(self._clean_text)
        print(f"📊 Loaded {len(df)} messages — {sum(df['label'] == 'spam')} spam messages.")
        return df

    def train(self, df=None, save_model=True):
        """Train the spam detection model and optionally save it"""
        if df is None:
            df = self.load_data()

        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        X = self.vectorizer.fit_transform(df['message'])
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = MultinomialNB()
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        print(f"✅ Training done! Accuracy: {accuracy_score(y_test, y_pred):.2%}")
        print("\n📈 Report:")
        print(classification_report(y_test, y_pred))
        print("🧮 Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        if save_model:
            self._save_model()

    def _save_model(self):
        """Save model and vectorizer locally"""
        joblib.dump(self.model, self.model_file)
        joblib.dump(self.vectorizer, self.vectorizer_file)
        print(f"💾 Model saved as '{self.model_file}' and vectorizer as '{self.vectorizer_file}'")

    def load_saved_model(self):
        """Load model and vectorizer if saved"""
        if Path(self.model_file).exists() and Path(self.vectorizer_file).exists():
            self.model = joblib.load(self.model_file)
            self.vectorizer = joblib.load(self.vectorizer_file)
            print("🔁 Loaded saved model!")
            return True
        return False

    def predict(self, text, show_confidence=True):
        """Predict if a message is spam or not"""
        if self.model is None or self.vectorizer is None:
            if not self.load_saved_model():
                raise Exception("⚠️ Model not found. Please train first.")

        cleaned = self._clean_text(text)
        vec = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(vec)[0]
        proba = self.model.predict_proba(vec)[0]

        print(f"\n📨 Message: {text}")
        print(f"🔮 Prediction: {prediction}")
        if show_confidence:
            print(f"📊 Confidence: {max(proba):.2%}")
            print(f"➡️ ham: {proba[0]:.2%}, spam: {proba[1]:.2%}")
        return prediction

def main():
    classifier = SpamClassifier()

    # Uncomment this **only on your very first run** to train and save the model
    # print("⚙️ First-time training in progress...")
    # classifier.train()

    # Every other time, this will load your model if already saved
    if not classifier.load_saved_model():
        print("⚠️ No saved model found, starting training...")
        classifier.train()

    # Interactive prediction menu
    while True:
        print("\n" + "="*50)
        print("📌 What would you like to do?")
        print("1. Test with sample emails")
        print("2. Try your own message")
        print("3. Retrain the model")
        print("4. Exit")
        choice = input("👉 Choose (1/2/3/4): ")

        if choice == '1':
            test_emails = [
                "Congratulations! You've won a $1000 gift card! Click here to claim",
                "Hi John, just checking in about tomorrow's meeting",
                "URGENT: Your bank account needs verification",
                "Team lunch this Friday at 12:30 PM"
            ]
            for email in test_emails:
                classifier.predict(email)

        elif choice == '2':
            email = input("\n✉️ Enter your message: ")
            classifier.predict(email)

        elif choice == '3':
            print("🔁 Retraining model from scratch...")
            classifier.train()

        elif choice == '4':
            print("👋 Exiting. See you again!")
            break

        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()