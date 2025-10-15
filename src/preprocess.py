# src/preprocess.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import os

# Téléchargement des ressources NLTK nécessaires (une seule fois)
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


def preprocess_text(text):
    """
    Nettoie et prétraite un texte de tweet.
    """
    # 1. Conversion en minuscules
    text = text.lower()

    # 2. Suppression des URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 3. Suppression des mentions (@username) et hashtags (#)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)

    # 4. Suppression des caractères non alphabétiques et de la ponctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 5. Tokenisation
    tokens = word_tokenize(text)

    # 6. Suppression des "Stop Words"
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # 7. Lemmatisation
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    return " ".join(lemmatized_tokens)


if __name__ == "__main__":
    # Charger les données brutes
    raw_data_path = os.path.join('data', 'raw_tweets.csv')
    df = pd.read_csv(raw_data_path)
    print("Début du prétraitement du texte...")

    # Appliquer la fonction de prétraitement à la colonne 'text'
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    print("Prétraitement terminé.")

    # Sélectionner les colonnes finales
    df_processed = df[['sentiment', 'cleaned_text']]

    # Séparer les données en ensembles d'entraînement et de test
    X = df_processed['cleaned_text']
    y = df_processed['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Sauvegarder les ensembles de données traités
    data_dir = 'data'
    train_df = pd.DataFrame({'text': X_train, 'sentiment': y_train})
    test_df = pd.DataFrame({'text': X_test, 'sentiment': y_test})
    train_df.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(data_dir, 'test.csv'), index=False)

    print("Données d'entraînement et de test sauvegardées.")
