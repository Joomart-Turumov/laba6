import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Загрузка необходимых данных для NLTK
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())  # Токенизация текста
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]  # Убираем стоп-слова
    return ' '.join(tokens)
