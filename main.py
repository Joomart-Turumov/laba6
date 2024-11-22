import arxiv
import pandas as pd
import nltk
import spacy
import plotly.express as px
import pyLDAvis
import pyLDAvis.gensim_models  # Для визуализации LDA через gensim
import gensim
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import streamlit as st

# Загрузка необходимых данных для NLTK
nltk.download('punkt')  # Загрузка ресурса для токенизации
nltk.download('stopwords')  # Загрузка списка стоп-слов
nltk.download('punkt_tab')  # Загрузка пункта для токенизации

# Загрузка данных
def collect_data(query="data science", max_results=100):
    # Обновленный метод получения данных с использованием Client
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
    results = search.results()
    papers = []
    for result in results:
        papers.append({
            "title": result.title,
            "summary": result.summary,
            "url": result.entry_id
        })
    return pd.DataFrame(papers)

# Предобработка текста
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())  # Токенизация текста
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]  # Убираем стоп-слова
    return ' '.join(tokens)

# Моделирование тем
def topic_modeling(text_data, n_topics=5):
    texts = [text.split() for text in text_data]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Используем NMF для тематического моделирования
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(text_data)
    nmf_model = NMF(n_components=n_topics, random_state=42)
    nmf_topics = nmf_model.fit_transform(X)
    
    # Используем gensim для LDA
    lda_model = gensim.models.LdaModel(corpus, num_topics=n_topics, id2word=dictionary, passes=15)
    
    return nmf_model, lda_model, vectorizer, nmf_topics, corpus, dictionary

# Визуализация тем
def visualize_topics(model, vectorizer, n_top_words=10):
    words = vectorizer.get_feature_names_out()
    topic_words = []
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[: -n_top_words - 1 : -1]
        topic_words.append([words[i] for i in top_words_idx])
    return topic_words

# Визуализация LDA
def visualize_lda(lda_model, corpus, dictionary):
    return pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)

# Streamlit UI
def run_streamlit():
    st.title("Тематика научных статей")
    
    # Ввод пользователя
    query = st.text_input("Введите тему для поиска научных статей:", "data science")
    max_results = st.slider("Количество статей:", min_value=10, max_value=100, value=50)
    
    # Сбор данных
    df = collect_data(query=query, max_results=max_results)
    st.write(f"Найдено {len(df)} статей по запросу '{query}'")
    st.dataframe(df)

    # Предобработка текста
    st.write("Предобрабатываем текст...")
    df['processed_summary'] = df['summary'].apply(preprocess_text)
    
    # Моделирование тем
    st.write("Моделируем темы...")
    nmf_model, lda_model, vectorizer, nmf_topics, corpus, dictionary = topic_modeling(df['processed_summary'])
    
    # Визуализация тем
    nmf_topic_words = visualize_topics(nmf_model, vectorizer)
    st.write("Темы, найденные с использованием NMF:")
    st.write(nmf_topic_words)
    
    # Визуализация LDA с использованием pyLDAvis
    lda_visualization = visualize_lda(lda_model, corpus, dictionary)
    st.write(pyLDAvis.display(lda_visualization))

# Запуск Streamlit
if __name__ == "__main__":
    run_streamlit()
