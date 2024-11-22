import streamlit as st
import pandas as pd
from data_collection import collect_data
from preprocessing import preprocess_text
from topic_modeling import topic_modeling
from visualization import visualize_topics, visualize_lda
import pyLDAvis

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
