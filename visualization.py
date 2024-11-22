import pyLDAvis
import pyLDAvis.gensim_models  # Для визуализации LDA через gensim

def visualize_topics(model, vectorizer, n_top_words=10):
    words = vectorizer.get_feature_names_out()
    topic_words = []
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[: -n_top_words - 1 : -1]
        topic_words.append([words[i] for i in top_words_idx])
    return topic_words

def visualize_lda(lda_model, corpus, dictionary):
    return pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
