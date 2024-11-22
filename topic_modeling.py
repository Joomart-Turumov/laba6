import gensim
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

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
