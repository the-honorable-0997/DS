#!/usr/bin/env python
# coding: utf-8

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer


def setup_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')


def main():
    setup_nltk()

    document = """
    Natural Language Processing is a subfield of Artificial Intelligence.
    It enables computers to understand, interpret, and generate human language.
    Machine learning models are trained on large datasets to perform NLP tasks.
    Text analytics involves extracting useful information from text data.
    """

    # Tokenization
    tokens = word_tokenize(document)
    print("\n--- TOKENS ---")
    print(tokens)

    # POS Tagging
    pt = pos_tag(tokens)
    print("\n--- POS TAGGING ---")
    print(pt)

    # Stopword Removal
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in tokens if w.lower() not in stop_words]

    print("\n--- AFTER STOPWORD REMOVAL ---")
    print(filtered)

    # Stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(w) for w in filtered]

    print("\n--- STEMMED WORDS ---")
    print(stemmed_words)

    # Lemmatization
    lmt = WordNetLemmatizer()
    lemmatized_words = [lmt.lemmatize(w) for w in filtered]

    print("\n--- LEMMATIZED WORDS ---")
    print(lemmatized_words)

    # TF-IDF
    text = [
        "Natural Language Processing is a subfield of Artificial Intelligence.",
        "Machine learning models are trained on large datasets to perform NLP tasks.",
        "Text analytics involves extracting useful information from text data.",
        "Deep learning is a part of machine learning and Artificial Intelligence.",
        "NLP enables computers to understand and generate human language."
    ]

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(text)

    print("\n--- TF-IDF MATRIX ---")
    print(tfidf.toarray())

    print("\n--- FEATURE NAMES ---")
    print(vectorizer.get_feature_names_out())


if __name__ == "__main__":
    main()