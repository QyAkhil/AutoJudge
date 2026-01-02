#Preprocessing for input data. Same preprocessing as the preprocessing done for input data  

import re
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack, csr_matrix

import nltk

for resource in ["punkt", "stopwords", "wordnet"]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource)


stop_words = set(stopwords.words('english'))
keep_words = {
    'if', 'no', 'not', 'nor', 'only', 'exists',
    'more', 'less', 'most', 'least', 'few', 'many',
    'find', 'print', 'given',
    'graph', 'path', 'edge', 'vertex', 'simple', 'constraint'
}
stop_words = stop_words - keep_words


lemmatizer = WordNetLemmatizer()


algo_keywords = {
    'dp', 'greedy', 'binary', 'search', 'dfs', 'bfs',
    'dijkstra', 'segment', 'tree', 'fenwick',
    'bitmask', 'combinatorics', 'modulo',
    'prime', 'graph', 'shortest', 'lca'
}


complexity_words = {
    'efficient', 'optimize', 'fast', 'logarithmic',
    'linear', 'quadratic', 'time', 'complexity',
    'constraints', 'limit','limits','bound','maximum', 'minimum', 'time limit',
    'upper bound'
}


def preprocess_for_model(title, description, input_format, output_format,
                         tags, time_limit, memory_limit,
                         tfidf_vectorizer, scaler, mlb_tags):

    # Text processing 
    combined_text = " ".join([str(title), str(description), str(input_format), str(output_format)])
    
    # Clean text
    combined_text = re.sub(r'[_]+', ' ', combined_text)
    combined_text = re.sub(r'[^\w\s]', ' ', combined_text)
    
    # Tokenize
    tokens = word_tokenize(combined_text.lower())
    
    # Remove stopwords and punctuation
    tokens_clean = [w for w in tokens if w not in stop_words and w not in string.punctuation]
    
    # Lemmatize
    tokens_lemma = [lemmatizer.lemmatize(w) for w in tokens_clean]
    
    # TF-IDF input
    tfidf_text = " ".join(tokens_lemma)
    X_text = tfidf_vectorizer.transform([tfidf_text])
    
    # Numeric features
    word_count = len(tokens)
    token_count = len(tokens_clean)
    algo_count = len(set(tokens_clean) & algo_keywords)
    complexity_terms = len(set(tokens) & complexity_words)
    
    numeric_features = np.array([[word_count, algo_count, complexity_terms, token_count,
                                  time_limit, memory_limit]])
    X_num = scaler.transform(numeric_features)
    
    # Tag features 
    X_tags = csr_matrix(mlb_tags.transform([tags]))  
    
    #  Combine all 
    X_final = hstack([X_text, X_num, X_tags])
    
    return X_final