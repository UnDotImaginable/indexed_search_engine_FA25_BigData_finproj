import spacy
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import math
from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
from collections import defaultdict

def find_links(doc_ids):
    links = {}

    conn = mysql.connector.connect(
        host="localhost",
        user="scraper",
        password="scraper123",
        database="STATE_UNION_ADDR"
    )

    cursor = conn.cursor()

    for doc_id in doc_ids:
        cursor.execute("SELECT Link FROM address_table WHERE doc_id = %s", (doc_id,))
        row = cursor.fetchone()
        
        if row:
            links[doc_id] = row[0]   # row[0] IS the Link.
        else:
            links[doc_id] = None     # Not found. 
    
    cursor.close()
    conn.close()

    
    return links




index_table = pd.read_csv("updated_index_table.csv")

app = Flask(__name__)
CORS(app)



@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()  
    query = data.get('query', '')
    
    print("Received query:", query)
    
    # return jsonify({"message": f"Server received: {query}"}), 200




    nlp = spacy.load("en_core_web_trf")
    doc = nlp(query)

    meaningful_tokens = []

    for token in doc:
        if token.is_stop:
            continue
        else:
            meaningful_tokens.append(token.lemma_.lower())
    

    docvecs = pd.read_csv("document_vectors.csv")
    doc_ids = docvecs['doc_id']
    docvecs = docvecs.drop(columns=['doc_id'])
    docvec_columns = list(docvecs.columns)

    term_frequencies = dict(Counter(meaningful_tokens))
    term_tfidf = dict()

    N = index_table['doc_id'].nunique()

    for query_term in term_frequencies.keys():
        if query_term not in index_table['term'].values:
            continue
    
        idf = index_table[index_table['term'] == query_term].iloc[0]['idf']
        term_tfidf[query_term] = math.log1p(term_frequencies[query_term]) * idf

    blank_query_vector = np.zeros(len(docvec_columns))

    for term, value in term_tfidf.items():
        if term in docvec_columns:
            idx = docvec_columns.index(term)
            blank_query_vector[idx] = value

    doc_matrix = docvecs.to_numpy()

    q_sim = cosine_similarity([blank_query_vector], doc_matrix)
    q1_scores = q_sim[0]

    top5_indices = np.argsort(q1_scores)[::-1][:5]

    top5_doc_ids = [doc_ids.iloc[idx] for idx in top5_indices]
    
    top5_pairs = find_links(top5_doc_ids)

    return jsonify({
        "top5": top5_pairs
    }), 200


    

if __name__ == "__main__":
    app.run(debug=True)