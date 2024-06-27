import os
import re
import nltk
import math
import sys
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from functools import lru_cache
import numpy as np
from finance_stemmer import PorterStemmer 

# RELEVANCE FEEDBACK FUNCTIONS START HERE
def relevanceFeedback(query, retrieved_docs):
    relevant_docs = set()
    irrelevant_docs = set()

    # Obtain user feedback on relevance of retrieved documents
    for doc_id, doc in retrieved_docs.items():
        relevance = input(f"Is document {doc_id} relevant? (Y/N): ")
        if relevance.lower() == 'y':
            relevant_docs.add(doc_id)
        else:
            irrelevant_docs.add(doc_id)

    return relevant_docs, irrelevant_docs

def adjustQuery(query, relevant_docs, irrelevant_docs, inverted_index, idf_list, alpha=1, beta=0.75, gamma=0.15):
    relevant_docs_terms = {}
    irrelevant_docs_terms = {}
    
    # Calculate the centroid of relevant documents
    for doc_id in relevant_docs:
        for term, freq in inverted_index[doc_id].items():
            if term not in relevant_docs_terms:
                relevant_docs_terms[term] = freq
            else:
                relevant_docs_terms[term] += freq
    
    # Calculate the centroid of irrelevant documents
    for doc_id in irrelevant_docs:
        for term, freq in inverted_index[doc_id].items():
            if term not in irrelevant_docs_terms:
                irrelevant_docs_terms[term] = freq
            else:
                irrelevant_docs_terms[term] += freq
    
    # Calculate the adjusted query vector
    adjusted_query = {}
    for term in query:
        # Initialize the term frequency in the adjusted query
        adjusted_query[term] = 0
        
        # Calculate the adjusted term frequency based on relevant and irrelevant documents
        if term in relevant_docs_terms:
            adjusted_query[term] += alpha * (relevant_docs_terms[term] / len(relevant_docs))
        if term in irrelevant_docs_terms:
            adjusted_query[term] -= beta * (irrelevant_docs_terms[term] / len(irrelevant_docs))
        
        # Incorporate the original query term frequency
        adjusted_query[term] += gamma * idf_list.get(term, 0)
    
    return adjusted_query

# END OF RELEVANCE FEEDBACK FUNCTIONS

# PREPROCESSING STARTS HERE
def removeSGML(text):
    clean_text = ""
    within_tag = False

    for char in text:
        if char == '<':
            within_tag = True
        elif char == '>': 
            within_tag = False
        elif not within_tag:
            clean_text += char

    return clean_text

def tokenizeText(text):
    t = removeSGML(text)
    tokens = []
    current_token = ""

    for char in t:
        #New word
        if char == ' ' and current_token:
            tokens.append(current_token)
            current_token = ""
        elif char == "\'":
            # Lot's of logic to deal with '
            if current_token.endswith('s') and current_token != "s":
                #For possesives'
                continue
            elif current_token.lower() in ["i", "you", "he", "she", "it", "we", "they"]:
                # For contractions
                current_token += char
            elif current_token:
                tokens.append(current_token)
                current_token = ""

        if char.isalnum() or char in ['-']:
            #Handles cases like check-in and normal words
            current_token += char
        elif char == '.':
            #Handles cases of U.S.A or decimals
            if current_token.replace('.', '').isalpha() or current_token.replace('.', '').isdigit():
                current_token += char
            elif current_token:
                tokens.append(current_token)
                current_token = ""

        elif char == ',':
            # Handles numbers like 332,087,410
            if current_token.replace(',', '').isdigit():
                current_token += char
            elif current_token:
                tokens.append(current_token)
                current_token = ""
            
        elif char == '/' and current_token.replace('/', '').isdigit(): 
            #Handles dates
            current_token += char

    if current_token:
        #Handles last word and deals with issues like fullstops
        if not current_token[-1].isalpha():
             tokens.append(current_token[:-1])
        else:
            tokens.append(current_token)       
    
    return tokens

def preprocess_sec_filings(file_path):
    # Read the SEC filing
    with open(file_path, 'r', encoding='utf-8') as file:
        filing_text = file.read()

    # Use BeautifulSoup to extract text from HTML
    soup = BeautifulSoup(filing_text, 'html.parser')
    text = soup.get_text()

    # Convert to lowercase
    text = text.lower()

    # Tokenize the text using custom function
    tokens = tokenizeText(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

# PREPROCESSING ENDS HERE

# IR STARTS HERE
def index_document(doc_ids):
    """
    Creates an inverted index and calculates the TF-IDF weights for each document.

    Args:
    - doc_ids (dict): A dictionary where key is document ID and value is list of words in that document.

    Returns:
    - tuple: A tuple containing the inverted index and the IDF list.
    """
    inverted_index = {}
    idf_list = {}
    tf_idf = {}
    total_docs = len(doc_ids)

    # Building inverted index and calculating document frequency
    for doc_id, tokens in doc_ids.items():
        for token in tokens:
            if token not in idf_list:
                idf_list[token] = set()
            idf_list[token].add(doc_id)
            if doc_id not in inverted_index:
                inverted_index[doc_id] = {}
            if token not in inverted_index[doc_id]:
                inverted_index[doc_id][token] = 0
            inverted_index[doc_id][token] += 1

    # Calculating IDF and updating TF-IDF for each document
    for term, docs in idf_list.items():
        idf = math.log10(total_docs / len(docs))
        for doc in docs:
            term_freq = inverted_index[doc][term]
            max_freq = max(inverted_index[doc].values())
            normalized_tf = term_freq / max_freq
            tf_idf.setdefault(doc, {})[term] = normalized_tf * idf

    return inverted_index, idf_list, tf_idf

def tf_idf_search(documents, query):
    """
    Searches documents by converting them into TF-IDF weights and using cosine similarity to rank them.

    Args:
    - documents (list of str): List of documents.
    - query (list of str): Search query as a list of words.

    Returns:
    - list: Ranked list of documents based on the relevance to the query.
    """
    doc_ids = {i: tokenizeText(doc) for i, doc in enumerate(documents)}
    inverted_index, idf_list, _ = index_document(doc_ids)

    # Building query TF-IDF vector
    query_tf = {}
    for word in query:
        if word in idf_list:
            if word not in query_tf:
                query_tf[word] = 0
            query_tf[word] += 1
    
    # Normalize query vector
    for term in query_tf:
        query_tf[term] = (query_tf[term] / max(query_tf.values())) * idf_list[term]

    # Calculate cosine similarity
    cos_sim = {}
    for doc_id, freqs in inverted_index.items():
        dot_product = sum(freqs.get(term, 0) * weight for term, weight in query_tf.items())
        doc_norm = np.sqrt(sum(value ** 2 for value in freqs.values()))
        query_norm = np.sqrt(sum(value ** 2 for value in query_tf.values()))
        if doc_norm * query_norm != 0:
            cos_sim[doc_id] = dot_product / (doc_norm * query_norm)
        else:
            cos_sim[doc_id] = 0

    # Rank documents by similarity
    ranked_docs = [documents[doc_id] for doc_id in sorted(cos_sim, key=cos_sim.get, reverse=True)]
    return ranked_docs

#VECTOR SPACE MODEL-----------------------------------------------------------------------------------
def preprocess_text(text):
    text = removeSGML(text)
    tokens = tokenizeText(text)
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token, 0, len(token)-1) for token in tokens]
    
    return stemmed_tokens

document_frequencies = {}


def indexDocument(doc_id, doc_content, doc_weight_scheme, query_weight_scheme, inverted_index, doc_length, df_counts, total_docs):
    tokens = preprocess_text(doc_content)
    
    doc_length[doc_id] = len(tokens)
    
    tf = {}
    for token in tokens:
        if token not in tf:
            tf[token] = 0
        tf[token] += 1
        
        if token not in df_counts:
            df_counts[token] = set()
        df_counts[token].add(doc_id)
    
    max_tf = max(tf.values())  # Find the maximum term frequency in the document
    # Updating the inverted indices with the appropriate weight for each term in the document
    for term, freq in tf.items():
        if doc_weight_scheme == 'tfc':
            idf = math.log((total_docs + 1) / (1 + len(df_counts.get(term, []))))
            tf_idf = freq * idf
        elif doc_weight_scheme == 'atfa':
            # Implementing the ATFA scheme
            tf_idf = 0.5 + 0.5 * (freq / max_tf)
        else:
            raise ValueError("Invalid document weighting scheme")
        
        if term not in inverted_index:
            inverted_index[term] = {}
        inverted_index[term][doc_id] = tf_idf


def retrieveDocuments(query, inverted_index, df_counts, total_docs, doc_length, query_weight_scheme, doc_weight_scheme):
    query_tokens = preprocess_text(query)
    tf = {token: query_tokens.count(token) for token in query_tokens}
    max_tf = max(tf.values())
    
    query_vector = {}
    if query_weight_scheme == 'tfx': 
        for token in set(query_tokens):
            tf = query_tokens.count(token)
            idf = math.log((total_docs + 1) / (1 + len(df_counts.get(token, []))))
            tf_idf = tf * idf
            query_vector[token] = tf_idf
    elif query_weight_scheme == 'atfa': 
            freq = tf[token]
            atfa = 0.5 + 0.5 * (freq / max_tf) if freq > 0 else 0
            tf_idf = atfa * idf
            query_vector[token] = tf_idf
        
    
    query_magnitude = math.sqrt(sum(tf_idf ** 2 for tf_idf in query_vector.values()))
    cosine_similarities = {}

    for doc_id in doc_length.keys():
        dot_product = sum(query_vector.get(term, 0) * inverted_index.get(term, {}).get(doc_id, 0) for term in query_vector)
        doc_magnitude = math.sqrt(sum(inverted_index.get(term, {}).get(doc_id, 0)**2 for term in query_vector))
        
        if doc_magnitude > 0 and query_magnitude > 0: 
            cosine_similarity = dot_product / (doc_magnitude * query_magnitude)
            cosine_similarities[doc_id] = cosine_similarity
        else:
            cosine_similarities[doc_id] = 0
    
    return cosine_similarities

def runVSM(doc_weight_scheme, query_weight_scheme, documents, query):
    inverted_index = {}
    df_counts = {}
    doc_length = {}
    total_docs = 0

    #Going through every document in the list: TO FIGURE IT OUT - EDIT FOR DICTIONARY
    for doc_id, content in enumerate(documents, 1):
        indexDocument(str(doc_id), content, doc_weight_scheme, query_weight_scheme, inverted_index, doc_length, df_counts, total_docs)
        total_docs += 1

    #The query processing
    similarity_scores = retrieveDocuments(query, inverted_index, df_counts, total_docs, doc_length, query_weight_scheme, doc_weight_scheme)
    sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
    result = (1, sorted_scores)  # Since there's only one query, using a fixed query_id = 1, can change later

    #Temp write to file for testing
    output_filename = f'vsm_results.{doc_weight_scheme}.{query_weight_scheme}.txt'
    with open(output_filename, 'w') as outf:
        query_id, scores = result
        for doc_id, score in scores:
            outf.write(f'{query_id} {doc_id} {score:.4f}\n')

    return result
    
def generate_dictinoary_data(text):
    word_to_index = dict()
    index_to_word = dict()
    corpus = []
    count = 0
    vocab_size = 0
    
    for word in text:
        word = word.lower()
        corpus.append(word)
        if word not in word_to_index:
            word_to_index[word] = count
            index_to_word[count] = word
            count += 1
    
    vocab_size = len(word_to_index)
    length_of_corpus = len(corpus)
    
    return word_to_index, index_to_word, corpus, vocab_size, length_of_corpus


def get_one_hot_vectors(target_word,context_words,vocab_size,word_to_index):
    trgt_word_vector = np.zeros(vocab_size)
    index_of_word_dictionary = word_to_index.get(target_word) 
    
    trgt_word_vector[index_of_word_dictionary] = 1
    ctxt_word_vector = np.zeros(vocab_size)
    for word in context_words:
        index_of_word_dictionary = word_to_index.get(word) 
        ctxt_word_vector[index_of_word_dictionary] = 1
        
    return trgt_word_vector,ctxt_word_vector

def generate_training_data(corpus,window_size,vocab_size,word_to_index,length_of_corpus,sample=None):

    training_data =  []
    training_sample_words =  []
    for i,word in enumerate(corpus):

        index_target_word = i
        target_word = word
        context_words = []

        if i == 0:  
            context_words = [corpus[x] for x in range(i + 1 , window_size + 1)] 
        elif i == len(corpus)-1:
            context_words = [corpus[x] for x in range(length_of_corpus - 2 ,length_of_corpus -2 - window_size  , -1 )]
        else:
            before_target_word_index = index_target_word - 1
            for x in range(before_target_word_index, before_target_word_index - window_size , -1):
                if x >=0:
                    context_words.extend([corpus[x]])
            after_target_word_index = index_target_word + 1
            for x in range(after_target_word_index, after_target_word_index + window_size):
                if x < len(corpus):
                    context_words.extend([corpus[x]])
        trgt_word_vector,ctxt_word_vector = get_one_hot_vectors(target_word,context_words,vocab_size,word_to_index)
        training_data.append([trgt_word_vector,ctxt_word_vector])   
        
        if sample is not None:
            training_sample_words.append([target_word,context_words])   
        
    return training_data,training_sample_words


def forward_prop(weight_inp_hidden,weight_hidden_output,target_word_vector):
    
    hidden_layer = np.dot(weight_inp_hidden.T, target_word_vector)
    
    u = np.dot(weight_hidden_output.T, hidden_layer)
    
    y_predicted = softmax(u)
    
    return y_predicted, hidden_layer, u
  
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def calculate_error(y_pred,context_words):
    
    total_error = [None] * len(y_pred)
    index_of_1_in_context_words = {}
    
    for index in np.where(context_words == 1)[0]:
        index_of_1_in_context_words.update ( {index : 'yes'} )
        
    number_of_1_in_context_vector = len(index_of_1_in_context_words)
    
    for i,value in enumerate(y_pred):
        
        if index_of_1_in_context_words.get(i) != None:
            total_error[i]= (value-1) + ( (number_of_1_in_context_vector -1) * value)
        else:
            total_error[i]= (number_of_1_in_context_vector * value)
            
            
    return  np.array(total_error)

def backward_prop(weight_inp_hidden,weight_hidden_output,total_error, hidden_layer, target_word_vector,learning_rate):
    
    dl_weight_inp_hidden = np.outer(target_word_vector, np.dot(weight_hidden_output, total_error.T))
    dl_weight_hidden_output = np.outer(hidden_layer, total_error)
    
    weight_inp_hidden = weight_inp_hidden - (learning_rate * dl_weight_inp_hidden)
    weight_hidden_output = weight_hidden_output - (learning_rate * dl_weight_hidden_output)
    
    return weight_inp_hidden,weight_hidden_output

def calculate_loss(u,ctx):
    
    sum_1 = 0
    for index in np.where(ctx==1)[0]:
        sum_1 = sum_1 + u[index]
    
    sum_1 = -sum_1
    sum_2 = len(np.where(ctx==1)[0]) * np.log(np.sum(np.exp(u)))
    
    total_loss = sum_1 + sum_2
    return total_loss

def retrieve_similar_documents(documents, ids, query, window_size, learning_rate, epochs):
    texts = []
    # train this by not many documents
    for doc in documents[:10]:
        temp = tokenizeText(doc)
        texts.extend(temp)
    word_to_index, index_to_word, corpus, vocab_size, length_of_corpus = generate_dictinoary_data(texts)
    
    training_data, _ = generate_training_data(corpus, window_size, vocab_size, word_to_index, length_of_corpus)
    
    np.random.seed(0)
    weight_inp_hidden = np.random.rand(vocab_size, vocab_size)
    weight_hidden_output = np.random.rand(vocab_size, vocab_size)
    
    for _ in range(epochs):
        for target_word_vector, context_word_vector in training_data:
            y_pred, hidden_layer, u = forward_prop(weight_inp_hidden, weight_hidden_output, target_word_vector)

            error = calculate_error(y_pred, context_word_vector)
            
            weight_inp_hidden, weight_hidden_output = backward_prop(weight_inp_hidden, weight_hidden_output, error, hidden_layer, target_word_vector, learning_rate)   
    embeddings = weight_inp_hidden
    query_tokens = query.lower().split()
    query_vector = np.zeros(vocab_size)
    for token in query_tokens:
        if token in word_to_index:
            query_vector[word_to_index[token]] = 1
    similarity_scores = np.dot(embeddings, query_vector)
    most_similar_indices = np.argsort(similarity_scores)[::-1]
    similar_documents = [ids[i] for i in most_similar_indices]
    
    return similar_documents

#IR ENDS HERE



# DATA VISUALIZATION STARTS HERE
import visualize
# NOTE: was potentially throwing errors for me
# visualize.visualize(cos_sort)
# DATA VISUALIZATION ENDS HERE

def unified_information_retrieval(query, documents):
    # Heuristic for determining which IR method to use. Word2Vec is better for length > 5 because it captures more meaning
    if len(query.split()) > 5: 
        method = 'word2vec'
    elif "specific term" in query:
        method = 'tfidf'
    else:
        method = 'feedback'
    # Use TF-IDF and cosine similarity for searching
    if method == 'tfidf':
        return tf_idf_search(documents, tokenizeText(query))
        # Initialize Word2Vec model if needed
    if method == 'word2vec':
       window_size = 1
       learning_rate = 0.1
       epochs = 5
    #    please also provided the documents ids in this function, please check the example file to see the document ids should be
       similar_documents = retrieve_similar_documents(documents, ids, query, window_size, learning_rate, epochs)
       return similar_documents
    # Use relevance feedback mechanism
    else:
        # This will require actual user interaction which we can simulate or leave for real-time usage
        return "TODO: PUT RELEVANCE SEARCH STUFF HERE."


def main(documents):
    while True:
        # Prompt user for a query
        user_input = input("Enter your query or type 'TERMINATE' to exit: ")
        
        if user_input.lower() == 'terminate':  # Check if user wants to terminate
            print("Terminating the program.")
            break
        
        query_normalized = user_input.split()  # Simple normalization splitting by spaces

        # Check if the query includes a year for filtering
        year_in_query = [word for word in query_normalized if word.isdigit()]
        filtered_documents = documents
        if year_in_query:
            filtered_documents = boolean_search(documents, year_in_query) #TODO - ROOPE

        preprocessed_docs = preprocess_documents(filtered_documents) #TODO - DAVID
        adjusted_query = 0
        
        # Initialize relevance feedback loop control variable
        continue_feedback = True

        while continue_feedback:
            # Retrieve and rank documents
            ranked_docs = ranked_retrieval(preprocessed_docs, query_normalized) #TODO - JIAYING & AUTUMN

            # Generate summaries
            summaries = [text_rank_summary(doc) for doc in ranked_docs] #TODO - HANK

            # Print the summaries of the top documents
            print("\nTop Documents and Summaries:")
            for doc, summary in zip(ranked_docs, summaries):
                print(doc[:100] + "...\n" + "Summary: " + summary + "\n")

            # RELEVANCE FEEDBACK STARTS HERE
            relevant_docs, irrelevant_docs = relevanceFeedback(query_normalized, ranked_docs)
            adjusted_query = adjustQuery(query_normalized, relevant_docs, irrelevant_docs, inverted_index, idf_list)
            retrieved_docs_with_feedback = retrieveDocuments(adjusted_query, inverted_index, idf_list, weight_doc, weight_query)

            print("Documents after relevance feedback:")
            for doc_id, score in retrieved_docs_with_feedback.items():
                print(f"Document {doc_id}: Score {score}")

            # Ask user if they want to continue relevance feedback
            feedback_choice = input("Do you want to continue providing relevance feedback? (Y/N): ")
            if feedback_choice.lower() != 'y':
                continue_feedback = False
        #RELEVANCE FEEDBACK ENDS HERE

    # End of main function


# Example usage
documents = [
    "This is a document from the year 2020. It contains information about data science.",
    "Document from 2021 discussing advanced machine learning algorithms.",
    "Old document from 1990 about historical data methods."
]

main(documents)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_document_file>")
        sys.exit(1)

    document_path = sys.argv[1]
    try:
        with open(document_path, 'r') as file:
            documents = file.read().splitlines()
    except FileNotFoundError:
        print(f"Error: The file at {document_path} was not found.")
        sys.exit(1)

    main(documents)
