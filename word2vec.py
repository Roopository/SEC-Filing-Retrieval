import os
import re
import nltk
import math
import sys
import pathlib
from nltk.corpus import stopwords
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from bs4 import BeautifulSoup

# # from functools import lru_cache# # 
import numpy as np

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
    print(len(texts))
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

def main():
    documents = []
    ids = []
    input_path = sys.argv[1]
    input = pathlib.Path(input_path)
    for file_path in input.iterdir():
        ids.append(file_path.stem)
        with open(file_path, 'r', encoding='ISO-8859-1') as re_file:
            texts = re_file.read()
            documents.append(texts)
    query = "money I need"
    window_size = 1
    learning_rate = 0.1
    epochs = 5

    similar_documents = retrieve_similar_documents(documents, ids, query, window_size, learning_rate, epochs)
    print(similar_documents)


if __name__ == "__main__":
    main()
