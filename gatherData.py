import requests
import sys
import os
import logging
import threading
import nltk
from nltk.corpus import stopwords
from sec_api import QueryApi
from bs4 import BeautifulSoup
from solution import tokenizeText

"""
Overview:

Script to run at the beginning, before other processes, that will gather the requisite data
from the SEC databases, store in local folders (either created or pre-existing)
and then run minimal pre-processing before handing off to larger IR methods.

"""

# Input: None
# Output: boolean indicating if process was successful or not
# Modifies: checks local paths and confirms correct folder to save forms
def checkLocalDir(): 
    return os.path.exists(os.path.join(os.getcwd(), 'data'))

# Input: None
# Output: boolean indicating if process was successful or not
# Modifies: creates new folder to store filings
def setupLocalDir(): 
    cwd = os.getcwd()
    final_path = os.path.join(cwd, 'data')
    os.makedirs(final_path)

    return os.path.exists(final_path)

# Input: None
# Output: array of HTTP requests representing the chunked GETS for relevant filings
def prepEDGARRequests():
    base_url = "https://data.sec.gov/submissions/CIK{}"
    ciks = []
    with open('tickerToCik.txt', 'r') as f:
        lines = f.readlines()
        for l in lines:
            arr = l.split()
            ticker = arr[0]
            cik = (str)(arr[1]).zfill(10)
            ciks.append(cik)
    
    requestURLs = [base_url.format(c) for c in ciks]

    return requestURLs

# Input: None
# Output: array of query objects for the sec_api QueryAPI object to pull SEC filing URLs
def prepSECRequests():
    ciks = []
    with open('tickerToCik.txt', 'r') as f:
        lines = f.readlines()
        for l in lines:
            arr = l.split()
            ticker = arr[0]
            cik = (str)(arr[1])
            # cik = (str)(arr[1]).zfill(10)
            ciks.append(cik)
    
    requestURLs = []
    # following line only for small batch testing
    ciks = ciks[:10]
    for c in ciks:
        query = {
            "query": f"cik:{c}", 
            "from": "0", # start returning matches from position null, i.e. the first matching filing 
            "size": "10",  # return just one filing
            "sort": [{ "filedAt": { "order": "desc" } }]
            }
        requestURLs.append(query)

    return requestURLs

# Input: single HTTP request
# Output: None
# Modifies: calls the response processing and storing functions to store files locally
def processEDGARRequest(r):
    # h = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:12.0) Gecko/20100101 Firefox/12.0'}
    edgar_resp = requests.get(r, headers={'User-Agent': "UMich EECS 486 Group 41 lisbonne@umich.edu"})
    processed_form = processForm(edgar_resp)
    storeForm(processed_form)

# Input: single SEC_API QueryApi request object
# Output: None
# Modifies: calls the response processing and storing functions to store files locally
def processSECRequest(q):
    api_key_file = os.path.join(os.getcwd(), 'data/sec_api_key.txt')
    with open(api_key_file, 'r') as f:
        key = f.readline()
        queryApi = QueryApi(api_key=key)

    response = queryApi.get_filings(q)
    filings = response['filings']
    # call process form to filter the retrieved text file and save only the helpful sections
    for f in filings:
        processed_response = processForm(f)

# Input: array of HTTP requests
# Output: files (either .zip or plain .txt) from SEC
# NOTE: inits a new thread to run code at intervals to avoid over-spamming the SEC api endpoints (limit @ 10/sec)
def threadRequests(allRequests, interval=0.5): 
    # setup a threading timer to run "processRequest()" every half second, iterating through the array of requests provided
    threading.Timer(interval, processSECRequest, allRequests).start()

# Input: grab and process SEC filing from url retrieved from SEC_API
# Output: parsed/cleaned version of the file for storage and later IR
def processForm(form): 
    link = form['linkToFilingDetails']
    # link = form['linkToHtml']
    new_filename = 'data/'
    new_filename += form['companyName']
    new_filename += form['filedAt']
    new_filename = new_filename.split(' ')
    new_filename = "".join(new_filename)
    response = requests.get(link, headers={'User-Agent': "UMich EECS 486 Group 41 lisbonne@umich.edu"})
    if response.status_code == 200:
        storeLinks(link)
        # pre-process the SEC html filing using bs4
        soup = BeautifulSoup(response.text, 'lxml')
        # store processed filing locally
        text = soup.get_text()

        # Convert to lowercase
        text = text.lower()

        # Tokenize the text using custom function
        tokens = tokenizeText(text)

        # Remove stopwords
        #stop_words = set(stopwords.words('english'))
        stop_words = set(["the", "and", "of"])
        tokens = [word for word in tokens if word not in stop_words]

        # Join tokens back into text
        preprocessed_text = ' '.join(tokens)

        storeForm(preprocessed_text, new_filename)


# Input: single file post parse, in the format of a dictionary: {'linkToFilingDetails': link to SEC file, 'content': the parsed BS4 content we want to save}
# Output: None
# Modifies: saves file into appropriate local folder
def storeForm(text, f_name): 
    # first need to check if the local "data" folder path is available
    if not os.path.exists(os.path.join(os.getcwd(), 'data')):
        return None
    # after confirming folder is present, save form
    f_name += '.txt'
    with open(f_name, 'w+') as f_out:
        f_out.write(text)
    print('finished storing form in /data')

# Input: sec document link
# Output: None
# Modifies: stores a visited SEC filing link in the designated 'links.txt' file locally
def storeLinks(sec_link): 
    sec_link += '\n'
    f_name = os.path.join(os.getcwd(), 'links.txt')
    if not os.path.exists(f_name):
        # logger.info("POTENTIAL ERROR: Can't find local links.txt file, creating a new file for future reference and storage of visted filing links.")
        with open(f_name, 'x') as f_out:
            f_out.write(sec_link)
    else:
        with open(f_name, 'a') as f_out:
            f_out.write(sec_link)
        
# Input: None
# Output: filenames for all local files to be retrieved
# NOTE: this function is intended to be imported by "solution.py" to aid in reading in files for pre-processing work
def loadForms(): 
    if not checkLocalDir():
        logging.info("WARNING: No local data directory found. Cannot load local files.")
        return None
    path = os.path.join(os.getcwd(), 'data')
    filenames = os.listdir(path)
    return filenames

def loadAllData(): 
    # first check local directories
    if not checkLocalDir():
        # create new local dir if needed
        setupLocalDir()

    # next, prepare all http request urls
    all_requests = prepSECRequests()

    # start the threaded timer for processing the requests
    # threadRequests(all_requests)
    # ^ this will start calling all the requests individually every half second, and should terminate when the args iterable is done
    for r in all_requests:
        processSECRequest(r)
        print("processed another one")


loadAllData()
