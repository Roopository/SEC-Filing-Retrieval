# SEC Filing IR System



### `Goal & Objectives`

The primary goal of this project is to develop an advanced Q&A system that extracts specific financial information from the SEC filings of public companies, enhancing the capability of investors, analysts, and researchers to efficiently sift through large volumes of financial data for more informed decision-making. The objectives include indexing thousands of SEC filings across various form types to establish a comprehensive database, and designing a sophisticated search system that accurately processes and responds to user queries by pinpointing relevant information within this database. Additionally, the system will return relevant files, summaries, and visualizations to aid users in making better investment decisions.

### `Getting Started`
1. **Start the Program**:
Run the script that contains the main function. Ensure that the documents variable is populated with your dataset of SEC filings. `TODO change the document variable part based on David's work`

2. **Enter a Query:**
Once the program is running, you will be prompted with: "Enter your query or type 'TERMINATE' to exit:". Input your query based on the information you are seeking from the SEC filings. The query can be as simple or specific as needed, and you can include a year to filter documents from a specific time frame.

3. **Query Processing:**
The system first checks if the query contains the term 'TERMINATE'. If found, the program will terminate.
Otherwise, the query undergoes simple normalization by splitting it into words.

4. **Document Processing and Ranking:**
The filtered documents are then preprocessed and ranked based on their relevance to the query. This involves several steps such as boolean search for filtering, preprocessing the content of the documents, and applying ranking algorithms to determine the most relevant documents.

5. **View Summaries:**
For each of the top-ranked documents, a summary is generated. The system then displays the beginning of each document followed by its respective summary to provide a concise overview of the content.

6. **Repeat or Terminate:**
After displaying the summaries, the system will prompt you again for a new query. You can continue to input new queries to retrieve different information, or type 'TERMINATE' to end the session.


###  `Methods & Descriptions`

### ```VSM.py```
Here's a detailed documentation of the methods in the Python script, providing an overview of each function's purpose, parameters, and return values:

### 1. preprocess_text
```python
def preprocess_text(text):
```
**Purpose:** Processes a given text by removing SGML tags, tokenizing, and stemming the words.

**Parameters:**
- `text` (str): The text to be processed.

**Returns:**
- `List[str]`: A list of stemmed tokens from the input text.

### 2. indexDocument
```python
def indexDocument(doc_id, doc_content, doc_weight_scheme, query_weight_scheme, inverted_index, doc_length, df_counts, total_docs):
```
**Purpose:** Indexes a single document, calculating term frequencies and updating the inverted index and document frequencies.

**Parameters:**
- `doc_id` (str): Unique identifier for the document.
- `doc_content` (str): The content of the document.
- `doc_weight_scheme` (str): The weighting scheme used for document vectors.
- `query_weight_scheme` (str): The weighting scheme used for query vectors (needed for IDF calculations).
- `inverted_index` (dict): The inverted index being built.
- `doc_length` (dict): Dictionary to store the length of each document.
- `df_counts` (dict): Dictionary to track document frequencies of terms.
- `total_docs` (int): Total number of documents processed so far.

**Returns:**
- None. Modifies `inverted_index`, `doc_length`, and `df_counts` in place.

### 3. retrieveDocuments
```python
def retrieveDocuments(query, inverted_index, df_counts, total_docs, doc_length, query_weight_scheme, doc_weight_scheme):
```
**Purpose:** Retrieves documents that are relevant to a given query using a vector space model.

**Parameters:**
- `query` (str): The search query.
- `inverted_index` (dict): The inverted index of documents.
- `df_counts` (dict): Document frequency counts for terms.
- `total_docs` (int): Total number of documents indexed.
- `doc_length` (dict): A dictionary with lengths of each document.
- `query_weight_scheme` (str): Weighting scheme for the query.
- `doc_weight_scheme` (str): Weighting scheme for the documents (not used directly in this function).

**Returns:**
- `Dict[int, float]`: A dictionary with document IDs as keys and their cosine similarity scores as values.

### 4. rocchio_update
```python
def rocchio_update(query_vector, relevant_docs, irrelevant_docs, inverted_index, alpha=1.0, beta=0.75, gamma=0.5):
```
**Purpose:** Updates the query vector using the Rocchio algorithm to reflect user feedback on document relevance.

**Parameters:**
- `query_vector` (dict): The original query vector.
- `relevant_docs` (list): List of document IDs deemed relevant by the user.
- `irrelevant_docs` (list): List of document IDs deemed irrelevant by the user.
- `inverted_index` (dict): The inverted index of documents.
- `alpha`, `beta`, `gamma` (float): Parameters controlling the influence of the original query, relevant documents, and irrelevant documents, respectively.

**Returns:**
- `Dict[str, float]`: A normalized updated query vector.

### 5. VSM
```python
def VSM(doc_weight_scheme, query_weight_scheme, docs_folder, query_file):
```
**Purpose:** Orchestrates the complete process of reading documents, indexing them, handling queries, and potentially refining queries based on feedback.

**Parameters:**
- `doc_weight_scheme` (str): The document weighting scheme.
- `query_weight_scheme` (str): The query weighting scheme.
- `docs_folder` (str): Path to the folder containing documents.
- `query_file` (str): Path to the file containing initial queries.

**Returns:**
- None. Outputs results directly to the user and may engage in interactive query refinement.

### 6. build_query_vector
```python
def build_query_vector(query, inverted_index, query_weight_scheme, total_docs, df_counts):
```
**Purpose:** Builds a query vector from a text string using specified weighting schemes.

**Parameters:**
- `query` (str): The search query.
- `inverted_index` (dict): The inverted index for reference (not directly used).
- `query_weight_scheme` (str): The scheme used to weight the query.
- `total_docs` (int): Total number of documents in the corpus.
- `df_counts` (dict): Document frequency counts for terms.

**Returns:**
- `Dict[str, float]`: A dictionary representing the query vector.

### 7. vector_to_query
```python
def vector_to_query(vector):
```
**Purpose:** Converts a vector representation back into a

 query string, primarily used after query refinement.

**Parameters:**
- `vector` (Dict[str, float]): The query vector to be converted.

**Returns:**
- `str`: A string that represents the query based on the vector's terms and their weights.

These detailed method documentations can be integrated into the project's developer documentation or used in code comments to improve code readability and maintenance.
