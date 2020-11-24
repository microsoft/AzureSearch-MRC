from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import logging
from nltk import sent_tokenize
import nltk.data
import nltk.downloader

try:
    nltk.data.path.append("./models/nltk/")
    logging.info('Importing nltk from local, specified folder.')
except:
    nltk.download('punkt')
finally:
    nltk.data.find('tokenizers/punkt')

# Local debugging
try:
    import sys
    import os
    sys.path.append('./')
    import configparser
    config = configparser.ConfigParser()
    config.read('config.ini')
    service_name = config['search']['service_name']
    index_name = config['search']['index_name']
    api_key = config['search']['key']
    logging.info('extracted keys from config')
except Exception as e:
    logging.warning(f'{e}')
    service_name = os.environ.get('service_name')
    index_name = os.environ.get('index_name')
    api_key = os.environ.get('api_key')
    logging.info('extracted keys from environment')

# Extract relevant text parts
def extract_relevant_text(highlight, paragraphs, tokenize):
    ''' Extracts relevant sentences around highlights to get optimized text and content
    This ensures that we have better data for the MRC scoring, as not only partial sentences are extracted'''
    if not isinstance(paragraphs, list):
        paragraphs = [paragraphs]
    # For every highlight, look up the paragraphs
    for paragraph in paragraphs:
        # If the highlight is in the respective paragraph, tokenize the sentences
        if tokenize:
            highlight = sent_tokenize(highlight)[0]
        if highlight in paragraph:
            sentences = sent_tokenize(paragraph)
            # Loop through every sentence and detect the highlight
            for sentence in sentences:
                # As soon as we found the highlight, extract the sentences around it
                if highlight in sentence:
                    if sentences.index(sentence) == 0:
                        return ". ".join(sentences[:2])
                    elif sentences.index(sentence) == len(sentences) - 1:
                        return ". ".join(sentences[-2:])
                    else:
                        return ". ".join(sentences[sentences.index(sentence)-1:sentences.index(sentence)+1])

# Request to Cognitive Search
def main(question, n=5, threshold=5, tokenize=True):
    # Create a SearchClient to send queries
    endpoint = f"https://{service_name}.search.windows.net/"
    credential = AzureKeyCredential(api_key)
    client = SearchClient(endpoint=endpoint,
                        index_name=index_name,
                        credential=credential)

    # Get top n results
    results = client.search(search_text=question, search_fields='paragraphs', highlight_fields='paragraphs-3', select='paragraphs,metadata_storage_name,document_id,document_uri,title', top=n)
    documents = []
    meta = []
    for result in results:
        if result['@search.score'] > threshold:
            if len(result['paragraphs']) == 0: 
                continue
            search_highlights = list(dict.fromkeys(result['@search.highlights']['paragraphs']))
            for highlight in search_highlights:
                h = highlight.replace('<em>', '').replace('</em>', '')
                relevant_text = extract_relevant_text(h, result['paragraphs'], tokenize)
                if relevant_text is None:
                    logging.info("Text is none, continue")
                    continue
                # We only proceed with the first 500 characters due to MRC limitations
                elif relevant_text[:500] in documents:
                    logging.info("Text already exists, continue")
                    continue
                documents.append(relevant_text[:500])
                meta.append(dict(
                    metadata_storage_name = result['metadata_storage_name'],
                    document_id = result['document_id'],
                    document_uri = result['document_uri'],
                    title = result['title']
                ))
    return documents, meta

if __name__ == "__main__":
    main("Who is the Boss of Microsoft?")