import logging
import json
import azure.functions as func

from . import bm25
from . import retriever
from . import reader
from . import helper

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # Collect parameters
    question, n_doc, treshold, tokenize, bm_n_doc = helper.get_config(req)

    if question:
        # Fetch relevant documents, if any
        documents, sources = retriever.main(question, n_doc, treshold, tokenize)

        # Apply BM25, if more than n documents
        if documents and len(documents) > bm_n_doc:
            documents, sources = bm25.main(question, documents, sources, bm_n_doc)

        # Extract relevant answers, if any
        if documents:
            answers = reader.main(question, documents, sources)
        else:
            answers = []

        # Format response
        res = json.dumps(dict(
            answers = answers,
            counts = dict(
                documents = len(documents),
                answers = len(answers)
            )
        ))
        return func.HttpResponse(res, mimetype='application/json')
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a question in the query string or in the request body for a personalized response.",
             status_code=200
        )
