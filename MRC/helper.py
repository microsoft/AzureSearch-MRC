import logging
import sys

def main():
    return None

def get_config(req):
    '''
    Get config and set parameters
    '''
    # Assuming that we receive the params via url
    question = req.params.get('question')
    n_doc = req.params.get('az_documents')
    treshold = req.params.get('az_treshold')
    tokenize = req.params.get('az_tokenize')
    bm_n_doc = req.params.get('bm_ndoc')
    # We check, whether we have received a value for every parameter
    if not any([question, n_doc, treshold, tokenize, bm_n_doc]):
        # Otherwise we try the request body
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            question = req_body.get('question')
            n_doc = req_body.get('az_documents')
            treshold = req_body.get('az_treshold')
            tokenize = req_body.get('az_tokenize')
            bm_n_doc = req_body.get('bm_ndoc')
    # Now we check whether we have everything - if we have some missings, we just set the defaults
    if not all([question, n_doc, treshold, tokenize, bm_n_doc]):
        logging.warning('Received one or multiple empty parameters, filling up with default values')
        if not n_doc:
            n_doc = 5
        if not treshold:
            treshold = 5
        if not bm_n_doc:
            bm_n_doc = 3
        if str(tokenize).lower() == "false":
            tokenize = False
        else:
            tokenize = True
    elif all([question, n_doc, treshold, tokenize, bm_n_doc]):
        pass
    logging.warning(f'[INFO] - Working with following parameters: n_doc: {n_doc}, treshold: {treshold}, bm_n_doc: {bm_n_doc}, tokenize: {tokenize}.')
    return question, int(n_doc), int(treshold), tokenize, int(bm_n_doc)

def load_models(model_name_or_path):
    '''
    Load models, tokenizers and configuration
    Pick one from the models below and set in in reader.py:
        - model_name_or_path = "deepset/bert-large-uncased-whole-word-masking-squad2"
        - model_name_or_path = "deepset/roberta-base-squad2"
        - model_name_or_path = "twmkn9/albert-base-v2-squad2"
        - model_name_or_path = "distilbert-base-cased-distilled-squad"
    '''
    # 
    if model_name_or_path == "twmkn9/albert-base-v2-squad2":
        from transformers import (
            AlbertConfig,
            AlbertForQuestionAnswering,
            AlbertTokenizer,
            squad_convert_examples_to_features
        )
        config_class, model_class, tokenizer_class = (
            AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer)
    elif model_name_or_path == "deepset/bert-large-uncased-whole-word-masking-squad2":
        from transformers import (
            BertConfig,
            BertForQuestionAnswering,
            BertTokenizer,
            squad_convert_examples_to_features
        )
        config_class, model_class, tokenizer_class = (
            BertConfig, BertForQuestionAnswering, BertTokenizer)
    elif model_name_or_path == "distilbert-base-cased-distilled-squad":
        from transformers import (
            DistilBertConfig,
            DistilBertForQuestionAnswering,
            DistilBertTokenizer,
            squad_convert_examples_to_features
        )
        config_class, model_class, tokenizer_class = (
            DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer)
    elif model_name_or_path == "deepset/roberta-base-squad2":
        from transformers import (
            RobertaConfig,
            RobertaForQuestionAnswering,
            RobertaTokenizer,
            squad_convert_examples_to_features
        )
        config_class, model_class, tokenizer_class = (
            RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer)
    else:
        logging.error(f'Model {model_name_or_path} is not available!')
        sys.exit()
    logging.info(f'Loaded {model_name_or_path} ...')
    return config_class, model_class, tokenizer_class, squad_convert_examples_to_features

if __name__ == "__main__":
    main()