import os
import logging
import torch
import time
from . import helper as he
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers.data.processors.squad import SquadResult, SquadV2Processor, SquadExample
from transformers.data.metrics.squad_metrics import compute_predictions_logits

# Define model - just uncomment the model you want to use
model_type = "bert" # may be bert, roberta, albert, distilbert, see helper.py
model_name_or_path = "deepset/bert-large-uncased-whole-word-masking-squad2"

# Set the flag for deployment
if os.path.exists(f'./models/{model_type}/config.json'):
    # Load supported models
    config_class, model_class, tokenizer_class, squad_convert_examples_to_features = he.load_models(model_name_or_path)
    model_name_or_path = f'./models/{model_type}/'
    logging.warning(f'[INFO] - Loading local model {model_type}.')
else:
    # Load supported models
    config_class, model_class, tokenizer_class, squad_convert_examples_to_features = he.load_models(model_name_or_path)
    logging.warning(f'[INFO] - Loading remote model {model_type}.')

# Config
n_best_size = 1
max_answer_length = 50
do_lower_case = True
null_score_diff_threshold = 0.0

def to_list(tensor):
    return tensor.detach().cpu().tolist()

# Setup model
config = config_class.from_pretrained(model_name_or_path)
tokenizer = tokenizer_class.from_pretrained(
    model_name_or_path, do_lower_case=do_lower_case)
model = model_class.from_pretrained(model_name_or_path, config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

processor = SquadV2Processor()

def run_prediction(question_text, context_texts):
    """
    Setup function to compute predictions
    """
    examples = []

    for i, context_text in enumerate(context_texts):
        example = SquadExample(
            qas_id=str(i),
            question_text=question_text,
            context_text=context_text,
            answer_text=None,
            start_position_character=None,
            title="Predict",
            is_impossible=False,
            answers=None,
        )
        examples.append(example)

    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=64,
        is_training=False,
        return_dataset="pt",
        threads=1,
    )

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=10)

    all_results = []

    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                #"token_type_ids": batch[2], #TODO: had to comment this?
            }

            example_indices = batch[3]

            outputs = model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs]

                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)

    # Output files are optional, may help debugging/learning locally
    # output_prediction_file = "/tmp/predictions.json"
    # output_nbest_file = "/tmp/nbest_predictions.json"
    # output_null_log_odds_file = "/tmp/null_predictions.json"

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        n_best_size,
        max_answer_length, # TODO
        do_lower_case,
        None,
        None,
        None,
        False, # verbose_logging
        True,  # version_2_with_negative
        null_score_diff_threshold,
        tokenizer,
    )

    return predictions

def main(question, documents, meta):
    # Run method
    _predictions = run_prediction(question, documents)
    logging.info(_predictions)
    predictions = []
    for prediction, m in zip(_predictions, meta):
        _prediction = _predictions[prediction]
        if _prediction != "" and _prediction != "empty":
            predictions.append(dict(
                answer = _prediction,
                title = m['title'],
                metadata_storage_name = m['metadata_storage_name'],
                document_id = m['document_id'],
                document_uri = m['document_uri']
            ))
    return predictions

if __name__ ==  '__main__':
    # Test question + context
    contexts = ["New Zealand (Māori: Aotearoa) is a sovereign island country in the southwestern Pacific Ocean. It has a total land area of 268,000 square kilometres (103,500 sq mi), and a population of 4.9 million. New Zealand's capital city is Wellington, and its most populous city is Auckland."]
    question = "How many people live in New Zealand?"
    print(main(question, contexts, [""]))