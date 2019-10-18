from roberta_models import RobertaSpanPredictionModel
from span_prediction_ropes import SpanPredictionRopesPredictor
from transformer_span_prediction import TransformerSpanPredictionReader

import argparse
import json
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from tqdm import tqdm

if __name__ == "__main__":
    # Parse arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("--archive_file", type = str, required = True,
                        help = "URL for a trained model file")
    parser.add_argument("--input_file", type=str, required=True,
                        help='path for drop input files')
    parser.add_argument("--output_file", type=str, required=True,
                        help="path for predictions output file")
    args = parser.parse_args()

    predictions = {}
    archive = load_archive(args.archive_file)
    predictor = Predictor.from_archive(archive, "span-prediction-ropes")
    input_json = json.load(open(args.input_file, encoding = "utf8"))

    for paragraph in tqdm(input_json['data'][0]['paragraphs']):
        background = paragraph['background']
        situation= paragraph['situation']

        for qa_pair in paragraph['qas']:
            question = qa_pair['question']
            question_id = qa_pair['id']
            predictions[question_id] = predictor.predict(question, background, situation)['best_span_str']

    # Write output file
    with open(args.output_file, "w", encoding = "utf8") as fout:
        json.dump(predictions, fout)
