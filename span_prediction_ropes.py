# pylint: disable=protected-access
from copy import deepcopy
from typing import Dict, List

from overrides import overrides
import numpy

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import (IndexField, ListField, LabelField, SpanField, SequenceLabelField,
                                  SequenceField)


@Predictor.register('span-prediction-ropes')
class SpanPredictionRopesPredictor(Predictor):

    def predict(self, question: str, background: str, situation: str = None) -> JsonDict:
        return self.predict_json({"background": background, "question": question, "situation": situation})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"question": "...", "passage": "..."}``.
        """
        question_text = json_dict["question"]
        background = json_dict["background"]
        situation = json_dict.get("situation")

        return self._dataset_reader.text_to_instance(question_text, background, situation)

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output = self.predict_instance(instance)
        # outputs = self._model.forward_on_instance(instance)

        best_span = output['best_span']
        confidence = output['start_probs'][best_span[0]] * output['end_probs'][best_span[1]]

        output['confidence'] = confidence

        return output
