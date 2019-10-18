from typing import Dict, List, Any
import collections
import itertools
import json
import logging
import numpy
import os
import re

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import JsonDict
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField
from allennlp.data.fields import MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# WIP: Note this dataset reader has partially implemented parameters, inherited from earlier readers
# Much of the code is modified from pytorch-transformer examples for SQuAD
# The treatment of tokens and their string positions is a bit of a mess

@DatasetReader.register("transformer_span_prediction")
class TransformerSpanPredictionReader(DatasetReader):
    """

    """

    def __init__(self,
                 pretrained_model: str,
                 max_pieces: int = 512,
                 syntax: str = "squad",
                 skip_id_regex: str = None,
                 add_prefix: Dict[str, str] = None,
                 ignore_main_context: bool = False,
                 ignore_situation_context: bool = False,
                 dataset_dir_out: str = None,
                 model_type: str = None,
                 doc_stride: int = 100,
                 is_training: bool = True,
                 context_selection: str = "first",
                 answer_can_be_in_question: bool = None,
                 do_lowercase: bool = None,
                 sample: int = -1) -> None:
        super().__init__()
        if do_lowercase is None:
            do_lowercase = '-uncased' in pretrained_model
        self._tokenizer = PretrainedTransformerTokenizer(pretrained_model,
                                                         do_lowercase=do_lowercase,
                                                         start_tokens = [],
                                                         end_tokens = [])
        self._tokenizer_internal = self._tokenizer._tokenizer
        token_indexer = PretrainedTransformerIndexer(pretrained_model, do_lowercase=do_lowercase)
        self._token_indexers = {'tokens': token_indexer}

        self._max_pieces = max_pieces
        self._sample = sample
        self._syntax = syntax
        self._skip_id_regex = skip_id_regex
        self._ignore_main_context = ignore_main_context
        self._ignore_situation_context = ignore_situation_context
        self._dataset_dir_out = dataset_dir_out
        self._model_type = model_type
        self._add_prefix = add_prefix or {}
        self._doc_stride = doc_stride
        self._answer_can_be_in_question = answer_can_be_in_question
        if self._answer_can_be_in_question is None:
            self._answer_can_be_in_question = syntax == "ropes"
        self._allow_no_answer = None
        self._is_training = is_training
        self._context_selection = context_selection
        self._global_debug_counters = {"best_window": 5}
        if model_type is None:
            for model in ["roberta", "bert", "openai-gpt", "gpt2", "transfo-xl", "xlnet", "xlm"]:
                if model in pretrained_model:
                    self._model_type = model
                    break

    @overrides
    def _read(self, file_path: str):
        self._dataset_cache = None
        if self._dataset_dir_out is not None:
            self._dataset_cache = []
        instances = self._read_internal(file_path)
        if self._dataset_cache is not None:
            if not isinstance(instances, list):
                instances = [instance for instance in Tqdm.tqdm(instances)]
            if not os.path.exists(self._dataset_dir_out):
                os.mkdir(self._dataset_dir_out)
            output_file = os.path.join(self._dataset_dir_out, os.path.basename(file_path))
            logger.info(f"Saving contextualized dataset to {output_file}.")
            with open(output_file, 'w') as file:
                for d in self._dataset_cache:
                    file.write(json.dumps(d))
                    file.write("\n")
        return instances

    def _read_internal(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        logger.info("Reading instances from jsonl dataset at: %s", file_path)
        examples = self._read_squad_examples(file_path)
        debug = 5

        for example in examples:
            debug -= 1
            if debug > 0:
                logger.info(example)
            yield self._example_to_instance(
                example=example,
                debug=debug)

    @overrides
    def text_to_instance(self,  # type: ignore
                         question: str,
                         background: str,
                         situation: str = None,
                         item_id: str = "NA",
                         debug: int = -1) -> Instance:
        # For use by predictor, does not support answer input atm
        paragraph_text = self._add_prefix.get("c", "") + background
        if self._ignore_main_context:
            paragraph_text = ""
        if situation and not self._ignore_situation_context:
            situation_context = self._add_prefix.get("s", "") + situation
            paragraph_text = paragraph_text + " " + situation_context
        question_text = self._add_prefix.get("q", "") + question
        # We're splitting into subtokens later anyway
        doc_tokens = [paragraph_text]
        question_tokens = [question_text]

        example = SpanPredictionExample(
            qas_id=item_id,
            doc_text=paragraph_text,
            question_text=question_text,
            doc_tokens=doc_tokens,
            question_tokens=question_tokens)
        return self._example_to_instance(example, debug)


    def _example_to_instance(self, example, debug):
        fields: Dict[str, Field] = {}
        features = self._transformer_features_from_example(example, debug)
        tokens_field = TextField(features.tokens, self._token_indexers)
        segment_ids_field = SequenceLabelField(features.segment_ids, tokens_field)

        fields['tokens'] = tokens_field
        fields['segment_ids'] = segment_ids_field

        metadata = {
            "id": features.unique_id,
            "question_text": example.question_text,
            "tokens": [x.text for x in features.tokens],
            "context_full": example.doc_text,
            "answer_texts": example.all_answer_texts,
            "answer_mask": features.p_mask
        }

        if features.start_position is not None:
            fields['start_positions'] = LabelField(features.start_position, skip_indexing=True)
            fields['end_positions'] = LabelField(features.end_position, skip_indexing=True)
            metadata['start_positions'] = features.start_position
            metadata['end_positions'] = features.end_position

        if debug > 0:
            logger.info(f"tokens = {features.tokens}")
            logger.info(f"segment_ids = {features.segment_ids}")
            logger.info(f"context = {example.doc_text}")
            logger.info(f"question = {example.question_text}")
            logger.info(f"answer_mask = {features.p_mask}")
            if features.start_position is not None and features.start_position >= 0:
                logger.info(f"start_position = {features.start_position}")
                logger.info(f"end_position = {features.end_position}")
                logger.info(f"orig_answer_text   = {example.orig_answer_text}")
                answer_text = self._string_from_tokens(features.tokens[features.start_position:(features.end_position + 1)])
                logger.info(f"answer from tokens = {answer_text}")

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    def _read_squad_examples(self, input_file):
        """Read a SQuAD-format json file into a list of SpanPredictionExample."""
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]

        version_2_with_negative = self._allow_no_answer

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                if self._syntax == "squad":
                    paragraph_text = paragraph["context"]
                elif self._syntax == "ropes":
                    paragraph_text = paragraph["background"]
                else:
                    raise ValueError(f"Invalid dataset syntax {self._syntax}!")

                paragraph_text = self._add_prefix.get("c", "") + paragraph_text
                if self._ignore_main_context:
                    paragraph_text = ""
                if self._syntax == "ropes" and not self._ignore_situation_context:
                        situation_text = paragraph["situation"]
                        situation_text = self._add_prefix.get("s", "") + situation_text
                        paragraph_text = paragraph_text + " " + situation_text

                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    question_text = self._add_prefix.get("q", "") + question_text
                    question_tokens = []
                    char_to_word_offset_question = []
                    prev_is_whitespace = True
                    for c in question_text:
                        if is_whitespace(c):
                            prev_is_whitespace = True
                        else:
                            if prev_is_whitespace:
                                question_tokens.append(c)
                            else:
                                question_tokens[-1] += c
                            prev_is_whitespace = False
                        char_to_word_offset_question.append(len(question_tokens) - 1)

                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    is_impossible = False
                    all_answer_texts = None
                    answer_in_passage = True
                    if "answers" in qa:
                        answer_in_passage = True
                        all_answer_texts = [a["text"] for a in qa["answers"]]
                        if version_2_with_negative:
                            is_impossible = qa.get("is_impossible")
                        if not version_2_with_negative or not is_impossible:
                            answer = qa["answers"][0]   # Use first answer for span labeling
                            orig_answer_text = answer["text"]
                            answer_offset = answer.get("answer_start")
                            answer_length = len(orig_answer_text)
                            start_position = 0
                            end_position = 0
                            if answer_offset is not None:
                                start_position = char_to_word_offset[answer_offset]
                                end_position = char_to_word_offset[answer_offset + answer_length - 1]
                            else:
                                # Have to find the answer, as in ROPES, for now looks for last occurrence,
                                # including in question (which comes last in full input)
                                answer_offset = _find_last_substring_index(orig_answer_text, question_text)
                                if answer_offset is not None:
                                    answer_in_passage = False
                                    start_position = char_to_word_offset_question[answer_offset]
                                    end_position = char_to_word_offset_question[answer_offset + answer_length -1]
                                else:
                                    answer_offset = _find_last_substring_index(orig_answer_text, paragraph_text)
                                    if answer_offset is not None:
                                        start_position = char_to_word_offset[answer_offset]
                                        end_position = char_to_word_offset[answer_offset + answer_length - 1]

                                if answer_offset is None:
                                    logger.warning(f"Couldn't find answer '{orig_answer_text}' in " +
                                                   f"question '{question_text}' or passage '{paragraph_text}'")
                                    if self._is_training:
                                        continue

                            # Only add answers where the text can be exactly recovered from the
                            # document. If this CAN'T happen it's likely due to weird Unicode
                            # stuff so we will just skip the example.
                            #
                            # Note that this means for training mode, every example is NOT
                            # guaranteed to be preserved.
                            if answer_in_passage:
                                actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                                cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
                                if actual_text.find(cleaned_answer_text) == -1:
                                    logger.warning("Could not find answer: '%s' vs. '%s'",
                                                   actual_text, cleaned_answer_text)
                                    if self._is_training:
                                        continue
                        else:
                            start_position = -1
                            end_position = -1
                            orig_answer_text = ""

                    example = SpanPredictionExample(
                        qas_id=qas_id,
                        doc_text=paragraph_text,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        question_tokens=question_tokens,
                        answer_in_passage=answer_in_passage,
                        orig_answer_text=orig_answer_text,
                        all_answer_texts=all_answer_texts,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=is_impossible)
                    examples.append(example)
                    if self._sample > 0 and len(examples) > self._sample:
                        return examples
        return examples

    @staticmethod
    def _truncate_tokens(context_tokens, question_tokens, choice_tokens, max_length):
        """
        Truncate context_tokens first, from the left, then question_tokens and choice_tokens
        """
        max_context_len = max_length - len(question_tokens) - len(choice_tokens)
        if max_context_len > 0:
            if len(context_tokens) > max_context_len:
                context_tokens = context_tokens[-max_context_len:]
        else:
            context_tokens = []
            while len(question_tokens) + len(choice_tokens) > max_length:
                if len(question_tokens) > len(choice_tokens):
                    question_tokens.pop(0)
                else:
                    choice_tokens.pop()
        return context_tokens, question_tokens, choice_tokens

    def _improve_answer_span(self, doc_tokens, input_start, input_end,
                             orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""

        # The SQuAD annotations are character based. We first project them to
        # whitespace-tokenized words. But then after WordPiece/etc tokenization, we can
        # often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).". However
        # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
        # the exact answer, 1895.
        #
        # However, this is not always possible. Consider the following:
        #
        #   Question: What country is the top exporter of electornics?
        #   Context: The Japanese electronics industry is the lagest in the world.
        #   Answer: Japan
        #
        # In this case, the annotator chose "Japan" as a character sub-span of
        # the word "Japanese". Since our WordPiece tokenizer does not split
        # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
        # in SQuAD, but does happen.
        tok_answer_text = self._string_from_tokens(self._tokenizer.tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = self._string_from_tokens(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)

    def _string_from_tokens(self, tokens):
        tokens_text = [x.text for x in tokens]
        if hasattr(self._tokenizer_internal, "convert_tokens_to_string"):
            return self._tokenizer_internal.convert_tokens_to_string(tokens_text)
        else:
            return " ".join(tokens_text)

    def _transformer_features_from_example(self, example, debug):

        cls_token = Token(self._tokenizer_internal.cls_token)
        sep_token = Token(self._tokenizer_internal.sep_token)
        cls_token_at_end = bool(self._model_type in ['xlnet'])
        cls_token_segment_id = 2 if self._model_type in ['xlnet'] else 0
        sequence_a_segment_id = 0
        sequence_b_segment_id = 1

        has_answer = example.orig_answer_text and not example.is_impossible

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = self._tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        q_tok_to_orig_index = []
        q_orig_to_tok_index = []
        all_query_tokens = []
        for (i, token) in enumerate(example.question_tokens):
            q_orig_to_tok_index.append(len(all_query_tokens))
            sub_tokens = self._tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                q_tok_to_orig_index.append(i)
                all_query_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if has_answer and example.answer_in_passage:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = self._improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position,
                example.orig_answer_text)
        elif has_answer and not example.answer_in_passage:
            tok_start_position = q_orig_to_tok_index[example.start_position]
            if example.end_position < len(example.question_tokens) - 1:
                tok_end_position = q_orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_query_tokens) - 1
            (tok_start_position, tok_end_position) = self._improve_answer_span(
                all_query_tokens, tok_start_position, tok_end_position,
                example.orig_answer_text)

        if len(all_query_tokens) > self._max_pieces:
            all_query_tokens = all_query_tokens[0:self._max_pieces]

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = self._max_pieces - len(all_query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, self._doc_stride)

        features_list = []

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = []

            # CLS token at the beginning
            if not cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = 0

            # Paragraph
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(0)
            paragraph_len = doc_span.length

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)

            # Query
            query_p_mask = 0 if self._answer_can_be_in_question else 1
            query_offset = len(tokens)
            for token in all_query_tokens:
                tokens.append(token)
                segment_ids.append(sequence_b_segment_id)
                p_mask.append(query_p_mask)

            # SEP token - won't worry about two tokens for Roberta here
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)

            # CLS token at the end
            if cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = len(tokens) - 1  # Index of classification token

            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None
            if has_answer and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if example.answer_in_passage and not (tok_start_position >= doc_start and
                                                              tok_end_position <= doc_end):
                    out_of_span = True
                if not example.answer_in_passage and tok_end_position >= len(all_query_tokens):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    if example.answer_in_passage:
                        doc_offset = 1
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset
                    else:
                        start_position = tok_start_position + query_offset
                        end_position = tok_end_position + query_offset

            if span_is_impossible:
                start_position = cls_index
                end_position = cls_index

            if debug > 100:
                logger.info("*** Example ***")
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % tokens)
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if span_is_impossible:
                    logger.info("impossible example")
                if has_answer and not span_is_impossible:
                    answer_text = self._string_from_tokens(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(f"answer from tokens: '{answer_text}'")

            features_list.append(
                InputFeatures(
                    unique_id=f"{example.qas_id}-{doc_span_index}",
                    example_index=None,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=None,
                    input_mask=None,
                    segment_ids=segment_ids,
                    cls_index=cls_index,
                    p_mask=p_mask,
                    paragraph_len=paragraph_len,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=span_is_impossible))
        # Just filter away impossible/missing spans for now (this uses labels, so not fair on dev/test):
        if example.orig_answer_text and self._is_training:
            features_list = list(filter(lambda x: not x.is_impossible and x.start_position is not None, features_list))
        if not self._is_training and len(features_list) > 1:
            # If we're not creating training data, just pick the first/last context window
            if self._context_selection == "last":
                features_list = features_list[-1:]
            else:
                features_list = features_list[:1]
        if len(features_list) > 1:
            top_score = -1
            # For now we'll just keep the first/last context which has the most answer tokens
            selected = None
            for f in features_list:
                if f.start_position is not None and f.end_position is not None:
                    score = sum(10 + 0 * int(f.token_is_max_context.get(i, False)) for i in range(f.start_position, f.end_position) )
                    if score > top_score or (self._context_selection == "last" and score >= top_score):
                        top_score = score
                        selected = f
            if self._global_debug_counters["best_window"] > 0:
                self._global_debug_counters["best_window"] -= 1
                logger.info(f"For answer '{example.orig_answer_text}', picked \n{self._string_from_tokens(selected.tokens)} "+
                f"\nagainst \n{[self._string_from_tokens(x.tokens) for x in features_list]} ")
        elif len(features_list) == 1:
            selected = features_list[0]
        else:
            selected = None
        return selected


class SpanPredictionExample(object):
    """
    A single training/test example for a span prediction dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 doc_text,
                 question_text,
                 doc_tokens,
                 question_tokens,
                 answer_in_passage=True,
                 orig_answer_text=None,
                 all_answer_texts=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_text = doc_text
        self.doc_tokens = doc_tokens
        self.question_tokens = question_tokens
        self.answer_in_passage = answer_in_passage
        self.orig_answer_text = orig_answer_text
        self.all_answer_texts = all_answer_texts
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "  qas_id: %s" % (self.qas_id)
        s += "\n  question_text: %s" % (
            self.question_text)
        s += "\n  doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        s += "\n  question_tokens: [%s]" % (" ".join(self.question_tokens))
        if self.orig_answer_text:
            s += f"\n  orig_answer_text: {self.orig_answer_text}"
        if self.start_position:
            s += "\n  start_position: %d" % (self.start_position)
        if self.end_position:
            s += "\n  end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += "\n  is_impossible: %r" % (self.is_impossible)
        return s

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_index,
                 p_mask,
                 paragraph_len,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def _find_last_substring_index(pattern_string, string):
    if not pattern_string:
        return None
    regex = re.escape(pattern_string)
    if pattern_string[0].isalpha() or pattern_string[0].isdigit():
        regex = "\\b" + regex
    if pattern_string[-1].isalpha() or pattern_string[-1].isdigit():
        regex = regex + "\\b"
    res = [match.start() for match in re.finditer(regex, string)]
    if len(res) == 0:
        regex_uncased = "(?i)" + regex
        res = [match.start() for match in re.finditer(regex_uncased, string.lower())]
    return res[-1] if res else None


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index
