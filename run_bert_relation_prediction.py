# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
from typing import Optional, Union, Tuple, List

import numpy as np
import torch
from flair.data import Sentence, Token
from flair.embeddings import StackedEmbeddings, WordEmbeddings, FlairEmbeddings, CharacterEmbeddings
from sklearn import metrics
from sklearn.metrics import f1_score
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import BertPreTrainedModel, BertModel
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.optimization import AdamW

# os.environ['CUDA_VISIBLE_DEVICES']= '1'
#torch.backends.cudnn.deterministic = True

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, input_sentence):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.input_sentence = input_sentence


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines



class KGProcessor(DataProcessor):
    """Processor for knowledge graph data set."""
    def __init__(self):
        self.labels = set()
    
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_dir)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", data_dir)

    def get_test_examples(self, data_dir):
      """See base class."""
      return self._create_examples(
          self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", data_dir)

    def get_relations(self, data_dir):
        """Gets all labels (relations) in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "relations.txt"), 'r') as f:
            lines = f.readlines()
            relations = []
            for line in lines:
                relations.append(line.strip())
        return relations

    def get_entities(self, data_dir):
        """Gets all entities in the knowledge graph."""
        # return list(self.labels)
        with open(os.path.join(data_dir, "entities.txt"), 'r') as f:
            lines = f.readlines()
            entities = []
            for line in lines:
                entities.append(line.strip())
        return entities

    def get_train_triples(self, data_dir):
        """Gets training triples."""
        return self._read_tsv(os.path.join(data_dir, "train.tsv"))

    def get_dev_triples(self, data_dir):
        """Gets validation triples."""
        return self._read_tsv(os.path.join(data_dir, "dev.tsv"))

    def get_test_triples(self, data_dir):
        """Gets test triples."""
        return self._read_tsv(os.path.join(data_dir, "test.tsv"))

    def _create_examples(self, lines, set_type, data_dir):
        """Creates examples for the training and dev sets."""
        # entity to text
        ent2text = {}
        with open(os.path.join(data_dir, "entity2text.txt"), 'r') as f:
            ent_lines = f.readlines()
            for line in ent_lines:
                temp = line.strip().split('\t')
                ent2text[temp[0]] = temp[1]

        if data_dir.find("FB15") != -1:
            with open(os.path.join(data_dir, "entity2text.txt"), 'r') as f:
                ent_lines = f.readlines()
                for line in ent_lines:
                    temp = line.strip().split('\t')
                    #first_sent_end_position = temp[1].find(".")
                    ent2text[temp[0]] = temp[1]#[:first_sent_end_position + 1]              

        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ent2text[line[0]]
            text_b = ent2text[line[2]]
            label = line[1]
            self.labels.add(label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class KGDataset(torch.utils.data.Dataset):
    def __init__(self, all_input_ids, input_sentences, all_input_mask, all_segment_ids, all_label_ids):
        self.all_input_ids = all_input_ids
        self.input_sentences = input_sentences
        # self.all_input_embeddings = all_input_embeddings
        self.all_input_mask = all_input_mask
        self.all_segment_ids = all_segment_ids
        self.all_label_ids = all_label_ids
    def __len__(self):
        return len(self.input_sentences)
    def __getitem__(self, item):
        return (self.all_input_ids[item], self.input_sentences[item],
                self.all_input_mask[item], self.all_segment_ids[item],
                self.all_label_ids[item])


class EmbeddingAdaptedBertClassifier(BertPreTrainedModel):
    def __init__(self, config, embeddings):
        super().__init__(config)
        self.embeddings = embeddings
        self.num_labels = config.num_labels
        self.config = config

        self.embedding_adapter = nn.Linear(self.embeddings.embedding_length, config.hidden_size);

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            selection: Tensor,
            input_sentences,
            embeddings_size: List[int],
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            # inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        self.embeddings.embed(list(input_sentences))
        inputs_embeds = torch.stack([torch.stack([t.embedding for t in s], dim=0) for s in input_sentences], dim=0).to(selection.device)

        if inputs_embeds is not None:
            embeddings_mask = torch.cat(
                [torch.ones(embeddings_size[i], device=selection.device) * selection[i] for i in
                 range(len(embeddings_size))], -1)
            selected_embeds = inputs_embeds * embeddings_mask
            inputs_embeds = self.embedding_adapter(selected_embeds)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class EmbedController(nn.Module):
    def __init__(
            self,
            num_actions,
            model_structure=None,
            state_size=20,
    ):
        super(EmbedController, self).__init__()
        self.previous_selection = None
        self.best_action = None
        self.num_actions = num_actions
        self.model_structure = model_structure
        self.state_size = state_size
        if self.model_structure is None:
            self.selector = nn.Parameter(
                torch.zeros(num_actions),
                requires_grad=True,
            )
        else:
            self.selector = torch.nn.Linear(state_size, num_actions)
            torch.nn.init.zeros_(self.selector.weight)
            torch.nn.init.zeros_(self.selector.bias)

    def sample(self, states=None, mask=None):
        value = self.get_value(states, mask)
        one_prob = torch.sigmoid(value)
        m = torch.distributions.Bernoulli(one_prob)
        selection = m.sample()
        # avoid all values are 0, or avoid the selection is the same as previous iteration in training
        if self.model_structure is None:
            while selection.sum() == 0 or (
                    self.previous_selection is not None and (self.previous_selection == selection).all()):
                selection = m.sample()
        else:
            for idx in range(len(selection)):
                while selection[idx].sum() == 0:
                    m_temp = torch.distributions.Bernoulli(one_prob[idx])
                    selection[idx] = m_temp.sample()

        log_prob = m.log_prob(selection)
        self.previous_selection = selection.clone()
        return selection, log_prob

    def forward(self, states=None, mask=None):
        value = self.get_value(states, mask)

        return torch.sigmoid(value)

    def get_value(self, states=None, mask=None):
        if self.model_structure is None:
            value = self.selector
        else:
            states = (states * mask.unsqueeze(-1)).sum(-2) / mask.sum(-1, keepdim=True)
            value = self.selector(states)
        return value


def save_ckpt(output_dir, model, tokenizer):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)

def create_dataloader(features, batch_size, sampler_class):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_sentences = [f.input_sentence for f in features]
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    def collate_fn(data):
        all_input_ids, all_input_sentences, all_input_mask, all_segment_ids, all_label_ids = zip(*data)
        all_input_ids = torch.stack(all_input_ids, dim=0)
        all_input_mask = torch.stack(all_input_mask, dim=0)
        all_segment_ids = torch.stack(all_segment_ids, dim=0)
        all_label_ids = torch.stack(all_label_ids, dim=0)
        return all_input_ids, all_input_sentences, all_input_mask, all_segment_ids, all_label_ids

    dataset = KGDataset(all_input_ids, all_input_sentences, all_input_mask, all_segment_ids, all_label_ids)
    sampler = sampler_class(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate_fn)
    return dataloader

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, print_info = True):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0 and print_info:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_sentence = Sentence(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        for _ in range(max_seq_length - len(input_ids)):
            input_sentence._add_token(Token("[PAD]"))
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        if ex_index < 5 and print_info:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              input_sentence=input_sentence,
                              ))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def _truncate_seq_triple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence triple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b) and len(tokens_a) > len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) > len(tokens_a) and len(tokens_b) > len(tokens_c):
            tokens_b.pop()
        elif len(tokens_c) > len(tokens_a) and len(tokens_c) > len(tokens_b):
            tokens_c.pop()
        else:
            tokens_c.pop()

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def simple_f1_score(preds, labels):
    return f1_score(preds, labels)

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "kg":
        return {"acc": simple_accuracy(preds, labels), "mavg": simple_f1_score(preds, labels)}
    else:
        raise KeyError(task_name)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "kg": KGProcessor,
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    args.seed = random.randint(1, 200)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_relations(args.data_dir)
    num_labels = len(label_list)

    entity_list = processor.get_entities(args.data_dir)
    #print(entity_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = 0
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))

    embeddings = StackedEmbeddings([
        WordEmbeddings('glove'),
        # CharacterEmbeddings(),
        # FlairEmbeddings('news-forward'),
        WordEmbeddings('glove'),
    ])

    embeddings_size = [emb.embedding_length for emb in embeddings.embeddings]

    model = EmbeddingAdaptedBertClassifier.from_pretrained(
        pretrained_model_name_or_path=args.bert_model,
        embeddings=embeddings,
        cache_dir=cache_dir,
        num_labels=num_labels,
    )

    controller = EmbedController(
        num_actions=len(embeddings.embeddings),
        state_size=embeddings.embedding_length,
    ).to(device)

    controller_learning_rate = 0.1

    controller_optimizer = Adam(controller.parameters(), lr=controller_learning_rate)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
        #model = torch.nn.parallel.data_parallel(model)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    num_warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False,)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=num_warmup_steps,
    #     num_training_steps=num_train_optimization_steps
    # )

    start_episode = 0
    max_episodes = 10
    baseline_score = 0

    action_dict = {}
    discount = 0.5

    train_features = convert_examples_to_features(
        train_examples, label_list, args.max_seq_length, tokenizer, embeddings)
    train_dataloader = create_dataloader(
        train_features,
        batch_size=args.train_batch_size,
        sampler_class=RandomSampler if args.local_rank == -1 else DistributedSampler,
    )

    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, args.max_seq_length, tokenizer, embeddings)
    eval_dataloader = create_dataloader(eval_features, batch_size=args.eval_batch_size, sampler_class=SequentialSampler)

    for episode in range(start_episode, max_episodes):
        logger.info(f"***** Start Episode {episode} *****")
        best_score = 0

        action, log_prob = controller.sample(None)
        selection = action

        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0
        if args.do_train:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)

            model.train()
            #print(model)
            for _ in trange(int(args.num_train_epochs), desc="Epoch"):
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    batch = tuple(t.to(device) if type(t) == Tensor else t for t in batch)
                    input_ids, input_sentences, input_mask, segment_ids, label_ids = batch

                    # define a new function to compute loss values for both output_modes
                    logits = model(input_sentences=input_sentences, selection=selection, embeddings_size=embeddings_size, token_type_ids=segment_ids, attention_mask=input_mask, labels=None).logits
                    #print(logits, logits.shape)

                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

                    # if n_gpu > 1:
                    #     loss = loss.mean() # mean() to average on multi-gpu.
                    # if args.gradient_accumulation_steps > 1:
                    #     loss = loss / args.gradient_accumulation_steps

                    loss.backward()

                    optimizer.step()
                    # scheduler.step()
                    # optimizer.zero_grad()

                    # tr_loss += loss.item()
                    # nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    global_step += 1
                print("Training loss: ", tr_loss, nb_tr_examples)

        if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            save_ckpt(args.output_dir, model, tokenizer)

            # Load a trained model and vocabulary that you have fine-tuned
            model = EmbeddingAdaptedBertClassifier.from_pretrained(args.output_dir, num_labels=num_labels, embeddings=embeddings)
            tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        else:
            model = EmbeddingAdaptedBertClassifier.from_pretrained(args.bert_model, num_labels=num_labels, embeddings=embeddings)
        model.to(device)

        if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)

            # Load a trained model and vocabulary that you have fine-tuned
            model = EmbeddingAdaptedBertClassifier.from_pretrained(args.output_dir, num_labels=num_labels, embeddings=embeddings)
            tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
            model.to(device)

            model.eval()
            eval_loss = 0
            nb_eval_steps = 0
            preds = []

            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = tuple(t.to(device) if type(t) == Tensor else t for t in batch)
                input_ids, input_sentences, input_mask, segment_ids, label_ids = batch

                with torch.no_grad():
                    logits = model(
                        input_sentences=input_sentences,
                        selection=selection,
                        embeddings_size=embeddings_size,
                        token_type_ids=segment_ids,
                        attention_mask=input_mask,
                        labels=None
                    ).logits

                # create eval loss and other metric required by the task
                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                print(label_ids.view(-1))

                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds[0] = np.append(
                        preds[0], logits.detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps
            preds = preds[0]
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
            preds = np.argmax(preds, axis=1)
            result = compute_metrics(task_name, preds, all_label_ids.numpy())
            loss = tr_loss/nb_tr_steps if args.do_train else None

            result['eval_loss'] = eval_loss
            result['global_step'] = global_step
            result['loss'] = loss

            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

            current_score = result['mavg'] * 100

            if current_score >= best_score:
                best_score = current_score

            if current_score >= baseline_score:
                logging.info(f"***** Saving the current overall best model, score: {current_score} *****")
                save_ckpt(os.path.join(args.output_dir, "best_model"), model, tokenizer)
                baseline_score = current_score

        logging.info(f"***** End episode {episode} *****")
        controller_optimizer.zero_grad()
        controller.zero_grad()
        if episode == 0:
            baseline_score = best_score
            logging.info(f"Setting baseline score to: {baseline_score}")

            best_action = action
            controller.best_action = action
            logging.info(f"Setting baseline action to: {best_action}")
        else:
            logging.info(f"previous distributions: ")
            print(controller(None))
            controller_loss = 0
            action_count = 0
            average_reward = 0
            reward_at_each_position = torch.zeros_like(action)
            count_at_each_position = torch.zeros_like(action)

            for prev_action in action_dict:
                reward = best_score - max(action_dict[prev_action]['scores'])
                prev_action = torch.Tensor(prev_action).type_as(action)

                reward = reward * (discount ** (torch.abs(action - prev_action).sum() - 1))
                average_reward += reward
                reward_at_each_position += reward * torch.abs(action - prev_action)
                count_at_each_position += torch.abs(action - prev_action)

                if torch.abs(action - prev_action).sum() > 0:
                    action_count += 1

            count_at_each_position[torch.where(count_at_each_position == 0)] += 1
            controller_loss -= (log_prob * reward_at_each_position).sum()

            controller_loss.backward()
            print("#=================")
            print(controller.selector)
            print(controller.selector.grad)
            controller_optimizer.step()
            print(controller.selector)
            print("#=================")

            logging.info(f"After distributions: ")
            print(controller(None))
            if best_score >= baseline_score:
                baseline_score = best_score
                best_action = action
                controller.best_action = action
                logging.info(f"Setting baseline score to: {baseline_score}")
                logging.info(f"Setting baseline action to: {best_action}")

            logging.info('=============================================')
            logging.info(f"Current Action: {action}")
            logging.info(f"Current best score: {best_score}")
            logging.info(f"Current total Reward: {average_reward}")
            logging.info(f"Current Reward at each position: {reward_at_each_position}")
            logging.info('=============================================')
            logging.info(f"Overall best Action: {best_action}")
            logging.info(f"Overall best score: {baseline_score}")
            logging.info(f"State dictionary: {action_dict}")
            logging.info('=============================================')

        curr_action = tuple(action.cpu().tolist())
        if curr_action not in action_dict:
            action_dict[curr_action] = {}
            action_dict[curr_action]['counts'] = 0
            action_dict[curr_action]['scores'] = []
        # self.action_dict[curr_action]['scores'].append(best_score)
        action_dict[curr_action]['counts'] += 1
        action_dict[curr_action]['scores'].append(best_score)

    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

            train_triples = processor.get_train_triples(args.data_dir)
            dev_triples = processor.get_dev_triples(args.data_dir)
            test_triples = processor.get_test_triples(args.data_dir)
            all_triples = train_triples + dev_triples + test_triples

            all_triples_str_set = set()
            for triple in all_triples:
                triple_str = '\t'.join(triple)
                all_triples_str_set.add(triple_str)

            eval_examples = processor.get_test_examples(args.data_dir)
            eval_features = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer, embeddings)
            logger.info("***** Running Prediction *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)

            eval_dataloader = create_dataloader(eval_features, batch_size=args.eval_batch_size, sampler_class=SequentialSampler)

            # Load a trained model and vocabulary that you have fine-tuned
            model = EmbeddingAdaptedBertClassifier.from_pretrained(os.path.join(args.output_dir, "best_model"), num_labels=num_labels, embeddings=embeddings)
            tokenizer = BertTokenizer.from_pretrained(os.path.join(args.output_dir, "best_model"), do_lower_case=args.do_lower_case)
            model.to(device)
            model.eval()
            eval_loss = 0
            nb_eval_steps = 0
            preds = []

            for batch in tqdm(eval_dataloader, desc="Testing"):
                batch = tuple(t.to(device) if type(t) == Tensor else t for t in batch)
                input_ids, input_sentences, input_mask, segment_ids, label_ids = batch

                with torch.no_grad():
                    logits = model(
                        input_sentences=input_sentences,
                        selection=selection,
                        embeddings_size=embeddings_size,
                        token_type_ids=segment_ids,
                        attention_mask=input_mask,
                        labels=None
                    ).logits

                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds[0] = np.append(
                        preds[0], logits.detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps
            preds = preds[0]
            print(preds, preds.shape)

            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

            ranks = []
            filter_ranks = []
            hits = []
            hits_filter = []
            for i in range(10):
                hits.append([])
                hits_filter.append([])

            for i, pred in enumerate(preds):
                rel_values = torch.tensor(pred)
                _, argsort1 = torch.sort(rel_values, descending=True)
                argsort1 = argsort1.cpu().numpy()

                rank = np.where(argsort1 == all_label_ids[i])[0][0]
                #print(argsort1, all_label_ids[i], rank)
                ranks.append(rank + 1)
                test_triple = test_triples[i]
                filter_rank = rank
                for tmp_label_id in argsort1[:rank]:
                    tmp_label = label_list[tmp_label_id]
                    tmp_triple = [test_triple[0], tmp_label, test_triple[2]]
                    #print(tmp_triple)
                    tmp_triple_str = '\t'.join(tmp_triple)
                    if tmp_triple_str in all_triples_str_set:
                        filter_rank -= 1
                filter_ranks.append(filter_rank + 1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

                    if filter_rank <= hits_level:
                        hits_filter[hits_level].append(1.0)
                    else:
                        hits_filter[hits_level].append(0.0)

            print("Raw mean rank: ", np.mean(ranks))
            print("Filtered mean rank: ", np.mean(filter_ranks))
            for i in [0,2,9]:
                print('Raw Hits @{0}: {1}'.format(i+1, np.mean(hits[i])))
                print('hits_filter Hits @{0}: {1}'.format(i+1, np.mean(hits_filter[i])))
            preds = np.argmax(preds, axis=1)

            result = compute_metrics(task_name, preds, all_label_ids)
            loss = tr_loss/nb_tr_steps if args.do_train else None

            result['eval_loss'] = eval_loss
            result['global_step'] = global_step
            result['loss'] = loss

            output_eval_file = os.path.join(args.output_dir, "test_results.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Test results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
            # relation prediction, raw
            print("Relation prediction hits@1, raw...")
            print(metrics.accuracy_score(all_label_ids, preds))

if __name__ == "__main__":
    main()
