import os
import json
import torch
from typing import List
from builders.vocab_builder import META_VOCAB
from .utils.utils import preprocess_sentence
from .utils.word_decomposation import is_Vietnamese, determine_non_Vietnamese_character
from typing import *


@META_VOCAB.register()
class VipherTokenizerV2:
    def __init__(self, config):
        """
        Initialize the tokenizer and build the vocabulary.

        Args:
            config: A configuration object with fields like:
                - config.pad_token, bos_token, eos_token, unk_token, space_token
                - config.path.train, dev, test: JSON data files
                - config.text, config.label: the JSON keys for text/label
                - config.schema, config.min_freq, etc. if needed
        """
        self._initialize_special_tokens(config)
        self.config = config

        # Vocab dictionaries for phonemes (onset, tone, rhyme)
        self.itos = {}
        self.stoi = {}

        # Label mappings
        self.i2l = {}
        self.l2i = {}

        # Build the vocab from JSON data
        self._make_vocab(config)

    def _initialize_special_tokens(self, config) -> None:
        """
        Define special tokens and their corresponding tuple-index forms
        (onset_idx, tone_idx, medial_idx, nucleus_idx, coda_idx).
        """
        self.pad_token = config.pad_piece

        # The base list of special tokens
        self.specials = [
            self.pad_token,
        ]

    def _make_vocab(self, config):
        """
        Build the onset/tone/rhyme vocabulary from the JSON data,
        differentiating Vietnamese from non-Vietnamese text.
        """
        json_paths = [config.path.train, config.path.dev, config.path.test]
        counter = set()

        labels = set()
        aspects = set()
        sentiments = set()

        # Collect token stats from each JSON
        for path in json_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"JSON path not found: {path}")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                if isinstance(item[config.text], list):
                    token_list = preprocess_sentence(" ".join(item[config.text]))
                else:
                    text = item[config.text]
                    token_list = preprocess_sentence(text)

                for token in token_list:
                    is_vn, word_split = is_Vietnamese(token)
                    if is_vn:
                        onset, medial, nucleus, coda, tone = word_split

                        # Update counters (skip if token is a special token)
                        if onset and onset not in self.specials:
                            counter.add(onset)
                        if tone and tone not in self.specials:
                            counter.add(tone)
                        if medial and medial not in self.specials:
                            counter.add(medial)
                        if nucleus and nucleus not in self.specials:
                            counter.add(nucleus)
                        if coda and coda not in self.specials:
                            counter.add(coda)
                    else:
                        # Non-Vietnamese word, split into characters
                        for char in token:
                            counter.update(char)

                if self.config.get("task_type", None) == "seq_labeling":
                    for label in item[config.label]:
                        label = label.split("-")[-1]
                        labels.add(label)
                elif self.config.get("task_type", None) == "aspect_based":
                    for label in item["label"]:
                        aspects.add(label["aspect"])
                        sentiments.add(label["sentiment"])
                else:
                    labels.add(item[config.label])

        # Build itos/stoi for onset, tone, rhyme
        # Prepend the specials at the start
        counter = list(counter)
        self.itos = {i: tok for i, tok in enumerate(self.specials + counter)}
        self.stoi = {tok: i for i, tok in enumerate(self.specials + counter)}
        self.pad_idx = self.stoi[self.pad_token]

        # Build label <-> index maps
        if self.config.get("task_type", None) == "aspect_based":
            aspects = sorted(
                {a for a in aspects if a is not None}, key=lambda x: x.lower()
            )
            sentiments = sorted(
                {s for s in sentiments if s is not None}, key=lambda x: x.lower()
            )

            self.i2a = {i: label for i, label in enumerate(aspects)}
            self.a2i = {label: i for i, label in enumerate(aspects)}

            self.i2s = {i: label for i, label in enumerate(sentiments, 1)}
            self.i2s[0] = None

            self.s2i = {label: i for i, label in enumerate(sentiments, 1)}
            self.s2i[None] = 0

        else:
            # Create label <-> index maps (sorted for consistent ordering)
            labels = sorted(list(labels))
            self.i2l = {i: label for i, label in enumerate(labels)}
            self.l2i = {label: i for i, label in enumerate(labels)}

    def encode_sentence(
        self, text: str, max_len: int = None
    ) -> Tuple[List[int], List[int]]:
        """
        Tokenize a sentence and return input IDs with a mapping from words to subwords.

        Args:
            text (str): The input text to tokenize.

        Returns:
            A tuple containing:
            - List[int]: Token IDs for the entire text.
            - List[int]: A list mapping each subword token ID to its original word index.
        """

        words = preprocess_sentence(
            text
        )  # Split text into words using your preprocess_sentence method
        input_ids = []
        for word in words:
            is_vn, word_split = is_Vietnamese(word)
            if is_vn:
                onset, medial, nucleus, coda, tone = word_split

                onset_idx = self.stoi.get(onset, self.pad_idx)
                tone_idx = self.stoi.get(tone, self.pad_idx)
                medial_idx = self.stoi.get(medial, self.pad_idx)
                nucleus_idx = self.stoi.get(nucleus, self.pad_idx)
                coda_idx = self.stoi.get(coda, self.pad_idx)
                input_ids.append(
                    (onset_idx, tone_idx, medial_idx, nucleus_idx, coda_idx)
                )
            else:
                for char in word:
                    onset_c, tone_c, medial_c, nucleus_c, coda_c = (
                        determine_non_Vietnamese_character(char)
                    )
                    o_idx = self.stoi.get(onset_c, self.pad_idx)
                    t_idx = self.stoi.get(tone_c, self.pad_idx)
                    m_idx = self.stoi.get(medial_c, self.pad_idx)
                    n_idx = self.stoi.get(nucleus_c, self.pad_idx)
                    c_idx = self.stoi.get(coda_c, self.pad_idx)
                    input_ids.append(
                        (o_idx, t_idx, m_idx, n_idx, c_idx)
                    )
        if max_len is not None:
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                input_ids[-1] = self.eos_idx
            elif len(input_ids) < max_len:
                # pad with the pad triplet
                input_ids += [(self.pad_idx, self.pad_idx, self.pad_idx, self.pad_idx, self.pad_idx)] * (max_len - len(input_ids))
                
        return torch.tensor(input_ids, dtype=torch.long)

    def encode_label(self, label: str) -> torch.Tensor:
        """
        Convert a string label to an integer label ID.
        """
        if self.config.get("task_type", None) == "seq_labeling":

            labels = [self.l2i[l] for l in label]

            return torch.Tensor(labels).long()
        elif self.config.get("task_type", None) == "aspect_based":

            label_vector = torch.zeros(self.total_aspects_labels["aspects"])
            for l in label:
                aspect = l["aspect"]
                sentiment = l["sentiment"]
                # active the OTHERS case
                if aspect == "OTHERS":
                    sentiment = "Positive"
                label_vector[self.a2i[aspect]] = self.s2i[sentiment]

            return torch.Tensor(label_vector).long()
        else:
            return torch.tensor([self.l2i[label]], dtype=torch.long)

    def decode_label(self, label_vecs: torch.Tensor) -> List[str]:
        """
        Convert integer label IDs back into strings.

        Args:
            label_vecs: 1D or 2D tensor of label IDs (e.g. shape [batch_size]).
        """

        if self.config.get("task_type", None) == "seq_labeling":
            results = []
            batch_labels = label_vecs.tolist()
            for labels in batch_labels:
                result = []
                for label in labels:
                    result.append(self.i2l[label])
                results.append(result)

            return results
        elif self.config.get("task_type", None) == "aspect_based":

            batch_decoded_labels = []

            # Iterate over each label vector in the batch
            for vec in label_vecs:
                instance_labels = []

                # Iterate over each aspect's sentiment value in the label vector
                for i, label_id in enumerate(vec):
                    label_id = label_id.item()  # Get the integer value of the label
                    if label_id == 0:
                        continue
                    aspect = self.i2a.get(i)

                    sentiment = self.i2s.get(label_id)
                    decoded_label = {"aspect": aspect, "sentiment": sentiment}
                    instance_labels.append(decoded_label)

                batch_decoded_labels.append(instance_labels)

            return batch_decoded_labels
        else:
            labels_out = []
            for vec in label_vecs:
                label_id = vec.item()
                labels_out.append(self.i2l[label_id])
            return labels_out

    @property
    def total_tokens(self) -> int:
        """
        Combined count of onset, tone, and rhyme tokens.
        """
        return len(self.stoi)

    @property
    def total_aspects_labels(self) -> dict:
        return {"aspects": len(self.i2a), "sentiment": len(self.i2s)}

    @property
    def total_labels(self) -> int:
        """
        Number of distinct labels encountered in the dataset.
        """
        return len(self.l2i)

    # @property
    # def get_pad_idx(self) -> int:
    #     """Get the ID of the padding token."""
    #     return self.pad_idx[0]
