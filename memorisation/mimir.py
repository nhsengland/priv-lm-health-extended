"""
Code to run memorisation-related metrics, adapted from the mimir library (https://github.com/iamgroot42/mimir)
which is associated with Duan et al., 2024:
```
@inproceedings{duan2024membership,
      title={Do Membership Inference Attacks Work on Large Language Models?}, 
      author={Michael Duan and Anshuman Suri and Niloofar Mireshghallah and Sewon Min and Weijia Shi and Luke Zettlemoyer and Yulia Tsvetkov and Yejin Choi and David Evans and Hannaneh Hajishirzi},
      year={2024},
      booktitle={Conference on Language Modeling (COLM)},
}
```
"""
import os
import time
from collections import defaultdict
from typing import List
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
from hf_olmo import *


class Model(nn.Module):
    """
    Base class (for LLMs).
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.model = None  # Set by child class
        self.tokenizer = None  # Set by child class
        self.device = None
        self.device_map = None
        self.name = None

    def to(self, device):
        """
        Shift model to a particular device.
        """
        self.model.to(device, non_blocking=True)

    def load(self):
        """
        Load model onto GPU (and compile, if requested) if not already loaded with device map.
        """
        if not self.device_map:
            start = time.time()
            try:
                self.model.cpu()
            except NameError:
                pass
            self.model.to(self.device, non_blocking=True)
            print(f"DONE ({time.time() - start:.2f}s)")

    def unload(self):
        """
        Unload model from GPU
        """
        start = time.time()
        try:
            self.model.cpu()
        except NameError:
            pass
        print(f"DONE ({time.time() - start:.2f}s)")

    def get_probabilities(
        self,
        text: str,
        tokens: np.ndarray = None,
        no_grads: bool = True,
        return_all_probs: bool = False,
    ):
        """
        Get the probabilities or log-softmaxed logits for a text under the current model.
        Args:
            text (str): The input text for which to calculate probabilities.
            tokens (numpy.ndarray, optional): An optional array of token ids. If provided, these tokens
            are used instead of tokenizing the input text. Defaults to None.

        Raises:
            ValueError: If the device or name attributes of the instance are not set.

        Returns:
            list: A list of probabilities.
        """
        with torch.set_grad_enabled(not no_grads):
            if self.device is None or self.name is None:
                raise ValueError("Please set self.device and self.name in child class")

            if tokens is not None:
                labels = torch.from_numpy(tokens.astype(np.int64)).type(
                    torch.LongTensor
                )
                if labels.shape[0] != 1:
                    # expand first dimension
                    labels = labels.unsqueeze(0)
            else:
                tokenized = self.tokenizer(text, return_tensors="pt")
                labels = tokenized.input_ids

            target_token_log_prob = []
            all_token_log_prob = []
            for i in range(0, labels.size(1), self.stride):
                begin_loc = max(i + self.stride - self.max_length, 0)
                end_loc = min(i + self.stride, labels.size(1))
                trg_len = end_loc - i  # may be different from stride on last loop
                input_ids = labels[:, begin_loc:end_loc].to(self.device)
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                logits = self.model(input_ids, labels=target_ids).logits
                if no_grads:
                    logits = logits.cpu()
                shift_logits = logits[..., :-1, :].contiguous()
                log_probabilities = torch.nn.functional.log_softmax(
                    shift_logits, dim=-1
                )
                shift_labels = target_ids[..., 1:]
                if no_grads:
                    shift_labels = shift_labels.cpu()
                shift_labels = shift_labels.contiguous()
                labels_processed = shift_labels[0]

                del input_ids
                del target_ids

                for i, token_id in enumerate(labels_processed):
                    if token_id != -100:
                        log_probability = log_probabilities[0, i, token_id]
                        if no_grads:
                            log_probability = log_probability.item()
                        target_token_log_prob.append(log_probability)
                        all_token_log_prob.append(log_probabilities[0, i])

            # Should be equal to # of tokens - 1 to account for shift
            assert len(target_token_log_prob) == labels.size(1) - 1
            all_token_log_prob = torch.stack(all_token_log_prob, dim=0)
            assert len(target_token_log_prob) == len(all_token_log_prob)

        if not no_grads:
            target_token_log_prob = torch.stack(target_token_log_prob)

        if not return_all_probs:
            return target_token_log_prob
        return target_token_log_prob, all_token_log_prob

    @torch.no_grad()
    def get_ll(self, text: str, tokens: np.ndarray = None, probs=None):
        """
        Get the log likelihood of each text under the base_model.

        Args:
            text (str): The input text for which to calculate the log likelihood.
            tokens (numpy.ndarray, optional): An optional array of token ids. If provided, these tokens
            are used instead of tokenizing the input text. Defaults to None.
            probs (list, optional): An optional list of probabilities. If provided, these probabilities
            are used instead of calling the `get_probabilities` method. Defaults to None.
        """
        all_prob = (
            probs if probs is not None else self.get_probabilities(text, tokens=tokens)
        )
        return -np.mean(all_prob)

    def load_model_properties(self):
        """
        Load model properties, such as max length and stride.
        """
        # TODO: getting max_length of input could be more generic
        if "silo" in self.name or "balanced" in self.name:
            self.max_length = self.model.model.seq_len
        elif hasattr(self.model.config, "max_position_embeddings"):
            self.max_length = self.model.config.max_position_embeddings
        elif hasattr(self.model.config, "n_positions"):
            self.max_length = self.model.config.n_positions
        else:
            # Default window size
            self.max_length = 1024
        self.stride = self.max_length // 2


class ReferenceModel(Model):
    def __init__(self, name: str):
        self.device = "cuda"
        self.name = name
        base_model_kwargs = {"revision": "main"}
        base_model_kwargs.update(dict(torch_dtype=torch.float16))
        self.tokenizer = AutoTokenizer.from_pretrained(
            name, use_fast=False, cache_dir=os.environ["HF_HOME"]
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            load_in_4bit=True,
            offload_folder="offload",
            device_map="auto",
            cache_dir=os.environ["HF_HOME"],
        )
        self.load_model_properties()


class LanguageModel(Model):
    def __init__(self, name, is_peft=False, **kwargs):
        super().__init__(**kwargs)
        self.device = "cuda"
        self.device_map = "auto"
        self.name = name
        base_model_kwargs = {"revision": "main"}
        base_model_kwargs.update(dict(torch_dtype=torch.float16))
        self.tokenizer = AutoTokenizer.from_pretrained(
            name, use_fast=False, cache_dir=os.environ["HF_HOME"]
        )
        if is_peft:
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                name,
                load_in_4bit=True,
                offload_folder="offload",
                device_map="auto",
                cache_dir=os.environ["HF_HOME"],
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                name,
                load_in_4bit=True,
                offload_folder="offload",
                device_map="auto",
                cache_dir=os.environ["HF_HOME"],
            )
        self.load_model_properties()

    @torch.no_grad()
    def get_ref(self, text: str, ref_model: ReferenceModel, tokens=None, probs=None):
        """
        Compute the loss of a given text calibrated against the text's loss under a reference model -- MIA baseline
        """
        lls = self.get_ll(text, tokens=tokens, probs=probs)
        lls_ref = ref_model.get_ll(text)

        return lls - lls_ref

    @torch.no_grad()
    def get_rank(self, text: str, log: bool = False):
        """
        Get the average rank of each observed token sorted by model likelihood
        """
        tokenized = self.tokenizer(text, return_tensors="pt").to(self.device)
        logits = self.model(**tokenized).logits[:, :-1]
        labels = tokenized.input_ids[:, 1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (
            logits.argsort(-1, descending=True) == labels.unsqueeze(-1)
        ).nonzero()

        assert (
            matches.shape[1] == 3
        ), f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:, -1], matches[:, -2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (
            timesteps == torch.arange(len(timesteps)).to(timesteps.device)
        ).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1  # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()

    @torch.no_grad()
    def get_lls(self, texts: List[str], batch_size: int = 6):
        # return [self.get_ll(text) for text in texts] # -np.mean([self.get_ll(text) for text in texts])
        # tokenized = self.tokenizer(texts, return_tensors="pt", padding=True)
        # labels = tokenized.input_ids
        total_size = len(texts)
        losses = []
        for i in range(0, total_size, batch_size):
            # Delegate batches and tokenize
            batch = texts[i : i + batch_size]
            tokenized = self.tokenizer(
                batch, return_tensors="pt", padding=True, return_attention_mask=True
            )
            label_batch = tokenized.input_ids

            # # mask out padding tokens
            attention_mask = tokenized.attention_mask
            assert attention_mask.size() == label_batch.size()

            needs_sliding = label_batch.size(1) > self.max_length // 2
            if not needs_sliding:
                label_batch = label_batch.to(self.device)
                attention_mask = attention_mask.to(self.device)

            # Collect token probabilities per sample in batch
            all_prob = defaultdict(list)
            for i in range(0, label_batch.size(1), self.stride):
                begin_loc = max(i + self.stride - self.max_length, 0)
                end_loc = min(i + self.stride, label_batch.size(1))
                trg_len = end_loc - i  # may be different from stride on last loop
                input_ids = label_batch[:, begin_loc:end_loc]
                mask = attention_mask[:, begin_loc:end_loc]
                if needs_sliding:
                    input_ids = input_ids.to(self.device)
                    mask = mask.to(self.device)

                target_ids = input_ids.clone()
                # Don't count padded tokens or tokens that already have computed probabilities
                target_ids[:, :-trg_len] = -100
                # target_ids[attention_mask == 0] = -100

                logits = self.model(
                    input_ids, labels=target_ids, attention_mask=mask
                ).logits.cpu()
                target_ids = target_ids.cpu()
                shift_logits = logits[..., :-1, :].contiguous()
                probabilities = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                shift_labels = target_ids[..., 1:].contiguous()

                for i, sample in enumerate(shift_labels):
                    for j, token_id in enumerate(sample):
                        if token_id != -100 and token_id != self.tokenizer.pad_token_id:
                            probability = probabilities[i, j, token_id].item()
                            all_prob[i].append(probability)

                del input_ids
                del mask

            # average over each sample to get losses
            batch_losses = [
                -np.mean(all_prob[idx]) for idx in range(label_batch.size(0))
            ]
            # print(batch_losses)
            losses.extend(batch_losses)
            del label_batch
            del attention_mask
        return losses  # np.mean(losses)

    def sample_from_model(self, texts: List[str], **kwargs):
        """
        Sample from base_model using ****only**** the first 30 tokens in each example as context
        """
        min_words = kwargs.get("min_words", 55)
        max_words = kwargs.get("max_words", 200)
        prompt_tokens = kwargs.get("prompt_tokens", 30)

        # encode each text as a list of token ids
        # if self.config.dataset_member == 'pubmed':
        #    texts = [t[:t.index(SEPARATOR)] for t in texts]
        #    all_encoded = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device, non_blocking=True)
        # else:
        if True:
            all_encoded = self.tokenizer(texts, return_tensors="pt", padding=True).to(
                self.device, non_blocking=True
            )
            all_encoded = {
                key: value[:, :prompt_tokens] for key, value in all_encoded.items()
            }

        decoded = ["" for _ in range(len(texts))]

        # sample from the model until we get a sample with at least min_words words for each example
        # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
        tries = 0
        while (
            m := min(len(x.split()) for x in decoded)
        ) < min_words and tries < 50:  # self.config.neighborhood_config.top_p:
            if tries != 0:
                print()
                print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")

            sampling_kwargs = {}
            if self.config.do_top_p:
                sampling_kwargs["top_p"] = 0.96  # self.config.top_p
            elif self.config.do_top_k:
                sampling_kwargs["top_k"] = 40  # self.config.top_k
            # min_length = 50 if config.dataset_member in ['pubmed'] else 150

            # outputs = base_model.generate(**all_encoded, min_length=min_length, max_length=max_length, do_sample=True, **sampling_kwargs, pad_token_id=base_tokenizer.eos_token_id, eos_token_id=base_tokenizer.eos_token_id)
            # removed minlen and attention mask min_length=min_length, max_length=200, do_sample=True,pad_token_id=base_tokenizer.eos_token_id,
            outputs = self.model.generate(
                **all_encoded,
                min_length=min_words * 2,
                max_length=max_words * 3,
                **sampling_kwargs,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            tries += 1

        return decoded

    @torch.no_grad()
    def get_entropy(self, text: str):
        """
        Get average entropy of each token in the text
        """
        tokenized = self.tokenizer(text, return_tensors="pt").to(self.device)
        logits = self.model(**tokenized).logits[:, :-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()

    @torch.no_grad()
    def get_max_norm(self, text: str, context_len=None, tk_freq_map=None):
        tokenized = self.tokenizer(text, return_tensors="pt").to(self.device)
        labels = tokenized.input_ids

        max_length = context_len if context_len is not None else self.max_length
        stride = max_length // 2  # self.stride
        all_prob = []
        for i in range(0, labels.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, labels.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = labels[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            outputs = self.model(input_ids, labels=target_ids)
            logits = outputs.logits
            # Shift so that tokens < n predict n
            # print(logits.shape)
            shift_logits = logits[..., :-1, :].contiguous()
            # shift_logits = torch.transpose(shift_logits, 1, 2)
            probabilities = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            shift_labels = target_ids[..., 1:].contiguous()
            labels_processed = shift_labels[0]

            for i, token_id in enumerate(labels_processed):
                if token_id != -100:
                    probability = probabilities[0, i, token_id].item()
                    max_tk_prob = torch.max(probabilities[0, i]).item()
                    tk_weight = (
                        max(tk_freq_map[token_id.item()], 1) / sum(tk_freq_map.values())
                        if tk_freq_map is not None
                        else 1
                    )
                    if tk_weight == 0:
                        print("0 count token", token_id.item())
                    tk_norm = tk_weight
                    all_prob.append((1 - (max_tk_prob - probability)) / tk_norm)

        # Should be equal to # of tokens - 1 to account for shift
        assert len(all_prob) == labels.size(1) - 1
        return -np.mean(all_prob)


class Attack:
    def __init__(
        self, target_model: Model, ref_model: Model = None, is_blackbox: bool = True
    ):
        self.target_model = target_model
        self.ref_model = ref_model
        self.is_loaded = False
        self.is_blackbox = is_blackbox

    def load(self):
        """
        Any attack-specific steps (one-time) preparation
        """
        if self.ref_model is not None:
            self.ref_model.load()
            self.is_loaded = True

    def unload(self):
        if self.ref_model is not None:
            self.ref_model.unload()
            self.is_loaded = False

    def _attack(self, **kwargs):
        """
        Actual logic for attack.
        """
        raise NotImplementedError("Attack must implement attack()")

    def attack(self, document, probs, **kwargs):
        """
        Score a document using the attack's scoring function. Calls self._attack
        """
        # Load attack if not loaded yet
        if not self.is_loaded:
            self.load()
            self.is_loaded = True

        detokenized_sample = kwargs.get("detokenized_sample", None)
        if self.config.pretokenized and detokenized_sample is None:
            raise ValueError("detokenized_sample must be provided")

        score = self._attack(document, probs=probs, **kwargs)

        return score


class MinKProbAttack(Attack):
    def __init__(self, model: Model):
        super().__init__(model, ref_model=None)

    @torch.no_grad()
    def _attack(self, document, probs, tokens=None, **kwargs):
        """
        Min-k % Prob Attack. Gets model probabilities and returns likelihood when computed over top k% of ngrams.
        """
        # Hyper-params specific to min-k attack
        k: float = kwargs.get("k", 0.2)
        window: int = kwargs.get("window", 1)
        stride: int = kwargs.get("stride", 1)

        all_prob = (
            probs
            if probs is not None
            else self.target_model.get_probabilities(document, tokens=tokens)
        )
        # iterate through probabilities by ngram defined by window size at given stride
        ngram_probs = []
        for i in range(0, len(all_prob) - window + 1, stride):
            ngram_prob = all_prob[i : i + window]
            ngram_probs.append(np.mean(ngram_prob))
        min_k_probs = sorted(ngram_probs)[: int(len(ngram_probs) * k)]

        return -np.mean(min_k_probs)
