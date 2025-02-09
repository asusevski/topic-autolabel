import re
from collections import Counter
from typing import List, Optional, Union

import ollama
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data_loader import DataType
from .video_loader import extract_video_frames, cleanup_frames


class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class TopicLabeler:
    def __init__(
        self,
        huggingface_model: str,
        ollama_model: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: Optional[int] = 8,
        dtype: DataType = "text",
    ):
        """
        Initialize the topic labeler with a specified LLM or LLM service.
        """
        if ollama_model == "":
            # TODO: gracefully fix this
            if dtype != "text":
                raise ValueError(
                    "Found non-text data and a HuggingFace model path, please only run with Ollama."
                )

            self.tokenizer = AutoTokenizer.from_pretrained(
                huggingface_model, padding_side="left"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                huggingface_model,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
            ).to(device)
            self.batch_size = batch_size
        else:
            self.device = device
            self.ollama_model = ollama_model
            self.batch_size = batch_size
        self.similarity_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        if torch.cuda.is_available():
            self.similarity_model.to(device)
        self.dtype = dtype

    def _create_prompt(
        self,
        text: Optional[str],
        candidate_labels: Optional[List[str]] = None,
        dtype: DataType = "text",
    ) -> str:
        """Generate appropriate prompt based on labeling mode."""
        if dtype == "text":
            if candidate_labels:
                return f"Given the following text, classify it into one of these categories: {', '.join(candidate_labels)}\n\nText: {text}\n\nThe category that best describes this text is:"
            return f"Use three words total (comma separated) to describe general topics in above texts. Under no circumstances use enumeration. Example format: Tree, Cat, Fireman\n\nText: {text}\nThree comma separated words:"
        else:
            if candidate_labels:
                return f"Classify the image into one of these categories: {', '.join(candidate_labels)}\n\nThe category that best describes this image is:"
            return "Use three words total (comma separated) to describe the image. Under no circumstances use enumeration. Example format: Tree, Cat, Fireman\n\nThree comma separated words:"

    @torch.no_grad()
    def _batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
    ) -> List[str]:
        """Generate responses for a batch of prompts."""
        if hasattr(self, "ollama_model"):
            responses = []
            for prompt in prompts:
                response = ollama.generate(model=self.ollama_model, prompt=prompt)
                responses.append(response.response)
            return responses

        # Tokenize all prompts at once
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.hf_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # Extract generated text for each sequence
        responses = []
        for i, output in enumerate(outputs):
            prompt_length = inputs["attention_mask"][i].sum()
            response = self.tokenizer.decode(
                output[prompt_length:], skip_special_tokens=True
            )
            responses.append(response.lower().strip())
        return responses

    def _filter_labels_semantic(
        self, label_counts: Counter, num_labels: int, similarity_threshold: float = 0.50
    ):
        """
        Filter labels semantically using SentenceTransformers by removing similar labels
        and keeping the most frequent ones.

        Args:
            label_counts: Counter object containing labels and their counts
            num_labels: Number of labels to return
            similarity_threshold: Threshold for cosine similarity (default: 0.7)

        Returns:
            List of filtered labels
        """
        # Sort labels by frequency
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        labels = [label for label, _ in sorted_labels]
        # Get embeddings for all labels
        embeddings = self.similarity_model.encode(
            labels, convert_to_tensor=True, device=self.device
        )
        # Calculate cosine similarity matrix
        similarity_matrix = torch.nn.functional.cosine_similarity(
            embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2
        )
        # Filter out similar labels
        filtered_indices = []
        for i in range(len(labels)):
            # Skip if this label is too similar to any already selected label
            too_similar = False
            for j in filtered_indices:
                if i != j and similarity_matrix[i, j] > similarity_threshold:
                    too_similar = True
                    break
            if not too_similar:
                filtered_indices.append(i)
            # Break if we have enough labels
            if len(filtered_indices) == num_labels:
                break
        # If we don't have enough labels after filtering, add the next most frequent ones
        if len(filtered_indices) < num_labels:
            remaining_indices = [
                i for i in range(len(labels)) if i not in filtered_indices
            ]
            filtered_indices.extend(
                remaining_indices[: num_labels - len(filtered_indices)]
            )

        return [labels[i] for i in filtered_indices[:num_labels]]

    def _process_open_ended_responses(
        self, responses: List[str], num_labels: int
    ) -> List[str]:
        """Process responses for open-ended labeling."""
        pattern = r"^\w+,\s*\w+,\s*\w+"
        word_lists = []

        for response in responses:
            words = re.findall(pattern, response)
            if words:
                word_lists.append(words[0].split(", "))
            else:
                word_lists.append([])
        # Get most common terms
        counts = Counter(word for sublist in word_lists for word in sublist)
        if len(counts) < num_labels:
            raise ValueError(
                f"Could not generate {num_labels} unique labels from the texts"
            )
        labels = self._filter_labels_semantic(counts, num_labels)
        return labels

    def generate_labels(
        self,
        texts: Optional[Union[str, List[str]]] = None,
        df: Optional[pd.DataFrame] = None,
        num_labels: int = 5,
        candidate_labels: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate labels for the given texts in batches.

        Args:
            texts: Single text or list of texts to label
            num_labels: Number of labels to generate for open-ended labeling
            candidate_labels: Optional list of predefined labels

        Returns:
            List of generated labels
        """
        if self.dtype == "image":
            prompt = self._create_prompt(
                text=None, candidate_labels=candidate_labels, dtype=self.dtype
            )
            assert type(df) is pd.DataFrame
            all_responses = []
            for fpath in df["filepath"]:
                response = ollama.chat(
                    model=self.ollama_model,
                    messages=[{"role": "user", "content": prompt, "images": [fpath]}],
                ).message.content
                all_responses.append(response)
            if candidate_labels is not None:
                return all_responses
            top_labels = self._process_open_ended_responses(all_responses, num_labels)
            # Re-label texts with top labels
            final_labels = []
            prompt = self._create_prompt(
                text=None, candidate_labels=top_labels, dtype=self.dtype
            )
            for fpath in df["filepath"]:
                response = ollama.chat(
                    model=self.ollama_model,
                    messages=[{"role": "user", "content": prompt, "images": [fpath]}],
                ).message.content
                final_labels.append(response)
            return final_labels
        elif self.dtype == "video":
            prompt = self._create_prompt(
                text=None, candidate_labels=candidate_labels, dtype=self.dtype
            )
            assert type(df) is pd.DataFrame
            all_video_responses = []
            for fpath in df["filepath"]:
                # split video into frames
                output_dir, frame_paths = extract_video_frames(fpath)
                all_responses = []
                try:
                    for frame_path in frame_paths:
                        response = ollama.chat(
                        model=self.ollama_model,
                            messages=[{"role":"user", "content": prompt, "images": [frame_path]}],
                        ).message.content
                        all_responses.append(response)
                        if candidate_labels is not None:
                            return all_responses
                        top_labels = self._process_open_ended_responses(all_responses,num_labels)
                        final_labels = []
                        prompt = self._create_prompt(
                            text=None, candidate_labels=top_labels, dtype=self.dtype
                        )
                        for frame_path in frame_paths:
                            response = ollama.chat(
                                model=self.ollama_model,
                                messages=[{"role": "user", "content": prompt, "images": [frame_path]}],
                            ).message.content
                            final_labels.append(response)
                        video_label = max(final_labels,key=final_labels.count)
                        all_video_responses.append(video_label)
                finally:
                    cleanup_frames(output_dir)
            return all_video_responses

        if isinstance(texts, str):
            texts = [texts]
        # Create dataset and dataloader for batch processing
        if hasattr(self, "ollama_model"):
            dataset = TextDataset(texts, tokenizer=None)
            max_tokens = 1_000  # TODO: fix this janky tmp, find out how enforce max tokens on ollama
        else:
            dataset = TextDataset(texts, self.tokenizer)
            max_tokens = (
                max(
                    len(self.tokenizer(x)["input_ids"]) + 2
                    for x in (candidate_labels or [])
                )
                if candidate_labels
                else 25
            )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        # Calculate max tokens based on labeling mode
        # NOTE: +2 is janky, but sometimes model outputs something like "a) answer" so meh
        # for that matter 25 is also janky -- solely for unsupervised labels
        all_responses = []
        # Process texts in batches
        for batch_texts in dataloader:
            prompts = [
                self._create_prompt(text, candidate_labels) for text in batch_texts
            ]
            responses = self._batch_generate(prompts, max_tokens)
            all_responses.extend(responses)
        if not candidate_labels:
            # Handle open-ended labeling
            top_labels = self._process_open_ended_responses(all_responses, num_labels)
            # Re-label texts with top labels
            final_labels = []
            for batch_texts in dataloader:
                prompts = [
                    self._create_prompt(text, top_labels) for text in batch_texts
                ]
                if not hasattr(self, "ollama_model"):
                    max_tokens = max(
                        len(self.tokenizer(x)["input_ids"]) + 2
                        for x in (top_labels or [])
                    )
                batch_responses = self._batch_generate(prompts, max_tokens)
                for response in batch_responses:
                    label_found = False
                    for label in top_labels:
                        if label in response:
                            final_labels.append(label)
                            label_found = True
                            break
                    if not label_found:
                        final_labels.append("<err>")
            return [
                response if response in top_labels else "<err>"
                for response in final_labels
            ]

        else:
            # Handle classification with candidate labels
            final_labels = []
            for response in all_responses:
                label_found = False
                for candidate_label in candidate_labels:
                    if candidate_label in response:
                        final_labels.append(candidate_label)
                        label_found = True
                        break
                if not label_found:
                    final_labels.append("<err>")
            all_responses = [response.strip().strip(".") for response in all_responses]
            return [
                response if response in candidate_labels else "<err>"
                for response in final_labels
            ]
