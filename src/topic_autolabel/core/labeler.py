import re
from collections import Counter
from typing import List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TopicLabeler:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the topic labeler with a specified LLM.
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(device)

    def _generate_prompt(
        self,
        text: str,
        candidate_labels: Optional[List[str]] = None,
    ) -> str:
        """
        Generate appropriate prompt based on whether we're doing open-ended
        labeling or classification with candidate labels.
        """
        if candidate_labels:
            prompt = f"""Given the following text, classify it into one of these categories: {', '.join(candidate_labels)}
            
Text: {text}

The category that best describes this text is:"""
        else:
            prompt = f""""Use three words total (comma separated)\
to describe general topics in above texts. Under no circumstances use enumeration. \
Example format: Tree, Cat, Fireman

Text: {text}
Three comma separated words:"""
        return prompt

    def generate_labels(
        self,
        texts: Union[str, List[str]],
        num_labels: int = 5,
        candidate_labels: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate labels for the given texts.
        """
        if isinstance(texts, str):
            texts = [texts]

        if candidate_labels:
            max_tokens = max(
                [len(self.tokenizer(x)["input_ids"]) for x in candidate_labels]
            )
            pattern = r""
        else:
            max_tokens = 50
            pattern = r"^\w+,\s*\w+,\s*\w+"

        labels = []
        for text in texts:
            prompt = self._generate_prompt(text, candidate_labels)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            prompt_length = inputs["input_ids"].shape[1]
            response = self.tokenizer.decode(outputs[0][prompt_length:])
            response = response.lower().strip()
            if candidate_labels:
                if response not in candidate_labels:
                    response = "<err>"
                labels.append(response)
            else:
                words = re.findall(pattern, response)
                if words:
                    words = words[0].split(", ")
                    labels.append(words)
                else:
                    labels.append([])

        if candidate_labels:
            return labels
        else:
            ## Re-label with most common terms
            counts = Counter(word for sublist in labels for word in sublist)
            try:
                assert (num_labels) <= len(counts)
            except AssertionError:
                raise Exception
            top_labels = [x[0] for x in counts.most_common(num_labels)]
            max_tokens = max([len(self.tokenizer(x)["input_ids"]) for x in top_labels])
            # TODO: filter top labels w/ embeddings and maybe remove generic labels via clustering?
            # re-label with top labels
            final_labels = []
            for text in texts:
                prompt = self._generate_prompt(text, top_labels)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=0.7,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                prompt_length = inputs["input_ids"].shape[1]
                response = self.tokenizer.decode(outputs[0][prompt_length:])
                response = response.lower().strip()
                found = False
                for label in top_labels:
                    if label in response:
                        final_labels.append(label)
                        found = True
                        break
                if not found:
                    final_labels.append("<err>")
            return final_labels
