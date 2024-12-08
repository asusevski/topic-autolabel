import re
from collections import Counter
from typing import List, Optional, Union
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 8
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
        self.batch_size = batch_size

    def _create_prompt(self, 
                       text: str, 
                       candidate_labels: Optional[List[str]] = None
    ) -> str:
        """Generate appropriate prompt based on labeling mode."""
        if candidate_labels:
            return f"Given the following text, classify it into one of these categories: {', '.join(candidate_labels)}\n\nText: {text}\n\nThe category that best describes this text is:"
        return f"Use three words total (comma separated) to describe general topics in above texts. Under no circumstances use enumeration. Example format: Tree, Cat, Fireman\n\nText: {text}\nThree comma separated words:"

    @torch.no_grad()
    def _batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int,
    ) -> List[str]:
        """Generate responses for a batch of prompts."""
        # Tokenize all prompts at once
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # Extract generated text for each sequence
        responses = []
        for i, output in enumerate(outputs):
            prompt_length = inputs["attention_mask"][i].sum()
            response = self.tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
            responses.append(response.lower().strip())
        
        return responses

    def _process_open_ended_responses(
        self,
        responses: List[str],
        num_labels: int
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
            raise ValueError(f"Could not generate {num_labels} unique labels from the texts")
            
        return [label for label, _ in counts.most_common(num_labels)]

    def generate_labels(
        self,
        texts: Union[str, List[str]],
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
        if isinstance(texts, str):
            texts = [texts]
        # Create dataset and dataloader for batch processing
        dataset = TextDataset(texts, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Calculate max tokens based on labeling mode
        max_tokens = (
            max(len(self.tokenizer(x)["input_ids"]) for x in (candidate_labels or []))
            if candidate_labels
            else 50
        )
        all_responses = []
        # Process texts in batches
        for batch_texts in dataloader:
            prompts = [self._create_prompt(text, candidate_labels) for text in batch_texts]
            responses = self._batch_generate(prompts, max_tokens)
            all_responses.extend(responses)
        if not candidate_labels:
            # Handle open-ended labeling
            top_labels = self._process_open_ended_responses(all_responses, num_labels)
            
            # Re-label texts with top labels
            final_labels = []
            for batch_texts in dataloader:
                prompts = [self._create_prompt(text, top_labels) for text in batch_texts]
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
            
            return final_labels
        else:
            # Handle classification with candidate labels
            return [
                response if response in candidate_labels else "<err>"
                for response in all_responses
            ]
