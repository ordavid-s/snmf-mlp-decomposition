import csv
import json
from typing import List, Tuple
import pandas as pd


class ConceptDataset:
    def __init__(self, path: str):
        """
        Initialize the dataset by loading the data from a CSV or JSON file.
        """
        self.path = path
        self.data = []
        
        # Load the CSV file
        if path.endswith('.csv'):
            with open(self.path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.data.append(row["prompt"])
        elif path.endswith('.json'):
            with open(self.path, 'r', encoding="utf-8") as f:
                data = json.load(f)
                for key, value in data.items():
                    self.data += value

    def __len__(self):
        # The length of the dataset is simply the number of rows
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return the prompt by index
        return self.data[idx]
    
    def get_batches(self, batch_size: int) -> List[dict]:
        """
        Group the data into batches of prompts.

        Args:
            batch_size (int): Number of samples per batch.

        Returns:
            List[dict]: A list of batches where each batch is a dictionary with key 'prompt'
                        containing a list of prompt strings.
        """
        batches = []
        for i in range(0, len(self.data), batch_size):
            batch_data = self.data[i:i + batch_size]
            batches.append({'prompt': list(batch_data)})
        return batches


class SupervisedConceptDataset:
    def __init__(self, path: str):
        """
        Initialize the dataset by loading the data from a CSV or JSON file.
        Expects each record to have 'prompt' and 'label' fields.
        For CSV files, it reads using pandas.
        For JSON files, it first tries to read a DataFrame with 'prompt' and 'label'
        columns; if not found, it assumes a dictionary structure where keys are labels
        and values are lists of prompts.
        """
        self.path = path
        self.data: List[Tuple[str, str]] = []
        
        if path.endswith('.csv'):
            # Load CSV using pandas
            df = pd.read_csv(self.path, encoding="utf-8")
            if 'prompt' in df.columns and 'label' in df.columns:
                # Drop any rows with missing prompt or label values
                df = df.dropna(subset=['prompt', 'label'])
                self.data = list(zip(df['prompt'], df['label']))
        
        elif path.endswith('.json'):
            try:
                # Try to load as a DataFrame; this will work if the JSON is a list of dicts
                df = pd.read_json(self.path, encoding="utf-8")
            except ValueError:
                # If the JSON structure is not suitable for read_json, try with orient='index'
                df = pd.read_json(self.path, orient='index', encoding="utf-8")
            
            if set(['prompt', 'label']).issubset(df.columns):
                # If we have the expected columns, use them
                df = df.dropna(subset=['prompt', 'label'])
                self.data = list(zip(df['prompt'], df['label']))
            elif set(['text', 'label']).issubset(df.columns):
                # If we have the expected columns, use them
                df = df.dropna(subset=['text', 'label'])
                self.data = list(zip(df['text'], df['label']))
            else:
                # Otherwise, assume the JSON is a dict where keys are labels and values are lists of prompts.
                # Fall back to using the json module.
                with open(self.path, 'r', encoding="utf-8") as f:
                    loaded_data = json.load(f)
                for label, prompts in loaded_data.items():
                    for prompt in prompts:
                        if prompt is not None and label is not None:
                            self.data.append((prompt, label))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Tuple[str, str]:
        # Return a tuple of (prompt, label) for the given index
        return self.data[idx]
    
    def get_batches(self, batch_size: int) -> List[dict]:
        """
        Group the data into batches of prompts and labels.

        Args:
            batch_size (int): Number of samples per batch.

        Returns:
            List[dict]: A list of batches, where each batch is a dictionary containing:
                        - 'prompt': a list of prompts
                        - 'label': a list of corresponding labels
        """
        batches = []
        for i in range(0, len(self.data), batch_size):
            batch = self.data[i:i + batch_size]
            prompts, labels = zip(*batch) if batch else ([], [])
            batches.append({'prompt': list(prompts), 'label': list(labels)})
        return batches
