import json
import os

class JsonHandler:
    def __init__(self, fields, path, auto_write=True):
        """
        Initialize the JsonHandler with a list of field names and a file path.
        
        Args:
            fields (list): List of keys that each row should have.
            path (str): Path to the JSON file where rows will be saved.
            auto_write (bool): If True, writes to disk after every add_row call.
                               If False, you must call write() manually to save.
        """
        self.fields = fields
        self.path = path
        self.auto_write = auto_write
        # Load existing data if the file exists; otherwise start with an empty list.
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    self.data = json.load(f)
                if not isinstance(self.data, list):
                    self.data = []
            except (json.JSONDecodeError, IOError):
                self.data = []
        else:
            self.data = []

    def add_row(self, **kwargs):
        """
        Add a new row to the data. Only keys provided in the initial fields are saved.
        Missing fields will be set to None.
        
        Args:
            **kwargs: Key-value pairs for the row.
        """
        # Build a row with only the specified fields.
        row = {field: kwargs.get(field) for field in self.fields}
        self.data.append(row)
        if self.auto_write:
            self.write()

    def write(self):
        """
        Write the current data to the JSON file.
        """
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

# Example usage:
if __name__ == "__main__":
    # Define the fields to save.
    fields = ["epoch", "accuracy", "loss"]
    # Initialize the handler with auto_write disabled for batch writing.
    handler = JsonHandler(fields, "model_outputs.json", auto_write=False)
    
    # Add some rows (these are only stored in memory).
    handler.add_row(epoch=1, accuracy=0.8)
    handler.add_row(epoch=2, accuracy=0.85, loss=0.4)
    
    # When ready, write all accumulated rows to disk.
    handler.write()

