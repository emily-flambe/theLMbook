import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://colab.research.google.com/github/aburkov/theLMbook/blob/main/emotion_classifier_CNN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell
def _():
    # Import required libraries
    import torch  # PyTorch for tensor operations and deep learning
    import torch.nn as nn  # Neural network module from PyTorch
    import numpy as np  # NumPy for numerical operations
    import re  # Regular expressions for text processing (if needed)
    import urllib.request  # For downloading files from URLs
    import gzip  # For handling compressed files
    import json  # For parsing JSON data
    import requests  # For making HTTP requests to download data
    import random  # For shuffling data and setting random seeds
    import pickle  # For saving and loading serialized objects
    import os  # For file system operations
    from tqdm import tqdm  # For displaying progress bars during loops
    from torch.utils.data import Dataset, DataLoader  # For creating custom datasets and data loaders in PyTorch

    def set_seed(seed):
        """
        Sets random seeds for reproducibility across different libraries.

        Args:
            seed (int): Seed value for random number generation.
        """
        random.seed(seed)
        torch.manual_seed(seed)  # Set seed for Python's built-in random module
        torch.cuda.manual_seed_all(seed)  # Set seed for CPU operations in PyTorch
        torch.backends.cudnn.deterministic = True  # Set seed for all GPUs
        torch.backends.cudnn.benchmark = False  # Use deterministic algorithms for cuDNN
      # Disable cuDNN auto-tuner for consistent behavior
    class Tokenizer:
        """
        Basic tokenizer that splits text on whitespace.
        """

        def tokenize(self, text):
            """
            Splits the input text into tokens based on whitespace.

            Args:
                text (str): Input text string to tokenize.

            Returns:
                list: List of word tokens.
            """
            words = text.split()
            return words  # Split text by whitespace

    class Embedder:
        """
        Embedder that converts tokens into their corresponding embeddings.

        Attributes:
            embeddings (dict): Dictionary mapping words to their vector representations.
            emb_dim (int): Dimensionality of the embeddings.
            seq_len (int): Fixed sequence length for input text.
        """

        def __init__(self, embeddings, emb_dim, seq_len):
            """
            Initializes the Embedder with embeddings, embedding dimension, and sequence length.

            Args:
                embeddings (dict): Pre-loaded word embeddings.
                emb_dim (int): Dimension of the embeddings.
                seq_len (int): Maximum number of tokens to consider.
            """
            self.embeddings = embeddings
            self.emb_dim = _emb_dim
            self.seq_len = seq_len

        def embed(self, tokens):
            """
            Converts a list of tokens into a tensor of embeddings.

            Tokens are looked up in the embeddings dictionary. If a token is not found,
            a zero vector is used. The sequence is truncated or padded to match the fixed sequence length.

            Args:
                tokens (list): List of token strings.

            Returns:
                torch.Tensor: Tensor of shape (seq_len, emb_dim) representing the embedded tokens.
            """
            embeddings = []
            for word in tokens[:self.seq_len]:
                if word in self.embeddings:  # Process each token up to the maximum sequence length
                    embeddings.append(torch.tensor(self.embeddings[word]))
                elif word.lower() in self.embeddings:
                    embeddings.append(torch.tensor(self.embeddings[word.lower()]))
                else:
                    embeddings.append(torch.zeros(self.emb_dim))
            if len(embeddings) < self.seq_len:
                padding_size = self.seq_len - len(embeddings)  # Use a zero vector for words not found in the embeddings
                embeddings.extend([torch.zeros(self.emb_dim)] * padding_size)
            return torch.stack(embeddings)
      # Pad sequence with zero vectors if the number of tokens is less than seq_len
    def load_embeddings(url, filename='vectors.dat'):
        """
        Downloads and loads word embeddings from a gzipped file.

        If the file does not exist locally, it is downloaded from the provided URL.  # Stack list of tensors into a single tensor of shape (seq_len, emb_dim)
        The file is expected to have a header indicating vocabulary size and embedding dimension,
        followed by each word and its corresponding binary vector.

        Args:
            url (str): URL to download the embeddings from.
            filename (str): Local filename to save/load the embeddings.

        Returns:
            tuple: A tuple containing:
                - vectors (dict): Mapping from words to their embedding vectors (as NumPy arrays).
                - emb_dim (int): Dimensionality of the embedding vectors.
        """
        if not os.path.exists(filename):
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc='Downloading') as progress_bar:

                def report_hook(count, block_size, total_size):
                    if total_size != -1:
                        progress_bar.total = total_size
                    progress_bar.update(block_size)
                urllib.request.urlretrieve(url, filename, reporthook=report_hook)
        else:  # Check if the embeddings file exists locally
            print(f'File {filename} already exists. Skipping download.')
        with gzip.open(filename, 'rb') as f:  # Download the embeddings file with a progress bar
            header = f.readline()
            vocab_size, _emb_dim = map(int, header.split())
            vectors = {}
            binary_len = np.dtype('float32').itemsize * _emb_dim
            with tqdm(total=vocab_size, desc='Loading word vectors') as pbar:
                for _ in range(vocab_size):
                    word = []
                    while True:
                        ch = f.read(1)
                        if ch == b' ':  # Open the gzipped embeddings file in binary read mode
                            word = b''.join(word).decode('utf-8')
                            break  # Read header line to get vocabulary size and embedding dimension
                        if ch != b'\n':
                            word.append(ch)
                    vector = np.frombuffer(f.read(binary_len), dtype='float32')
                    vectors[word] = vector
                    pbar.update(1)  # Calculate the number of bytes for each embedding vector
        return (vectors, _emb_dim)

    def load_and_split_data(url, test_ratio=0.1):  # Read each word and its corresponding embedding vector with a progress bar
        """
        Downloads, decompresses, and splits the dataset into training and testing sets.

        The dataset is expected to be a gzipped file where each line is a JSON object.  # Read characters one by one until a space is encountered (indicating end of word)

        Args:
            url (str): URL to download the dataset from.
            test_ratio (float): Proportion of data to be used as the test set.

        Returns:
            tuple: A tuple containing:
                - train_data (list): List of training examples.
                - test_data (list): List of testing examples.  # Read the binary vector data and convert it into a NumPy array of type float32
        """
        response = requests.get(url)
        content = gzip.decompress(response.content).decode()
        data = [json.loads(line) for line in content.splitlines()]
        random.shuffle(data)
        split_index = int(len(data) * (1 - test_ratio))
        return (data[:split_index], data[split_index:])

    def download_and_prepare_data(data_url, vectors_url, seq_len, batch_size):
        """
        Downloads and prepares the dataset and word embeddings for training and evaluation.

        This function downloads the text dataset and word embeddings, creates label mappings,
        initializes the tokenizer and embedder, and returns data loaders for training and testing.

        Args:
            data_url (str): URL to download the text dataset.
            vectors_url (str): URL to download the word embeddings.
            seq_len (int): Fixed sequence length for token embeddings.
            batch_size (int): Batch size for the data loaders.

        Returns:
            tuple: A tuple containing:  # Download the dataset from the provided URL
                - train_loader (DataLoader): DataLoader for training data.
                - test_loader (DataLoader): DataLoader for testing data.  # Decompress the gzipped content and decode it to a string
                - id_to_label (dict): Mapping from label IDs to label names.
                - num_classes (int): Number of unique classes.  # Parse each line as a JSON object
                - emb_dim (int): Dimensionality of the word embeddings.
        """  # Shuffle the data to ensure a random distribution
        train_split, test_split = load_and_split_data(data_url, test_ratio=0.1)
        embeddings, _emb_dim = load_embeddings(vectors_url)  # Determine the split index based on the test_ratio
        label_to_id, id_to_label, num_classes = create_label_mappings(train_split)
        tokenizer = Tokenizer()
        embedder = Embedder(embeddings, _emb_dim, seq_len)
        train_loader, test_loader = create_data_loaders(train_split, test_split, tokenizer, embedder, label_to_id, batch_size)
        return (train_loader, test_loader, id_to_label, num_classes, _emb_dim)

    class TextClassificationDataset(Dataset):
        """
        PyTorch Dataset for text classification.

        This dataset converts raw text and label data into a format that can be fed into the model.
        It tokenizes text and then embeds it using the provided tokenizer and embedder.

        Args:
            data (list): List of dictionaries containing "text" and "label" keys.
            tokenizer (Tokenizer): Tokenizer instance to split text into tokens.
            embedder (Embedder): Embedder instance to convert tokens into embeddings.
            label_to_id (dict): Mapping from label strings to numeric IDs.
        """

        def __init__(self, data, tokenizer, embedder, label_to_id):
            self.texts = [item['text'] for item in data]
            self.label_ids = [label_to_id[item['label']] for item in data]
            self.tokenizer = tokenizer
            self.embedder = embedder
      # Load and split the dataset into training and testing splits
        def __len__(self):
            """
            Returns the total number of examples in the dataset.  # Load pre-trained word embeddings and get the embedding dimension
            """
            return len(self.texts)
      # Create mappings between labels and their numeric IDs using the training data
        def __getitem__(self, idx):
            """
            Retrieves the embedded text and label for the example at the specified index.  # Initialize the tokenizer and embedder with the loaded embeddings and sequence length

            Args:
                idx (int): Index of the example to retrieve.
      # Create DataLoaders for both training and testing datasets
            Returns:
                tuple: A tuple containing:
                    - embeddings (torch.Tensor): Tensor of shape (seq_len, emb_dim).
                    - label (torch.Tensor): Tensor containing the label ID.
            """
            tokens = self.tokenizer.tokenize(self.texts[idx])
            embeddings = self.embedder.embed(tokens)
            return (embeddings, torch.tensor(self.label_ids[idx], dtype=torch.long))

    class CNNTextClassifier(nn.Module):
        """
        Convolutional Neural Network for text classification.

        This model applies two 1D convolutional layers followed by fully connected layers
        to classify input text based on its embedded representation.

        Args:
            emb_dim (int): Dimensionality of the input embeddings.
            num_classes (int): Number of target classes for classification.
            seq_len (int): Fixed sequence length of the input text.
            id_to_label (dict): Mapping from label IDs to label names.
        """

        def __init__(self, emb_dim, num_classes, seq_len, id_to_label):  # Extract texts and convert labels to their corresponding IDs
            super().__init__()
            self.config = {'emb_dim': _emb_dim, 'num_classes': num_classes, 'seq_len': seq_len, 'id_to_label': id_to_label}
            self.conv1 = nn.Conv1d(_emb_dim, 512, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(256 * seq_len, 128)
            self.fc2 = nn.Linear(128, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            """
            Defines the forward pass of the CNN model.

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim).

            Returns:
                torch.Tensor: Output logits tensor of shape (batch_size, num_classes).
            """
            x = x.permute(0, 2, 1)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = x.flatten(start_dim=1)
            x = self.fc1(x)
            return self.fc2(x)  # Tokenize the text at the given index

    def calculate_accuracy(model, dataloader, device):  # Convert tokens into embeddings using the embedder
        """
        Evaluates the model's accuracy on the provided dataset.  # Return the embeddings and corresponding label as a tensor

        Args:
            model (nn.Module): Trained model.
            dataloader (DataLoader): DataLoader for the dataset to evaluate.
            device: Device on which computations are performed.

        Returns:
            float: Accuracy as a fraction of correct predictions.
        """
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                embeddings, labels = batch
                embeddings = embeddings.to(device)
                labels = labels.to(device)
                outputs = model(embeddings)
                _, predicted = torch.max(outputs, 1)  # Save model configuration for later use (e.g., during inference)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        model.train()
        return accuracy

    def create_label_mappings(train_dataset):  # First convolutional layer: input channels = emb_dim, output channels = 512
        """
        Creates mappings between label strings and numeric IDs based on the training dataset.  # Second convolutional layer: input channels = 512, output channels = 256

        Args:  # Fully connected layer to reduce flattened features to 128 units
            train_dataset (list): List of training examples with a "label" field.
      # Output layer mapping to the number of classes
        Returns:
            tuple: A tuple containing:  # ReLU activation function
                - label_to_id (dict): Mapping from label string to numeric ID.
                - id_to_label (dict): Mapping from numeric ID to label string.
                - num_classes (int): Total number of unique classes.
        """
        unique_labels = sorted(set((item['label'] for item in train_dataset)))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        id_to_label = {i: label for label, i in label_to_id.items()}
        return (label_to_id, id_to_label, len(unique_labels))

    def create_data_loaders(train_split, test_split, tokenizer, embedder, label_to_id, batch_size):
        """
        Creates PyTorch DataLoaders for training and testing datasets.
      # Rearrange dimensions to (batch_size, emb_dim, seq_len)
        Args:  # Apply first convolution and ReLU activation
            train_split (list): List of training examples.  # Apply second convolution and ReLU activation
            test_split (list): List of testing examples.  # Flatten features for the fully connected layers
            tokenizer (Tokenizer): Tokenizer instance for processing text.  # Apply first fully connected layer
            embedder (Embedder): Embedder instance for converting tokens to embeddings.  # Return output logits from the final layer
            label_to_id (dict): Mapping from label strings to numeric IDs.
            batch_size (int): Batch size for the DataLoaders.

        Returns:
            tuple: A tuple containing:
                - train_loader (DataLoader): DataLoader for the training dataset.
                - test_loader (DataLoader): DataLoader for the testing dataset.
        """
        train_dataset = TextClassificationDataset(train_split, tokenizer, embedder, label_to_id)
        test_dataset = TextClassificationDataset(test_split, tokenizer, embedder, label_to_id)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size)
        return (train_loader, test_loader)

    def save_model(model, prefix):  # Set model to evaluation mode
        """
        Saves the trained model's state dictionary and configuration to disk.
      # Disable gradient calculations for efficiency
        Args:
            model (nn.Module): Trained model to be saved.
            prefix (str): Prefix for the saved file name.
        """
        torch.save({'state_dict': model.state_dict(), 'config': model.config}, f'{prefix}_model.pth')  # Forward pass
      # Get predicted class indices
    def load_model(prefix):
        """
        Loads a saved model from disk and prepares it for evaluation.
      # Set model back to training mode
        Args:
            prefix (str): Prefix used during model saving.

        Returns:
            nn.Module: The loaded CNNTextClassifier model in evaluation mode.
        """
        checkpoint = torch.load(f'{prefix}_model.pth', map_location=torch.device('cpu'))
        config = checkpoint['config']
        model = CNNTextClassifier(emb_dim=config['emb_dim'], num_classes=config['num_classes'], seq_len=config['seq_len'], id_to_label=config['id_to_label'])
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model

    def test_model(model, test_input, tokenizer=None, embedder=None):
        """
        Tests the model on a single input text and prints the predicted label.
      # Extract and sort unique labels from the training data
        Args:
            model (nn.Module): Trained text classification model.  # Create a mapping from each label to a unique integer ID
            test_input (str): Input text to classify.
            tokenizer (Tokenizer, optional): Tokenizer instance. If None, a new one is created.  # Create the reverse mapping from ID to label
            embedder (Embedder, optional): Embedder instance. If None, embeddings are loaded and a new embedder is created.

        Notes:
            This function prints the input text along with the predicted emotion.
        """
        if not tokenizer:
            tokenizer = Tokenizer()
        if not embedder:
            embeddings, _emb_dim = load_embeddings(vectors_url)
            embedder = Embedder(embeddings, _emb_dim, seq_len)
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            tokens = tokenizer.tokenize(test_input)
            embeddings = embedder.embed(tokens)
            embeddings = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0).to(device)
            outputs = model(embeddings)
            _, predicted = torch.max(outputs.data, 1)
            predicted_label = model.config['id_to_label'][predicted.item()]
        print(f'Input: {test_input}')
        print(f'Predicted emotion: {predicted_label}')
      # Initialize the custom dataset for training and testing data
    def set_hyperparameters():
        """
        Defines and returns hyperparameters for training.  # Create DataLoaders; enable shuffling for training data

        Returns:
            tuple: A tuple containing:
                - num_epochs (int): Number of training epochs.
                - seq_len (int): Fixed sequence length for input text.
                - batch_size (int): Batch size for training.
                - learning_rate (float): Learning rate for the optimizer.
        """
        num_epochs = 2
        seq_len = 100
        batch_size = 32
        learning_rate = 0.001
        return (num_epochs, seq_len, batch_size, learning_rate)
    if __name__ == '__main__':  # Save the model's state and configuration as a checkpoint file
        set_seed(42)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        data_url = 'https://www.thelmbook.com/data/emotions'
        vectors_url = 'https://www.thelmbook.com/data/word-vectors'
        num_epochs, seq_len, batch_size, learning_rate = set_hyperparameters()
        train_loader, test_loader, id_to_label, num_classes, _emb_dim = download_and_prepare_data(data_url, vectors_url, seq_len, batch_size)
        model = CNNTextClassifier(_emb_dim, num_classes, seq_len, id_to_label)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            num_batches = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
            for batch in progress_bar:  # Load checkpoint containing model state and configuration
                batch_embeddings, batch_labels = batch
                batch_embeddings = batch_embeddings.to(device)
                batch_labels = batch_labels.to(device)  # Reinitialize the model using the saved configuration
                optimizer.zero_grad()
                outputs = model(batch_embeddings)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1  # Load the saved weights into the model
                progress_bar.set_postfix({'Loss': total_loss / num_batches})
            avg_loss = total_loss / num_batches  # Set the model to evaluation mode
            test_acc = calculate_accuracy(model, test_loader, device)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {test_acc:.4f}')
        model_name = 'CNN_classifier'
        save_model(model, model_name)  # Initialize tokenizer if not provided  # Initialize embedder if not provided by loading embeddings using global vectors_url and seq_len  # Determine the device from the model's parameters  # Set model to evaluation mode  # Tokenize the input text  # Convert tokens into embeddings  # Convert embeddings to a float tensor, add batch dimension, and move to the correct device  # Perform a forward pass through the model to get predictions  # Map the predicted numeric label to the actual label string  # Set random seeds for reproducibility  # Determine computation device: use GPU if available, otherwise CPU  # URLs for downloading the dataset and word embeddings  # Set training hyperparameters  # Download and prepare data loaders, label mappings, and embedding dimensions  # Initialize the CNN text classifier model with the embedding dimension and label mappings  # Move model to the appropriate device  # Define the loss function and optimizer  # Cross-entropy loss for classification  # AdamW optimizer  # Training loop over multiple epochs  # Set model to training mode  # Initialize progress bar for the current epoch  # Reset gradients before backpropagation  # Forward pass through the model  # Compute loss  # Backpropagation  # Update model parameters  # Update progress bar with the current average loss  # Evaluate model accuracy on the test set after each epoch  # Save the trained model to disk with the specified prefix
    return (
        Embedder,
        Tokenizer,
        load_embeddings,
        load_model,
        model_name,
        seq_len,
        test_model,
        vectors_url,
    )


@app.cell
def _(
    Embedder,
    Tokenizer,
    load_embeddings,
    load_model,
    model_name,
    seq_len,
    test_model,
    vectors_url,
):
    if __name__ == '__main__':
        loaded_model = load_model(model_name)  # Load the previously saved model using the specified model name prefix.
        embeddings, _emb_dim = load_embeddings(vectors_url)  # The 'load_model' function reads the checkpoint and reconstructs the CNNTextClassifier.
        tokenizer = Tokenizer()
        embedder = Embedder(embeddings, _emb_dim, seq_len)
        test_input = "I'm so happy to be able to train a text classifier!"  # Load the word embeddings from the provided URL.
        test_model(loaded_model, test_input, tokenizer, embedder)  # This returns a dictionary mapping words to their embedding vectors and the embedding dimension.  # Initialize the tokenizer.  # The Tokenizer here simply splits text into tokens based on whitespace.  # Create an embedder instance using the loaded embeddings, embedding dimension, and fixed sequence length.  # The Embedder will convert tokenized text into a tensor of embeddings suitable for the model.  # Define a sample input text to classify.  # Use the test_model function to evaluate the loaded model on the sample input.  # This function tokenizes the text, converts it to embeddings, performs a forward pass through the model,  # and prints out the predicted label.
    return


if __name__ == "__main__":
    app.run()
