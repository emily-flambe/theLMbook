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
    <a href="https://colab.research.google.com/github/aburkov/theLMbook/blob/main/news_RNN_language_model.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RNN-based language model

    ## Utility functions and classes

    In the cell below, we import the dependencies and define the utility functions and the model class:
    """)
    return


@app.cell
def _():
    # Import required libraries
    import os  # For file and path operations (check_file_exists, extract_dataset)
    import urllib.request  # For downloading dataset files from URLs
    import tarfile  # For extracting .tar.gz dataset archives
    import torch  # Main PyTorch library for tensor operations and deep learning
    import torch.nn as nn  # Neural network modules, layers, and utilities
    from torch.utils.data import DataLoader, IterableDataset  # For efficient data loading and streaming
    import random  # For setting random seeds in reproducibility
    from tqdm import tqdm  # For progress bars in training and evaluation
    import math  # For computing perplexity using exp()
    import re  # For preprocessing text (replacing numbers with placeholders)
    from transformers import AutoTokenizer  # For loading a pre-trained tokenizer

    # ----------------------------
    # Utility Functions
    def set_seed(seed):
        """
        Sets random seeds for reproducibility across different Python libraries.
        This ensures that random operations give the same results across runs.

        Args:
            seed (int): Seed value for random number generation
        """
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Set seed for Python's built-in random module
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Set seed for PyTorch's CPU random number generator

    class IterableTextDataset(IterableDataset):  # Set seed for PyTorch's GPU random number generator
        """
        An iterable dataset for processing text data in a memory-efficient way.  # Requests cuDNN to use deterministic algorithms when possible
        Instead of loading all data into memory, it streams data from disk.  # Note: This may impact performance and might not guarantee determinism in all cases
        Inherits from PyTorch's IterableDataset for streaming support.
      # Disables cuDNN's auto-tuner which finds the best algorithm for your specific input size
        Args:  # Ensures consistent behavior but might be slower as it doesn't optimize for input sizes
            file_path (str): Path to the text file containing sentences
            tokenizer: Tokenizer object for converting text to tokens
            max_length (int): Maximum sequence length to process (default: 30)
        """

        def __init__(self, file_path, tokenizer, max_length=30):
            self.file_path = file_path
            self.tokenizer = _tokenizer
            self.max_length = max_length
            self._count_sentences()

        def __iter__(self):
            """
            Creates an iterator over the dataset.
            This method is called when iterating over the dataset.  # Store file path for reading data

            Yields:  # Store tokenizer for text processing
                tuple: (input_sequence, target_sequence) pairs for language modeling
                      input_sequence is the sequence up to the last token  # Set maximum sequence length to truncate long sequences
                      target_sequence is the sequence shifted one position right
            """
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    sentence = line.strip()
                    sentence = re.sub('\\d+', '###', sentence)
                    encoded_sentence = self.tokenizer.encode(sentence, max_length=self.max_length, truncation=True)
                    if len(encoded_sentence) >= 2:
                        input_seq = encoded_sentence[:-1]
                        target_seq = encoded_sentence[1:]
                        yield (torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long))

        def __len__(self):
            return self._num_sentences  # Open file in read mode with UTF-8 encoding

        def _count_sentences(self):  # Process each line (sentence) in the file
            print(f'Counting sentences in {self.file_path}...')
            with open(self.file_path, 'r', encoding='utf-8') as f:  # Remove leading/trailing whitespace
                self._num_sentences = sum((1 for _ in f))
            print(f'Found {self._num_sentences} sentences in {self.file_path}.')  # Replace all numbers with ### placeholder
      # This reduces vocabulary size and helps model generalize
    def create_collate_fn(tokenizer):
        """
        Creates a collate function for batching sequences of different lengths.  # Convert sentence to token IDs
        This function pads shorter sequences to match the longest sequence in the batch.

        Args:
            tokenizer: Tokenizer object containing padding token information

        Returns:
            function: Collate function that handles padding in batches  # Only use sequences with at least 2 tokens
        """  # (need at least one input and one target token)

        def collate_fn(batch):  # Input is all tokens except last
            input_seqs, target_seqs = zip(*batch)
            pad_index = _tokenizer.pad_token_id  # Target is all tokens except first
            input_padded = nn.utils.rnn.pad_sequence(input_seqs, batch_first=True, padding_value=pad_index)
            target_padded = nn.utils.rnn.pad_sequence(target_seqs, batch_first=True, padding_value=pad_index)  # Convert to PyTorch tensors and yield
            return (input_padded, target_padded)
        return collate_fn

    def check_file_exists(filename):
        """
        Checks if a file exists in the current directory.
        Args:
            filename (str): Name of the file to check
        Returns:
            bool: True if file exists, False otherwise
    ## ----------------------------
    ## Download and prepare data
        """
        return os.path.exists(filename)

    def download_file(url):
        """
        Downloads a file from the given URL if it doesn't exist locally.
        Uses a custom User-Agent to help prevent download blocks.

        Args:
            url (str): URL of the file to download
        Returns:
            str: Name of the downloaded file ("news.tar.gz")
        """
        filename = 'news.tar.gz'
        if not check_file_exists(filename):  # Separate inputs and targets from batch
            print(f'Downloading dataset from {url}...')
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})  # Get padding token ID from tokenizer
            with urllib.request.urlopen(req) as response:
                with open(filename, 'wb') as out_file:  # Pad input sequences to same length
                    out_file.write(response.read())
            print('Download completed.')  # Pad target sequences to same length
        else:
            print(f'{filename} already downloaded.')
        return filename

    def is_within_directory(directory, target):
        """
        Checks if a target path is within a specified directory by comparing absolute paths.

        Args:
            directory (str): Base directory path
            target (str): Target path to check
        Returns:
            bool: True if target's absolute path starts with directory's absolute path
        """
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        prefix = os.path.commonprefix([abs_directory, abs_target])
        return prefix == abs_directory

    def extract_dataset(filename):
        """
        Extracts train.txt and test.txt from the downloaded archive.
        Includes debug information about archive contents.

        Args:  # Always use news.tar.gz as the filename, regardless of URL
            filename (str): Name of the archive file
        Returns:
            tuple: Paths to extracted train and test files
        """
        data_dir = os.path.join(os.path.dirname(filename), 'news')
        train_path = os.path.join(data_dir, 'train.txt')
        test_path = os.path.join(data_dir, 'test.txt')
        if check_file_exists(train_path) and check_file_exists(test_path):
            print('Data files already extracted.')
            return (train_path, test_path)
        print('\nListing archive contents:')
        with tarfile.open(filename, 'r:gz') as tar:
            for member in tar.getmembers():
                print(f'Archive member: {member.name}')
            print('\nExtracting files...')
            tar.extractall('.')
        if not (check_file_exists(train_path) and check_file_exists(test_path)):
            raise FileNotFoundError(f'Required files not found in the archive. Please check the paths above.')
        print('Extraction completed.')
        return (train_path, test_path)

    def create_datasets(train_file, test_file, tokenizer, max_length=30):
        """
        Creates IterableTextDataset objects for training and testing.
        These datasets will stream data from disk instead of loading it all into memory.

        Args:
            train_file (str): Path to training data file
            test_file (str): Path to test data file
            tokenizer: Tokenizer object for text processing

        Returns:
            tuple: (train_dataset, test_dataset) - Dataset objects for training and testing
        """
        train_dataset = IterableTextDataset(train_file, _tokenizer, max_length)
        test_dataset = IterableTextDataset(test_file, _tokenizer, max_length)
        print(f'Training sentences: {len(train_dataset)}')
        print(f'Test sentences: {len(test_dataset)}')
        return (train_dataset, test_dataset)

    def create_dataloaders(train_dataset, test_dataset, batch_size, collate_fn):
        """
        Creates DataLoader objects for efficient data iteration.

        Args:
            train_dataset: Training dataset
            test_dataset: Test dataset
            batch_size (int): Number of sequences per batch
            collate_fn: Function to handle padding and batch creation

        Returns:
            tuple: (train_dataloader, test_dataloader) - DataLoader objects for
                   iterating over batches of data with proper padding
        """
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)  # Extract to current directory first
        return (train_dataloader, test_dataloader)

    def download_and_prepare_data(url, batch_size, tokenizer, max_length=30):
        """
        Main function to handle the complete data preparation pipeline.
        Downloads data, extracts it, and creates necessary dataset objects.

        Args:
            url (str): URL where the dataset archive can be downloaded
            batch_size (int): Batch size for data loading
            tokenizer: Tokenizer object for text processing
            max_length (int): Maximum sequence length for tokenization (default: 30)

        Returns:
            tuple: (train_dataloader, test_dataloader) - Ready-to-use data loaders
        """
        filename = download_file(url)
        train_file, test_file = extract_dataset(filename)
        train_dataset, test_dataset = create_datasets(train_file, test_file, _tokenizer, max_length)
        collate_fn = create_collate_fn(_tokenizer)
        return create_dataloaders(train_dataset, test_dataset, batch_size, collate_fn)
      # Create training dataset
    def compute_loss_and_perplexity(model, dataloader, tokenizer, criterion, device, max_sentences=1000):
        """  # Create test dataset
        Evaluates model performance by computing loss and perplexity on data.

        Args:  # Print dataset sizes
            model (nn.Module): The language model to evaluate
            dataloader (DataLoader): Data loader containing batched sequences
            tokenizer: Tokenizer for handling special tokens like padding
            criterion: Loss function (usually CrossEntropyLoss)
            device: Device to run computation on (cuda/cpu)
            max_sentences (int): Maximum number of sentences to evaluate (default: 1000)
                               Limits evaluation to a subset for faster validation

        Returns:
            tuple: (average_loss, perplexity)
                   - average_loss: Mean loss per token (excluding padding)
                   - perplexity: exp(average_loss), lower is better
        """
        _model.eval()
        total_loss = 0.0
        total_tokens = 0
        sentences_processed = 0
        with torch.no_grad():
            for input_seq, target_seq in tqdm(dataloader, desc='Evaluating', leave=False):
                input_seq = input_seq.to(device)  # Create training data loader
                target_seq = target_seq.to(device)
                batch_size_current = input_seq.size(0)
                logits = _model(input_seq)
                logits = logits.reshape(-1, logits.size(-1))  # Function to handle padding
                target = target_seq.reshape(-1)  # Number of worker processes (0 = single process)
                mask = target != _tokenizer.pad_token_id
                loss = criterion(logits[mask], target[mask])  # Create test data loader
                loss_value = loss.item() * mask.sum().item()
                total_loss += loss_value
                total_tokens += mask.sum().item()
                sentences_processed += batch_size_current
                if sentences_processed >= max_sentences:
                    break
        _average_loss = total_loss / total_tokens
        _perplexity = math.exp(_average_loss)
        return (_average_loss, _perplexity)

    def perform_model_evaluation(model, test_dataloader, criterion, tokenizer, device, contexts):
        """
        Perform evaluation of the model including loss calculation, perplexity, and text generation.

        Args:
            model: The neural network model
            test_dataloader: DataLoader containing test/validation data
            criterion: Loss function
            tokenizer: Tokenizer for text generation
            device: Device to run computations on (cuda/cpu)
            contexts: List of context strings for text generation

        Returns:  # Step 1: Download dataset archive from URL
            tuple: (average_loss, perplexity)
        """
        _model.eval()  # Step 2: Extract training and test files from archive
        _average_loss, _perplexity = compute_loss_and_perplexity(_model, test_dataloader, _tokenizer, criterion, device, max_sentences=1000)
        print(f'Validation Average Loss: {_average_loss:.4f}, Perplexity: {_perplexity:.2f}')
        print('Generating text based on contexts using generate_text:\n')  # Step 3: Create dataset objects for streaming data
        for _context in _contexts:
            _generated_text = generate_text(model=_model, start_string=_context, tokenizer=_tokenizer, device=device, max_length=50)
            print(f'\nContext: {_context}')  # Step 4: Create function to handle batch creation
            print(f'\nGenerated text: {_generated_text}\n')
        return (_average_loss, _perplexity)
      # Step 5: Create and return data loaders
    def generate_text(model, start_string, tokenizer, device, max_length=50):
        """
        Generates text continuation from a given start string using greedy decoding.
        This method always chooses the most likely next token.

        Args:
            model (nn.Module): Trained language model
            start_string (str): Initial text to continue from
            tokenizer: Tokenizer for text processing
            device: Device to run generation on (cuda/cpu)
            max_length (int): Maximum length of generated sequence

        Returns:
            str: Generated text continuation
        """
        _model.eval()
        tokens = _tokenizer.encode(start_string, return_tensors='pt', max_length=max_length, truncation=True).to(device)
        generated = tokens
        for _ in range(max_length):
            output = _model(generated)
            next_token_logits = output[0, -1, :]  # Set model to evaluation mode (disables dropout, etc.)
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
            generated = torch.cat((generated, next_token_id), dim=1)
            if next_token_id.item() == _tokenizer.eos_token_id:  # Initialize counters for loss calculation
                break  # Accumulator for total loss across all batches
        _generated_text = _tokenizer.decode(generated.squeeze().tolist())  # Counter for total number of tokens (excluding padding)
        return _generated_text  # Counter for number of sentences processed

    def save_model(model, tokenizer, file_prefix):  # Disable gradient computation for efficiency
        model_state = {'state_dict': _model.state_dict(), 'vocab_size': _model.vocab_size, 'emb_dim': _model.emb_dim, 'num_layers': _model.num_layers, 'pad_index': _model.pad_index, 'training': _model.training}
        torch.save(model_state, f'{file_prefix}_model.pth')  # Iterate through data with progress bar
        _tokenizer.save_pretrained(f'{file_prefix}_tokenizer')
      # Move input and target sequences to specified device
    def load_model(file_prefix):  # Shape: (batch_size, seq_len)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Shape: (batch_size, seq_len)
        model_state = torch.load(f'{file_prefix}_model.pth', map_location=device, weights_only=True)
        _model = RecurrentLanguageModel(model_state['vocab_size'], model_state['emb_dim'], model_state['num_layers'], model_state['pad_index']).to(device)  # Get current batch size (might be smaller for last batch)
        _model.load_state_dict(model_state['state_dict'])
        _model.eval()
        _tokenizer = AutoTokenizer.from_pretrained(f'{file_prefix}_tokenizer')  # Forward pass through the model
        return (_model, _tokenizer)  # Shape: (batch_size, seq_len, vocab_size)

    def get_hyperparameters():  # Reshape logits and target for loss calculation
        """  # Shape: (batch_size * seq_len, vocab_size)
        Returns default hyperparameters for model training.  # Shape: (batch_size * seq_len)

        Returns:  # Create mask to exclude padding tokens
            tuple: (emb_dim, num_layers, batch_size, learning_rate, num_epochs)
        """
        emb_dim = 128  # Compute loss only on non-padded tokens
        num_layers = 2
        batch_size = 128
        learning_rate = 0.001  # Update counters
        num_epochs = 1  # Total loss for this batch
        context_size = 30  # Accumulate batch loss
        return (emb_dim, num_layers, batch_size, learning_rate, num_epochs, context_size)  # Count non-padding tokens  # Update sentence counter and check if we've reached maximum  # Calculate final metrics  # Normalize loss by number of tokens  # Convert loss to perplexity  # Switch to evaluation mode  # Compute metrics  # Generate text using the contexts  # The loaded language model  # Context to continue  # Tokenizer for text conversion  # CPU or GPU device  # Maximum length of generated sequence  # Set model to evaluation mode  # Convert start string to token ids and move to device  # return_tensors="pt" returns PyTorch tensor instead of list  # Initialize generated sequence with input tokens  # Generate new tokens one at a time  # Get model's predictions  # Shape: (1, seq_len, vocab_size)  # Get logits for the next token (last position)  # Shape: (vocab_size)  # Choose token with highest probability (greedy decoding)  # unsqueeze twice to match expected shape (1, 1)  # Add new token to generated sequence  # Stop if end of sequence token is generated  # Convert token ids back to text  # Save training state  # Load state dict to the correct device first  # Create model and move it to device before loading state dict  # Load state dict after model is on correct device  # Keep model on device  # Set to evaluation mode  # Embedding dimension  # Number of RNN layers  # Training batch size  # Learning rate for optimization  # Number of training epochs  # Maximum input sequence length

    # ----------------------------
    # Recurrent Language Model Class (merged from a separate cell to avoid a
    # cyclic dependency with the dataset/utility cell above).
    def initialize_weights(model):
        """
        Initializes model weights using Xavier uniform initialization for multi-dimensional
        parameters and uniform initialization for biases and other 1D parameters.

        Args:
            model (nn.Module): PyTorch model whose weights need to be initialized
        """
        for name, param in _model.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)  # Loop through all named parameters in the model
            else:
                nn.init.uniform_(param)  # Check if parameter has more than 1 dimension (e.g., weight matrices)

    class ElmanRNNUnit(nn.Module):  # Use Xavier uniform initialization for weight matrices
        """  # This helps prevent vanishing/exploding gradients by keeping the variance constant
        Implementation of a single Elman RNN unit (a simple recurrent neural network cell).
        This is the basic building block of our RNN that processes one time step of input.
      # For 1D parameters (like biases), use simple uniform initialization
        Args:
            emb_dim (int): Dimension of the embedding/hidden state vectors
        """

        def __init__(self, emb_dim):
            super(ElmanRNNUnit, self).__init__()
            self.Uh = nn.Parameter(torch.rand(emb_dim, emb_dim))
            self.Wh = nn.Parameter(torch.rand(emb_dim, emb_dim))
            self.b = nn.Parameter(torch.rand(emb_dim))

        def forward(self, x, h):
            """
            Computes one step of the RNN unit.  # Hidden-to-hidden weight matrix: transforms previous hidden state
      # Shape: (emb_dim, emb_dim)
            Args:
                x (torch.Tensor): Current input tensor of shape (batch_size, emb_dim)
                h (torch.Tensor): Previous hidden state of shape (batch_size, emb_dim)  # Input-to-hidden weight matrix: transforms current input
      # Shape: (emb_dim, emb_dim)
            Returns:
                torch.Tensor: New hidden state of shape (batch_size, emb_dim)
      # Bias term added to the sum of transformations
            The formula implemented is: h_new = tanh(x @ Wh + h @ Uh + b)  # Shape: (emb_dim,)
            where @ represents matrix multiplication
            """
            input_transform = x @ self.Wh
            hidden_transform = h @ self.Uh
            return torch.tanh(input_transform + hidden_transform + self.b)

    class ElmanRNN(nn.Module):
        """
        Multi-layer Elman RNN implementation that processes entire sequences.
        Stacks multiple RNN units to create a deeper network that can learn more complex patterns.

        Args:
            emb_dim (int): Dimension of embeddings and hidden states
            num_layers (int): Number of stacked RNN layers
        """

        def __init__(self, emb_dim, num_layers):  # 1. Transform current input: x @ Wh
            super().__init__()
            self.emb_dim = emb_dim
            self.num_layers = num_layers  # 2. Transform previous hidden state: h @ Uh
            self.rnn_units = nn.ModuleList([ElmanRNNUnit(emb_dim) for _ in range(num_layers)])

        def forward(self, x):  # 3. Add both transformations and bias
            """  # 4. Apply tanh activation function to get new hidden state
            Processes input sequence through all RNN layers.  # tanh squashes values to range (-1, 1), helping prevent exploding gradients

            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, seq_len, emb_dim)

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, seq_len, emb_dim)
            """
            batch_size, seq_len, emb_dim = x.size()
            h_prev = [torch.zeros(batch_size, emb_dim, device=x.device) for _ in range(self.num_layers)]
            outputs = []
            for t in range(seq_len):
                input_t = x[:, t]
                for l, rnn_unit in enumerate(self.rnn_units):
                    h_new = rnn_unit(input_t, h_prev[l])
                    h_prev[l] = h_new
                    input_t = h_new
                outputs.append(input_t)  # Create a list of RNN units, one for each layer
            return torch.stack(outputs, dim=1)  # ModuleList is used so PyTorch tracks all parameters

    class RecurrentLanguageModel(nn.Module):
        """
        Complete language model implementation combining embedding layer,
        multi-layer RNN, and output projection layer.

        The model architecture is:
        1. Input tokens -> Embedding Layer -> Embedded Vectors
        2. Embedded Vectors -> RNN Layers -> Context Vectors
        3. Context Vectors -> Linear Layer -> Vocabulary Predictions

        Args:
            vocab_size (int): Size of the vocabulary (number of unique tokens)
            emb_dim (int): Dimension of embeddings and hidden states
            num_layers (int): Number of RNN layers  # Get dimensions from input tensor
            pad_index (int): Index used for padding tokens
        """
      # Initialize hidden states for each layer with zeros
        def __init__(self, vocab_size, emb_dim, num_layers, pad_index):  # Each hidden state has shape (batch_size, emb_dim)
            super().__init__()
            self.vocab_size = vocab_size
            self.emb_dim = emb_dim
            self.num_layers = num_layers
            self.pad_index = pad_index
            self.embedding = nn.Embedding(vocab_size, emb_dim, pad_index)  # Will store outputs for each time step
            self.rnn = ElmanRNN(emb_dim=emb_dim, num_layers=num_layers)
            self.fc = nn.Linear(emb_dim, vocab_size)
      # Process each time step
        def forward(self, x):
            """  # Get input for current time step
            Processes input sequences through the entire model.

            Args:  # Process through each layer
                x (torch.Tensor): Input tensor of token indices
                                Shape: (batch_size, seq_len)  # Compute new hidden state for this layer

            Returns:
                torch.Tensor: Output logits for next token prediction  # Update hidden state for this layer
                             Shape: (batch_size, seq_len, vocab_size)

            Process:  # Output of this layer becomes input to next layer
            1. Convert token indices to embeddings
            2. Process embeddings through RNN layers
            3. Project RNN outputs to vocabulary size  # Add final layer's output to results
            """
            embeddings = self.embedding(x)
            rnn_output = self.rnn(embeddings)  # Stack all time steps' outputs into a single tensor
            logits = self.fc(rnn_output)  # Shape: (batch_size, seq_len, emb_dim)
            return logits  # Save model parameters  # Embedding layer: converts token indices to dense vectors  # pad_index tokens will be mapped to zero vectors  # RNN layers for processing sequences  # Final linear layer to convert RNN outputs to vocabulary predictions  # Output size is vocab_size to get logits for each possible token  # Convert token indices to embeddings  # Shape: (batch_size, seq_len) -> (batch_size, seq_len, emb_dim)  # Process through RNN layers  # Shape: (batch_size, seq_len, emb_dim) -> (batch_size, seq_len, emb_dim)  # Project to vocabulary size to get logits  # Shape: (batch_size, seq_len, emb_dim) -> (batch_size, seq_len, vocab_size)

    return (
        AutoTokenizer,
        RecurrentLanguageModel,
        compute_loss_and_perplexity,
        download_and_prepare_data,
        generate_text,
        get_hyperparameters,
        initialize_weights,
        load_model,
        nn,
        save_model,
        set_seed,
        torch,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model Classes

    RNN language model classes and the initialization method are defined in the cell above (merged for marimo compatibility).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training the language model

    In the cell below, we load the data, train, and save the language model:
    """)
    return


@app.cell
def _(
    AutoTokenizer,
    RecurrentLanguageModel,
    compute_loss_and_perplexity,
    download_and_prepare_data,
    generate_text,
    get_hyperparameters,
    initialize_weights,
    nn,
    save_model,
    set_seed,
    torch,
    tqdm,
):
    # ----------------------------
    # Main training loop for a Recurrent Neural Network Language Model
    # This script handles the entire training process including data loading,
    # model training, validation, and text generation
    if __name__ == '__main__':
        set_seed(42)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        emb_dim, num_layers, batch_size, learning_rate, num_epochs, context_size = get_hyperparameters()  # Initialize random seeds to ensure reproducible results
        _tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3.5-mini-instruct')
        vocab_size = len(_tokenizer)
        data_url = 'https://www.thelmbook.com/data/news'  # Check for CUDA-capable GPU and set the device accordingly
        train_dataloader, test_dataloader = download_and_prepare_data(data_url, batch_size, _tokenizer, context_size)
        _model = RecurrentLanguageModel(vocab_size, emb_dim, num_layers, _tokenizer.pad_token_id)
        _model.to(device)  # Retrieve model architecture and training hyperparameters from configuration
        initialize_weights(_model)  # emb_dim: dimensionality of token embeddings and hidden states
        total_params = sum((p.numel() for p in _model.parameters() if p.requires_grad))  # num_layers: number of recurrent layers in the model
        print(f'Total trainable parameters: {total_params}\n')  # batch_size: mini-batch size
        criterion = nn.CrossEntropyLoss(ignore_index=_tokenizer.pad_token_id)  # learning_rate: step size for optimizer updates
        optimizer = torch.optim.AdamW(_model.parameters(), lr=learning_rate)  # num_epochs: number of complete passes through the training dataset
        eval_interval = 200000  # context_size: maximum input sequence length
        examples_processed = 0
        _contexts = ['Moscow', 'New York', 'The hurricane', 'The President']
        for epoch in range(num_epochs):  # Initialize the tokenizer using Microsoft's Phi-3.5-mini model
            _model.train()
            print(f'Starting Epoch {epoch + 1}/{num_epochs}, Model in training mode: {_model.training}')
            total_loss = 0.0  # Get the size of the vocabulary that the model needs to handle
            total_tokens = 0
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')
            for batch_idx, (input_seq, target_seq) in enumerate(progress_bar):  # Download the news dataset and create DataLoader objects for training and testing
                input_seq = input_seq.to(device)  # DataLoaders handle batching and shuffling
                target_seq = target_seq.to(device)
                batch_size_current, seq_len = input_seq.shape
                optimizer.zero_grad()
                output = _model(input_seq)  # Initialize the RNN language model with specified architecture parameters
                output = output.reshape(batch_size_current * seq_len, vocab_size)  # vocab_size: determines output layer dimensionality
                target = target_seq.reshape(batch_size_current * seq_len)  # emb_dim: size of word embeddings and hidden states
                non_padding_token_count = (target != _tokenizer.pad_token_id).sum().item()  # num_layers: number of RNN layers
                loss = criterion(output, target)  # pad_token_id: special token ID used for padding shorter sequences
                loss.backward()
                optimizer.step()
                loss_value = loss.item() * non_padding_token_count  # Move the model to GPU if available
                total_loss += loss_value
                total_tokens += non_padding_token_count
                examples_processed += batch_size_current  # Initialize model weights using custom initialization scheme
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})  # This is important for stable training of deep neural networks
                if examples_processed >= eval_interval:
                    avg_loss = total_loss / total_tokens
                    print(f'\nAfter {examples_processed} examples, Average Loss: {avg_loss:.4f}')  # Calculate and display the total number of trainable parameters in the model
                    _model.eval()
                    _average_loss, _perplexity = compute_loss_and_perplexity(_model, test_dataloader, _tokenizer, criterion, device, max_sentences=1000)
                    print(f'Validation Average Loss: {_average_loss:.4f}, Perplexity: {_perplexity:.2f}')
                    print('Generating text based on contexts using generate_text:\n')  # Initialize the loss function (Cross Entropy) for training
                    for _context in _contexts:  # ignore_index=pad_token_id ensures that padding tokens don't contribute to the loss
                        _generated_text = generate_text(model=_model, start_string=_context, tokenizer=_tokenizer, device=device, max_length=50)  # This prevents the model from learning to predict padding tokens
                        print(f'\nContext: {_context}')
                        print(f'\nGenerated text: {_generated_text}')
                    _model.train()  # Initialize the AdamW optimizer with specified learning rate
                    examples_processed = 0
                    total_loss = 0.0
                    total_tokens = 0  # Set evaluation interval (number of examples after which to perform validation)
            if total_tokens > 0:  # 200,000 examples provides a good balance between training time and monitoring frequency
                avg_loss = total_loss / total_tokens
                print(f'\nEpoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')  # Counter for tracking progress toward next evaluation
            else:
                print(f'\nEpoch {epoch + 1}/{num_epochs} completed.')  # Define test contexts for generating sample text during evaluation
            _model.eval()
            _average_loss, _perplexity = compute_loss_and_perplexity(_model, test_dataloader, _tokenizer, criterion, device, max_sentences=1000)
            print(f'Validation Average Loss: {_average_loss:.4f}, Perplexity: {_perplexity:.2f}\n')
            print('Generating text based on contexts using generate_text:\n')
            for _context in _contexts:
                _generated_text = generate_text(model=_model, start_string=_context, tokenizer=_tokenizer, device=device, max_length=50)
                print(f'\nContext: {_context}')
                print(f'\nGenerated text: {_generated_text}')  # Main training loop - iterate through specified number of epochs
            _model.train()
        _model_name = 'RNN_LM'  # Set model to training mode
        save_model(_model, _tokenizer, _model_name)  # Initialize tracking variables for this epoch  # Accumulator for loss across all batches  # Counter for actual tokens processed (excluding padding)  # Create progress bar for monitoring training progress  # Iterate through batches in the training data  # Move input and target sequences to GPU if available  # Get current batch dimensions for reshaping operations  # Clear gradients from previous batch  # This is necessary as PyTorch accumulates gradients by default  # Forward pass: get model predictions for this batch  # output shape: (batch_size, seq_len, vocab_size)  # Reshape output and target tensors for loss computation  # - output: reshape to (batch_size * seq_len, vocab_size) for CrossEntropyLoss  # - target: reshape to (batch_size * seq_len) to match CrossEntropyLoss requirements  # Count number of non-padding tokens in target  # This is needed because to calculate the average loss for multiple batches we need to divide the total loss  # by the number of tokens in these batches, but criterion(output, target) returns the average loss per token in a batch.  # So, we will multiply the loss per token by the number of tokens to get the loss per batch  # Compute loss between model predictions and actual targets  # Backward pass: compute gradients of loss with respect to model parameters  # Update model parameters using calculated gradients  # Calculate actual loss value for this batch  # Multiply the loss per token by number of non-padding tokens to get total loss for the batch  # Accumulate total loss for epoch statistics  # Accumulate total number of actual tokens processed  # Increment counter for examples processed  # Update progress bar with current batch loss  # Periodic evaluation after processing specified number of examples  # Calculate average loss over the last eval_interval examples  # Switch to evaluation mode  # Compute validation metrics  # average_loss: mean loss on validation set  # perplexity: exponential of average loss, lower is better  # sentences_processed: number of validation sequences evaluated  # Generate sample texts to qualitatively assess model performance  # Generate text continuation for each test context  # The loaded language model  # Context to continue  # Tokenizer for text conversion  # CPU or GPU device  # Maximum length of generated sequence  # Switch back to training mode for continued training  # Reset counters for next evaluation interval  # End-of-epoch reporting  # Calculate and display average loss for the epoch  # Handle edge case where no tokens were processed  # Perform end-of-epoch validation  # Generate text continuation for each test context  # Limit generation to 50 tokens for brevity  # Reset to training mode for next epoch  # Save the trained model and tokenizer for later use  # This includes model architecture, weights, and tokenizer configuration
    return criterion, device, test_dataloader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Testing the model

    In the cell below, we load and test the language model:
    """)
    return


@app.cell
def _(
    compute_loss_and_perplexity,
    criterion,
    device,
    generate_text,
    load_model,
    test_dataloader,
):
    # ----------------------------
    # Model tests
    if __name__ == '__main__':
        _model_name = 'RNN_LM'
        _model, _tokenizer = load_model(_model_name)
        _model.eval()
        _average_loss, _perplexity = compute_loss_and_perplexity(_model, test_dataloader, _tokenizer, criterion, device, max_sentences=1000)
        print(f'Validation Average Loss: {_average_loss:.4f}, Perplexity: {_perplexity:.2f}\n')
        print('Testing the model:\n')  # Load the previously saved model and tokenizer from disk
        _contexts = ['Moscow', 'New York', 'A hurricane', 'The President']  # This recreates the exact model state from after training
        for _context in _contexts:
            _generated_text = generate_text(model=_model, start_string=_context, tokenizer=_tokenizer, device=device, max_length=50)
            print(f'\nPrompt: {_context}')
            print(f'\nGenerated response: {_generated_text}')  # Print header for test section  # Define a list of test prompts to evaluate model performance  # Iterate through each test prompt and generate text  # Generate text using greedy decoding (most likely tokens)  # The loaded language model  # Context to continue  # Tokenizer for text conversion  # CPU or GPU device  # Maximum length of generated sequence  # Print the original prompt and model's response
    return


@app.cell
def _():
    from google.colab import runtime
    runtime.unassign()
    return


if __name__ == "__main__":
    app.run()
