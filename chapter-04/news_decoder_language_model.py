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
    <a href="https://colab.research.google.com/github/aburkov/theLMbook/blob/main/news_decoder_language_model.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="display: flex; justify-content: center;">
        <div style="background-color: #f4f6f7; padding: 15px; width: 80%;">
            <table style="width: 100%">
                <tr>
                    <td style="vertical-align: middle;">
                        <span style="font-size: 14px;">
                            A notebook for <a href="https://www.thelmbook.com" target="_blank" rel="noopener">The Hundred-Page Language Models Book</a> by Andriy Burkov<br><br>
                            Code repository: <a href="https://github.com/aburkov/theLMbook" target="_blank" rel="noopener">https://github.com/aburkov/theLMbook</a>
                        </span>
                    </td>
                    <td style="vertical-align: middle;">
                        <a href="https://www.thelmbook.com" target="_blank" rel="noopener">
                            <img src="https://thelmbook.com/img/book.png" width="80px" alt="The Hundred-Page Language Models Book">
                        </a>
                    </td>
                </tr>
            </table>
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Decoder-based language model

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
    import torch.nn.functional as F  # For softmax
    from torch.utils.data import DataLoader, IterableDataset  # For efficient data loading
    import random  # For setting random seeds
    from tqdm import tqdm  # For progress bars
    import math  # For computing perplexity using exp()
    import re  # For preprocessing text (replacing numbers with placeholders)
    from transformers import AutoTokenizer  # For loading pre-trained tokenizer
    #import tempfile         # For temporary file handling during extraction
    #import shutil           # For file operations during extraction

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
    # Dataset Class
        """

        def __init__(self, file_path, tokenizer, max_length=30):
            self.file_path = file_path
            self.tokenizer = _tokenizer
            self.max_length = max_length
            self._count_sentences()

        def __iter__(self):
            """
            Creates an iterator over the dataset.
            This method is called when iterating over the dataset.

            Yields:
                tuple: (input_sequence, target_sequence) pairs for language modeling  # Store file path for reading data
                      input_sequence is the sequence up to the last token
                      target_sequence is the sequence shifted one position right  # Store tokenizer for text processing
            """
            with open(self.file_path, 'r', encoding='utf-8') as f:  # Set maximum sequence length to truncate long sequences
                for line in f:
                    sentence = line.strip()
                    sentence = re.sub('\\d+', '###', sentence)
                    encoded_sentence = self.tokenizer.encode(sentence, max_length=self.max_length, truncation=True)
                    if len(encoded_sentence) >= 2:
                        input_seq = encoded_sentence[:-1]
                        target_seq = encoded_sentence[1:]
                        yield (torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long))

        def __len__(self):
            return self._num_sentences

        def _count_sentences(self):
            print(f'\nCounting sentences in {self.file_path}...')  # Open file in read mode with UTF-8 encoding
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self._num_sentences = sum((1 for _ in f))  # Process each line (sentence) in the file
            print(f'\nFound {self._num_sentences} sentences in {self.file_path}.')
      # Remove leading/trailing whitespace
    def create_collate_fn(tokenizer):
        """  # Replace all numbers with ### placeholder
        Creates a collate function for batching sequences of different lengths.  # This reduces vocabulary size and helps model generalize
        This function pads shorter sequences to match the longest sequence in the batch.

        Args:  # Convert sentence to token IDs
            tokenizer: Tokenizer object containing padding token information

        Returns:
            function: Collate function that handles padding in batches
        """

        def collate_fn(batch):  # Only use sequences with at least 2 tokens
            input_seqs, target_seqs = zip(*batch)  # (need at least one input and one target token)
            pad_index = _tokenizer.pad_token_id
            input_padded = nn.utils.rnn.pad_sequence(input_seqs, batch_first=True, padding_value=pad_index)  # Input is all tokens except last
            target_padded = nn.utils.rnn.pad_sequence(target_seqs, batch_first=True, padding_value=pad_index)
            return (input_padded, target_padded)  # Target is all tokens except first
        return collate_fn
      # Convert to PyTorch tensors and yield
    def check_file_exists(filename):
        """
        Checks if a file exists in the current directory.
        Args:
            filename (str): Name of the file to check
        Returns:
            bool: True if file exists, False otherwise
        """
        return os.path.exists(filename)

    ## ----------------------------
    ## Download and prepare data
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
        if not check_file_exists(filename):
            print(f'\nDownloading dataset from {url}...')
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:  # Separate inputs and targets from batch
                with open(filename, 'wb') as out_file:
                    out_file.write(response.read())  # Get padding token ID from tokenizer
            print('\nDownload completed.')
        else:  # Pad input sequences to same length
            print(f'\n{filename} already downloaded.')
        return filename  # Pad target sequences to same length

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

        Args:
            filename (str): Name of the archive file
        Returns:
            tuple: Paths to extracted train and test files  # Always use news.tar.gz as the filename, regardless of URL
        """
        data_dir = os.path.join(os.path.dirname(filename), 'news')
        train_path = os.path.join(data_dir, 'train.txt')
        test_path = os.path.join(data_dir, 'test.txt')
        if check_file_exists(train_path) and check_file_exists(test_path):
            print('\nData files already extracted.')
            return (train_path, test_path)
        print('\nListing archive contents:')
        with tarfile.open(filename, 'r:gz') as tar:
            for member in tar.getmembers():
                print(f'\nArchive member: {member.name}')
            print('\nExtracting files...')
            tar.extractall('.')
        if not (check_file_exists(train_path) and check_file_exists(test_path)):
            raise FileNotFoundError(f'\nRequired files not found in the archive. Please check the paths above.')
        print('\nExtraction completed.')
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
        print(f'\nTraining sentences: {len(train_dataset)}')
        print(f'\nTest sentences: {len(test_dataset)}')
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
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)
        return (train_dataloader, test_dataloader)

    def download_and_prepare_data(url, batch_size, tokenizer, max_length=30):  # Extract to current directory first
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

    def compute_loss_and_perplexity(model, dataloader, tokenizer, criterion, device, max_sentences=1000):
        """
        Evaluates model performance by computing loss and perplexity on data.  # Create training dataset

        Args:  # Create test dataset
            model (nn.Module): The language model to evaluate
            dataloader (DataLoader): Data loader containing batched sequences
            tokenizer: Tokenizer for handling special tokens like padding  # Print dataset sizes
            criterion: Loss function (usually CrossEntropyLoss)
            device: Device to run computation on (cuda/cpu)
            max_sentences (int): Maximum number of sentences to evaluate (default: 1000)
                               Limits evaluation to a subset for faster validation

        Returns:
            tuple: (average_loss, perplexity, sentences_processed)
                   - average_loss: Mean loss per token (excluding padding)
                   - perplexity: exp(average_loss), lower is better
        """
        _model.eval()
        total_loss = 0.0
        total_tokens = 0
        sentences_processed = 0
        with torch.no_grad():
            for input_seq, target_seq in tqdm(dataloader, desc='Evaluating', leave=False):
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)
                batch_size_current = input_seq.size(0)
                logits = _model(input_seq)  # Create training data loader
                logits = logits.reshape(-1, logits.size(-1))
                target = target_seq.reshape(-1)
                mask = target != _tokenizer.pad_token_id
                loss = criterion(logits[mask], target[mask])  # Function to handle padding
                loss_value = loss.item() * mask.sum().item()  # Number of worker processes (0 = single process)
                total_loss += loss_value
                total_tokens += mask.sum().item()  # Create test data loader
                sentences_processed += batch_size_current
                if sentences_processed >= max_sentences:
                    break
        average_loss = total_loss / total_tokens
        perplexity = math.exp(average_loss)
        return (average_loss, perplexity)

    def generate_text(model, start_string, tokenizer, device, max_length=50):
        """
        Generates text continuation from a given start string using greedy decoding.

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
        input_indices = _tokenizer.encode(start_string, add_special_tokens=False)  # Step 1: Download dataset archive from URL
        input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)
        generated_indices = input_indices.copy()
        for _ in range(max_length - len(input_indices)):  # Step 2: Extract training and test files from archive
            logits = _model(input_tensor)
            logits = logits[:, -1, :]
            if _tokenizer.unk_token_id is not None:  # Step 3: Create dataset objects for streaming data
                logits[:, _tokenizer.unk_token_id] = float('-inf')
            next_token = torch.argmax(logits, dim=-1)
            generated_indices.append(next_token.item())  # Step 4: Create function to handle batch creation
            if next_token.item() == _tokenizer.eos_token_id:
                break
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)  # Step 5: Create and return data loaders
        return _tokenizer.decode(generated_indices, skip_special_tokens=True)

    def save_model(model, tokenizer, model_name):
    # Evaluation Functions
        """
        Saves the model state dictionary and tokenizer using the specified model name.

        Args:
            model (nn.Module): The trained model to save
            tokenizer: The tokenizer used with the model
            model_name (str): Name to use for the saved model files
        """
        save_dir = os.path.join('models', _model_name)
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f'{_model_name}.pth')
        torch.save({'model_state_dict': _model.state_dict(), 'model_config': {'vocab_size': len(_tokenizer), 'emb_dim': _model.embedding.embedding_dim, 'num_heads': len(_model.layers[0].attn.heads), 'num_blocks': len(_model.layers), 'pad_idx': _model.embedding.padding_idx}}, model_path)
        tokenizer_path = os.path.join(save_dir, 'tokenizer')
        _tokenizer.save_pretrained(tokenizer_path)
        print(f"Model and tokenizer saved as '{_model_name}'")

    def load_model(model_name, device=None):
        """
        Loads a saved model and tokenizer using the model name.

        Args:  # Set model to evaluation mode (disables dropout, etc.)
            model_name (str): Name of the model to load
            device: Device to load the model onto (if None, uses available device)
      # Initialize counters for loss calculation
        Returns:  # Accumulator for total loss across all batches
            tuple: (loaded_model, loaded_tokenizer)  # Counter for total number of tokens (excluding padding)
        """  # Counter for number of sentences processed
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Disable gradient computation for efficiency
        save_dir = os.path.join('models', _model_name)
        if not os.path.exists(save_dir):  # Iterate through data with progress bar
            raise FileNotFoundError(f"No saved model found with name '{_model_name}'")
        tokenizer_path = os.path.join(save_dir, 'tokenizer')  # Move input and target sequences to specified device
        _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)  # Shape: (batch_size, seq_len)
        model_path = os.path.join(save_dir, f'{_model_name}.pth')  # Shape: (batch_size, seq_len)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        _model = DecoderLanguageModel(vocab_size=checkpoint['model_config']['vocab_size'], emb_dim=checkpoint['model_config']['emb_dim'], num_heads=checkpoint['model_config']['num_heads'], num_blocks=checkpoint['model_config']['num_blocks'], pad_idx=checkpoint['model_config']['pad_idx'])  # Get current batch size (might be smaller for last batch)
        _model.load_state_dict(checkpoint['model_state_dict'])
        _model.to(device)
        _model.eval()  # Forward pass through the model
        print(f"\nModel '{_model_name}' loaded successfully")  # Shape: (batch_size, seq_len, vocab_size)
        return (_model, _tokenizer)
      # Reshape logits and target for loss calculation
    def get_hyperparameters():  # Shape: (batch_size * seq_len, vocab_size)
        emb_dim = 128  # Shape: (batch_size * seq_len)
        num_heads = 8
        num_blocks = 2  # Create mask to exclude padding tokens
        batch_size = 128
        learning_rate = 0.001
        num_epochs = 1  # Compute loss only on non-padded tokens
        context_size = 30
        return (emb_dim, num_heads, num_blocks, batch_size, learning_rate, num_epochs, context_size)  # Update counters  # Total loss for this batch  # Accumulate batch loss  # Count non-padding tokens  # Update sentence counter and check if we've reached maximum  # Calculate final metrics  # Normalize loss by number of tokens  # Convert loss to perplexity  # Set model to evaluation mode to disable dropout and other training-specific behaviors  # Convert input string to token indices  # Convert indices to tensor and move to specified device (GPU/CPU)  # Keep track of all generated tokens, starting with input sequence  # Generate tokens until we hit max length or end-of-sequence token  # Get model predictions for the entire sequence  # Only take predictions for the last token position  # Prevent the model from generating unknown tokens by setting their probability to negative infinity  # Greedy decoding: select the token with highest probability  # Add the chosen token to our generated sequence  # If we generate an end-of-sequence token, stop generation  # Add the new token to input tensor for next iteration  # Convert token indices back to text, removing any special tokens  # Create the models directory if it doesn't exist  # Save the model state dictionary and configuration  # Save the tokenizer  # Check if model exists  # Load the tokenizer  # Load the model state and config  # Create a new model instance with the saved configuration  # Load the saved state dictionary

    # ----------------------------
    # Weight Initialization and Core Functions
    # This section contains utility functions for weight initialization
    # and core computational functions used throughout the model
    def initialize_weights(model):
        """
        Initialize the weights of different model components using appropriate schemes.
        Each layer type receives specialized initialization for optimal training.
        """
        for module in _model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)  # Xavier uniform initialization for linear layers
            elif isinstance(module, nn.Embedding):  # Helps maintain variance across network layers
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():  # Initialize biases to zero
                        module.weight[module.padding_idx].fill_(0)
            elif isinstance(module, AttentionHead):  # Initialize embedding layers with normal distribution
                nn.init.xavier_uniform_(module.W_Q)
                nn.init.xavier_uniform_(module.W_K)
                nn.init.xavier_uniform_(module.W_V)  # Ensure padding tokens have zero embeddings
            elif isinstance(module, MultiHeadAttention):
                nn.init.xavier_uniform_(module.W_O)
            elif isinstance(module, DecoderLanguageModel):
                nn.init.xavier_uniform_(module.output)  # Initialize query, key, and value projection matrices
            elif isinstance(module, RMSNorm):  # Xavier uniform helps maintain good gradient flow
                nn.init.ones_(module.scale)
            elif isinstance(module, MLP):
                nn.init.xavier_uniform_(module.W_1)
                nn.init.xavier_uniform_(module.W_2)
                nn.init.zeros_(module.B_1)  # Initialize output projection matrix for attention mechanism
                nn.init.zeros_(module.B_2)

    def rope(x, theta_base=10000.0):  # Initialize final output projection layer
        """
        Implements Rotary Position Embedding (RoPE) for transformer attention.
        RoPE encodes position information through rotation matrices applied to pairs of dimensions.  # Initialize RMSNorm scale parameters to ones
      # This starts with identity transformation
        Args:
            x: Input tensor of shape (batch_size, seq_len, emb_dim)
            theta_base: Base for computing rotation frequencies (default: 10000.0)  # Initialize feed-forward network parameters

        Returns:
            Tensor with position information encoded through rotations
        """
        batch_size, seq_len, emb_dim = x.size()
        assert emb_dim % 2 == 0, 'Embedding dimensionality must be even for RoPE'
        pos = torch.arange(0, seq_len, dtype=torch.float32, device=x.device)
        pos = pos.unsqueeze(0).expand(batch_size, seq_len)
        p = torch.arange(1, emb_dim // 2 + 1, dtype=torch.float32, device=x.device)
        theta_p = 1.0 / theta_base ** (2 * (p - 1) / emb_dim)
        pos = pos.unsqueeze(-1)
        theta = pos * theta_p
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        x_rotated_1 = x1 * cos_theta - x2 * sin_theta
        x_rotated_2 = x1 * sin_theta + x2 * cos_theta
        x_rotated = torch.stack((x_rotated_1, x_rotated_2), dim=-1).reshape(batch_size, seq_len, emb_dim)
        return x_rotated
      # Generate sequence position indices
    class RMSNorm(nn.Module):
        """
        Root Mean Square Layer Normalization
        A simplified alternative to Layer Normalization that only uses RMS statistics  # Compute frequency bands for each dimension pair
        """  # Modified: frequencies start from p=1 and use (p-1) in exponent

        def __init__(self, emb_dim, epsilon=1e-08):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(emb_dim))  # Compute rotation angles for each position and frequency
            self.epsilon = epsilon

        def forward(self, x):
            squared_x = x ** 2  # Compute rotation components
            mean_squared = torch.mean(squared_x, dim=-1, keepdim=True)
            rms = torch.sqrt(mean_squared + self.epsilon)
            x_normalized = x / rms
            output = x_normalized * self.scale  # Split input into alternating dimensions
            return output  # Dimensions at indices 0,2,4,...
      # Dimensions at indices 1,3,5,...
    class AttentionHead(nn.Module):
        """  # Apply 2D rotations to each pair
        Single head of self-attention
        Transforms input using learned projections and computes scaled dot-product attention
        """
      # Recombine rotated pairs into final output
        def __init__(self, emb_dim, d_h):
            super().__init__()
            self.W_Q = nn.Parameter(torch.rand(emb_dim, d_h))
            self.W_K = nn.Parameter(torch.rand(emb_dim, d_h))
            self.W_V = nn.Parameter(torch.rand(emb_dim, d_h))
    # Model Components
    # This section contains the building blocks of the transformer decoder
    # including normalization, attention, and feed-forward layers
            self.d_h = d_h

        def forward(self, x, mask):
            Q = x @ self.W_Q
            K = x @ self.W_K
            V = x @ self.W_V
            Q, K = (rope(Q), rope(K))
            scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_h)
            masked_scores = scores.masked_fill(mask == 0, float('-inf'))
            attention_weights = torch.softmax(masked_scores, dim=-1)  # Learnable scale parameter
            return attention_weights @ V  # Small constant for numerical stability

    class MultiHeadAttention(nn.Module):
        """  # Compute root mean square normalization
        Multi-head attention mechanism
        Allows the model to jointly attend to information from different positions
        """

        def __init__(self, emb_dim, num_heads):  # Normalize and scale
            super().__init__()
            d_h = emb_dim // num_heads
            self.heads = nn.ModuleList([AttentionHead(emb_dim, d_h) for _ in range(num_heads)])
            self.W_O = nn.Parameter(torch.rand(emb_dim, emb_dim))

        def forward(self, x, mask):
            head_outputs = [head(x, mask) for head in self.heads]
            x = torch.cat(head_outputs, dim=-1)
            return x @ self.W_O

    class MLP(nn.Module):
        """  # Initialize projection matrices for queries, keys, and values
        Multi-Layer Perceptron for transformer feed-forward network
        Uses a larger intermediate dimensionality (4x) with ReLU activation
        """
      # Dimensionality of attention head
        def __init__(self, emb_dim):
            super().__init__()
            self.W_1 = nn.Parameter(torch.rand(emb_dim, emb_dim * 4))  # Project input into query, key, and value spaces
            self.B_1 = nn.Parameter(torch.rand(emb_dim * 4))
            self.W_2 = nn.Parameter(torch.rand(emb_dim * 4, emb_dim))
            self.B_2 = nn.Parameter(torch.rand(emb_dim))

        def forward(self, x):  # Apply rotary position embeddings to queries and keys
            x = x @ self.W_1 + self.B_1
            x = torch.relu(x)
            x = x @ self.W_2 + self.B_2  # Compute attention scores with scaling factor
            return x

    class DecoderBlock(nn.Module):  # Apply causal mask and attention weights
        """
        Single transformer decoder block
        Combines self-attention and feed-forward layers with residual connections
        """

        def __init__(self, emb_dim, num_heads):
            super().__init__()
            self.norm1 = RMSNorm(emb_dim)
            self.attn = MultiHeadAttention(emb_dim, num_heads)
            self.norm2 = RMSNorm(emb_dim)
            self.mlp = MLP(emb_dim)

        def forward(self, x, mask):  # Dimensionality of each attention head
            attn_out = self.attn(self.norm1(x), mask)
            x = x + attn_out  # Create multiple attention heads
            mlp_out = self.mlp(self.norm2(x))
            x = x + mlp_out
            return x

    class DecoderLanguageModel(nn.Module):
        """  # Output projection matrix
        Complete decoder-only transformer language model
        Processes input sequences using multiple decoder blocks and projects to vocabulary
        """
      # Process input through each attention head
        def __init__(self, vocab_size, emb_dim, num_heads, num_blocks, pad_idx):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)  # Concatenate outputs and project to final dimensionality
            self.layers = nn.ModuleList([DecoderBlock(emb_dim, num_heads) for _ in range(num_blocks)])
            self.output = nn.Parameter(torch.rand(emb_dim, vocab_size))

        def forward(self, x):
            x = self.embedding(x)
            _, seq_len, _ = x.size()
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            for layer in self.layers:
                x = layer(x, mask)
            return x @ self.output  # Initialize weights and biases for two-layer feed-forward network  # First linear transformation and activation  # Second linear transformation  # Layer components  # Self-attention sub-block with residual connection  # Feed-forward sub-block with residual connection  # Token embedding layer  # Stack of decoder blocks  # Output projection to vocabulary size  # Embed input tokens  # Create causal attention mask  # Process through decoder blocks  # Project to vocabulary distribution

    # ----------------------------
    # Main training loop for a Decoder Language Model
    # This script handles the entire training process including data loading,
    # model training, validation, and text generation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if __name__ == '__main__':
        set_seed(42)
        emb_dim, num_heads, num_blocks, batch_size, learning_rate, num_epochs, context_size = get_hyperparameters()
        _tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3.5-mini-instruct')  # Initialize random seeds to ensure reproducible results
        pad_idx = _tokenizer.pad_token_id
        data_url = 'https://www.thelmbook.com/data/news'  # Retrieve model architecture and training hyperparameters from configuration
        train_dataloader, test_dataloader = download_and_prepare_data(data_url, batch_size, _tokenizer, context_size)  # emb_dim: dimensionality of input token and intermediary embeddings
        vocab_size = len(_tokenizer)  # num_heads: number of attention heads in each transformer block
        print(f'\nVocabulary size: {vocab_size}\n')  # num_blocks: number of transformer blocks in the model
        _model = DecoderLanguageModel(vocab_size, emb_dim, num_heads, num_blocks, pad_idx)  # batch_size: mini-batch size
        _model.to(device)  # learning_rate: step size for optimizer updates
        initialize_weights(_model)  # num_epochs: number of complete passes through the training dataset
        optimizer = torch.optim.AdamW(_model.parameters(), lr=learning_rate)  # context_size: maximum input sequence length
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        total_params = sum((p.numel() for p in _model.parameters() if p.requires_grad))
        print(f'\nTotal trainable parameters: {total_params}\n')  # Initialize the tokenizer using Microsoft's Phi-3.5-mini model
        eval_interval = 200000
        examples_processed = 0  # Get padding token index for padding shorter sequences
        _contexts = ['Moscow', 'New York', 'A hurricane', 'The President']
        for epoch in range(num_epochs):
            _model.train()  # Check for CUDA-capable GPU and set the device accordingly
            total_loss = 0.0
            total_tokens = 0
            progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')  # Download the news dataset and create DataLoader objects for training and testing
            for batch_idx, (input_seq, target_seq) in enumerate(progress_bar):  # DataLoaders handle batching and shuffling
                input_seq = input_seq.to(device)
                target_seq = target_seq.to(device)
                optimizer.zero_grad()
                logits = _model(input_seq)
                logits = logits.reshape(-1, logits.size(-1))
                target = target_seq.reshape(-1)  # Get the size of the vocabulary that the model needs to handle
                mask = target != pad_idx
                loss = criterion(logits[mask], target[mask])
                loss.backward()
                optimizer.step()  # Initialize the Decoder language model with specified architecture parameters
                loss_value = loss.item() * mask.sum().item()  # vocab_size: determines output layer dimensionality
                total_loss += loss_value  # emb_dim: size of token embeddings and intermediary embeddings
                total_tokens += mask.sum().item()  # num_heads: number of attention heads per transformer block
                examples_processed += input_seq.size(0)  # num_blocks: number of transformer blocks in the model
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})  # pad_idx: special token ID used for padding shorter sequences
                if examples_processed >= eval_interval:
                    avg_loss = total_loss / total_tokens
                    print(f'\nAfter {examples_processed} examples, Average Loss: {avg_loss:.4f}')
                    _model.eval()
                    average_loss, perplexity = compute_loss_and_perplexity(_model, test_dataloader, _tokenizer, criterion, device, max_sentences=1000)  # Move the model to GPU if available
                    print(f'\nValidation Average Loss: {average_loss:.4f}, Perplexity: {perplexity:.2f}')
                    _model.eval()
                    for _context in _contexts:  # Initialize model weights using custom initialization scheme
                        _generated_text = generate_text(model=_model, start_string=_context, tokenizer=_tokenizer, device=device, max_length=50)  # This is important for stable training of deep neural networks
                        print(f'\nContext: {_context}')
                        print(f'\nGenerated text: {_generated_text}\n')
                    _model.train()  # Initialize the AdamW optimizer with specified learning rate
                    examples_processed = 0
                    total_loss = 0.0
                    total_tokens = 0  # Initialize the loss function (Cross Entropy) for training
            if total_tokens > 0:  # ignore_index=pad_idx ensures that padding tokens don't contribute to the loss
                avg_loss = total_loss / total_tokens
                print(f'\nEpoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
            else:  # Calculate and display the total number of trainable parameters in the model
                print(f'\nEpoch {epoch + 1}/{num_epochs} completed.')
            _model.eval()
            print('\nGenerating text based on contexts using generate_text:\n')
            for _context in _contexts:  # Set evaluation interval (number of examples after which to perform validation)
                _generated_text = generate_text(model=_model, start_string=_context, tokenizer=_tokenizer, device=device, max_length=50)  # 200,000 examples provides a good balance between training time and monitoring frequency
                print(f'\nContext: {_context}')
                print(f'\nGenerated text: {_generated_text}\n')  # Counter for tracking progress toward next evaluation
            average_loss, perplexity = compute_loss_and_perplexity(_model, test_dataloader, _tokenizer, criterion, device, max_sentences=1000)
            print(f'\nValidation Average Loss: {average_loss:.4f}, Perplexity: {perplexity:.2f}')  # Define test contexts for generating sample text during evaluation
            _model.train()
        _model_name = 'Decoder_LM'
        save_model(_model, _tokenizer, _model_name)  # Main training loop - iterate through specified number of epochs  # Set model to training mode  # Initialize tracking variables for this epoch  # Accumulator for loss across all batches  # Counter for actual tokens processed (excluding padding)  # Create progress bar for monitoring training progress  # Iterate through batches in the training data  # Move input and target sequences to GPU if available  # Clear gradients from previous batch  # Forward pass: get model predictions for this batch  # output shape: (batch_size, seq_len, vocab_size)  # Reshape logits and target tensors for loss computation  # Create mask to exclude padding tokens from loss calculation  # Compute loss between model predictions and actual targets  # Using masked versions to ignore padding tokens  # Backward pass: compute gradients of loss with respect to model parameters  # Update model parameters using calculated gradients  # Calculate actual loss value for this batch accounting for padding  # Accumulate total loss and tokens for epoch statistics  # Update progress bar with current batch loss  # Periodic evaluation after processing specified number of examples  # Calculate average loss over the last eval_interval examples  # Switch to evaluation mode  # Compute validation metrics  # Record validation  # Generate sample texts to qualitatively assess model performance  # Generate text continuation for each test context  # Switch back to training mode for continued training  # Reset counters for next evaluation interval  # End-of-epoch reporting  # Calculate and display average loss for the epoch  # Handle edge case where no tokens were processed  # Perform end-of-epoch validation  # Generate sample texts for qualitative assessment  # Reset to training mode for next epoch  # Save the trained model and tokenizer for later use  # This includes model architecture, weights, and tokenizer configuration
    return device, generate_text, load_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Testing the model

    In the cell below, we load and test the language model:
    """)
    return


@app.cell
def _(device, generate_text, load_model):
    # ----------------------------
    # Model tests
    if __name__ == '__main__':
        _model_name = 'Decoder_LM'
        _model, _tokenizer = load_model(_model_name)
        _model.eval()
        print('\nTesting the model:\n')
        _contexts = ['Moscow', 'New York', 'A hurricane', 'The President']
        for _context in _contexts:  # Load the previously saved model and tokenizer from disk
            _generated_text = generate_text(model=_model, start_string=_context, tokenizer=_tokenizer, device=device, max_length=50)  # This recreates the exact model state from after training
            print(f'\nPrompt: {_context}')
            print(f'\nGenerated response: {_generated_text}\n')  # Print header for test section  # Define a list of test prompts to evaluate model performance  # Iterate through each test prompt and generate text  # Generate text using greedy decoding (most likely tokens)  # The loaded language model  # Text to continue  # Tokenizer for text conversion  # CPU or GPU device  # Maximum length of generated sequence  # Print the original prompt and model's response
    return


@app.cell
def _():
    from google.colab import runtime
    runtime.unassign()
    return


if __name__ == "__main__":
    app.run()
