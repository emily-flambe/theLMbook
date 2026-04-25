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
    <a href="https://colab.research.google.com/github/aburkov/theLMbook/blob/main/byte_pair_encoding.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
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
    # Training the BPE model

    Below, we load the data and train the BPE model:
    """)
    return


@app.cell
def _():
    # Import required libraries
    import os  # For file operations and path handling
    import urllib.request  # For downloading files
    import tarfile  # For extracting tar files
    import pickle  # For saving/loading tokenizer
    import re  # For regex in merge operations
    import time  # For timing operations
    from collections import defaultdict  # For counting tokens and pairs

    def download_file(url, filename):
        """
        Downloads a file from a URL if it doesn't exist locally.
        Prevents redundant downloads by checking file existence.

        Args:
            url (str): URL to download the file from
            filename (str): Local path to save the downloaded file

        Returns:
            None: Prints status messages about download progress
        """
        if not os.path.exists(filename):  # Check if file already exists to avoid re-downloading
            print(f'Downloading dataset from {url}...')
            urllib.request.urlretrieve(url, filename)
            print('Download completed.')
        else:
            print(f'{filename} already downloaded.')

    def is_within_directory(directory, target):
        """
        Security check to prevent path traversal attacks by verifying target path.
        Ensures extracted files remain within the intended directory.

        Args:
            directory (str): Base directory path to check against
            target (str): Target path to validate

        Returns:
            bool: True if target is within directory, False otherwise
        """
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)  # Convert both paths to absolute form for comparison
        prefix = os.path.commonprefix([abs_directory, abs_target])
        return prefix == abs_directory
      # Get common prefix to check containment
    def safe_extract_tar(tar_file, required_files):
        """
        Safely extracts specific files from a tar archive with security checks.
        Prevents path traversal attacks and extracts only required files.

        Args:
            tar_file (str): Path to the tar archive file
            required_files (list): List of filenames to extract

        Returns:
            None: Extracts files and prints progress

        Raises:
            Exception: If path traversal attempt is detected
        """
        with tarfile.open(tar_file, 'r:gz') as tar:
            for member in tar.getmembers():
                if not is_within_directory('.', member.name):
                    raise Exception('Attempted Path Traversal in Tar File')
            for member in tar.getmembers():  # Perform security check on all archive members
                if any((member.name.endswith(file) for file in required_files)):
                    member.name = os.path.basename(member.name)
                    tar.extract(member, '.')
                    print(f'Extracted {member.name}')
      # Extract only the specified files
    def create_word_generator(filepath):
        """
        Creates a generator that yields words from a text file one at a time.  # Remove path prefix for safety
        Memory efficient way to process large text files.

        Args:
            filepath (str): Path to text file to read

        Returns:
            generator: Yields individual words from the file
        """

        def generator():
            with open(filepath, 'r') as f:
                for line in f:
                    for _word in line.split():
                        yield _word
        return generator()

    def download_and_prepare_data(url):
        """
        Downloads, extracts, and prepares dataset for training.
        Handles both downloading and extraction with security checks.

        Args:
            url (str): URL of the dataset to download

        Returns:
            generator: Word generator for the training data
        """
        required_files = ['train.txt', 'test.txt']
        filename = os.path.basename(url)
        download_file(url, filename)
        if not all((os.path.exists(file) for file in required_files)):
            print('Extracting files...')
            safe_extract_tar(filename, required_files)
            print('Extraction completed.')
        else:
            print("'train.txt' and 'test.txt' already extracted.")
        return create_word_generator('train.txt')  # Download dataset if needed

    def initialize_vocabulary(corpus):
        """  # Extract required files if they don't exist
        Creates initial vocabulary from corpus by splitting words into characters.
        Adds word boundary marker '_' and tracks unique characters.

        Args:
            corpus (iterable): Iterator or list of words to process

        Returns:
            tuple: (vocabulary dict mapping tokenized words to counts,  # Create and return word generator
                   set of unique characters in corpus)
        """
        vocabulary = defaultdict(int)
        charset = set()
        for _word in corpus:
            word_with_marker = '_' + _word
            characters = list(word_with_marker)
            charset.update(characters)
            tokenized_word = ' '.join(characters)
            vocabulary[tokenized_word] += 1
        return (vocabulary, charset)

    def get_pair_counts(vocabulary):
        """
        Counts frequencies of adjacent symbol pairs in the vocabulary.  # Track word counts and unique characters
        Used to identify most common pairs for merging.

        Args:
            vocabulary (dict): Dictionary mapping tokenized words to their counts
      # Add word boundary marker and split into characters
        Returns:
            defaultdict: Maps token pairs to their frequency counts
        """  # Update set of unique characters
        pair_counts = defaultdict(int)
        for tokenized_word, count in vocabulary.items():  # Create space-separated string of characters
            _tokens = tokenized_word.split()
            for i in range(len(_tokens) - 1):  # Increment count for this tokenized word
                pair = (_tokens[i], _tokens[i + 1])
                pair_counts[pair] += count
        return pair_counts

    def merge_pair(vocab, pair):
        """
        Merges all occurrences of a specific symbol pair in the vocabulary.
        Uses regex for accurate token boundary matching.

        Args:
            vocab (dict): Current vocabulary dictionary
            pair (tuple): Pair of tokens to merge

        Returns:
            dict: New vocabulary with specified pair merged
        """
        new_vocab = {}
        bigram = re.escape(' '.join(pair))  # Split word into tokens
        pattern = re.compile('(?<![^\\s])' + bigram + '(?![^\\s])')
        for tokenized_word, count in vocab.items():  # Count adjacent pairs weighted by word frequency
            new_tokenized_word = pattern.sub(''.join(pair), tokenized_word)
            new_vocab[new_tokenized_word] = count
        return new_vocab

    def byte_pair_encoding(corpus, vocab_size):
        """
        Implements the BPE algorithm to learn a subword vocabulary.
        Iteratively merges most frequent character pairs until target vocabulary size is reached.

        Args:
            corpus (iterable): Iterator or list of words to learn BPE from
            vocab_size (int): Target vocabulary size to stop merging at

        Returns:
            tuple: (final vocabulary dict, list of merge operations,
                   set of base characters, set of all tokens)
        """
        vocab, charset = initialize_vocabulary(corpus)
        merges = []  # Create regex pattern for matching the pair
        _tokens = set(charset)
        while len(_tokens) < vocab_size:
            pair_counts = get_pair_counts(vocab)
            if not pair_counts:  # Apply merge to all words in vocabulary
                break
            most_frequent_pair = max(pair_counts, key=pair_counts.get)
            merges.append(most_frequent_pair)
            vocab = merge_pair(vocab, most_frequent_pair)
            new_token = ''.join(most_frequent_pair)
            _tokens.add(new_token)
        return (vocab, merges, charset, _tokens)

    def tokenize_word(word, merges, charset, unk_token='<UNK>'):
        """
        Tokenizes a single word using learned BPE merges.
        Handles unknown characters with UNK token.

        Args:
            word (str): Word to tokenize
            merges (list): List of learned merge operations
            charset (set): Set of known characters
            unk_token (str): Token to use for unknown characters
      # Initialize vocabulary with character-level tokens
        Returns:
            list: List of tokens for the word
        """
        _word = '_' + _word
        _tokens = [char if char in charset else unk_token for char in _word]  # Keep merging pairs until we reach target vocab size
        for left, right in merges:
            i = 0  # Get counts of all adjacent token pairs
            while i < len(_tokens) - 1:
                if _tokens[i:i + 2] == [left, right]:
                    _tokens[i:i + 2] = [left + right]
                else:
                    i += 1  # Find and record the most frequent pair
        return _tokens

    def build_merge_map(merges):
        """  # Update vocabulary by merging the most frequent pair
        Creates a mapping from token pairs to their merged forms.
        Preserves merge order for consistent tokenization.
      # Add the new merged token to our token set
        Args:
            merges (list): List of merge operations

        Returns:
            dict: Maps token pairs to (merged_token, merge_priority) tuples
        """
        merge_map = {}
        for i, (left, right) in enumerate(merges):
            merged_token = left + right
            merge_map[left, right] = (merged_token, i)
        return merge_map

    def tokenize_word_fast(word, merge_map, vocabulary, charset, unk_token='[UNK]'):
        """
        Optimized tokenization function using pre-computed merge map.
        Produces identical results to original algorithm but faster.

        Args:
            word (str): Word to tokenize
            merge_map (dict): Mapping of token pairs to merged forms  # Add word boundary marker and convert to characters
            vocabulary (dict): Current vocabulary dictionary
            charset (set): Set of known characters
            unk_token (str): Token to use for unknown characters
      # Apply merges in order
        Returns:
            list: List of tokens for the word
        """
        word_with_prefix = '_' + _word
        if word_with_prefix in vocabulary:
            return [word_with_prefix]
        _tokens = [char if char in charset else unk_token for char in word_with_prefix]
        while True:
            pairs_with_positions = []
            for i in range(len(_tokens) - 1):
                pair = (_tokens[i], _tokens[i + 1])
                if pair in merge_map:
                    merged_token, merge_priority = merge_map[pair]
                    pairs_with_positions.append((i, pair, merged_token, merge_priority))
            if not pairs_with_positions:
                break
            pairs_with_positions.sort(key=lambda x: (x[3], x[0]))
            pos, pair, merged_token, _ = pairs_with_positions[0]
            _tokens[pos:pos + 2] = [merged_token]
        return _tokens

    def save_tokenizer(merges, charset, tokens, vocab, filename='tokenizer.pkl'):  # Build map with merge priorities
        """
        Saves tokenizer state to a pickle file for later use.

        Args:
            merges (list): List of merge operations
            charset (set): Set of known characters
            tokens (set): Set of all tokens
            vocab (dict): Vocabulary dictionary mapping words to counts
            filename (str): Path to save tokenizer state

        Returns:
            None: Saves tokenizer to disk
        """
        with open(filename, 'wb') as f:
            pickle.dump({'merges': merges, 'charset': charset, 'tokens': _tokens, 'vocab': vocab}, f)

    def load_tokenizer(filename='tokenizer.pkl'):
        """
        Loads tokenizer state from a pickle file.

        Args:  # Check if word exists in vocabulary as-is
            filename (str): Path to saved tokenizer state

        Returns:
            dict: Dictionary containing tokenizer components
        """  # Initialize with characters, replacing unknown ones
        with open(filename, 'rb') as f:
            return pickle.load(f)
    if __name__ == '__main__':  # Keep merging until no more merges possible
        vocab_size = 5000
        max_corpus_size = 500000  # Find all possible merge operations
        data_url = 'https://www.thelmbook.com/data/news'
        word_gen = download_and_prepare_data(data_url)
        word_list = []
        for _word in word_gen:
            word_list.append(_word)
            if len(word_list) >= max_corpus_size:
                break
        print('Training BPE tokenizer...')  # Exit if no more merges possible
        vocab, merges, charset, _tokens = byte_pair_encoding(word_list, vocab_size)
        print('Saving the tokenizer...')
    # Main function for downloading, training BPE, saving, and loading tokenizer
        save_tokenizer(merges, charset, _tokens, vocab)  # Sort by merge priority and position for consistency  # Apply first valid merge  # Configuration parameters  # Target vocabulary size  # Maximum number of words to process  # Dataset source  # Download and prepare training data  # Collect corpus up to maximum size  # Train BPE tokenizer  # Save trained tokenizer
    return (
        build_merge_map,
        load_tokenizer,
        time,
        tokenize_word,
        tokenize_word_fast,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Testing the trained BPE tokenizer

    Once the BPE tokenizer is trained, we can load it and apply to a new text:
    """)
    return


@app.cell
def _(
    build_merge_map,
    load_tokenizer,
    time,
    tokenize_word,
    tokenize_word_fast,
):
    if __name__ == '__main__':
        print('Loading the tokenizer...')
        tokenizer = load_tokenizer()
        sentence = "Let's proceed to the language modeling part."
        start_time = time.time()  # Tokenize the sample sentence using the loaded tokenizer
        tokenized_sentence = [tokenize_word(_word, tokenizer['merges'], tokenizer['charset']) for _word in sentence.split()]
        elapsed = time.time() - start_time
        print('\nSentence tokenized with the straightforward implementation:')
        for _word, _tokens in zip(sentence.split(), tokenized_sentence):
            print(f'{_word} -> {_tokens}')
        print('--- Elapsed: %s seconds ---' % elapsed)
        merge_map = build_merge_map(tokenizer['merges'])
        start_time = time.time()
        fast_tokenized_sentence = [tokenize_word_fast(_word, merge_map, tokenizer['vocab'], tokenizer['charset']) for _word in sentence.split()]
        elapsed = time.time() - start_time
        print('\nSentence tokenized with a fast implementation:')
        for _word, _tokens in zip(sentence.split(), fast_tokenized_sentence):
            print(f'{_word} -> {_tokens}')
        print('--- Elapsed: %s seconds ---' % (time.time() - start_time))
        print('\nVocabulary size:', len(tokenizer['tokens']))
    return


if __name__ == "__main__":
    app.run()
