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
    <a href="https://colab.research.google.com/github/aburkov/theLMbook/blob/main/count_language_model.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="display: flex; justify-content: center;">
        <div style="background-color: #f4f6f7; padding: 15px; width: 80%;">
            <table style="width: 100%">
                <tbody><tr>
                    <td style="vertical-align: middle;">
                        <span style="font-size: 14px;">
                            A notebook for <a rel="noopener" href="https://www.thelmbook.com">The Hundred-Page Language Models Book</a> by Andriy Burkov<br><br>
                            Code repository: <a rel="noopener" href="https://github.com/aburkov/theLMbook">https://github.com/aburkov/theLMbook</a>
                        </span>
                    </td>
                    <td style="vertical-align: middle;">
                        <a rel="noopener" href="https://www.thelmbook.com">
                            <img data-canonical-src="https://thelmbook.com/img/book.png" alt="The Hundred-Page Language Models Book" width="80px" src="https://camo.githubusercontent.com/f3418fcb7da7c0ac7557629af1c0ccdf35ccff5ebe788c9987a4874c1a103c84/68747470733a2f2f7468656c6d626f6f6b2e636f6d2f696d672f626f6f6b2e706e67">
                        </a>
                    </td>
                </tr>
            </tbody></table>
        </div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Count-based language model

    ## Utility functions and classes

    In the cell below, we import the dependencies and define the utility functions and the model class:
    """)
    return


@app.cell
def _():
    # Import required libraries
    import re  # For regular expressions (text tokenization)
    import requests  # For downloading the corpus
    import gzip  # For decompressing the downloaded corpus
    import io  # For handling byte streams
    import math  # For mathematical operations (log, exp)
    import random  # For random number generation
    from collections import defaultdict  # For efficient dictionary operations
    import pickle, os  # For saving and loading the model

    def set_seed(seed):
        """
        Sets random seeds for reproducibility.

        Args:
            seed (int): Seed value for the random number generator
        """
        random.seed(seed)

    def download_corpus(url):
        """
        Downloads and decompresses a gzipped corpus file from the given URL.

        Args:
            url (str): URL of the gzipped corpus file

        Returns:
            str: Decoded text content of the corpus

        Raises:
            HTTPError: If the download fails
        """
        print(f'Downloading corpus from {url}...')
        response = requests.get(url)
        response.raise_for_status()  # Raises an exception for bad HTTP responses
        print('Decompressing and reading the corpus...')
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
            corpus = f.read().decode('utf-8')
        print(f'Corpus size: {len(corpus)} characters')
        return corpus

    class CountLanguageModel:
        """
        Implements an n-gram language model using count-based probability estimation.
        Supports variable context lengths up to n-grams.
        """

        def __init__(self, n):
            """
            Initialize the model with maximum n-gram length.

            Args:
                n (int): Maximum length of n-grams to use
            """
            self.n = n
            self.ngram_counts = [{} for _ in range(n)]  # Maximum n-gram length
            self.total_unigrams = 0  # List of dictionaries for each n-gram length
      # Total number of tokens in training data
        def predict_next_token(self, context):
            """
            Predicts the most likely next token given a context.
            Uses backoff strategy: tries largest n-gram first, then backs off to smaller n-grams.

            Args:
                context (list): List of tokens providing context for prediction

            Returns:
                str: Most likely next token, or None if no prediction can be made
            """
            for n in range(self.n, 1, -1):
                if len(context) >= n - 1:  # Start with largest n-gram, back off to smaller ones
                    context_n = tuple(context[-(n - 1):])
                    counts = self.ngram_counts[n - 1].get(context_n)  # Get the relevant context for this n-gram
                    if counts:
                        return max(counts.items(), key=lambda x: x[1])[0]
            unigram_counts = self.ngram_counts[0].get(())  # Return most frequent token
            if unigram_counts:  # Backoff to unigram if no larger context matches
                return max(unigram_counts.items(), key=lambda x: x[1])[0]
            return None

        def get_probability(self, token, context):
            for n in range(self.n, 1, -1):
                if len(context) >= n - 1:
                    context_n = tuple(context[-(n - 1):])
                    counts = self.ngram_counts[n - 1].get(context_n)
                    if counts:
                        total = sum(counts.values())
                        count = counts.get(token, 0)
                        if count > 0:
                            return count / total
            unigram_counts = self.ngram_counts[0].get(())
            count = unigram_counts.get(token, 0)
            V = len(unigram_counts)
            return (count + 1) / (self.total_unigrams + V)

    def train(model, tokens):
        """
        Trains the language model by counting n-grams in the training data.

        Args:
            model (CountLanguageModel): Model to train
            tokens (list): List of tokens from the training corpus
        """
        for n in range(1, _model.n + 1):
            counts = _model.ngram_counts[n - 1]
            for i in range(len(tokens) - n + 1):  # Train models for each n-gram size from 1 to n
                context = tuple(tokens[i:i + n - 1])
                next_token = tokens[i + n - 1]
                if context not in counts:  # Slide a window of size n over the corpus
                    counts[context] = defaultdict(int)
                counts[context][next_token] = counts[context][next_token] + 1  # Split into context (n-1 tokens) and next token
        _model.total_unigrams = len(tokens)

    def generate_text(model, context, num_tokens):
        """  # Initialize counts dictionary for this context if needed
        Generates text by repeatedly sampling from the model.

        Args:
            model (CountLanguageModel): Trained language model  # Increment count for this context-token pair
            context (list): Initial context tokens
            num_tokens (int): Number of tokens to generate
      # Store total number of tokens for unigram probability calculations
        Returns:
            str: Generated text including initial context
        """
        generated = list(context)
        while len(generated) - len(context) < num_tokens:
            next_token = _model.predict_next_token(generated[-(_model.n - 1):])
            generated.append(next_token)
            if len(generated) - len(context) >= num_tokens and next_token == '.':
                break
        return ' '.join(generated)

    def compute_perplexity(model, tokens, context_size):
        """
        Computes perplexity of the model on given tokens.
      # Start with the provided context
        Args:
            model (CountLanguageModel): Trained language model
            tokens (list): List of tokens to evaluate on  # Generate new tokens until we reach the desired length
            context_size (int): Maximum context size to consider
      # Use the last n-1 tokens as context for prediction
        Returns:
            float: Perplexity score (lower is better)
        """
        if not tokens:  # Stop if we've generated enough tokens AND found a period
            return float('inf')  # This helps ensure complete sentences
        total_log_likelihood = 0
        num_tokens = len(tokens)
        for i in range(num_tokens):
            context_start = max(0, i - context_size)  # Join tokens with spaces to create readable text
            context = tuple(tokens[context_start:i])
            token = tokens[i]
            probability = _model.get_probability(token, context)
            total_log_likelihood += math.log(probability)
        average_log_likelihood = total_log_likelihood / num_tokens
        perplexity = math.exp(-average_log_likelihood)
        return perplexity

    def tokenize(text):
        """
        Tokenizes text into words and periods.

        Args:
            text (str): Input text to tokenize
      # Handle empty token list
        Returns:
            list: List of lowercase tokens matching words or periods
        """
        return re.findall('\\b[a-zA-Z0-9]+\\b|[.]', text.lower())  # Initialize log likelihood accumulator

    def download_and_prepare_data(data_url):
        """
        Downloads and prepares training and test data.  # Calculate probability for each token given its context

        Args:  # Get appropriate context window, handling start of sequence
            data_url (str): URL of the corpus to download

        Returns:
            tuple: (training_tokens, test_tokens) split 90/10
        """  # Get probability of this token given its context
        corpus = download_corpus(data_url)
        tokens = tokenize(corpus)
        split_index = int(len(tokens) * 0.9)  # Add log probability to total (using log for numerical stability)
        train_corpus = tokens[:split_index]
        test_corpus = tokens[split_index:]
        return (train_corpus, test_corpus)  # Calculate average log likelihood

    def save_model(model, model_name):
        """  # Convert to perplexity: exp(-average_log_likelihood)
        Saves the trained language model to disk.  # Lower perplexity indicates better model performance

        Args:
            model (CountLanguageModel): Trained model to save
            model_name (str): Name to use for the saved model file

        Returns:
            str: Path to the saved model file

        Raises:
            IOError: If there's an error writing to disk
        """
        os.makedirs('models', exist_ok=True)
        model_path = os.path.join('models', f'{model_name}.pkl')
        try:
            print(f'Saving model to {model_path}...')
            with open(model_path, 'wb') as f:
                pickle.dump({'n': _model.n, 'ngram_counts': _model.ngram_counts, 'total_unigrams': _model.total_unigrams}, f)
            print('Model saved successfully.')
            return model_path
        except IOError as e:
            print(f'Error saving model: {e}')
            raise

    def load_model(model_name):
        """
        Loads a trained language model from disk.  # Download and extract the corpus

        Args:
            model_name (str): Name of the model to load  # Convert text to tokens

        Returns:
            CountLanguageModel: Loaded model instance  # Split into training (90%) and test (10%) sets

        Raises:
            FileNotFoundError: If the model file doesn't exist
            IOError: If there's an error reading the file
        """
        model_path = os.path.join('models', f'{model_name}.pkl')
        try:
            print(f'Loading model from {model_path}...')
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            _model = CountLanguageModel(model_data['n'])
            _model.ngram_counts = model_data['ngram_counts']
            _model.total_unigrams = model_data['total_unigrams']
            print('Model loaded successfully.')
            return _model
        except FileNotFoundError:
            print(f'Model file not found: {model_path}')
            raise
        except IOError as e:
            print(f'Error loading model: {e}')
            raise  # Create models directory if it doesn't exist

    def get_hyperparameters():
        """  # Construct file path
        Returns model hyperparameters.

        Returns:
            int: Size of n-grams to use in the model
        """
        n = 5
        return n  # Create new model instance  # Restore model state

    return (
        CountLanguageModel,
        compute_perplexity,
        download_and_prepare_data,
        generate_text,
        get_hyperparameters,
        load_model,
        save_model,
        set_seed,
        tokenize,
        train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training the model

    In the cell below, we load the data, train, and save the model:
    """)
    return


@app.cell
def _(
    CountLanguageModel,
    compute_perplexity,
    download_and_prepare_data,
    get_hyperparameters,
    save_model,
    set_seed,
    train,
):
    # Main model training block
    if __name__ == '__main__':
        set_seed(42)  # Initialize random seeds for reproducibility
        n = get_hyperparameters()
        model_name = 'count_model'
        data_url = 'https://www.thelmbook.com/data/brown'
        train_corpus, test_corpus = download_and_prepare_data(data_url)
        print('\nTraining the model...')  # Download and prepare the Brown corpus
        _model = CountLanguageModel(n)
        train(_model, train_corpus)
        print('\nModel training complete.')
        perplexity = compute_perplexity(_model, test_corpus, n)  # Train the model and evaluate its performance
        print(f'\nPerplexity on test corpus: {perplexity:.2f}')
        save_model(_model, model_name)
    return (model_name,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Testing the model

    Below, we load the trained model and use it to generate text:
    """)
    return


@app.cell
def _(generate_text, load_model, model_name, tokenize):
    # Main model testing block
    if __name__ == '__main__':
        _model = load_model(model_name)
        contexts = ['i will build a', 'the best place to', 'she was riding a']
        for context in contexts:
            tokens = tokenize(context)  # Test the model with some example contexts
            next_token = _model.predict_next_token(tokens)
            print(f'\nContext: {context}')
            print(f'Next token: {next_token}')
            print(f'Generated text: {generate_text(_model, tokens, 10)}')  # Generate completions for each context
    return


if __name__ == "__main__":
    app.run()
