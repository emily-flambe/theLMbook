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
    <a href="https://colab.research.google.com/github/aburkov/theLMbook/blob/main/sampling_method.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
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
    # Token sampling method

    ## Method implementation

    In the cell below, we implement the token sampling method that combines temperature, top-k, and top-p:
    """)
    return


@app.cell
def _():
    import numpy as np

    def validate_inputs(logits, vocabulary, temperature, top_k, top_p):
        """
        Validate all input parameters for the token sampling process.

        Args:
            logits (list): Raw model output scores for each token
            vocabulary (list): List of all possible tokens
            temperature (float): Temperature parameter for logits scaling
            top_k (int): Number of highest probability tokens to keep
            top_p (float): Cumulative probability threshold for nucleus sampling

        Raises:
            ValueError: If any parameters are invalid or out of expected ranges
        """
        if len(logits) != len(vocabulary):
            raise ValueError("Mismatch between logits and vocabulary sizes.")
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        if top_k < 0 or top_k > len(logits):
            raise ValueError("top_k must be between 0 and len(logits).")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in the range (0, 1].")

    def get_token_counts(prev_tokens, vocabulary):
        """
        Count the frequency of each token in the previous generation history.

        Args:
            prev_tokens (list): Previously generated tokens
            vocabulary (list): List of all possible tokens

        Returns:
            dict: Mapping of token indices to their frequencies
        """
        token_counts = {}
        if prev_tokens is not None:
            for token in prev_tokens:
                if token in vocabulary:
                    idx = vocabulary.index(token)
                    token_counts[idx] = token_counts.get(idx, 0) + 1
        return token_counts

    def apply_presence_penalty(logits, token_counts, presence_penalty):
        """
        Apply presence penalty to tokens that have appeared before.

        Args:
            logits (numpy.ndarray): Token logits
            token_counts (dict): Mapping of token indices to their frequencies
            presence_penalty (float): Fixed penalty to subtract from logits of present tokens

        Returns:
            numpy.ndarray: Modified logits with presence penalty applied

        Note:
            Unlike frequency penalty, this applies the same penalty regardless of frequency
        """
        if presence_penalty != 0.0:
            for idx in token_counts:
                logits[idx] -= presence_penalty
        return logits

    def apply_frequency_penalty(logits, token_counts, frequency_penalty):
        """
        Apply frequency penalty proportional to token occurrence count.

        Args:
            logits (numpy.ndarray): Token logits
            token_counts (dict): Mapping of token indices to their frequencies
            frequency_penalty (float): Penalty factor to multiply with token frequency

        Returns:
            numpy.ndarray: Modified logits with frequency penalty applied

        Note:
            Penalty increases linearly with token frequency
        """
        if frequency_penalty != 0.0:
            for idx, count in token_counts.items():
                logits[idx] -= frequency_penalty * count
        return logits

    def apply_temperature(logits, temperature):
        """
        Apply temperature scaling to logits to control randomness.

        Args:
            logits (numpy.ndarray): Token logits
            temperature (float): Temperature parameter (>1 increases randomness, <1 decreases it)

        Returns:
            numpy.ndarray: Temperature-scaled and normalized logits

        Note:
            - Higher temperature makes distribution more uniform
            - Lower temperature makes distribution more peaked
            - Normalizes by subtracting max for numerical stability
        """
        if temperature != 1.0:
            logits = logits / temperature
        return logits - np.max(logits)

    def apply_top_k_filtering(logits, top_k, min_tokens_to_keep=1):
        """
        Apply top-k filtering to keep only the k highest probability tokens.

        Args:
            logits (numpy.ndarray): Token logits
            top_k (int): Number of top tokens to keep
            min_tokens_to_keep (int): Minimum number of tokens to keep regardless of top-k

        Returns:
            numpy.ndarray: Modified logits with all but top-k tokens set to -inf

        Note:
            Ensures at least min_tokens_to_keep tokens remain available for sampling
        """
        if top_k > 0:
            indices_to_remove = np.argsort(logits)[:-min_tokens_to_keep]
            indices_to_keep = np.argsort(logits)[-top_k:]
            for idx in indices_to_remove:
                if idx not in indices_to_keep:
                    logits[idx] = float('-inf')
        return logits

    def apply_top_p_filtering(logits, top_p, min_tokens_to_keep=1):
        """
        Apply nucleus (top-p) filtering to keep tokens comprising top p probability mass.

        Args:
            logits (numpy.ndarray): Token logits
            top_p (float): Cumulative probability threshold (0 to 1)
            min_tokens_to_keep (int): Minimum number of tokens to keep regardless of top-p

        Returns:
            numpy.ndarray: Modified logits with unlikely tokens set to -inf

        Note:
            1. Converts logits to probabilities
            2. Sorts tokens by probability
            3. Keeps minimal set of tokens whose cumulative probability >= top_p
            4. Ensures at least min_tokens_to_keep tokens remain
        """
        if top_p < 1.0:
            probs = np.exp(logits)
            probs = probs / probs.sum()

            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)

            sorted_indices_to_remove = sorted_indices[cumulative_probs > top_p]

            if len(sorted_indices_to_remove) > len(sorted_indices) - min_tokens_to_keep:
                sorted_indices_to_remove = sorted_indices_to_remove[
                    :len(sorted_indices) - min_tokens_to_keep
                ]

            logits[sorted_indices_to_remove] = float('-inf')
        return logits

    def convert_to_probabilities(logits):
        """
        Convert logits to a valid probability distribution using softmax.

        Args:
            logits (numpy.ndarray): Token logits

        Returns:
            numpy.ndarray: Probability distribution summing to 1
        """
        probs = np.exp(logits)
        return probs / probs.sum()

    def sample_token(logits, vocabulary, temperature=0.7, top_k=0, top_p=1.0,
                    repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
                    prev_tokens=None):
        """
        Main function for sampling the next token using various sampling strategies.
        Applies sampling methods in the same order as the transformers library.

        Args:
            logits (list): Raw model output scores for each token
            vocabulary (list): List of all possible tokens
            temperature (float): Temperature for logits scaling (default: 0.7)
            top_k (int): Number of highest probability tokens to keep (default: 0, disabled)
            top_p (float): Cumulative probability threshold for nucleus sampling (default: 1.0)
            repetition_penalty (float): Penalty for repeated tokens (default: 1.0, no penalty)
            presence_penalty (float): Fixed penalty for token presence (default: 0.0)
            frequency_penalty (float): Penalty scaled by token frequency (default: 0.0)
            prev_tokens (list): Previously generated tokens (default: None)

        Returns:
            str: Sampled token from vocabulary

        Process:
            1. Validate all input parameters
            2. Apply repetition, presence, and frequency penalties
            3. Apply temperature scaling
            4. Apply top-k and top-p filtering
            5. Convert to probability distribution and sample
        """
        validate_inputs(logits, vocabulary, temperature, top_k, top_p)

        logits = np.array(logits, dtype=np.float64)

        # 1. Apply penalties
        token_counts = get_token_counts(prev_tokens, vocabulary)
        logits = apply_presence_penalty(logits, token_counts, presence_penalty)
        logits = apply_frequency_penalty(logits, token_counts, frequency_penalty)

        # 2. Apply temperature scaling
        logits = apply_temperature(logits, temperature)

        # 3. Apply filtering
        logits = apply_top_k_filtering(logits, top_k)
        logits = apply_top_p_filtering(logits, top_p)

        # 4. Convert to probabilities and sample
        probabilities = convert_to_probabilities(logits)
        return np.random.choice(vocabulary, p=probabilities)

    if __name__ == "__main__":
        # Create a test vocabulary and corresponding logits
        vocabulary = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
        logits = np.array([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5])

        print("Test vocabulary:", vocabulary)
        print("Initial logits:", logits)
        print("\nSampling with different parameters:")

        # Test 1: Default parameters
        print("\nTest 1: Default parameters (temperature=0.7, no top-k/p filtering)")
        samples = [sample_token(logits.copy(), vocabulary) for _ in range(5)]
        print("Samples:", samples)

        # Test 2: High temperature (more random)
        print("\nTest 2: High temperature (temperature=2.0)")
        samples = [sample_token(logits.copy(), vocabulary, temperature=2.0) for _ in range(5)]
        print("Samples:", samples)

        # Test 3: Low temperature (more deterministic)
        print("\nTest 3: Low temperature (temperature=0.2)")
        samples = [sample_token(logits.copy(), vocabulary, temperature=0.2) for _ in range(5)]
        print("Samples:", samples)

        # Test 4: Top-k filtering
        print("\nTest 4: Top-k filtering (top_k=3)")
        samples = [sample_token(logits.copy(), vocabulary, top_k=3) for _ in range(5)]
        print("Samples:", samples)

        # Test 5: Top-p filtering
        print("\nTest 5: Top-p filtering (top_p=0.9)")
        samples = [sample_token(logits.copy(), vocabulary, top_p=0.9) for _ in range(5)]
        print("Samples:", samples)

        # Test 6: Combined filtering
        print("\nTest 6: Combined filtering (temperature=0.5, top_k=3, top_p=0.9)")
        samples = [sample_token(logits.copy(), vocabulary, temperature=0.5, top_k=3, top_p=0.9)
                  for _ in range(5)]
        print("Samples:", samples)

        # Demonstrate error handling
        print("\nError handling examples:")
        try:
            # Test with mismatched sizes
            sample_token(logits[:5], vocabulary)
        except ValueError as e:
            print("Expected error:", e)

        try:
            # Test with invalid temperature
            sample_token(logits, vocabulary, temperature=0)
        except ValueError as e:
            print("Expected error:", e)
    return


if __name__ == "__main__":
    app.run()
