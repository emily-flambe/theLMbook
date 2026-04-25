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
    <a href="https://colab.research.google.com/github/aburkov/theLMbook/blob/main/emotion_classifier_LR.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
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


@app.cell
def _():
    # Import required libraries
    import gzip             # For decompressing gzipped data files
    import json             # For parsing JSON-formatted data
    import random           # For shuffling dataset and setting seeds
    import requests         # For downloading dataset from URL
    from sklearn.feature_extraction.text import CountVectorizer # Text vectorization utility
    from sklearn.linear_model import LogisticRegression         # Logistic regression model
    from sklearn.metrics import accuracy_score                  # For model evaluation

    # ----------------------------
    # Utility Functions
    # ----------------------------

    def set_seed(seed):
        """
        Sets random seed for reproducibility.

        Args:
            seed (int): Seed value for random number generation
        """
        random.seed(seed)

    def download_and_split_data(data_url, test_ratio=0.1):
        """
        Downloads emotion classification dataset from URL and splits into train/test sets.
        Handles decompression and JSON parsing of the raw data.

        Args:
            data_url (str): URL of the gzipped JSON dataset
            test_ratio (float): Proportion of data to use for testing (default: 0.1)

        Returns:
            tuple: (X_train, y_train, X_test, y_test) containing:
                - X_train, X_test: Lists of text examples for training and testing
                - y_train, y_test: Lists of corresponding emotion labels
        """
        # Download and decompress the dataset
        response = requests.get(data_url)
        content = gzip.decompress(response.content).decode()

        # Parse JSON lines into list of dictionaries
        dataset = [json.loads(line) for line in content.splitlines()]

        # Shuffle dataset for random split
        random.shuffle(dataset)

        # Split into train and test sets
        split_index = int(len(dataset) * (1 - test_ratio))
        train, test = dataset[:split_index], dataset[split_index:]

        # Separate text and labels
        X_train = [item["text"] for item in train]
        y_train = [item["label"] for item in train]
        X_test = [item["text"] for item in test]
        y_test = [item["label"] for item in test]

        return X_train, y_train, X_test, y_test

    # ----------------------------
    # Main Execution
    # ----------------------------

    # Set random seed for reproducibility
    set_seed(42)

    # Download and prepare dataset
    data_url = "https://www.thelmbook.com/data/emotions"
    X_train_text, y_train, X_test_text, y_test = download_and_split_data(
        data_url, test_ratio=0.1
    )

    print("Number of training examples:", len(X_train_text))
    print("Number of test examples:", len(X_test_text))

    # ----------------------------
    # Baseline Model
    # ----------------------------

    # Initialize text vectorizer with basic parameters
    # max_features=10_000: Limit vocabulary to top 10k most frequent words
    # binary=True: Convert counts to binary indicators (0/1)
    vectorizer = CountVectorizer(max_features=10_000, binary=True)

    # Transform text data to numerical features
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Initialize and train logistic regression model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # Make predictions on train and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate and display accuracy metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"\nTrain accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    # ----------------------------
    # Improved Model
    # ----------------------------

    print("\n--- Better hyperparameters ---")

    # Initialize vectorizer with improved parameters
    # max_features=20000: Increased vocabulary size
    # ngram_range=(1, 2): Include both unigrams and bigrams
    vectorizer = CountVectorizer(max_features=20000, ngram_range=(1, 2))

    # Transform text data with new vectorizer
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Train and evaluate model with same parameters
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    return


if __name__ == "__main__":
    app.run()
