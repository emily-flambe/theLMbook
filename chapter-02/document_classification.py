import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    import re, torch, torch.nn as nn

    return re, torch


@app.cell
def _(re):
    def tokenize(text):
        """
        Tokenizes a single document.
        """
        return re.findall(r"\w+", text.lower())

    def get_vocabulary(texts):
        """
        Constructs the vocabulary for a corpus.
        """
        tokens = {token for text in texts for token in tokenize(text)}
        return {word: idx for idx, word in enumerate(sorted(tokens))}

    def doc_to_bow(doc, vocabulary):
        """
        Feature extraction function that converts a document to a feature vector.
        """
        tokens = set(tokenize(doc))
        bow = [0] * len(vocabulary)
        for token in tokens:
            if token in vocabulary:
                bow[vocabulary[token]] = 1
        return bow

    return doc_to_bow, get_vocabulary


@app.cell
def _(torch):
    torch.manual_seed(42)

    docs = [
        "Movies are fun for everyone.",
        "Watching movies is great fun.",
        "Enjoy a great movie today.",
        "Research is interesting and important.",
        "Learning math is very important.",
        "Science discovery is interesting.",
        "Rock is great to listen to.",
        "Listen to music for fun.",
        "Music is fun for everyone.",
        "Listen to folk music!"
    ]

    raw_labels = [1, 1, 1, 3, 3, 3, 2, 2, 2, 2]
    num_classes = len(set(raw_labels))
    return docs, raw_labels


@app.cell
def _(docs, get_vocabulary):
    vocabulary = get_vocabulary(docs)
    return (vocabulary,)


@app.cell
def _(vocabulary):
    vocabulary
    return


@app.cell
def _(doc_to_bow, docs, raw_labels, torch, vocabulary):
    # Transform our documents and labels into numbers

    vectors = torch.tensor(
        data=[doc_to_bow(doc, vocabulary) for doc in docs],
        dtype=torch.float32
    )

    labels = torch.tensor(data=raw_labels, dtype=torch.long) - 1 # converts labels 1, 2, 3 to 0, 1, 2 because PyTorch expects class indices to begin at 0
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
