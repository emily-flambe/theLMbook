import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    import re, torch, torch.nn as nn

    return nn, re, torch


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
    return docs, num_classes, raw_labels


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
    return labels, vectors


@app.cell
def _(nn):
    # Now we will build a MLP using the PyTorch Module API in PyTorch, one of their APIs for model definition (the other is the sequential API)

    class SimpleClassifier(nn.Module):
        """
        Implements a feedforward neural network with two layers.

        Intentionally omits a final softmax layer, as PyTorch's CrossEntropyLoss combines softmax and cross-entropy loss internally, eliminating the need for an explicit softmax in the model's forward pass.
        """
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__() # QUESTION what is this?
            # fc1 = fully connected layer 1
            self.fc1 = nn.Linear(input_dim, hidden_dim) 
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            x = self.fc1(x) # input x is passed to the first fully connected layer, which shapes it to have the original number of inputs and "hidden_dim" number of outputs
            x = self.relu(x) # output from fc1 is fed through ReLU activation function, maintaining its shape from fc1 (input_dim x hidden_dim)
            x = self.fc2(x) # second fully connected layer with the outputs from the ReLU layer (hidden_dim) being the number of inputs, and producing output_dim outputs
            return x

    return (SimpleClassifier,)


@app.cell
def _(SimpleClassifier, num_classes, vocabulary):
    input_dim = len(vocabulary)
    hidden_dim = 50 # QUESTION is this arbitrary or what?
    output_dim = num_classes

    model = SimpleClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    model
    return (model,)


@app.cell
def _(labels, model, nn, torch, vectors):
    # Now we define the loss function, choose the gradient descent algorithm, and set up the training loop:

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for step in range(3000):
        optimizer.zero_grad()
        loss = criterion(model(vectors), labels)
        loss.backward()
        optimizer.step() # QUESTION: how does step() work?
    return loss, step


@app.cell
def _(step):
    step
    return


@app.cell
def _(loss):
    loss
    return


@app.cell
def _(doc_to_bow, model, torch, vocabulary):
    # Once training is complete, we can test the model on a new document
    new_docs = [
        "Listening to rock music is fun.",
        "I love science very much."
    ]
    class_names = [
        "Cinema",
        "Music",
        "Science"
    ]

    new_doc_vectors = torch.tensor(
        data=[doc_to_bow(new_doc, vocabulary) for new_doc in new_docs],
        dtype=torch.float32
    )

    with torch.no_grad(): # no_grad() disables the default gradient tracking during testing or inference.
        outputs = model(new_doc_vectors) # processes all inputs simultaneously during inference
        predicted_ids = torch.argmax(outputs, dim=1) + 1 # torch.argmax identifies the highest logit (output probability score)'s index corresponding to the predicted class; add 1 back to compensate for earlier shift from 1-based to 0-based indexing QUESTION: why was all that index shifting really necessary? Why not just have the labels be zero-indexed from the start? QUESTION: what are logits?

    for i, new_doc in enumerate(new_docs):
        print(f'{new_doc}: {class_names[predicted_ids[i].item() - 1]}')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
