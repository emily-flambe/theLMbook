# /// script
# dependencies = ["datasets", "evaluate", "flagembedding", "peft", "transformers"]
# ///

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
    <a href="https://colab.research.google.com/github/aburkov/theLMbook/blob/main/document_classifier_with_LLMs_as_labelers.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tutorial: Training a Document Classifier by a Taxonomy

    In this tutorial, we will train a document classifier capable of classfying documents according to a given taxonomy.

    Contrary to many exising tutorials that start with a labeled dataset ready for training, we will:

    1. Download a corpus of unlabeled documents
    2. Download a real-world taxonomy
    3. Use an ensemble of language model APIs to label the documents by the taxonomy with a high accuracy
    4. Build a labeled dataset for supervised learning from the documents and LLM-assigned labels
    5. Use Colab and SwarmoOne to train a series of models of different sizes to see how the classifier quality depends on the model size

    Let's go!

    ## Getting the Documents

    The documents we will train our neural networks to classify will be titles and abstracts from a collection of STEM papers. First, we download from [Kaggle](https://www.kaggle.com/) a raw file with the [ArXiv](https://arxiv.org/) paper metadata for our collection of papers:
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's print two records from this dataset in their raw state:
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We see that the abstract is formatted as a concatenation of short chunks whith new line characters at the end of each chunk. Let's parse the entire dataset by reading only the ID, title, and abstract into a list. We will also "clean" the abstract by concatenating the chunks and removing the unnecessary new line characters:
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Getting the Taxonomy

    Now, we have an unlabeled dataset in the `records` list. We want to train a classifier that take a titles and an abstract of an article and predicts its topic according to a given taxonomy.

    We will use the [IPTC NewsCodes](https://iptc.org/standards/newscodes/) taxonomy which is used by the world's largest news agencies, such as Agence France-Presse, Associated Press, and Reuters.

    The taxonomy consists of 1124 leaf elements, organized into up to 6 levels. Let's download it and see what it look like:
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can see that each element in the taxonomy has its `URI`, `code`, `name`, and `definition`, and it has a dedicated level in the taxonomy, from `Level1/NewsCode` to `Level6/NewsCode`.

    For example, `arts, culture, entertainment, and media` has the code `medtop:01000000`, and we see that it's a level 1 code. We also see that it isn't a leaf code because it has a code below it at level 2: `medtop:20000002` (`arts and entertainment`). The latter isn't leaf-level either because it has four codes below it at level 3: `medtop:20000003` (`animation`), `medtop:20000004` (`cartoon`), `medtop:20000005` (`cinema`), `medtop:20000007` (`dance`).

    `animation`, `cartoon`, and `cinema`, on the other hand, are leaf-level elements because they don't have any lower-level codes under them.

    `dance`, while being on the same level as `animation`, `cartoon`, and `cinema`, does have codes on a level below it: `medtop:20000008` (`ballet`), `medtop:20000009` (`modern dance`), `medtop:20000010` (`traditional dance`), so it's not a leaf-level code. But `ballet`, `modern dance`, and `traditional dance` are leaf-level codes because they don't have any code below them.

    Because we are building a classifier where all classes should be at the same level of abstraction, we are interested in the elements that are at leaf positions. Let's transform our hierarchical taxonomy into a flat taxonomy by only keeping the leaf-level elements, attaching their "parent" elements to them in a chain-like format:
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As you can see, we also attached to each code it's title and definition. They will be helpful for LMs that we will use to label the data.

    Now we have a multiclass classification problem with 1124 classes and inputs consisting of a title and abstract. To train this classifier, we need labeled examples of (input, ouput) pairs where the input is a combination of a title and an abstract and the output (called a label) is one of the 1124 classes.

    To help a language model choose the right label for a given input, let's convert the categories to a human-readable format by separating levels with a `>`:
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we have our classes. What makes them appropriate classes for a multiclass classification problem? They are of the same level of abstraction and they have unique descriptive naming.

    ## Building the Labeled Dataset

    Now, let's take a sample of documents and use off-the-shelf language models to label them.

    For this tutorial, we will use an **ensemble** of fast but reasonably "smart" models available on [OpenRouter](https://openrouter.ai/models):

    - `meta-llama/llama-4-maverick`
    - `openai/gpt-4.1-nano`
    - `deepseek/deepseek-chat`
    - `google/gemini-2.5-flash-preview`
    - `x-ai/grok-3-mini-beta`

    > We choose fast models so that we shouldn't wait for too long for these models to have labeled all the documents.

    > At the time of writing this tutorial, for these models to be available, the OpenRouter user must enable the model provider to use the inputs to train their models (use [this link](https://openrouter.ai/settings/privacy) to enable this).

    When we have an ensemble of classifiers, we call each classifier with the same input and their majority predicted class as the final predicted class.

    The general principle in an ensemble is that the models must be uncorrelated. This is because correlated models would likely agree on a wrong prediction, while uncorrelated models, when they are wrong, usually predict different classes not forming a majorty.

    This uncorrelatedness is generally hard to achieve with off-the-shelf commercial LMs because they are likely trained mostly on the same Web data. However, its better to use LMs from different providers, because some of them might have used some proprietary corpus, like Google might have used Google Books corpus or Meta might have used Facebook data.

    Furthermore, depending on the importance of the final classifier (a nice to have one vs. mission critical) we might prefer more expensive and slower "reasoning" models with hopefully higher accuracy.

    > If you don't want to pay for labeling the data with language models, you can download the final [training](https://drive.google.com/file/d/1aO1zTkGI2mChnVq0qVx7vk44Uquph3QU/view?usp=sharing) and [test](https://drive.google.com/file/d/1Hi2KGGAxLACfUxZDZGXwuLWTQeXAo2TW/view?usp=sharing) labeled dataset. The files are in the Pickle format. Once dowloaded, read the tutorial without executing the cells until the **Training a Classifier** section. Upload the downloaded Pickle files to the files folder in your Colab notebook and replace the code reading the files from Google Drive to reading the downloaded Pickle files from a local directory.

    First, let's sample the documents we will label. First, we shuffle `records` and keep 20,000 for training and 500 for testing:
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will use OpenRouter with OpenAI API to access the models:
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To call LMs, we need a prompt. The prompt must clearly explain to the LM the context of what it's asked to do and provide all options to choose from. Because we have more than a 1000 classes, it's not the best idea to put all of them into the prompt and ask the model to choose the best one.

    There are several reasons why it's not a good idea. First, if the model is expensive and the dataset to label is large, a long prompt will increase the labeling cost. Second, despite some models supporting very long contexts (1M tokens and longer), their ability to use such long contexts decreses as the input size grows, which in our case would result in labeling quality going down. Third, when given too many choices, the model has higer chances to disregard some of them or even to output non-existing class (that is to hallucinate one).

    To reduce the list of options each model will be asked to choose from, different tehcniques can be used. For example:

    1. Embed documents and labels into dense vectors using pretrained embedding models, them only show the LM labels that have a high cosine similarity with the document, say top 10 most similar labels to the document.
    2. Use the LMs the first time to pick the most relevant higher level taxonomy catgories and then only show it the second time the labels that have these higher-level categories as a parents.

    In our case, we can use the Level 1 parents of our leaf taxonomy elements, but for the sake of generality (and fun!), let's use the embedding-based approach.

    We will use `BAAI/bge-m3` as the embedding model. It shows high scores on the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) and has a size small enough to run on a CPU. Let's install the dependency needed to use this model and load the model:
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, let's process our unlabeled examples by finding top 100 most likely labels for them. (100 is a magic number that feels to be a good reduction from a 1000 candidates but not too small to miss some important classes from consideration; in practical implementations, this number could be funetuned as a hyperparameter.) For this, we will embed both the document and the labels and then compare embedding vectors using cosine similarity.

    Since the labels remain constant during inference, we can compute their embeddings once and reuse them for all documents. This eliminates redundant computations and speeds up the process significantly:
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Processing documents in larger batches reduces the overhead of individual encoding calls and takes advantage of vectorized operations. Here's how we encode documents and attach top labels:
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we have top 100 classes for each document, we will ask three language models to choose the most relevant one for a given document among the 100 options.

    To ask an LM, we need a prompt. Let's use the following one:

    ```
    You are a document classification assistant. Your task is to classify the given document into the most appropriate category from the provided list of categories. The document consists of a title and an abstract. The categories are provided with their IDs and full hierarchical paths and definitions in the following format:

    ID: [ID], PATH: Level 1 > Level 2 > ... > Level 6 (definition)

    For example:

    ID: medtop:20000842, PATH: sport > competition discipline > track and field > relay run (A team of athletes run a relay over identical distances)

    Here is the document to classify:

    Title: {title}

    Abstract: {abstract}

    Here are the possible categories:

    {category_list}

    Select the most appropriate category for this document from the list above. Provide only the ID of the selected category (e.g., medtop:20000842). Do not invent a new category; choose only from the provided options. Do not mention any of the irrelevat options in your output. Your output must only contain the most appropriate category and notthing else.
    ```

    To call the LMs and get the results, let's first define helper functions:
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, let's run the labeling function (`process_records`) and save the labeled data to our Google Drive account so that we don't need to reprocess it if something goes wrong with the Colab execution environment:
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's now load the saved labeled examples, verify that they are all either have a majority label or the label is None:
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we can see, 2350 records haven't got a label (about 10%) because there was no consensus among the three LMs.

    At this stage, we have three options for these unlabeled examples:

    1. to discard them,
    2. to try more expensive and potentially more capable language models or, if it doesn't help,
    3. to label them by hand.

    Because labeling the examples using an ensemble of 5 LLMs already costed us about $100, we will choose to discard them to keep the cost of the tutorial manageable. In a business context, we would use more expensive reasoning models to label these more challenging examples. The presence of such challenging examples in the training set would be beneficial for the classifier's quality.

    > $100 for labeleing 20,000 examples might seem high, but it's not. Doing the labeling by hand would be close to impossible due to the high complexity of the texts that would require very expensive experts to label them. And even if we had such experts, they would have to keep in mind too many classes (knowing by heart their definitions) to choose from. An enterprise like this, before LLMs, would cost hundreds of thousands of dollars. This would also take months and not hours that LLMs took.

    > In a business scenario, it's recommended to take a small random sample of labeled examples and validate them by an ensemble of human labelers to make sure that LLMs did a good job. However, if 3+ out of 5 LLMs trained by 5 different provideres agreed on the same label out ot 100, it's a strong indicator of a correct label.

    Let's remove the unlabeled examples from our training and test datasets:
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Looking at the Data

    Let's see what our label distribution looks like:
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As we can see, the class distribution is highly imbalanced. We have 210 classes represented by fewer than 10 documents, which isn't good. Some class imbalance doesn't hurt, but here, we have only about 20-30 classes that were attached to the majority of documents.

    In a business scenario, we would have to increase the presence of documents of minority classes in the training data. Otherwise, our model will not learn from just a few representatives of these classes.

    To do that, we would:

    1. Embed all unlabeled documents as well as all labeled documents labeled with minority classes,
    2. Find the similar unlabeled documents to the labeled documents with minority labels according to cosine similarity
    3. Use LMs to label these similar documents until a substantial number of examples get these minority labels (at least 10 per class, but the more the better with the goal of having a better balanced distribition of labels by class)

    In this tutorial, we will make a different business decision which will save us model delivery time and money, but will require additional spendings in production.

    **The decision is the following:** we will assign the "Other" label to all documents that were assigned an original label with less than 10 examples.

    In a business scenario, this would mean that we would keep using the ensemble of LMs to label documents for which Other has been predicted by our classifier. Given that the number of such documents will be small, the cost of calling LMs will be greatly reduced compared to calling LMs for all documents:
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training a Classifier

    We are now ready to train. Let's first read the labeled documents from the pickle files and convert them into the format suitable for training with the `Hugging Face` API:
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we have the data in the format expected by the Hugging Face API, so we can start training. We will finetune RoBERTa as our **baseline**:
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So, our baseline is 80.46% with RoBERTa-base with all default hyperparatemers except for a short warmup.

    The accuracy number on its own doesn't tell us whether the model is good in practice or not. First of all, our data has a serious class imbalance, which makes the accuracy as a metric less informative. Furthermore, in a multiclass classification, when the classes are very granular (which is our case with more tham 1200 options), two or more classes can often be appropriate for the same input, especially in predicting a topic of a document: one document can contain a mixture of several topics.

    For us, 80.46% accuracy is a baseline which we will use as the minimum performance level when considering more complex solutions. More specifically, we will compare it to the accuracy of larger models we will finetune on the same dataset.

    Let's finetune a larger model: Qwen/Qwen2.5-1.5B-Instruct. It's an instruct-finetued language model with 1.5B parameters.

    This model has two advantages over RoBERTa:

    - It has a much longer context of 32k tokens (compared to 512 tokens in RoBERTa), which allows us to put more information in the input
    - It understand instructions, so we can use the prompt we used above to get the labels from off-the-shelf LLMs, which will simplify for the model the task of selecting the right model.

    Let's prepare the training data for Qwen:
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, let's finetune it using LoRA to save time and GPU expenses:

    > LoRA is a technique that consists of injecting several new learnable parameters into the LM and keep the original model parameters "frozen." This allows speeding up the training and using larger batches because when we train fewer parameters, the gradients of these parameters also take less GPU memory.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We only managed to train for 6 epochs, because Colab has a limit on the session length, unless we opt for Colab Pro, which is quite expensive.

    As we can see, the highest accuracy with `Qwen/Qwen2.5-1.5B-Instruct` is 86.67% which is 6.21 percentage points higher than RoBERTa's 80.46%. It's a great increase given that all we did is to finetune a larger model. We didn't play with the label distribution in the training set trying to make it more flat, we didn't do any hyperparameter tuning, and we didn't labeled more data.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's train a 7B model and see if it allows increasing accuracy. With Colab, training models larger than 2B parameters is practically challenging. The process will be slow and take several days. So, we need Colab Pro and a lot of time. Instead, we will use [SwarmOne](https://swarmone.ai/) who generously shared credits with me to finish this tutorial.

    Let's train a 7B model. With Colab, training models larger than 2B parameters is practically challenging. The process will be slow and take several days. So, we need Colab Pro and a lot of time. Instead, we will use [SwarmOne](https://swarmone.ai/) who generously shared credits with me to finish this tutorial.

    > SwarmOne let's you train models of different size without thinking about GPUs. You simply choose the model size you want to train and the system finds the best hardware and the best way to distribute the training for this model size and the shape of the training data.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For `Qwen/Qwen2.5-7B-Instruct`, and the following hyperparameters:

    ```
        lora_r = 64
        lora_alpha = 128
        max_length = 5120
        batch_size = 6
    ```

    we have the accuracy of 90.14%:
    ![Screenshot 2025-05-21 at 6.15.08 PM.png](data:image/png;base64,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)
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For `Qwen/Qwen2.5-14B-Instruct` with the following values of hyperparamaters:

    ```
        lora_r = 64
        lora_alpha = 128
        max_length = 5120
        batch_size = 6
    ```

    we have the accuracy of 90.60%:

    ![Screenshot 2025-05-21 at 6.22.37 PM.png](data:image/png;base64,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)
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Given that we don't see a substantial increase in accuracy by doubling the size of the model, we stop.

    In a business context, for each size of the model, depending on the available budget, we would run several experiments by changing such hyperparameters as model size, batch size, context size, LoRA's `r` and `alpha`, try different initialization algorithms for LoRA parameters, and try full finetune instead of LoRA. This would most likely result in an experiment yielding a model with a better accuracy.

    Overall, for a multiclass classifier with 38 labels (37 labels from the taxonomy plus `Other`), any result above 90% is good. Most of the "errors" are classes that would equally fit.

    But if we had a budget for more experiments, for a longer context than 5000 tokens (to include more labels in the text format into the prompt), for labeling not 20,000 but 200,000 documents, and for building a better balanced dataset, the actual production accuracy might be much higher.
    """)
    return

@app.cell
def _(mo):
    _ = mo  # keep dep ordering
    import kagglehub
    import os

    # Download latest version
    path = kagglehub.dataset_download("Cornell-University/arxiv")

    print("Path to dataset files:", path)
    print("Files in dataset directory:", os.listdir(path))

    import json
    from pprint import pprint
    # Construct the full path to the JSON file
    file_path = os.path.join(path, 'arxiv-metadata-oai-snapshot.json')

    # Open the file and read the first two lines
    with open(file_path, 'r') as file:
        # Read and parse the first record
        first_line = file.readline()
        first_record = json.loads(first_line)

        # Read and parse the second record
        second_line = file.readline()
        second_record = json.loads(second_line)

    # Print the first two records
    print("First record:")
    pprint(first_record)
    print("\nSecond record:")
    pprint(second_record)

    def clean_abstract(text):
        """Clean the abstract by preserving paragraphs and removing unnecessary newlines within paragraphs."""
        paragraphs = text.split('\n\n')
        cleaned_paragraphs = []
        for paragraph in paragraphs:
            cleaned_paragraph = ' '.join((line.strip() for line in paragraph.split('\n')))  # Split into paragraphs using double newlines
            cleaned_paragraphs.append(cleaned_paragraph)
        return '\n\n'.join(cleaned_paragraphs)

    def clean_title(text):  # Replace single newlines with spaces within each paragraph
        """Clean the title by removing extra spaces and joining lines with spaces."""
        return ' '.join((line.strip() for line in text.split('\n')))
    records = []  # Join paragraphs back with double newlines
    with open(file_path, 'r') as file_1:
        for line in file_1:
            record = json.loads(line)
            filtered_record = {'id': record.get('id', ''), 'title': clean_title(record.get('title', '')), 'abstract': clean_abstract(record.get('abstract', ''))}
            records.append(filtered_record)
    for i, record in enumerate(records[:2], start=1):
        print(f'Record {i}:')
        print(f"ID: {record['id']}")
    # Open the file and read the first two lines
        print(f"Title: {record['title']}")
        print(f"Abstract: {record['abstract']}")
    # Print the first two records
        print()

    import pandas as pd
    import requests
    from io import BytesIO

    # The URL of the Excel file with the Media Topics taxonomy
    url = "https://www.iptc.org/std/NewsCodes/IPTC-MediaTopic-NewsCodes.xlsx"

    # Download the file
    response = requests.get(url)

    # Create a file-like object from the downloaded content
    excel_data = BytesIO(response.content)

    # Load the spreadsheet, skipping the first row and using the second row as headers
    df = pd.read_excel(excel_data, skiprows=1)

    # Columns to select
    selected_columns = [
        'NewsCode-URI',
        'NewsCode-QCode (flat)',
        'Level1/NewsCode',
        'Level2/NewsCode',
        'Level3/NewsCode',
        'Level4/NewsCode',
        'Level5/NewsCode',
        'Level6/NewsCode',
        'Name (en-US)',
        'Definition (en-US)'
    ]

    # Select only the specified columns
    df_selected = df[selected_columns]

    # Display the first 20 rows
    df_selected.head(20)

    level_cols = [f'Level{i}/NewsCode' for i in range(1, 7)]  # ['Level1/NewsCode', ..., 'Level6/NewsCode']
    path_1 = []
    # Initialize the path stack and a list for leaf paths
    leaf_paths = []
    for i_1 in range(len(df)):
        row = df.iloc[i_1]
    # Process each row
        non_empty_levels = row[level_cols].notna()
        if non_empty_levels.sum() != 1:
            continue
        col = non_empty_levels.idxmax()  # Determine the level by finding the non-empty level column
        current_level = level_cols.index(col) + 1
        qcode = row['NewsCode-QCode (flat)']
        name = row['Name (en-US)']  # Skip rows with invalid level data (not exactly one level column filled)
        definition = row['Definition (en-US)']
        element = (qcode, name, definition)
        while len(path_1) >= current_level:  # Level number (1 to 6)
            path_1.pop()
        path_1.append(element)  # Extract the code, name, and definition
        is_leaf = False
        if i_1 == len(df) - 1:
            is_leaf = True
        else:
            next_row = df.iloc[i_1 + 1]  # Create a tuple for the current element
            next_non_empty_levels = next_row[level_cols].notna()
            if next_non_empty_levels.sum() == 1:
                next_col = next_non_empty_levels.idxmax()  # Update the path stack to match the current level
                next_level = level_cols.index(next_col) + 1
                if next_level <= current_level:
                    is_leaf = True
            else:
                is_leaf = True  # Check if this row is a leaf
        if is_leaf:
            leaf_paths.append(path_1.copy())  # Last row is a leaf
    print('Number of leaf-level codes:', len(leaf_paths))
    for path_1 in leaf_paths[:20]:
    # Output each leaf path
        print(path_1)  # If the next row is invalid, treat the current row as a leaf  # If it's a leaf, store the current path

    def format_path(path):
        """
        Formats a taxonomy path into a string with names separated by '>' and the leaf definition in parentheses.
        Args:
            path (list of tuples): Each tuple contains (code, name, definition).
        Returns:
            str: Formatted string, e.g., 'name1 > name2 > name3 (definition3)'
        """
        names = [element[1] for element in path]
        joined_names = ' > '.join(names)
        leaf_definition = path[-1][2]
        return f'{joined_names} ({leaf_definition})'
    print('Number of leaf-level codes:', len(leaf_paths))
    for path_2 in leaf_paths[:20]:
        print(format_path(path_2))

    import random
    random.seed(42)

    train_size = 20_000
    test_size = 500

    shuffled_records = random.sample(records, train_size + test_size)
    train_records = shuffled_records[:train_size]
    test_records = shuffled_records[train_size:]

    from openai import AsyncOpenAI

    def get_responses(client, models, prompt):
        """
        Get responses from multiple models for a given prompt.

        Args:
            client: An initialized OpenAI client object
            models: List of model names (strings) to query
            prompt: String containing the prompt to send to each model

        Returns:
            List of strings, where each string is the response from a model
            or an error message if the query fails
        """
        responses = []
        for model in models:
            try:
                completion = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://www.thelmbook.com",
                        "X-Title": "The Hundred-Page Language Models Book",
                    },
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                response = completion.choices[0].message.content
                responses.append(response)
            except Exception as e:
                responses.append(f"Error with model {model}: {str(e)}")
        return responses

    models = ["meta-llama/llama-4-maverick", "openai/gpt-4.1-nano", "deepseek/deepseek-chat", "google/gemini-2.5-flash-preview", "x-ai/grok-3-mini-beta"]

    client = AsyncOpenAI(
      base_url="https://openrouter.ai/api/v1",
      # create an account on https://openrouter.ai/
      # and then create your own key;
      api_key="PUT YOUR KEY HERE",
      timeout=5.0
    )

    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    import torch
    from FlagEmbedding import BGEM3FlagModel
    import time

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize model on GPU
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    # Create mapping from formatted label to leaf-level code ID
    label_to_code = {format_path(path): path[-1][0] for path in leaf_paths}

    # Create mapping from leaf-level code ID to formatted label
    code_to_label = {path[-1][0]: format_path(path) for path in leaf_paths}

    # Format labels for embedding
    labels = [format_path(path) for path in leaf_paths]

    # Compute label embeddings once
    start_time = time.perf_counter()
    label_embeddings_np = model.encode(labels, batch_size=32)['dense_vecs']  # Returns NumPy array
    label_embeddings = torch.from_numpy(label_embeddings_np).to(device)      # Move to GPU
    end_time = time.perf_counter()
    print(f"Time to embed all labels: {end_time - start_time:.2f} seconds")

    from pathlib import Path
    import pickle
    from google.colab import drive

    def attach_top_labels(records, label_embeddings, top_k=100, batch_size=128):
        for i in range(0, len(records), batch_size):
            batch_records = records[i:i + batch_size]
            batch_documents = [f"Title: {r['title']}\nAbstract: {r['abstract']}" for r in batch_records]
            doc_embeddings_np = model.encode(batch_documents, batch_size=32)['dense_vecs']
            doc_embeddings = torch.from_numpy(doc_embeddings_np).to(device)
            similarity_matrix = torch.mm(doc_embeddings, label_embeddings.t())
            top_indices = torch.topk(similarity_matrix, k=top_k, dim=1).indices.cpu().numpy()
            for j, record in enumerate(batch_records):
                top_formatted_labels = [labels[idx] for idx in top_indices[j]]
                top_labels = [(label_to_code[label], label) for label in top_formatted_labels]
                record['top_labels'] = top_labels
        return records
    drive.mount('/content/drive', force_remount=False)
    # Mount Google Drive (if not already mounted)
    drive_dir = Path('/content/drive/MyDrive/labeled_classifier_data')
    drive_dir.mkdir(parents=True, exist_ok=True)
    # Define the Google Drive directory
    train_top_labels_pkl = drive_dir / 'train_records_with_top_labels.pkl'
    test_top_labels_pkl = drive_dir / 'test_records_with_top_labels.pkl'
    # Ensure the directory exists

    def load_or_compute_top_labels(records, pkl_path, label_embeddings, top_k=100, batch_size=512):
    # Define paths for records with top labels
        """
        Load records from a pickle file if it exists, otherwise compute top labels and save them.

    # Function to load or compute top labels for records
        Args:
            records (list): List of record dictionaries to process.
            pkl_path (Path): Path object for the pickle file.
            label_embeddings (torch.Tensor): Precomputed label embeddings.
            top_k (int): Number of top labels to attach (default: 100).
            batch_size (int): Batch size for processing (default: 512).

        Returns:
            list: Records with top labels attached.
        """
        if pkl_path.exists():
            records = pickle.loads(pkl_path.read_bytes())
            print(f'Loaded records with top_labels from {pkl_path}')
        else:
            attach_top_labels(records, label_embeddings, top_k=top_k, batch_size=batch_size)
            with pkl_path.open('wb') as f:
                pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'Computed top labels and saved to {pkl_path}')
        return records
    train_records_1 = load_or_compute_top_labels(train_records, train_top_labels_pkl, label_embeddings)
    test_records_1 = load_or_compute_top_labels(test_records, test_top_labels_pkl, label_embeddings)
    print('Sample train record:')
    print(f"Title: {train_records_1[1]['title']}")
    print('Top labels:')
    # Load or compute top labels for train and test records
    for code, label in train_records_1[1]['top_labels'][:5]:
    # Example output to verify
        print(f'{code}: {label}')  # Show first 5 for brevity

    import re
    import asyncio
    from collections import Counter

    def batched(iterable, n):
        """Yield successive n-sized chunks from iterable."""
        for i in range(0, len(iterable), n):
            yield iterable[i:i + n]

    async def get_valid_id(client, model, prompt, valid_ids, max_retries=3):
        """
        Asynchronously call an LLM until it returns a valid ID or retries are exhausted.

        Args:
            client: An AsyncOpenAI client instance
            model: String, the model name to query
            prompt: String, the prompt to send
            valid_ids: List of valid category IDs
            max_retries: Integer, maximum number of retries

        Returns:
            String, the valid ID, or None if all retries fail
        """
        for attempt in range(max_retries):
            start_time = time.perf_counter()
            try:
                completion = await client.chat.completions.create(extra_headers={'HTTP-Referer': 'https://www.thelmbook.com', 'X-Title': 'The Hundred-Page Language Models Book'}, model=model, messages=[{'role': 'user', 'content': prompt}])
                elapsed = time.perf_counter() - start_time
                if completion is None or not getattr(completion, 'choices', None) or (not completion.choices):
                    print(f'Attempt {attempt + 1}: No choices from {model}')
                    continue
                choice = completion.choices[0]
                if not getattr(choice, 'message', None) or choice.message.content is None:
                    print(f'Attempt {attempt + 1}: No content from {model}')
                    continue
                response = choice.message.content.strip()
                extracted_id = extract_id(response)
                if extracted_id in valid_ids:  #print(completion)
                    return extracted_id
                print(f'Attempt {attempt + 1}: Invalid ID «{response}»')
            except Exception as exc:
                elapsed = time.perf_counter() - start_time
                print(f'{model} | attempt {attempt + 1} | {elapsed:.2f}s (error)')
                print(f'Attempt {attempt + 1}: {exc}')
        print(f'Failed to get valid ID from {model} after {max_retries} attempts')
        return None

    def generate_category_list(top_labels):
        """
        Generate a formatted string of categories with IDs and paths.
        """
        return '\n'.join((f'ID: {code}, PATH: {label}' for code, label in top_labels))

    def extract_id(response):
        """
        Extract the *last* category ID (medtop:<digits>) found in the model’s response.
        This avoids grabbing earlier examples the model might list before its final choice.
        """
        ids = re.findall('medtop:\\d+', response)
        return ids[-1] if ids else None

    async def process_single_record(record, client, models, majority_threshold):
        """Process a single record by calling 5 LLMs in parallel and determining the majority label."""
        title = record['title']
        abstract = record['abstract']
        category_list = generate_category_list(record['top_labels'])
    # ──────────────────────────────────────────────────────────────
    # Helpers
        valid_ids = [code for code, _ in record['top_labels']]
        prompt = f'\nYou are a document classification assistant. Your task is to classify the given document into the most appropriate category from the provided list of categories. The document consists of a title and an abstract. The categories are provided with their IDs and full hierarchical paths and definitions in the following format:\n\nID: [ID], PATH: Level 1 > Level 2 > ... > Level 6 (definition)\n\nFor example:\n\nID: medtop:20000842, PATH: sport > competition discipline > track and field > relay run (A team of athletes run a relay over identical distances)\n\nHere is the document to classify:\n\nTitle: {title}\n\nAbstract: {abstract}\n\nHere are the possible categories:\n\n{category_list}\n\nSelect the most appropriate category for this document from the list above. Provide only the ID of the selected category (e.g., medtop:20000842). Do not invent a new category; choose only from the provided options. Do not mention any of the irrelevat options in your output. Your output must only contain the most appropriate category and notthing else.\n'
        tasks = [get_valid_id(client, model, prompt, valid_ids) for model in models]
        model_ids = await asyncio.gather(*tasks)
        counter = Counter((mid for mid in model_ids if mid is not None))
        majority_id = None
        if counter:
            top_id, freq = counter.most_common(1)[0]
            if freq >= majority_threshold:
                majority_id = top_id
        record['model_ids'] = model_ids
        record['majority_id'] = majority_id

    async def process_records(records, client, models):
        """Process records in batches of 10 concurrently with a 1-second delay between batches."""
        majority_threshold = len(models) // 2 + 1
        processed = 0
        for batch in batched(records, 10):
            batch_tasks = [process_single_record(record, client, models, majority_threshold) for record in batch]
            await asyncio.gather(*batch_tasks)
            processed = processed + len(batch)
            print(f'Processed {processed} documents out of {len(records)}')
            await asyncio.sleep(1)
        have_majority = sum((1 for record in records if record['majority_id'] is not None))
        print(f'Found {have_majority} documents with a majority label')  # Create tasks for all 5 models  # Await all responses concurrently  # Perform majority voting  #print("Have majority!")  # Update the record in place  # e.g., 3 for 5 models  # Process records in batches of 10  # Create tasks for each record in the batch  # Execute all tasks concurrently  # Update and report progress  # Add 1-second delay  # Final summary

    def load_records(drive_folder='/content/drive/MyDrive/labeled_classifier_data'):
        """
        Mounts Google\xa0Drive, loads train_records.pkl and test_records.pkl,
        verifies that every record has a 'majority_id' key (raises ValueError if any are missing),
        and returns a list of those records whose majority_id is None.
        """
        drive.mount('/content/drive', force_remount=False)
        drive_dir = Path(drive_folder)
        train_path = drive_dir / 'train_records.pkl'
        test_path = drive_dir / 'test_records.pkl'
        train_records = pickle.loads(train_path.read_bytes())  # Mount Drive (will be a no‑op if already mounted)
        test_records = pickle.loads(test_path.read_bytes())
        all_recs = train_records + test_records
        missing = [r for r in all_recs if 'majority_id' not in r]
        if missing:
            raise ValueError(f"{len(missing)} records are missing the 'majority_id' field")
        unlabeled_records = [r for r in all_recs if r.get('majority_id') is None]
        return (train_records, test_records, unlabeled_records)  # Load pickles
    train_records_3, test_records_3, unlabeled_records = load_records()
    print(f'Found {len(unlabeled_records)} records without a majority label')  # Verify every record has the 'majority_id' field  # Return those records where majority_id is None

    labeled_only_train_records = [record for record in train_records_3 if record['majority_id'] is not None]
    labeled_only_test_records = [record for record in test_records_3 if record['majority_id'] is not None]
    drive_dir_2 = Path('/content/drive/MyDrive/labeled_classifier_data')
    # Define the Google Drive directory path (consistent with previous code)
    drive_dir_2.mkdir(parents=True, exist_ok=True)
    labeled_only_train_pkl = drive_dir_2 / 'labeled_only_train_records.pkl'
    # Ensure the directory exists, creating it if necessary
    labeled_only_test_pkl = drive_dir_2 / 'labeled_only_test_records.pkl'
    with labeled_only_train_pkl.open('wb') as f:
    # Define new, descriptive file paths
        pickle.dump(labeled_only_train_records, f, protocol=pickle.HIGHEST_PROTOCOL)
    with labeled_only_test_pkl.open('wb') as f:
        pickle.dump(labeled_only_test_records, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Save the filtered datasets to the new pickle files
    print(f'Saved {len(labeled_only_train_records)} labeled train records to {labeled_only_train_pkl}')
    # Print confirmation with the number of records saved and file locations
    print(f'Saved {len(labeled_only_test_records)} labeled test records to {labeled_only_test_pkl}')

    import matplotlib.pyplot as plt

    def plot_label_distribution(train_records, top_n=20):
        """
        Plots the distribution of labels in the training examples.

        Parameters:
        - train_records (list): List of dictionaries, each containing 'majority_id' and 'top_labels'.
        - top_n (int): Number of top labels to display in the bar plot (default: 20).
        """
        code_to_label = {}
        for record in train_records:  # Build a mapping from label codes to their formatted names using 'top_labels'
            for code, label in record.get('top_labels', []):
                if code not in code_to_label:
                    code_to_label[code] = label
        labels = [record['majority_id'] for record in train_records if record.get('majority_id') is not None]
        label_counts = Counter(labels)
        print(f'Total labeled examples: {len(labels)}')
        print(f'Number of unique labels: {len(label_counts)}')  # Extract labels from 'majority_id', filtering out records where it is None

        def shorten_label(label):
            label = label.split(' (')[0]  # Count the frequency of each label
            parts = label.split(' > ')
            if len(parts) >= 2:
                return ' > '.join(parts[-2:])  # Print basic statistics
            else:
                return parts[-1]
        top_labels = label_counts.most_common(top_n)
        top_label_names = [shorten_label(code_to_label.get(code, code)) for code, count in top_labels]  # Helper function to shorten label names to the last two levels
        top_counts = [count for code, count in top_labels]
        plt.figure(figsize=(12, 10))  # Remove definition in parentheses, if present
        plt.barh(top_label_names, top_counts)
        plt.xlabel('Number of Documents')  # Split by ' > ' to get hierarchy levels
        plt.title(f'Top {top_n} Most Frequent Labels in Training Data')
        plt.gca().invert_yaxis()  # Return the last two levels if available, otherwise the last level
        plt.tight_layout()
        plt.show()
        counts = list(label_counts.values())
        plt.figure(figsize=(10, 6))
        plt.hist(counts, bins=20, edgecolor='black')
        plt.xlabel('Number of Documents per Label')  # --- Bar Plot: Top N Most Frequent Labels ---
        plt.ylabel('Number of Labels')  # Get the top N labels and their counts
        plt.title('Distribution of Label Frequencies in Training Data')
        plt.tight_layout()  # Shorten label names, falling back to code if not found in mapping
        plt.show()
    import numpy as np

    def print_label_statistics(train_records, threshold=20):  # Create the horizontal bar plot
        """
        Prints statistics about the label distribution in the training examples.

        Parameters:
        - train_records (list): List of dictionaries, each containing 'majority_id'.  # Highest count at the top
        """
        labels = [record['majority_id'] for record in train_records if record.get('majority_id') is not None]
        label_counts = Counter(labels)
        counts = list(label_counts.values())  # --- Histogram: Distribution of Label Frequencies ---
        num_classes = len(label_counts)  # Get all frequency counts
        if num_classes > 0:
            average_docs_per_class = np.mean(counts)
            max_docs = np.max(counts)  # Create the histogram
            min_docs = np.min(counts)
            classes_with_less_than_threshold = sum((1 for count in counts if count < threshold))
        else:
            average_docs_per_class = max_docs = min_docs = classes_with_less_than_threshold = 0
        print(f'Average number of documents per class: {average_docs_per_class:.2f}')
        print(f'Maximum number of documents in a class: {max_docs}')
        print(f'Minimum number of documents in a class: {min_docs}')
        print(f'Number of classes with fewer than {threshold} documents: {classes_with_less_than_threshold}')
    print_label_statistics(train_records_3, 20)
    plot_label_distribution(train_records_3, top_n=100)  # Extract labels from 'majority_id', filtering out records where it is None  # Count the frequency of each label  # Calculate statistics  # Print the requested statistics

    def relabel_rare_classes(train_records, test_records, min_docs=10):
        """
        Processes training and testing records by keeping all examples with non-None labels and changing
        labels with fewer than min_docs occurrences in the training set to "Other." Prints statistics
        about the original training label distribution.

        Parameters:
        - train_records (list): List of dictionaries, each containing 'majority_id'.
        - test_records (list): List of dictionaries, each containing 'majority_id'.
        - min_docs (int): Minimum number of documents required per class to keep its original label (default: 10).

        Returns:
        - updated_train_records (list): Training records with rare labels changed to "Other".
        - updated_test_records (list): Testing records with labels not in common training labels changed to "Other".
        """
        train_labels = [record['majority_id'] for record in train_records if record.get('majority_id') is not None]
        label_counts = Counter(train_labels)
        counts = list(label_counts.values())
        num_classes = len(label_counts)  # Step 1: Extract non-None labels from training records
        if num_classes > 0:
            average_docs_per_class = np.mean(counts)
            max_docs = np.max(counts)  # Step 2: Count label frequencies
            min_docs_stat = np.min(counts)
            classes_with_less_than_min = sum((1 for count in counts if count < min_docs))
        else:  # Step 3: Calculate and print statistics
            average_docs_per_class = max_docs = min_docs_stat = classes_with_less_than_min = 0
        print(f'Average number of documents per class: {average_docs_per_class:.2f}')
        print(f'Maximum number of documents in a class: {max_docs}')
        print(f'Minimum number of documents in a class: {min_docs_stat}')
        print(f'Number of classes with fewer than {min_docs} documents: {classes_with_less_than_min}')
        common_labels = {label for label, count in label_counts.items() if count >= min_docs}

        def update_label(record):
            return record['majority_id'] if record['majority_id'] in common_labels else 'Other'
        updated_train_records = [{**record, 'majority_id': update_label(record)} for record in train_records if record.get('majority_id') is not None]
        if updated_train_records:
            other_count = sum((1 for record in updated_train_records if record['majority_id'] == 'Other'))
            ratio = other_count / len(updated_train_records)
            print(f"Ratio of training documents labeled as 'Other': {ratio * 100:.2f}%")
        else:
            print('No labeled training documents.')  # Step 4: Identify common labels (those with at least min_docs)
        updated_test_records = [{**record, 'majority_id': update_label(record)} for record in test_records if record.get('majority_id') is not None]
        return (updated_train_records, updated_test_records)
    relabeled_train_records, relabeled_test_records = relabel_rare_classes(train_records_3, test_records_3, min_docs=20)  # Step 5: Helper function to decide the new label
    print(len(relabeled_train_records))
    drive_dir_3 = Path('/content/drive/MyDrive/labeled_classifier_data')
    drive_dir_3.mkdir(parents=True, exist_ok=True)
    relabeled_train_pkl = drive_dir_3 / 'relabeled_train_records.pkl'  # Step 6: Create updated training records
    relabeled_test_pkl = drive_dir_3 / 'relabeled_test_records.pkl'
    with relabeled_train_pkl.open('wb') as f_1:
        pickle.dump(relabeled_train_records, f_1, protocol=pickle.HIGHEST_PROTOCOL)
    with relabeled_test_pkl.open('wb') as f_1:
        pickle.dump(relabeled_test_records, f_1, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved {len(relabeled_train_records)} relabeled train records to {relabeled_train_pkl}')  # Calculate and print the ratio of 'Other' labels in training set
    # Save the relabeled datasets to the new pickle files
    # Define new, descriptive file paths
    # Print confirmation with the number of records saved and file locations
    print(f'Saved {len(relabeled_test_records)} relabeled test records to {relabeled_test_pkl}')  # Step 7: Create updated testing records

    import pickle
    from google.colab import drive
    from pathlib import Path
    from datasets import Dataset
    from transformers import AutoTokenizer

    drive.mount('/content/drive')

    # Define the directory and file paths
    drive_dir = Path("/content/drive/MyDrive/labeled_classifier_data")
    train_pkl = drive_dir / "relabeled_train_records.pkl"
    test_pkl = drive_dir / "relabeled_test_records.pkl"

    # Load the training and test records from pickle files
    with train_pkl.open('rb') as f:
        train_records = pickle.load(f)

    with test_pkl.open('rb') as f:
        test_records = pickle.load(f)

    # Get unique labels from the training records and create a mapping to integer IDs
    train_labels = set(record['majority_id'] for record in train_records)
    print("The number of unique labels in the training data:", len(train_labels))
    label_to_id = {label: idx for idx, label in enumerate(sorted(train_labels))}

    # Filter test records to only include labels present in the training set
    test_records = [record for record in test_records if record['majority_id'] in train_labels]

    # Create lists of texts and label IDs for training and test sets
    train_texts = [f"Title: {record['title']}\nAbstract: {record['abstract']}" for record in train_records]
    train_label_ids = [label_to_id[record['majority_id']] for record in train_records]

    test_texts = [f"Title: {record['title']}\nAbstract: {record['abstract']}" for record in test_records]
    test_label_ids = [label_to_id[record['majority_id']] for record in test_records]

    # Create Hugging Face Dataset objects
    train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_label_ids})
    test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_label_ids})

    # Load the RoBERTa tokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    # Define the tokenization function
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

    # Apply tokenization to the datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # At this point, train_dataset and test_dataset are ready for fine-tuning
    print("Training dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))

    import wandb
    os.environ['WANDB_PROJECT'] = 'document-classifier'
    # Set the WandB project name via environment variable
    # Log in to WandB using your API key
    wandb.login(key='PUT YOUR KEY HERE')  # Replace with your actual WandB API key

    import evaluate
    from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
    device_1 = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device_1}')
    model_name = 'roberta-base'
    num_labels = len(label_to_id)
    # Check if GPU is available and print the device being used
    model_1 = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model_1.to(device_1)
    accuracy_metric = evaluate.load('accuracy')
    # Define the model name

    def compute_metrics(p):
    # Load the RoBERTa model for sequence classification
    # Replace 'label_to_id' with your actual label mapping
        preds = np.argmax(p.predictions, axis=1)  # Number of classes in your dataset
        return accuracy_metric.compute(predictions=preds, references=p.label_ids)
    training_args = TrainingArguments(output_dir='./results', num_train_epochs=20, per_device_train_batch_size=32, per_device_eval_batch_size=32, warmup_steps=250, logging_steps=10, eval_strategy='epoch', report_to=['wandb'], run_name=f'{model_name}-finetune')
    trainer = Trainer(model=model_1, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset, compute_metrics=compute_metrics)
    # Load the accuracy metric from the evaluate library
    trainer.train()
    eval_results = trainer.evaluate()
    # Define the compute_metrics function to calculate accuracy
    print('Evaluation results:', eval_results)
    # Set up training arguments with WandB logging enabled and dynamic run name
    # Create the Trainer object with compute_metrics for accuracy logging
    # Fine-tune the model
    # Evaluate the model and log results to WandB
    wandb.finish()  # Directory to save model outputs  # Number of training epochs  # Batch size for training  # Batch size for evaluation  # Number of warmup steps  # Log every 10 steps  # Evaluate at the end of each epoch  # Log metrics to WandB  # Dynamic run name including model name  # Replace with your training dataset  # Replace with your evaluation dataset  # Function to compute accuracy

    import os
    import wandb
    import torch
    import pickle
    import numpy as np
    import evaluate
    from collections import Counter
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from datasets import Dataset
    from pathlib import Path
    from google.colab import drive
    from peft import LoraConfig, get_peft_model

    def print_trainable_ratio(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ratio = trainable_params / total_params
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Ratio of trainable parameters: {ratio * 100:.2f}%")

    # Mount Google Drive to access dataset files
    drive.mount('/content/drive')

    # Define the directory and file paths
    drive_dir = Path("/content/drive/MyDrive/labeled_classifier_data")
    train_pkl = drive_dir / "relabeled_train_records.pkl"
    test_pkl = drive_dir / "relabeled_test_records.pkl"

    # Load the training and test records from pickle files
    with train_pkl.open('rb') as f:
        train_records = pickle.load(f)

    with test_pkl.open('rb') as f:
        test_records = pickle.load(f)

    # Identify common labels from train_records (excluding "Other")
    common_labels = set(record['majority_id'] for record in train_records if record['majority_id'] != "Other")

    # Create label mapping from all possible majority_id values (including "Other")
    all_labels = set(record['majority_id'] for record in train_records)
    label_to_id = {label: idx for idx, label in enumerate(sorted(all_labels))}

    # Create reverse mapping for label IDs to label names
    id_to_label = {id: label for label, id in label_to_id.items()}

    # Function to generate category list for prompts
    def generate_category_list(top_labels, common_labels):
        filtered_labels = [(code, label) for code, label in top_labels if code in common_labels]
        category_list = "\n".join(f"ID: {code}, PATH: {label}" for code, label in filtered_labels)
        category_list += "\nID: Other, PATH: Other (Documents that do not fit into any specific category)"
        return category_list

    # Function to create prompts
    def create_prompt(record, common_labels):
        title = record['title']
        abstract = record['abstract']
        category_list = generate_category_list(record['top_labels'], common_labels)
        prompt = f"""
    You are a document classification assistant. Your task is to classify the given document into the most appropriate category from the provided list of categories. The document consists of a title and an abstract. The categories are provided with their IDs and full hierarchical paths and definitions in the following format:

    ID: [ID], PATH: Level 1 > Level 2 > ... > Level 6 (definition)

    For example:

    ID: medtop:20000842, PATH: sport > competition discipline > track and field > relay run (A team of athletes run a relay over identical distances)

    Here is the document to classify:

    Title: {title}

    Abstract: {abstract}

    Here are the possible categories:

    {category_list}

    Select the most appropriate category for this document from the list above. Provide only the ID of the selected category. If none of the specific categories are appropriate, choose 'Other'.
    """
        return prompt.strip()

    # Create prompts and label IDs for training and test sets
    train_prompts = [create_prompt(record, common_labels) for record in train_records]
    train_label_ids = [label_to_id[record['majority_id']] for record in train_records]

    test_prompts = [create_prompt(record, common_labels) for record in test_records]
    test_label_ids = [label_to_id[record['majority_id']] for record in test_records]

    print("First training example:")
    print("Input text:")
    print(train_prompts[0])
    print("\nOutput label:", id_to_label[train_label_ids[0]])

    # Create Hugging Face Dataset objects
    train_dataset = Dataset.from_dict({'text': train_prompts, 'label': train_label_ids})
    test_dataset = Dataset.from_dict({'text': test_prompts, 'label': test_label_ids})

    model_name_1 = 'Qwen/Qwen2.5-1.5B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name_1)
    # Load the Qwen tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
    # Define the tokenization function
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=768)
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    # Apply tokenization to the datasets
    print('Training dataset size:', len(train_dataset))
    print('Test dataset size:', len(test_dataset))
    model_2 = AutoModelForSequenceClassification.from_pretrained(model_name_1, num_labels=len(label_to_id), pad_token_id=tokenizer.eos_token_id)
    # Print dataset sizes for verification
    device_2 = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_2.to(device_2)
    lora_config = LoraConfig(r=32, lora_alpha=64, target_modules=['q_proj', 'v_proj'], lora_dropout=0.1, bias='none', task_type='SEQ_CLS')
    # Load Qwen model for sequence classification
    model_2 = get_peft_model(model_2, lora_config)
    print_trainable_ratio(model_2)
    model_2.to(device_2)
    training_args_1 = TrainingArguments(output_dir='./results', num_train_epochs=20, per_device_train_batch_size=6, per_device_eval_batch_size=6, warmup_steps=250, logging_steps=10, learning_rate=2e-05, weight_decay=0.1, eval_strategy='epoch', report_to=['wandb'], run_name=f'{model_name_1}-finetune-lora')
    accuracy_metric_1 = evaluate.load('accuracy')

    def compute_metrics_1(p):
        preds = np.argmax(p.predictions, axis=1)
        return accuracy_metric_1.compute(predictions=preds, references=p.label_ids)
    label_names = [id_to_label[i] for i in range(len(label_to_id))]
    trainer_1 = Trainer(model=model_2, args=training_args_1, train_dataset=train_dataset, eval_dataset=test_dataset, compute_metrics=compute_metrics_1)
    trainer_1.train()
    eval_results_1 = trainer_1.evaluate()
    print('Evaluation results:', eval_results_1)
    wandb.finish()
    # Set up training arguments
    # Load accuracy metric for evaluation
    # Define compute_metrics function
    # Initialize Trainer
    # Train the model
    # Evaluate the model on the test dataset
    # Optional: Finish WandB run
    # Optional: Exit the script cleanly
    os._exit(0)  # Ensure the model remains on the correct device after LoRA application
    return


if __name__ == "__main__":
    app.run()
