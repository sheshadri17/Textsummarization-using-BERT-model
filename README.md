# Develop a Text Classifier [BERT] using Transformers and PyTorch on your Custom Dataset

In recent years `Natural language understanding [NLU]` has made extraordinary advancements in many fields like text correction, text prediction, text translation, text classification, and many more... Recently Meta has released a 200 language Language Translation GitHub repository [FAIRSEQ](https://ai.facebook.com/tools/fairseq/), in the world of `NLP or NLU` there is one name that is not that famous but has many tools under its sleeve is [HuggingFace](https://huggingface.co/), they have a huge library for `NLP` and `Image Classification`


### Install required packages

Once the virtual environment is activated run the following command to get the required packages...

> `pip install transformers[torch] pandas datasets pyarrow scikit-learn`

> **NOTE** - this will take some time to complete, and depends on your laptop or pc and connection speed.

### Get the dataset

Here,  I am using a dataset downloaded from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), it is a movie review dataset, and it has around 50K samples with two labels `positive` and `negative`, but this can be implemented to more than two classes as well.

### Read the dataset

Now, let's read the dataset...

```python
    import pandas as pd
    df = pd.read_csv('./input/dataset.csv')
    df.head()
```
You will see that there are two columns, `review` and `sentiment`, the problem we have is a binary classification of the column `review`.

### Process the dataset

We need our data in a certain format before we pass it to the `Bert base uncased`, this model requires `input_ids`, `token_type_ids`, `attention_mask`, `label`, and `text`, also there is a particular way we need to pass the data and that is why we have installed `pyarrow` and `datasets`.

> **NOTE** - When you work on a different model that `Bert base uncased`, you need to make sure the data is in the format required by that model.

To tokenize the text, we will use `AutoTokenizer`. The following piece of code will initialize the tokenizer.

```python
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
```
Now, let's process the data in the CSV file, for this I will write a function, in this function the tokenizer uses `max_length=128`, you can increase that, but since I am just showing the workings, I will use `128`.

```python
    from typing import Dict

    def process_data(row) -> Dict:
        # Clean the text
        text = row['review']
        text = str(text)
        text = ' '.join(text.split())
        # Get tokens
        encodings = tokenizer(text, padding="max_length", truncation=True, max_length=128)
        # Convert string to integers
        label = 0
        if row['sentiment'] == 'positive':
            label += 1

        encodings['label'] = label
        encodings['text'] = text

        return encodings
```
Let's check the working of the function...

```python
    print(process_data({
        'review': 'this is a sample review of a movie.',
        'sentiment': 'positive'
    }))
```
We will pass each row of the dataset and process the `review` and convert the `sentiment` into `int` since the dataset consists of 15K samples, but as I said, for demonstration purposes, I will only use 1K samples.

```python
    # Store the encodings into an array to generate dataset
    processed_data = []

    for i in range(len(df[:1000])):
        processed_data.append(process_data(df.iloc[i]))
```
### Generate the dataset

Now, let's generate the dataset in a format required by the `Trainer` module of the `transformers` library. You can read mode about the `Trainer` [here](https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/trainer#transformers.Trainer).

> **NOTE** - The `Trainer` module accepts Datagenerator from PyTorch, so you can use that to generate the data on the go, in case you have a bigger dataset.

The code piece below converts the list of encodings into a data frame and split that into a training and validation set of data.

```python
    from sklearn.model_selection import train_test_split

    new_df = pd.DataFrame(processed_data)

    train_df, valid_df = train_test_split(
        new_df,
        test_size=0.2,
        random_state=2022
    )
```
Now, let's convert the `train_df` and `valid_df` into `Dataset` accepted by the `Trainer` module.

```python
    import pyarrow as pa
    from datasets import Dataset

    train_hg = Dataset(pa.Table.from_pandas(train_df))
    valid_hg = Dataset(pa.Table.from_pandas(valid_df))
```

## Train and Evaluate the model
---
Since our dataset is now ready, we can safely move forward to training the model on our custom dataset.

The following piece of code will initialize a `Bert base uncased` model for our training.

```python
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )
```
The model is ready, we are getting closer to the results, let's create a `Trainer`, it will require `TrainingArguments` as well. The following code will initialize both of them.

Under `TrainingArguments` you will see a `outpit_dir` argument, it is used by the module to write training logs.

```python
    from transformers import TrainingArguments, Trainer

    training_args = TrainingArguments(output_dir="./result", evaluation_strategy="epoch")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hg,
        eval_dataset=valid_hg,
        tokenizer=tokenizer
    )
```

Let's train our model...

```python
    trainer.train()
```
once the training is complete, we can evaluate the model as well...

```python
    trainer.evaluate()
```
### Save the model

We will save the model at the desired location...

```python
    model.save_pretrained('./model/')
```
### Load the model

When you need to load the model and initialize the tokenizer as well...

model...

```python
    from transformers import AutoModelForSequenceClassification

    new_model = AutoModelForSequenceClassification.from_pretrained('./model/')
```
tokenizer...

```python
    from transformers import AutoTokenizer

    new_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
```

### Get predictions

Now, that we have loaded our model, we are ready for the prediction...

The following function will use the model and tokenizer to get the prediction from a piece of text.

```python
    import torch
    import numpy as np

    def get_prediction(text):
        encoding = new_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}

        outputs = new_model(**encoding)

        logits = outputs.logits

        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        probs = probs.detach().numpy()
        label = np.argmax(probs, axis=-1)
        
        if label == 1:
            return {
                'sentiment': 'Positive',
                'probability': probs[1]
            }
        else:
            return {
                'sentiment': 'Negative',
                'probability': probs[0]
            }
```
let's see if it works or not, fingers are crossed....

```python
    get_prediction('I am happy to see you.')
```


