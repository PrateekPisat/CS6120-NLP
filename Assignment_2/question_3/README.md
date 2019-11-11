# Assignment 2

## Setup

`pip install -r deps`<br>
`python -m spacy download en`

**NOTE**: I have not inlcuded the entire word2vec model in the .zip for obvious reasons. Enter the path to the model on you computer in the provided `.env` file and the do a <br>
`source .env`

## Question 3

The text files are available in the `reports/` directory.<br>

### Models

There are three separate vector generation function:
`get_bow_vectors`, `get_google_vectors`, `get_tf_idf_vectors`. These are used to build a model. Out of which models generated from `get_bow_vectors` and `get_tf_idf_vectors` have been comented out since the model generated using google's word2vec gives the best results(the other 2 overfit).<br>

To use the other 2 models uncomment lines 361 and 362.

To run the code for this question use:

```bash
python[3] sentiment.py
```

This will automatically create/update `pos.txt` and `neg.txt` in the `reports/` directory.
