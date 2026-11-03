# MLE_CaseStudy_4
Deliverables

This folder contains exactly three submission files: this README, `MLE_CaseStudy_4.ipynb`, and `Amazon_Product_Reviews.csv`.

## Data provenance and labels

The CSV is the full, unmodified 30,846-row download from Kaggle's **Sentiment Analysis: Amazon Product Reviews** dataset: <https://www.kaggle.com/datasets/miriamodeyianypeter/sentiment-analysis-amazon-product-reviews>, retrieved 3 November 2026. It includes Amazon product-review text and `star_rating`. The notebook derives classes from the observed ratings: 1–2 = negative, 3 = neutral, and 4–5 = positive. It does not reuse the source's binary `sentiment` field, because that field cannot support the required three classes.

## Run instructions

Install `pandas`, `scikit-learn`, `torch`, `transformers`, `datasets`, `accelerate`, `matplotlib`, and `seaborn`; then run all cells. Internet access is needed only the first time to obtain `bert-base-uncased`. A CUDA GPU is recommended but not required. The notebook uses a fixed random seed and records actual held-out test metrics after it trains.

## Code walkthrough

1. **Load/validate:** read the adjacent CSV, combine title and body, remove empty reviews, and map star ratings to the documented three labels.
2. **Split:** stratified train/validation/test splits preserve class proportions and keep test data unseen until final evaluation.
3. **LSTM baseline:** Keras is deliberately not used. A PyTorch tokenizer builds its vocabulary from training text only; an embedding → bidirectional LSTM → linear classifier learns the baseline. The loop reports validation loss and macro-F1 each epoch.
4. **BERT:** `AutoTokenizer` and `AutoModelForSequenceClassification` load Hugging Face `bert-base-uncased` with three output labels. `Trainer` fine-tunes it and computes accuracy, macro precision/recall/F1, a classification report, and confusion matrix on the true test split.
5. **Comparison:** a table compares the two models from the values generated in that run. The notebook does not embed fabricated accuracy figures.
6. **Deployment:** expose BERT behind a REST endpoint with input validation, batching, request logging without raw review text, monitoring for latency/class drift, model versioning, rate limits, and a human-review route for low-confidence predictions. Use the smaller LSTM only where its measured quality/latency trade-off is acceptable.

## Responsible use

Star rating is a proxy label, not ground-truth emotion. Inspect errors across product categories and short/long reviews; protect review text and identifiers; monitor performance drift after release. Do not claim a result before running the notebook in the intended environment.
