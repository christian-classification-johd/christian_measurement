# Some details about the files

## data

- `data/processed_texts/` has all the input text data and metadata, incouding both lemmatized and unlemmatized versions of the corpus
- `data/new_lemmatized_texts_2026` has the new updated corpus
- `data/stopwords` has stopwords used in the analyses.

## scripts

`scripts` has all of the classifier models, model evaluation, temporal confounding regression analyses, and lexical features in addition to the data processing scripts.

## results

- `results/predictions` contains the model predictions for both the transformer- and non-transformer-based models.
- `restults/figures` has figures detailing top drivers of classification for each of the models.
- `results/temporal_confounding` has the results of the INLA temporal confounding analyses.
- `results/model_evaluation` has code for the analysis of model performance.
- `results/word_interaction` contains scripts to perform analyses to visualize the interactions between words in the corpus.
