# Some details about the files

## Data files in the `data` folder

- `data/processed_texts/corpus_full.csv` has all the texts and their metadata (e.g., data, genre)
- `data/processed_texts/corpus_600ad.csv` has all the texts and their metadata (e.g., data, genre)
- `data/processed_texts/latinise_metadata_2024.csv` has the old metadata
- `data/processed_texts/latinise_metadata_2026.csv` has the new metadata
- the raw lemmatized texts are in `/data/lemmatized_texts_2026`
- `latin_sample_full_metadata_only.csv` contains a sample corpus' metadata for manual labeling
- `latin_sample_600_metadata_only.csv` contains a sample corpus' for texts written before 600AD and their metadata for manual labeling
- `latin_sample_2_full_metadata_only.csv` contains another sample corpus' metadata for manual labeling that does not include the texts in the first sample.
- `latin_sample_2_600_metadata_only.csv` contains another sample corpus' for texts written before 600AD and their metadata for manual labeling that does not include the texts in the first sample.
- `latin_sample_final.csv` contains the manually labeled training dataset
- `corpus_600ad_with_NB_ridge_SVM_XGB_LGB.csv` has the model predictions along with the probabilities of christian label assignment by the different models.
 
## Data processing sripts in the `scripts` folder

- `scripts/data_processing_full` has code to merge the metadata and texts and to create sample corpora
- `scripts/data_processing_600ad` has code to merge the metadata and texts written up until 600ad and to create sample corpora

## Analysis files in the `scripts` folder

- `text_classification_full.qmd` contains the code for data preparation and the classification models trained on the training data:
  - **Naive Bayes classifer** using word frequencies
  - **Logistic regression** model using *tf-idf* indices
  - **SVM** model using *tf-idf* indices
  - **XGBoost** model using *tf-idf* indices
  - **LightGBM** model using *tf-idf* indices
- `text_classification_600ad.qmd` has the same models as `text_classification_full.qmd` but only trains models on texts written before 600 AD
  - This file also contains code for *k*-fold validation
- `text_classification_nb.qmd` has a **Gaussian Naive Bayes model** for text classification using *tf-idf* scores of words as a continuous measure
