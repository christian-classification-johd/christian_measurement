import pandas as pd
from pathlib import Path

CORPUS_PATH = Path("../../results/predictions/non_transformer_lemmatized/corpus_600ad_model_outputs_oof.csv")
BERT_PATH = Path("../../results/predictions/bert_outputs_unlemmatized/bert_full_predictions.csv")
OUTPUT_PATH = Path("../../results/predictions/corpus_600ad_combined_predictions.csv")

corpus = pd.read_csv(CORPUS_PATH)
bert = pd.read_csv(BERT_PATH)

corpus = corpus.rename(columns={
    "prob_ridge": "prob_elastic_net",
    "score_ridge_logit": "score_elastic_net_logit",
    "label_ridge_youden": "label_elastic_net_youden",
    "label_ridge_f1": "label_elastic_net_f1",
})

bert_cols_to_merge = [
    "row_id",
    "prob_bert",
    "score_bert_logit",
    "score_bert_margin",
    "label_bert_youden",
    "label_bert_f1",
]
bert_subset = bert[bert_cols_to_merge]

combined = corpus.merge(bert_subset, on="row_id", how="left")

metadata_cols = [
    "row_id",
    "id",
    "title",
    "creator",
    "date_range_start",
    "date_range_end",
    "type",
    "file",
    "n_chars_raw",
    "n_chars",
    "n_words_raw",
    "n_words",
]

label_cols = [
    "manual_label",
    "prediction_source",
]

model_cols = [
    "prob_nb", "score_nb_logit", "label_nb_youden", "label_nb_f1",
    "prob_elastic_net", "score_elastic_net_logit", "label_elastic_net_youden", "label_elastic_net_f1",
    "prob_svm", "score_svm_margin", "score_svm_logit", "label_svm_youden", "label_svm_f1",
    "prob_xgb", "score_xgb_margin", "score_xgb_logit", "label_xgb_youden", "label_xgb_f1",
    "prob_lgb", "score_lgb_margin", "score_lgb_logit", "label_lgb_youden", "label_lgb_f1",
    "prob_bert", "score_bert_logit", "score_bert_margin", "label_bert_youden", "label_bert_f1",
]

col_order = metadata_cols + label_cols + model_cols
combined = combined[col_order]

combined.to_csv(OUTPUT_PATH, index=False)

print(f"Combined file saved to: {OUTPUT_PATH}")
print(f"  Rows: {len(combined)}")
print(f"  Columns: {len(combined.columns)}")
print(f"  Labeled rows: {combined['manual_label'].notna().sum()}")
print(f"  Unlabeled rows: {combined['manual_label'].isna().sum()}")
