import os, json, math, random, re, copy, time, unicodedata
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_linear_schedule_with_warmup

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

ROOT = Path('.')
DATA_DIR = Path('../../data/processed_texts/600ad')
STOPWORD_DIR = Path('../../data/stopwords')
BERT_OUTPUT_DIR = Path('../../results/predictions/bert_outputs_unlemmatized/600ad')
TEMPORAL_SWEEP_DIR = BERT_OUTPUT_DIR / 'temporal_sweep'
CHECKPOINT_DIR = TEMPORAL_SWEEP_DIR / 'checkpoints_temporal_sweep'
OUTPUT_DIR = TEMPORAL_SWEEP_DIR / 'model_outputs_temporal_sweep'
PROGRESS_PATH = TEMPORAL_SWEEP_DIR / 'model_progress_log.csv'

N_REMOVE_GRID = [0, 25, 50, 100, 200]
MIN_DOCFREQ_RANK = 5
MASK_WITH_TOKEN = None

LEMMA_CORPUS_CANDIDATES = [
    DATA_DIR / 'corpus_600ad.csv',
    DATA_DIR / 'corpus_600ad_lemmatized.csv',
    DATA_DIR / 'corpus_600ad_lemmas.csv',
]
SURFACE_CORPUS_CANDIDATES = [
    DATA_DIR / 'corpus_600ad_unlemmatized.csv',
    DATA_DIR / 'corpus_600ad_surface.csv',
    DATA_DIR / 'corpus_600ad_raw.csv',
    DATA_DIR / 'corpus_600ad.csv',
]
LABEL_PATH_CANDIDATES = [
    DATA_DIR / 'corpus_600ad_labeled.csv',
]
CONFIG_CANDIDATES = [
    BERT_OUTPUT_DIR / 'bert_run_config.json',
    Path('bert_run_config.json'),
    Path('bert_run_config(7).json'),
]
STOPWORD_CANDIDATES_1 = [STOPWORD_DIR / 'stopwords_latin1.txt']
STOPWORD_CANDIDATES_2 = [STOPWORD_DIR / 'stopwords_latin2.txt']

SURFACE_TEXT_COL_CANDIDATES = ['text_unlemmatized', 'text_surface', 'surface_text', 'raw_text', 'text_raw', 'text']
LEMMA_TEXT_COL_CANDIDATES = ['text_lemmatized', 'lemma_text', 'lemmatized_text', 'lemmas', 'text_lemma', 'text']
ID_COL_CANDIDATES = ['id']
DATE_START_COL_CANDIDATES = ['date_range_start', 'date_start']
DATE_END_COL_CANDIDATES = ['date_range_end', 'date_end']
LABEL_COL_CANDIDATES = ['label', 'manual_label']

for p in [TEMPORAL_SWEEP_DIR, CHECKPOINT_DIR, OUTPUT_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def find_first_existing(candidates):
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    raise FileNotFoundError('None of these paths exists: ' + ' ; '.join(map(str, candidates)))


def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f'Could not find any of {candidates} in columns: {list(df.columns)}')
    return None


def log_progress(step, detail=''):
    row = pd.DataFrame([{'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'step': step, 'detail': detail}])
    print(f"[{row.iloc[0]['time']}] {step}" + (f" | {detail}" if detail else ''))
    if PROGRESS_PATH.exists():
        old = pd.read_csv(PROGRESS_PATH)
        pd.concat([old, row], ignore_index=True).to_csv(PROGRESS_PATH, index=False)
    else:
        row.to_csv(PROGRESS_PATH, index=False)


def safe_prob(p, eps=1e-15):
    p = np.asarray(p, dtype=float)
    return np.clip(p, eps, 1 - eps)


def safe_logit(p, eps=1e-15):
    p = safe_prob(p, eps)
    return np.log(p / (1 - p))


def auc_safe(y, p):
    y01 = (pd.Series(y).astype(str) == 'christian').astype(int).to_numpy()
    p = safe_prob(p)
    if len(np.unique(y01)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y01, p))
    except Exception:
        return np.nan


def log_loss_safe(y, p):
    y01 = (pd.Series(y).astype(str) == 'christian').astype(int).to_numpy()
    p = safe_prob(p)
    return float(-(y01 * np.log(p) + (1 - y01) * np.log(1 - p)).mean())


def brier_safe(y, p):
    y01 = (pd.Series(y).astype(str) == 'christian').astype(int).to_numpy()
    p = safe_prob(p)
    return float(np.mean((y01 - p) ** 2))


def hard_metrics(y, pred_label):
    y01 = (pd.Series(y).astype(str) == 'christian').astype(int).to_numpy()
    p01 = (pd.Series(pred_label).astype(str) == 'christian').astype(int).to_numpy()
    tp = int(np.sum((y01 == 1) & (p01 == 1)))
    tn = int(np.sum((y01 == 0) & (p01 == 0)))
    fp = int(np.sum((y01 == 0) & (p01 == 1)))
    fn = int(np.sum((y01 == 1) & (p01 == 0)))
    precision = np.nan if (tp + fp) == 0 else tp / (tp + fp)
    recall = np.nan if (tp + fn) == 0 else tp / (tp + fn)
    specificity = np.nan if (tn + fp) == 0 else tn / (tn + fp)
    f1 = np.nan if (not np.isfinite(precision) or not np.isfinite(recall) or (precision + recall) == 0) else 2 * precision * recall / (precision + recall)
    bal_acc = np.nanmean([recall, specificity])
    den = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0))
    mcc = np.nan if den == 0 else ((tp * tn) - (fp * fn)) / den
    return {
        'accuracy': (tp + tn) / len(y01),
        'balanced_accuracy': bal_acc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'mcc': mcc,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }


def choose_threshold(y, p, metric='youden'):
    y01 = (pd.Series(y).astype(str) == 'christian').astype(int).to_numpy()
    p = safe_prob(p)
    cand = np.unique(np.concatenate([[0.0, 0.5, 1.0], p]))
    if metric == 'youden':
        best_val, best_th = -np.inf, 0.5
        for th in cand:
            pred = (p >= th).astype(int)
            tp = np.sum((pred == 1) & (y01 == 1))
            tn = np.sum((pred == 0) & (y01 == 0))
            fp = np.sum((pred == 1) & (y01 == 0))
            fn = np.sum((pred == 0) & (y01 == 1))
            recall = np.nan if (tp + fn) == 0 else tp / (tp + fn)
            specificity = np.nan if (tn + fp) == 0 else tn / (tn + fp)
            youden = (0 if np.isnan(recall) else recall) + (0 if np.isnan(specificity) else specificity) - 1
            if youden > best_val:
                best_val, best_th = youden, float(th)
        return best_th
    best_val, best_th = -np.inf, 0.5
    for th in cand:
        pred = (p >= th).astype(int)
        tp = np.sum((pred == 1) & (y01 == 1))
        fp = np.sum((pred == 1) & (y01 == 0))
        fn = np.sum((pred == 0) & (y01 == 1))
        den = 2 * tp + fp + fn
        f1 = 0 if den == 0 else 2 * tp / den
        if f1 > best_val:
            best_val, best_th = f1, float(th)
    return best_th


def tokenize_basic(text):
    if not isinstance(text, str):
        return []
    text = unicodedata.normalize('NFC', text)
    toks = re.findall(r"\w+", text, flags=re.UNICODE)
    out = []
    for t in toks:
        t = t.casefold()
        if any(ch.isalpha() for ch in t):
            out.append(t)
    return out


def load_stopwords():
    custom = {'m', 'c', 'l', 'd', 'gt', 'lt', 'br', 'href', 'fridericus', 'heinricus', 'conradus', 'meus', 'tuus'}
    p1 = find_first_existing(STOPWORD_CANDIDATES_1)
    p2 = find_first_existing(STOPWORD_CANDIDATES_2)
    s1 = [x.strip() for x in Path(p1).read_text(encoding='utf-8').splitlines() if x.strip()]
    s2 = []
    for line in Path(p2).read_text(encoding='utf-8').splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            s2.append(line)
    stops = {x for x in s1 + s2 if x != '*'} | custom
    return {unicodedata.normalize('NFC', x).casefold() for x in stops if x}


def load_data():
    lemma_path = find_first_existing(LEMMA_CORPUS_CANDIDATES)
    surface_path = find_first_existing(SURFACE_CORPUS_CANDIDATES)
    label_path = find_first_existing(LABEL_PATH_CANDIDATES)
    lemma_df = pd.read_csv(lemma_path)
    surface_df = pd.read_csv(surface_path)
    labels = pd.read_csv(label_path)
    id_col_lemma = pick_col(lemma_df, ID_COL_CANDIDATES)
    id_col_surface = pick_col(surface_df, ID_COL_CANDIDATES)
    id_col_labels = pick_col(labels, ID_COL_CANDIDATES)
    label_col = pick_col(labels, LABEL_COL_CANDIDATES)
    for df, id_col in [(lemma_df, id_col_lemma), (surface_df, id_col_surface), (labels, id_col_labels)]:
        df[id_col] = df[id_col].astype(str)
    start_col = pick_col(lemma_df, DATE_START_COL_CANDIDATES)
    end_col = pick_col(lemma_df, DATE_END_COL_CANDIDATES)
    lemma_text_col = pick_col(lemma_df, LEMMA_TEXT_COL_CANDIDATES)
    surface_text_col = pick_col(surface_df, SURFACE_TEXT_COL_CANDIDATES)
    lemma_keep = lemma_df[[id_col_lemma, lemma_text_col, start_col, end_col]].copy()
    lemma_keep.columns = ['id', 'lemma_text', 'date_range_start', 'date_range_end']
    surface_keep = surface_df[[id_col_surface, surface_text_col]].copy()
    surface_keep.columns = ['id', 'surface_text']
    labels_keep = labels[[id_col_labels, label_col]].copy()
    labels_keep.columns = ['id', 'manual_label']
    labels_keep = labels_keep[labels_keep['manual_label'].isin(['christian', 'non_christian'])].drop_duplicates('id')
    df = lemma_keep.merge(surface_keep, on='id', how='inner')
    df = df[df['lemma_text'].notna() & df['surface_text'].notna()].copy()
    df['date_range_start'] = pd.to_numeric(df['date_range_start'], errors='coerce')
    df['date_range_end'] = pd.to_numeric(df['date_range_end'], errors='coerce')
    df['date_mid'] = (df['date_range_start'] + df['date_range_end']) / 2
    df = df.reset_index(drop=True)
    df['row_id'] = np.arange(1, len(df) + 1)
    labeled = df.merge(labels_keep, on='id', how='inner').copy()
    labeled = labeled.reset_index(drop=True)
    return df, labeled, {'lemma_path': str(lemma_path), 'surface_path': str(surface_path), 'label_path': str(label_path), 'lemma_text_col': lemma_text_col, 'surface_text_col': surface_text_col}


def rank_date_words(texts, date_mid, stopwords, min_docfreq_rank=5):
    docs = []
    dates = []
    for txt, dt in zip(texts, date_mid):
        toks = [t for t in tokenize_basic(txt) if t not in stopwords]
        if toks and np.isfinite(dt):
            docs.append(' '.join(toks))
            dates.append(float(dt))
    if len(docs) < 5:
        return pd.DataFrame(columns=['term', 'rho', 'abs_rho', 'docfreq'])
    min_df_use = max(min_docfreq_rank, math.ceil(len(docs) * 0.02))
    vec = CountVectorizer(tokenizer=str.split, preprocessor=None, token_pattern=None, lowercase=False, min_df=min_df_use)
    X = vec.fit_transform(docs)
    if X.shape[1] == 0:
        return pd.DataFrame(columns=['term', 'rho', 'abs_rho', 'docfreq'])
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    X_prop = sparse.diags(1.0 / row_sums) @ X
    rhos = []
    dates = np.asarray(dates, dtype=float)
    for j in range(X_prop.shape[1]):
        col = X_prop[:, j].toarray().ravel()
        if np.std(col) == 0 or np.std(dates) == 0:
            rhos.append(np.nan)
        else:
            rhos.append(float(spearmanr(col, dates).statistic))
    terms = np.array(vec.get_feature_names_out())
    docfreq = np.asarray((X > 0).sum(axis=0)).ravel().astype(int)
    out = pd.DataFrame({'term': terms, 'rho': rhos, 'abs_rho': np.abs(rhos), 'docfreq': docfreq})
    out = out.dropna(subset=['rho']).sort_values(['abs_rho', 'term'], ascending=[False, True]).reset_index(drop=True)
    return out


def build_lemma_surface_map(lemma_texts, surface_texts):
    mapping = defaultdict(Counter)
    matched_docs = 0
    skipped_docs = 0
    for lemma_text, surface_text in zip(lemma_texts, surface_texts):
        lemmas = tokenize_basic(lemma_text)
        surfaces = tokenize_basic(surface_text)
        if not lemmas or not surfaces or len(lemmas) != len(surfaces):
            skipped_docs += 1
            continue
        matched_docs += 1
        for lem, surf in zip(lemmas, surfaces):
            mapping[lem][surf] += 1
    return mapping, matched_docs, skipped_docs


def expand_lemmas_to_surface_forms(lemmas, mapping):
    forms = set()
    for lem in lemmas:
        forms.update(mapping.get(lem, {}).keys())
    return forms


TOKEN_OR_GAP_RE = re.compile(r"\w+|\W+", flags=re.UNICODE)
WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def mask_surface_forms(text, surface_forms, mask_token):
    if not surface_forms or not isinstance(text, str):
        return text
    parts = TOKEN_OR_GAP_RE.findall(unicodedata.normalize('NFC', text))
    out = []
    for part in parts:
        if WORD_RE.fullmatch(part):
            norm = part.casefold()
            out.append(mask_token if norm in surface_forms else part)
        else:
            out.append(part)
    return ''.join(out)


def stratified_train_val_split(y, val_frac, seed):
    y = np.asarray(y)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, va_idx = next(splitter.split(np.zeros(len(y)), y))
    return tr_idx, va_idx


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_length, padding=False)
        self.labels = None if labels is None else np.asarray(labels, dtype=int)

    def __len__(self):
        return len(self.enc['input_ids'])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def make_loader(texts, labels, tokenizer, max_length, batch_size, shuffle):
    ds = TextDataset(texts, labels, tokenizer, max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


def train_one_model(train_texts, train_labels, val_texts, val_labels, cfg, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoModelForSequenceClassification.from_pretrained(cfg['model_name'], num_labels=2)
    model.to(device)

    train_loader = make_loader(train_texts, train_labels, tokenizer, cfg['max_length'], cfg['train_batch_size'], True)
    val_loader = make_loader(val_texts, val_labels, tokenizer, cfg['max_length'], cfg['eval_batch_size'], False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    total_steps = max(1, math.ceil(len(train_loader) / cfg['grad_accum_steps']) * cfg['n_epochs'])
    warmup_steps = int(cfg['warmup_ratio'] * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    best_state = None
    best_val = np.inf
    bad_epochs = 0

    for epoch in range(cfg['n_epochs']):
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / cfg['grad_accum_steps']
            loss.backward()
            if step % cfg['grad_accum_steps'] == 0 or step == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        model.eval()
        losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                losses.append(float(out.loss.detach().cpu()))
        val_loss = float(np.mean(losses)) if losses else np.inf
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs > cfg['patience']:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    return model, tokenizer, device


def predict_logits(model, tokenizer, device, texts, batch_size, max_length):
    loader = make_loader(texts, None, tokenizer, max_length, batch_size, False)
    model.eval()
    outs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.detach().cpu().numpy()
            outs.append(logits)
    return np.vstack(outs) if outs else np.zeros((0, 2), dtype=float)


def fit_temperature(logits, labels):
    if len(logits) == 0 or len(np.unique(labels)) < 2:
        return 1.0
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    temp = nn.Parameter(torch.ones(1, dtype=torch.float32))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temp], lr=0.1, max_iter=50, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits_t / temp.clamp(min=1e-3), labels_t)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
        value = float(temp.detach().clamp(min=1e-3).item())
        if not np.isfinite(value):
            return 1.0
        return value
    except Exception:
        return 1.0


def logits_to_scores_probs(logits, temperature=1.0):
    logits = np.asarray(logits, dtype=float) / max(float(temperature), 1e-6)
    score = logits[:, 1] - logits[:, 0]
    prob = 1.0 / (1.0 + np.exp(-score))
    return score, safe_prob(prob)


def summarize_metrics(y, prob, pred_youden, pred_f1, model_name, n_removed):
    hy = hard_metrics(y, pred_youden)
    hf = hard_metrics(y, pred_f1)
    row = {
        'model': model_name,
        'n_removed': int(n_removed),
        'auc': auc_safe(y, prob),
        'log_loss': log_loss_safe(y, prob),
        'brier': brier_safe(y, prob),
    }
    row.update({f'{k}_youden': v for k, v in hy.items()})
    row.update({f'{k}_f1': v for k, v in hf.items()})
    return row


def upsert_rows(existing, new_rows, keys):
    if existing is None or existing.empty:
        return new_rows.copy()
    existing = existing.copy()
    new_rows = new_rows.copy()
    for col in new_rows.columns:
        if col not in existing.columns:
            existing[col] = np.nan
    for col in existing.columns:
        if col not in new_rows.columns:
            new_rows[col] = np.nan
    existing = existing[new_rows.columns]
    mask = pd.Series(True, index=existing.index)
    for _, row in new_rows[keys].drop_duplicates().iterrows():
        m = np.ones(len(existing), dtype=bool)
        for k in keys:
            m &= existing[k].astype(str).fillna('') == str(row[k])
        mask &= ~m
    return pd.concat([existing.loc[mask], new_rows], ignore_index=True)


def merge_prediction_columns(existing_path, new_df, key_candidates=('row_id', 'id')):
    new_df = new_df.copy()
    if existing_path.exists():
        old = pd.read_csv(existing_path)
        keys = [k for k in key_candidates if k in old.columns and k in new_df.columns]
        if keys:
            overlap = [c for c in new_df.columns if c in old.columns and c not in keys]
            old = old.drop(columns=overlap, errors='ignore')
            merged = old.merge(new_df, on=keys, how='outer')
            return merged
    return new_df


def load_config():
    path = find_first_existing(CONFIG_CANDIDATES)
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    return cfg, str(path)


def main():
    log_progress('load_config', '')
    cfg, cfg_path = load_config()
    stopwords = load_stopwords()
    full_df, lab, meta = load_data()
    log_progress('data_loaded', json.dumps({'config_path': cfg_path, **meta}))
    n_nc = int((lab['manual_label'] == 'non_christian').sum())
    n_nc_dated = int(((lab['manual_label'] == 'non_christian') & np.isfinite(lab['date_mid'])).sum())
    log_progress('date_coverage', f'non-Christian labeled = {n_nc}, with valid date_mid = {n_nc_dated}')

    label_to_int = {'non_christian': 0, 'christian': 1}
    model_name = 'bert_temporal_sweep'
    outer_skf = StratifiedKFold(n_splits=cfg.get('outer_folds', 5), shuffle=True, random_state=SEED)

    oof_prob = {n: np.full(len(lab), np.nan) for n in N_REMOVE_GRID}
    oof_score = {n: np.full(len(lab), np.nan) for n in N_REMOVE_GRID}
    oof_pred_y = {n: np.array([''] * len(lab), dtype=object) for n in N_REMOVE_GRID}
    oof_pred_f = {n: np.array([''] * len(lab), dtype=object) for n in N_REMOVE_GRID}
    oof_fold = {n: np.zeros(len(lab), dtype=int) for n in N_REMOVE_GRID}

    removed_lemmas_rows = []
    removed_surface_rows = []
    fold_metric_rows = []
    fold_threshold_rows = []
    temperature_rows = []

    y_all = lab['manual_label'].to_numpy()
    for outer_i, (tr_idx, te_idx) in enumerate(outer_skf.split(np.zeros(len(lab)), y_all), start=1):
        log_progress('outer_fold_start', f'fold {outer_i} of {cfg.get("outer_folds", 5)}')
        train_df = lab.iloc[tr_idx].reset_index(drop=True)
        test_df = lab.iloc[te_idx].reset_index(drop=True)

        nc_train = train_df[(train_df['manual_label'] == 'non_christian') & np.isfinite(train_df['date_mid'])].copy()
        ranking = rank_date_words(nc_train['lemma_text'].tolist(), nc_train['date_mid'].to_numpy(), stopwords, MIN_DOCFREQ_RANK)
        mapping, matched_docs, skipped_docs = build_lemma_surface_map(train_df['lemma_text'].tolist(), train_df['surface_text'].tolist())
        log_progress('date_rank_done', f'outer {outer_i} | ranked {len(ranking)} lemmas | aligned_docs = {matched_docs} | skipped_docs = {skipped_docs}')

        for n_remove in N_REMOVE_GRID:
            top_lemmas = ranking['term'].head(n_remove).tolist() if n_remove > 0 else []
            surface_forms = sorted(expand_lemmas_to_surface_forms(top_lemmas, mapping))
            if top_lemmas:
                removed_lemmas_rows.extend([
                    {'outer_fold': outer_i, 'n_removed': n_remove, 'rank': j + 1, 'term': term, 'rho': float(ranking.iloc[j]['rho']), 'abs_rho': float(ranking.iloc[j]['abs_rho'])}
                    for j, term in enumerate(top_lemmas)
                ])
            if surface_forms:
                removed_surface_rows.extend([
                    {'outer_fold': outer_i, 'n_removed': n_remove, 'term': term} for term in surface_forms
                ])


            mask_token = MASK_WITH_TOKEN or "[MASK]"
            train_masked = [mask_surface_forms(x, set(surface_forms), mask_token) for x in train_df['surface_text'].tolist()]
            test_masked = [mask_surface_forms(x, set(surface_forms), mask_token) for x in test_df['surface_text'].tolist()]

            inner_tr_rel, inner_va_rel = stratified_train_val_split(train_df['manual_label'].to_numpy(), cfg.get('val_frac', 0.1), SEED + outer_i * 1000 + n_remove)
            inner_train_texts = [train_masked[i] for i in inner_tr_rel]
            inner_val_texts = [train_masked[i] for i in inner_va_rel]
            inner_train_labels = train_df['manual_label'].iloc[inner_tr_rel].map(label_to_int).to_numpy()
            inner_val_labels = train_df['manual_label'].iloc[inner_va_rel].map(label_to_int).to_numpy()

            model, tokenizer, device = train_one_model(inner_train_texts, inner_train_labels, inner_val_texts, inner_val_labels, cfg, SEED + outer_i * 1000 + n_remove)
            val_logits = predict_logits(model, tokenizer, device, inner_val_texts, cfg['eval_batch_size'], cfg['max_length'])
            temp = fit_temperature(val_logits, inner_val_labels)
            val_score, val_prob = logits_to_scores_probs(val_logits, temp)
            th_y = choose_threshold(train_df['manual_label'].iloc[inner_va_rel].tolist(), val_prob, 'youden')
            th_f = choose_threshold(train_df['manual_label'].iloc[inner_va_rel].tolist(), val_prob, 'f1')
            test_logits = predict_logits(model, tokenizer, device, test_masked, cfg['eval_batch_size'], cfg['max_length'])
            test_score, test_prob = logits_to_scores_probs(test_logits, temp)

            oof_prob[n_remove][te_idx] = test_prob
            oof_score[n_remove][te_idx] = test_score
            oof_pred_y[n_remove][te_idx] = np.where(test_prob >= th_y, 'christian', 'non_christian')
            oof_pred_f[n_remove][te_idx] = np.where(test_prob >= th_f, 'christian', 'non_christian')
            oof_fold[n_remove][te_idx] = outer_i

            metrics_row = summarize_metrics(test_df['manual_label'].tolist(), test_prob, oof_pred_y[n_remove][te_idx], oof_pred_f[n_remove][te_idx], model_name, n_remove)
            metrics_row['outer_fold'] = outer_i
            fold_metric_rows.append(metrics_row)
            fold_threshold_rows.append({'outer_fold': outer_i, 'model': model_name, 'n_removed': n_remove, 'threshold_youden': th_y, 'threshold_f1': th_f})
            temperature_rows.append({'outer_fold': outer_i, 'model': model_name, 'n_removed': n_remove, 'temperature': temp})
            log_progress('outer_fold_n_done', f'outer {outer_i} | n_remove = {n_remove} | masked_surface_forms = {len(surface_forms)}')

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        log_progress('outer_fold_done', f'fold {outer_i} of {cfg.get("outer_folds", 5)}')

    oof_rows = []
    for n_remove in N_REMOVE_GRID:
        oof_rows.append(summarize_metrics(lab['manual_label'].tolist(), oof_prob[n_remove], oof_pred_y[n_remove], oof_pred_f[n_remove], model_name, n_remove))
    oof_metrics_by_n = pd.DataFrame(oof_rows).sort_values(['model', 'n_removed']).reset_index(drop=True)
    fold_metric_df = pd.DataFrame(fold_metric_rows).sort_values(['model', 'n_removed', 'outer_fold']).reset_index(drop=True)
    fold_threshold_df = pd.DataFrame(fold_threshold_rows).sort_values(['model', 'n_removed', 'outer_fold']).reset_index(drop=True)
    temperature_df = pd.DataFrame(temperature_rows).sort_values(['model', 'n_removed', 'outer_fold']).reset_index(drop=True)
    removed_lemmas_df = pd.DataFrame(removed_lemmas_rows).sort_values(['outer_fold', 'n_removed', 'rank']).reset_index(drop=True) if removed_lemmas_rows else pd.DataFrame(columns=['outer_fold', 'n_removed', 'rank', 'term', 'rho', 'abs_rho'])
    removed_surface_df = pd.DataFrame(removed_surface_rows).sort_values(['outer_fold', 'n_removed', 'term']).reset_index(drop=True) if removed_surface_rows else pd.DataFrame(columns=['outer_fold', 'n_removed', 'term'])

    oof_long = []
    for n_remove in N_REMOVE_GRID:
        oof_long.append(pd.DataFrame({
            'row_id': lab['row_id'],
            'id': lab['id'],
            'manual_label': lab['manual_label'],
            'model': model_name,
            'n_removed': n_remove,
            'outer_fold': oof_fold[n_remove],
            'prob': oof_prob[n_remove],
            'score': oof_score[n_remove],
            'pred_youden': oof_pred_y[n_remove],
            'pred_f1': oof_pred_f[n_remove],
        }))
    oof_long = pd.concat(oof_long, ignore_index=True)

    oof_wide = lab[['row_id', 'id', 'manual_label']].copy()
    for n_remove in N_REMOVE_GRID:
        oof_wide[f'outer_fold_bert_n{n_remove}'] = oof_fold[n_remove]
        oof_wide[f'prob_bert_n{n_remove}'] = oof_prob[n_remove]
        oof_wide[f'score_bert_n{n_remove}'] = oof_score[n_remove]
        oof_wide[f'label_bert_youden_n{n_remove}'] = oof_pred_y[n_remove]
        oof_wide[f'label_bert_f1_n{n_remove}'] = oof_pred_f[n_remove]

    full_removed_lemmas = []
    full_removed_surface = []
    deployment_thresholds = []
    full_predictions_long_rows = []
    full_predictions_wide = full_df[['row_id', 'id']].copy()

    nc_full = lab[(lab['manual_label'] == 'non_christian') & np.isfinite(lab['date_mid'])].copy()
    full_ranking = rank_date_words(nc_full['lemma_text'].tolist(), nc_full['date_mid'].to_numpy(), stopwords, MIN_DOCFREQ_RANK)
    full_mapping, matched_docs, skipped_docs = build_lemma_surface_map(lab['lemma_text'].tolist(), lab['surface_text'].tolist())
    log_progress('full_rank_done', f'ranked {len(full_ranking)} lemmas | aligned_docs = {matched_docs} | skipped_docs = {skipped_docs}')

    for n_remove in N_REMOVE_GRID:
        top_lemmas = full_ranking['term'].head(n_remove).tolist() if n_remove > 0 else []
        surface_forms = sorted(expand_lemmas_to_surface_forms(top_lemmas, full_mapping))
        if top_lemmas:
            full_removed_lemmas.extend([
                {'n_removed': n_remove, 'rank': j + 1, 'term': term, 'rho': float(full_ranking.iloc[j]['rho']), 'abs_rho': float(full_ranking.iloc[j]['abs_rho'])}
                for j, term in enumerate(top_lemmas)
            ])
        if surface_forms:
            full_removed_surface.extend([{'n_removed': n_remove, 'term': term} for term in surface_forms])


        mask_token = MASK_WITH_TOKEN or "[MASK]"
        lab_masked = [mask_surface_forms(x, set(surface_forms), mask_token) for x in lab['surface_text'].tolist()]
        full_masked = [mask_surface_forms(x, set(surface_forms), mask_token) for x in full_df['surface_text'].tolist()]

        tr_rel, va_rel = stratified_train_val_split(lab['manual_label'].to_numpy(), cfg.get('val_frac', 0.1), SEED + 900000 + n_remove)
        fit_texts = [lab_masked[i] for i in tr_rel]
        val_texts = [lab_masked[i] for i in va_rel]
        fit_labels = lab['manual_label'].iloc[tr_rel].map(label_to_int).to_numpy()
        val_labels = lab['manual_label'].iloc[va_rel].map(label_to_int).to_numpy()

        model, tokenizer, device = train_one_model(fit_texts, fit_labels, val_texts, val_labels, cfg, SEED + 900000 + n_remove)
        val_logits = predict_logits(model, tokenizer, device, val_texts, cfg['eval_batch_size'], cfg['max_length'])
        temp = fit_temperature(val_logits, val_labels)
        all_logits = predict_logits(model, tokenizer, device, full_masked, cfg['eval_batch_size'], cfg['max_length'])
        all_score, all_prob = logits_to_scores_probs(all_logits, temp)
        th_y = choose_threshold(lab['manual_label'].tolist(), oof_prob[n_remove], 'youden')
        th_f = choose_threshold(lab['manual_label'].tolist(), oof_prob[n_remove], 'f1')
        deployment_thresholds.append({'n_removed': n_remove, 'model': model_name, 'threshold_youden': th_y, 'threshold_f1': th_f, 'threshold_scope': 'full_labeled_oof_calibrated', 'temperature_full_fit': temp})

        full_predictions_wide[f'prob_bert_n{n_remove}'] = all_prob
        full_predictions_wide[f'score_bert_n{n_remove}'] = all_score
        full_predictions_wide[f'label_bert_youden_n{n_remove}'] = np.where(all_prob >= th_y, 'christian', 'non_christian')
        full_predictions_wide[f'label_bert_f1_n{n_remove}'] = np.where(all_prob >= th_f, 'christian', 'non_christian')

        full_predictions_long_rows.append(pd.DataFrame({
            'row_id': full_df['row_id'],
            'id': full_df['id'],
            'model': model_name,
            'n_removed': n_remove,
            'prob': all_prob,
            'score': all_score,
            'pred_youden': np.where(all_prob >= th_y, 'christian', 'non_christian'),
            'pred_f1': np.where(all_prob >= th_f, 'christian', 'non_christian'),
        }))
        log_progress('full_fit_done_n', f'n_remove = {n_remove} | masked_surface_forms = {len(surface_forms)}')

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    full_predictions_long = pd.concat(full_predictions_long_rows, ignore_index=True)
    deployment_thresholds_df = pd.DataFrame(deployment_thresholds).sort_values(['model', 'n_removed']).reset_index(drop=True)
    full_removed_lemmas_df = pd.DataFrame(full_removed_lemmas).sort_values(['n_removed', 'rank']).reset_index(drop=True) if full_removed_lemmas else pd.DataFrame(columns=['n_removed', 'rank', 'term', 'rho', 'abs_rho'])
    full_removed_surface_df = pd.DataFrame(full_removed_surface).sort_values(['n_removed', 'term']).reset_index(drop=True) if full_removed_surface else pd.DataFrame(columns=['n_removed', 'term'])

    oof_metrics_by_n.to_csv(OUTPUT_DIR / 'bert_oof_metrics_by_n.csv', index=False)
    fold_metric_df.to_csv(OUTPUT_DIR / 'bert_fold_metrics_by_n.csv', index=False)
    fold_threshold_df.to_csv(OUTPUT_DIR / 'bert_fold_thresholds_by_n.csv', index=False)
    temperature_df.to_csv(OUTPUT_DIR / 'bert_fold_temperatures_by_n.csv', index=False)
    removed_lemmas_df.to_csv(OUTPUT_DIR / 'removed_date_lemmas_by_fold.csv', index=False)
    removed_surface_df.to_csv(OUTPUT_DIR / 'removed_surface_forms_by_fold.csv', index=False)
    oof_long.to_csv(OUTPUT_DIR / 'bert_oof_predictions_by_n_long.csv', index=False)
    full_predictions_long.to_csv(OUTPUT_DIR / 'bert_full_predictions_by_n_long.csv', index=False)
    deployment_thresholds_df.to_csv(OUTPUT_DIR / 'bert_deployment_thresholds_by_n.csv', index=False)
    full_removed_lemmas_df.to_csv(OUTPUT_DIR / 'removed_date_lemmas_full_fit.csv', index=False)
    full_removed_surface_df.to_csv(OUTPUT_DIR / 'removed_surface_forms_full_fit.csv', index=False)

    metrics_path = BERT_OUTPUT_DIR / 'bert_metrics_oof.csv'
    if metrics_path.exists():
        old_metrics = pd.read_csv(metrics_path)
        merged_metrics = upsert_rows(old_metrics, oof_metrics_by_n, ['model', 'n_removed'])
    else:
        merged_metrics = oof_metrics_by_n
    merged_metrics.to_csv(metrics_path, index=False)

    oof_path = BERT_OUTPUT_DIR / 'bert_oof_predictions.csv'
    merge_prediction_columns(oof_path, oof_wide).to_csv(oof_path, index=False)

    full_pred_path = BERT_OUTPUT_DIR / 'bert_full_predictions.csv'
    merge_prediction_columns(full_pred_path, full_predictions_wide).to_csv(full_pred_path, index=False)

    log_progress('all_done', 'bert temporal-sweep pipeline finished')


if __name__ == '__main__':
    main()
