"""LendingClub dataset loading and preprocessing.

Feature definitions and cleaning rules are extracted from the project notebook.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

YEAR_MIN, YEAR_MAX = 2012, 2015      # only loans issued in this range are used

FINAL_STATUSES = ['Fully Paid', 'Charged Off', 'Default']

TRAIN_FRAC  = 0.7                    # repo: np.split(..., [int(.7*len(df))])

SEED        = 42                     # makes the random split / model randomness reproducible

SAMPLE_FRAC = 0.1                  # e.g. 0.05 for a quick smoke test; None = full data

CLS_USECOLS = ['loan_amnt', 'funded_amnt', 'term', 'int_rate', 'installment', 'emp_length', 'home_ownership',
 'annual_inc', 'verification_status', 'loan_status', 'purpose', 'zip_code', 'addr_state', 'dti',
 'delinq_2yrs', 'earliest_cr_line', 'fico_range_low', 'fico_range_high', 'inq_last_6mths',
 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal',
 'revol_util', 'total_acc', 'out_prncp', 'out_prncp_inv', 'last_fico_range_high',
 'last_fico_range_low', 'collections_12_mths_ex_med', 'mths_since_last_major_derog', 'policy_code',
 'application_type', 'annual_inc_joint', 'dti_joint', 'acc_now_delinq', 'tot_coll_amt',
 'tot_cur_bal', 'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il',
 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'all_util', 'total_rev_hi_lim', 'inq_fi',
 'total_cu_tl', 'inq_last_12m', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy', 'bc_util',
 'chargeoff_within_12_mths', 'delinq_amnt', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op',
 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc',
 'mths_since_recent_bc_dlq', 'mths_since_recent_inq', 'mths_since_recent_revol_delinq',
 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl',
 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats',
 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 'pct_tl_nvr_dlq',
 'percent_bc_gt_75', 'pub_rec_bankruptcies', 'tax_liens', 'tot_hi_cred_lim', 'total_bal_ex_mort',
 'total_bc_limit', 'total_il_high_credit_limit']

REG_USECOLS = ['loan_amnt', 'funded_amnt', 'total_pymnt', 'term', 'int_rate', 'installment', 'issue_d',
 'last_pymnt_d', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'purpose',
 'zip_code', 'addr_state', 'dti', 'delinq_2yrs', 'earliest_cr_line', 'fico_range_low',
 'fico_range_high', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record',
 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'out_prncp', 'out_prncp_inv',
 'last_fico_range_high', 'last_fico_range_low', 'collections_12_mths_ex_med',
 'mths_since_last_major_derog', 'policy_code', 'application_type', 'annual_inc_joint', 'dti_joint',
 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 'open_act_il', 'open_il_12m',
 'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m',
 'all_util', 'total_rev_hi_lim', 'inq_fi', 'total_cu_tl', 'inq_last_12m', 'acc_open_past_24mths',
 'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths', 'delinq_amnt',
 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl',
 'mort_acc', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq', 'mths_since_recent_inq',
 'mths_since_recent_revol_delinq', 'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl',
 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0',
 'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m',
 'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'pub_rec_bankruptcies', 'tax_liens', 'tot_hi_cred_lim',
 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit']

NAN_MAX_COLS = ['mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog',
 'mths_since_rcnt_il', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq', 'mths_since_recent_inq',
 'mths_since_recent_revol_delinq', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op',
 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl']

NAN_ZERO_COLS = ['tot_coll_amt', 'tot_cur_bal', 'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m',
 'total_bal_il', 'open_rv_12m', 'open_rv_24m', 'inq_fi', 'inq_last_12m', 'num_accts_ever_120_pd',
 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats', 'num_tl_120dpd_2m', 'num_tl_30dpd',
 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m', 'mort_acc', 'percent_bc_gt_75', 'tot_hi_cred_lim',
 'total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit', 'bc_open_to_buy',
 'acc_open_past_24mths', 'total_cu_tl', 'total_rev_hi_lim']

NAN_MEAN_COLS = ['dti', 'annual_inc_joint', 'dti_joint', 'il_util', 'all_util', 'bc_util', 'pct_tl_nvr_dlq',
 'avg_cur_bal', 'revol_util']

ORIG_DTYPES = {'addr_state': 'category',
 'annual_inc': 'float64',
 'application_type': 'category',
 'delinq_2yrs': 'int64',
 'dti': 'float64',
 'emp_length': 'category',
 'fico_range_high': 'int64',
 'fico_range_low': 'int64',
 'home_ownership': 'category',
 'inq_last_6mths': 'int64',
 'int_rate': 'object',
 'loan_amnt': 'int64',
 'loan_status': 'category',
 'open_acc': 'int64',
 'pub_rec': 'int64',
 'purpose': 'category',
 'revol_bal': 'int64',
 'revol_util': 'object',
 'term': 'category',
 'total_acc': 'int64',
 'verification_status': 'category',
 'zip_code': 'category'}

CAT_COLS = ['term', 'emp_length', 'home_ownership', 'verification_status', 'purpose',
            'zip_code', 'addr_state', 'application_type']

INT_REQUIRED = [c for c, t in ORIG_DTYPES.items() if t == 'int64']

RAW_COLS = sorted(set(CLS_USECOLS) | set(REG_USECOLS) | {'issue_d', 'last_pymnt_d', 'total_pymnt'})

def load_raw(csv_path, year_min, year_max, chunksize=200_000):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'{csv_path} not found. Put accepted_2007_to_2018Q4.csv next to the notebook (or set CSV_PATH).')
    if os.path.getsize(csv_path) < 1_000_000:
        raise ValueError(f'{csv_path} is only {os.path.getsize(csv_path)} bytes - it looks like a shortcut/alias, '
                         'not the real dataset. Copy the actual CSV file (~1.5 GB) instead.')
    header = pd.read_csv(csv_path, nrows=0).columns
    missing = [c for c in RAW_COLS if c not in header]
    if missing:
        raise KeyError(f'Columns missing from the CSV: {missing}')

    str_cols = {c: 'object' for c in CAT_COLS + ['int_rate', 'revol_util', 'issue_d', 'last_pymnt_d',
                                                  'earliest_cr_line', 'loan_status']}
    parts, n_read = [], 0
    for chunk in pd.read_csv(csv_path, usecols=RAW_COLS, dtype=str_cols, chunksize=chunksize, low_memory=False):
        n_read += len(chunk)
        year = pd.to_datetime(chunk['issue_d'], format='%b-%Y', errors='coerce').dt.year   # footer/junk rows -> NaT
        keep = year.between(year_min, year_max) & chunk['loan_status'].isin(FINAL_STATUSES)
        parts.append(chunk[keep])
        print(f'  read {n_read:>9,} rows, kept so far {sum(len(p) for p in parts):>9,}', end='\r', flush=True)
    raw = pd.concat(parts, ignore_index=True)
    # A footer/junk row can make a numeric column come back as text in one chunk -> force numeric dtypes.
    num_cols = [c for c in RAW_COLS if c not in str_cols]
    raw[num_cols] = raw[num_cols].apply(pd.to_numeric, errors='coerce')
    print(f'\nRows read: {n_read:,} | kept (issued {year_min}-{year_max}, final status): {len(raw):,}')
    return raw, n_read

def clean_common(df):
    """Shared steps from utils.py: percentage parsing, NaN rules, earliest_cr_line -> days."""
    df = df.copy()
    # parse_percentage(): '13.99%' -> 0.1399  (also works if the file already stores plain numbers like 13.99)
    for c in ['revol_util', 'int_rate']:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace('%', '', regex=False), errors='coerce') / 100.0

    for col in df.columns:                                    # same loop / same order as the repo
        if col in NAN_MEAN_COLS:
            df[col] = df[col].fillna(np.nanmean(df[col].values))
        if col in NAN_ZERO_COLS:
            df[col] = df[col].fillna(0.0)
        if col in NAN_MAX_COLS:
            df[col] = df[col].fillna(np.nanmax(df[col].values) * 5)

    # repo: pd.to_timedelta(pd.to_datetime(x)).dt.days  == days since 1970-01-01
    ecl = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
    df['earliest_cr_line'] = (ecl - pd.Timestamp('1970-01-01')).dt.days
    return df

def fill_residual_nan(df, tag, meta=None):
    """Safety net (not in the repo): any NaN still left would crash scikit-learn. Normally 0 cells."""
    num = df.select_dtypes('number')
    na = num.columns[num.isna().any()]
    n_cells = int(num[na].isna().sum().sum()) if len(na) else 0
    for c in na:
        m = df[c].mean()
        df[c] = df[c].fillna(0.0 if np.isnan(m) else m)
    if meta is not None:
        meta[f'residual_nan_cells_{tag}'] = n_cells
    if n_cells:
        print(f'[{tag}] filled {n_cells:,} residual NaN cells in {len(na)} columns with the column mean')
    return df

def split_and_scale(df, label_col, tag, train_frac=TRAIN_FRAC, seed=SEED, meta=None):
    """70/30 random split + StandardScaler fitted on the train part only (as in the repo)."""
    n = len(df)
    perm = np.random.RandomState(seed).permutation(n)
    n_train = int(train_frac * n)
    tr_idx, te_idx = perm[:n_train], perm[n_train:]
    feats = [c for c in df.columns if c != label_col]
    X = df[feats].to_numpy(dtype=np.float64)
    y = df[label_col].to_numpy()
    train_x, test_x = X[tr_idx], X[te_idx]
    train_y, test_y = y[tr_idx], y[te_idx]
    del X
    scaler = StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    if meta is not None:
        meta[f'n_features_{tag}'] = len(feats)
        meta[f'n_train_{tag}'] = int(len(train_y))
        meta[f'n_test_{tag}'] = int(len(test_y))
    print(f'[{tag}] features: {len(feats):,} | train: {train_x.shape} | test: {test_x.shape}')
    return train_x, train_y, test_x, test_y

def prepare_data(raw, meta=None, train_frac=0.7, seed=42):
    """Classification dataset  (utils.prepare_data)."""
    df = clean_common(raw[CLS_USECOLS])
    df['loan_status'] = df['loan_status'].map({'Charged Off': 0, 'Default': 0, 'Fully Paid': 1})
    df = df.dropna(subset=['loan_status'])
    df['loan_status'] = df['loan_status'].astype(int)
    df = fill_residual_nan(df, 'classification', meta)
    df = pd.get_dummies(df, dtype=np.uint8)                 # one-hot (uint8 as in older pandas)
    if meta is not None:
        meta['class_counts'] = {'Default/Charged Off (0)': int((df['loan_status'] == 0).sum()),
                                'Fully Paid (1)': int((df['loan_status'] == 1).sum())}
        meta['n_loans_classification'] = int(len(df))
    return split_and_scale(df, 'loan_status', 'classification', train_frac, seed, meta)

def prepare_regression_data(raw, meta=None, train_frac=0.7, seed=42):
    """Regression dataset  (utils.prepare_regression_data) - label = (total_pymnt/funded_amnt)^(365/(days+30))."""
    df = clean_common(raw[REG_USECOLS])
    df = df.dropna(subset=['last_pymnt_d'])
    issue = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
    last = pd.to_datetime(df['last_pymnt_d'], format='%b-%Y', errors='coerce')
    df['label'] = (df['total_pymnt'] / df['funded_amnt']) ** (365.0 / ((last - issue).dt.days + 30))
    df = df.drop(['total_pymnt', 'issue_d', 'last_pymnt_d'], axis=1)
    bad = ~np.isfinite(df['label'])
    if meta is not None:
        meta['regression_rows_dropped_bad_label'] = int(bad.sum())
    df = df[~bad]
    df = fill_residual_nan(df, 'regression', meta)
    df = pd.get_dummies(df, dtype=np.uint8)
    if meta is not None:
        meta['n_loans_regression'] = int(len(df))
    return split_and_scale(df, 'label', 'regression', train_frac, seed, meta)

def load_project_data(csv_path, year_min=YEAR_MIN, year_max=YEAR_MAX,
                      sample_frac=SAMPLE_FRAC, chunksize=200_000, seed=SEED):
    """Load raw loans, filter years/statuses, drop malformed required fields,
    and optionally sample using the notebook's configured fraction.
    """
    raw, rows_read = load_raw(csv_path, year_min, year_max, chunksize=chunksize)
    raw = raw.dropna(subset=INT_REQUIRED).reset_index(drop=True)
    if sample_frac is not None:
        raw = raw.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)
    return raw, rows_read
