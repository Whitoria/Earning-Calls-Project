"""
XGBoost Training Pipeline
─────────────────────────
Data sources:
  market+transcript.db  →  call_transcripts, prices
  new_sec_data.db       →  nlp_metrics

Target: N-day post-earnings stock return (tertile classification)

Usage:
  python train_pipeline.py
  python train_pipeline.py --days_after 5
  python train_pipeline.py --time_split --time_split_year 2022
  python train_pipeline.py --mda_only          # restrict NLP to MD&A section only
"""

import argparse
import logging
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, label_binarize
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    market_db        = "data/market+transcript.db",   # contains call_transcripts + prices
    sec_db           = "data/new_sec_data.db",         # contains nlp_metrics
    output_dir       = "model_output",
    days_after       = 1,                         # trading days after call to measure return
    test_size        = 0.20,
    time_split       = False,                     # True = cut by year; False = group shuffle
    time_split_year  = 2022,
    use_sec          = False,                      # False = skip SEC NLP features entirely
    mda_only         = True,                      # True = MD&A section only; False = avg all
    n_estimators     = 300,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.80,
    colsample_bytree = 0.70,
    seed             = 42,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_call_transcripts(db: str) -> pd.DataFrame:
    con = sqlite3.connect(db)
    df = pd.read_sql("SELECT * FROM call_transcripts", con)
    con.close()
    df["report_date"] = pd.to_datetime(df["report_date"])
    log.info(f"call_transcripts: {len(df):,} rows")
    return df


def load_prices(db: str) -> pd.DataFrame:
    con = sqlite3.connect(db)
    df = pd.read_sql("SELECT symbol, report_date, close, company_id FROM prices", con)
    con.close()
    df["report_date"] = pd.to_datetime(df["report_date"])
    df = df.sort_values(["symbol", "report_date"]).reset_index(drop=True)
    log.info(f"prices: {len(df):,} rows")
    return df


def load_nlp_metrics(db: str, mda_only: bool) -> pd.DataFrame:
    con = sqlite3.connect(db)
    if mda_only:
        df = pd.read_sql(
            "SELECT * FROM nlp_metrics WHERE section_name = 'Item 7: MD&A'", con
        )
        if df.empty:
            log.warning("No MD&A rows found — falling back to all sections")
            df = pd.read_sql("SELECT * FROM nlp_metrics", con)
    else:
        df = pd.read_sql("SELECT * FROM nlp_metrics", con)
    con.close()

    # ── Decode filing date from accession_number ──────────────────────────
    # Format: XXXXXXXXXX-YY-NNNNNN  where YY = 2-digit year, NNNNNN = sequence
    # e.g. 0001045810-26-000021 → filed in 2026, sequence 21
    # We reconstruct an approximate date as YYYY-01-01 from the sequence number
    # and use the full accession number to get a more precise date where possible.
    def parse_accession_date(acc: str) -> pd.Timestamp | None:
        try:
            # accession_number may be stored with or without dashes
            # normalise to dashed format: XXXXXXXXXX-YY-NNNNNN
            acc = str(acc).strip()
            if "-" not in acc and len(acc) == 18:
                acc = f"{acc[:10]}-{acc[10:12]}-{acc[12:]}"
            parts = acc.split("-")
            if len(parts) != 3:
                return None
            yy = int(parts[1])
            # 2-digit year: 00-29 → 2000-2029, 30-99 → 1930-1999
            year = 2000 + yy if yy < 30 else 1900 + yy
            # Sequence encodes rough filing order within the year
            # Use it to spread filings across the year (seq/max * 365 days)
            seq = int(parts[2])
            # Cap at 365 days to avoid overflow
            day_offset = min(seq % 366, 365)
            return pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=day_offset)
        except Exception:
            return None

    df["filing_date"] = df["accession_number"].apply(parse_accession_date)

    n_dated = df["filing_date"].notna().sum()
    log.info(f"nlp_metrics: {len(df):,} rows, {n_dated:,} with decoded filing dates  (mda_only={mda_only})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. BUILD TARGET — post-earnings return
# ─────────────────────────────────────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame, calls: pd.DataFrame, days_after: int) -> pd.DataFrame:
    prices_by_sym = {s: g.reset_index(drop=True) for s, g in prices.groupby("symbol")}
    pairs = calls[["ticker", "report_date"]].drop_duplicates()

    records = []
    for _, row in pairs.iterrows():
        sym_prices = prices_by_sym.get(row["ticker"])
        if sym_prices is None:
            continue
        future = sym_prices[sym_prices["report_date"] >= row["report_date"]]
        if len(future) < days_after + 1:
            continue
        p0 = future.iloc[0]["close"]
        p1 = future.iloc[days_after]["close"]
        if p0 == 0:
            continue
        records.append({
            "ticker":       row["ticker"],
            "report_date":  row["report_date"],
            "stock_return": float((p1 - p0) / p0),
        })

    ret = pd.DataFrame(records)
    log.info(f"Returns: {len(ret):,} matched  |  {len(pairs) - len(ret):,} skipped (no price data)")
    return ret


# ─────────────────────────────────────────────────────────────────────────────
# Switching the Label classification to something different

# def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
#     low  = df["stock_return"].quantile(0.33)
#     high = df["stock_return"].quantile(0.67)

#     df["label"] = pd.cut(
#         df["stock_return"],
#         bins=[-np.inf, low, high, np.inf],
#         labels=["negative", "neutral", "positive"],
#     ).astype(str)

#     log.info(f"Label thresholds — negative < {low:+.4f}  |  neutral  |  > {high:+.4f} positive")
#     log.info(f"Label counts:\n{df['label'].value_counts().to_string()}")
#     return df
# ─────────────────────────────────────────────────────────────────────────────


def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    pos_thr = 0.03
    neg_thr = -0.03

    df["label"] = np.where(
        df["stock_return"] >= pos_thr,
        "positive",
        np.where(
            df["stock_return"] <= neg_thr,
            "negative",
            "neutral"
        )
    )

    log.info(
        f"Label thresholds — negative < -3% | neutral | > +3% positive"
    )

    log.info(f"Label counts:\n{df['label'].value_counts().to_string()}")

    return df



# ─────────────────────────────────────────────────────────────────────────────
# 3. MERGE & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

CALL_FEATURE_COLS = [
    "mean_pos", "mean_neg", "mean_neu",
    "std_pos", "std_neg",
    "neg_spike_count", "sentence_count",
    "forward_looking_count", "forward_looking_ratio",
    "prep_mean_pos", "prep_mean_neg",
    "qa_mean_pos", "qa_mean_neg",
    "qa_sentiment_drop", "qa_neg_increase",
    "sentiment_vs_prior",
]

NLP_FEATURE_COLS = [
    "mda_word_count", "mda_avg_sentence_length", "mda_type_token_ratio",
    "mda_readability_flesch", "mda_readability_fog",
    "mda_positive_score", "mda_negative_score", "mda_polarity_score",
    "mda_forward_looking_score", "mda_risk_density",
]


def build_dataset(
    calls: pd.DataFrame,
    prices: pd.DataFrame,
    nlp: pd.DataFrame,
    days_after: int,
) -> pd.DataFrame:

    # ── Target ───────────────────────────────────────────────────────────────
    returns = compute_returns(prices, calls, days_after)
    returns = assign_labels(returns)

    df = calls.merge(returns, on=["ticker", "report_date"], how="inner")
    log.info(f"After returns join: {len(df):,} rows")

    # ── NLP metrics — most recent filing BEFORE each call date ─────────────
    nlp_agg_cols = [
        "word_count", "avg_sentence_length", "type_token_ratio",
        "readability_flesch", "readability_fog",
        "positive_score", "negative_score", "polarity_score",
        "forward_looking_score", "risk_density",
    ]
    nlp_agg_cols = [c for c in nlp_agg_cols if c in nlp.columns]

    if not nlp.empty and "filing_date" in nlp.columns and nlp["filing_date"].notna().any():
        # Time-aware join: match each call to the most recent SEC filing before it
        nlp_dated = nlp.dropna(subset=["filing_date"])[[
            "company_id", "filing_date"] + nlp_agg_cols
        ].sort_values("filing_date")

        df_sorted = df.sort_values("report_date").reset_index()
        merged_nlp = pd.merge_asof(
            df_sorted[["index", "company_id", "report_date"]],
            nlp_dated,
            left_on="report_date",
            right_on="filing_date",
            by="company_id",
            direction="backward",
        ).set_index("index")

        nlp_renamed = merged_nlp[nlp_agg_cols].add_prefix("mda_")
        df = df.join(nlp_renamed)

        matched = df[[f"mda_{nlp_agg_cols[0]}"]].notna().sum().iloc[0]
        log.info(f"After time-aware NLP join: {len(df):,} rows, {matched:,} rows with SEC data ({matched/len(df)*100:.1f}%)")
    elif not nlp.empty:
        log.warning("No filing dates decoded — falling back to per-company average NLP join")
        nlp_agg = (
            nlp.groupby("company_id")[nlp_agg_cols]
            .mean()
            .reset_index()
            .add_prefix("mda_")
            .rename(columns={"mda_company_id": "company_id"})
        )
        df = df.merge(nlp_agg, on="company_id", how="left")
        log.info(f"After NLP join (flat avg): {len(df):,} rows")
    else:
        log.info("SEC NLP join skipped — call features only")

    # ── Engineered features ──────────────────────────────────────────────────
    df = df.sort_values(["ticker", "report_date"]).reset_index(drop=True)

    # Divergence between call optimism and SEC filing tone
    if "mda_polarity_score" in df.columns:
        df["feat_sentiment_divergence"] = df["mean_pos"] - df["mda_polarity_score"]

    # Call negativity vs filing risk density
    if "mda_risk_density" in df.columns:
        df["feat_risk_escalation"] = df["mean_neg"] - (df["mda_risk_density"] / 1000)

    # Readability shift vs prior filing (per ticker)
    if "mda_readability_flesch" in df.columns:
        df["feat_readability_delta"] = (
            df.groupby("ticker")["mda_readability_flesch"].diff()
        )

    # Sentiment momentum (lag-2)
    df["feat_sentiment_lag2"] = (
        df.groupby("ticker")["sentiment_vs_prior"].shift(2)
    )

    # Q&A negativity relative to prepared remarks
    df["feat_qa_neg_ratio"] = df["qa_mean_neg"] / (df["mean_neg"] + 1e-9)

    # Forward-looking ratio change vs prior call
    df["feat_fl_ratio_delta"] = (
        df.groupby("ticker")["forward_looking_ratio"].diff()
    )

    # Negative spike rate normalised by sentence count
    df["feat_spike_rate"] = df["neg_spike_count"] / (df["sentence_count"] + 1)

    # ── Lagged return features (momentum) ────────────────────────────────────
    # Prior quarter return — did the stock already have momentum going in?
    df["feat_prior_return"] = df.groupby("ticker")["stock_return"].shift(1)

    # Was the prior quarter positive?
    df["feat_prior_positive"] = (df["feat_prior_return"] > 0).astype(float)

    # Consecutive quarters moving in the same direction (streak length)
    df["feat_return_streak"] = (
        df.groupby("ticker")["feat_prior_positive"]
        .transform(lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )

    # 2-quarter and 4-quarter average return (trend signal)
    df["feat_return_2q_avg"] = (
        df.groupby("ticker")["stock_return"]
        .transform(lambda x: x.shift(1).rolling(2).mean())
    )
    df["feat_return_4q_avg"] = (
        df.groupby("ticker")["stock_return"]
        .transform(lambda x: x.shift(1).rolling(4).mean())
    )

    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    eng   = [c for c in df.columns if c.startswith("feat_")]
    call  = [c for c in CALL_FEATURE_COLS if c in df.columns]
    nlp   = [c for c in NLP_FEATURE_COLS  if c in df.columns]
    feats = list(dict.fromkeys(call + nlp + eng))
    log.info(
        f"Features: {len(feats)} total  "
        f"(call={len(call)}, nlp={len(nlp)}, engineered={len(eng)})"
    )
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# 4. SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_data(df, feature_cols, cfg):
    X  = df[feature_cols]
    le = LabelEncoder()
    y  = le.fit_transform(df["label"])

    if cfg["time_split"]:
        cutoff     = pd.Timestamp(f"{cfg['time_split_year']}-01-01")
        train_mask = df["report_date"] < cutoff
        tr, te     = np.where(train_mask)[0], np.where(~train_mask)[0]
        log.info(f"Time split @ {cfg['time_split_year']}: train={len(tr):,}  test={len(te):,}")
    else:
        gss = GroupShuffleSplit(1, test_size=cfg["test_size"], random_state=cfg["seed"])
        tr, te = next(gss.split(X, y, groups=df["ticker"]))
        log.info(f"Group shuffle split: train={len(tr):,}  test={len(te):,}")

    return X.iloc[tr], X.iloc[te], y[tr], y[te], le


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X_train, X_test, y_train, y_test, cfg):
    model = XGBClassifier(
        n_estimators          = cfg["n_estimators"],
        max_depth             = cfg["max_depth"],
        learning_rate         = cfg["learning_rate"],
        subsample             = cfg["subsample"],
        colsample_bytree      = cfg["colsample_bytree"],
        use_label_encoder     = False,
        eval_metric           = "mlogloss",
        early_stopping_rounds = 20,
        random_state          = cfg["seed"],
        n_jobs                = -1,
    )
    log.info("Training XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    log.info(f"Best iteration: {model.best_iteration}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, le, df_test, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    preds       = model.predict(X_test)
    proba       = model.predict_proba(X_test)
    class_names = le.classes_

    # ── Classification report ────────────────────────────────────────────────
    report = classification_report(y_test, preds, target_names=class_names)
    log.info(f"\n{report}")
    (out / "classification_report.txt").write_text(report)

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    log.info("Saved confusion_matrix.png")

    # ── ROC-AUC (one-vs-rest) ────────────────────────────────────────────────
    y_bin = label_binarize(y_test, classes=range(len(class_names)))
    try:
        auc = roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")
        log.info(f"Macro ROC-AUC (OvR): {auc:.4f}")
        (out / "roc_auc.txt").write_text(f"Macro ROC-AUC (OvR): {auc:.4f}\n")
    except Exception as e:
        log.warning(f"ROC-AUC failed: {e}")

    # ── Per-ticker accuracy ──────────────────────────────────────────────────
    if "ticker" in df_test.columns:
        records = []
        test_reset = df_test.reset_index(drop=True)
        for tick, grp in test_reset.groupby("ticker"):
            idx = grp.index.tolist()
            yt  = y_test[idx]
            yp  = preds[idx]
            records.append({
                "ticker":   tick,
                "n_calls":  len(yt),
                "accuracy": float((yt == yp).mean()),
            })
        tick_df = pd.DataFrame(records).sort_values("accuracy", ascending=False)
        tick_df.to_csv(out / "per_ticker_accuracy.csv", index=False)

        fig, ax = plt.subplots(figsize=(max(8, len(tick_df) * 0.4), 5))
        ax.bar(tick_df["ticker"], tick_df["accuracy"], color="steelblue")
        ax.axhline(0.333, color="red", linestyle="--", label="Random baseline (33%)")
        ax.set_xlabel("Ticker")
        ax.set_ylabel("Accuracy")
        ax.set_title("Per-Ticker Accuracy")
        ax.legend()
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(out / "per_ticker_accuracy.png", dpi=150)
        plt.close(fig)
        log.info("Saved per_ticker_accuracy.png + .csv")

    # ── Learning curve ───────────────────────────────────────────────────────
    results = model.evals_result()
    if results:
        val_loss = results["validation_0"]["mlogloss"]
        fig, ax  = plt.subplots(figsize=(8, 4))
        ax.plot(val_loss, label="Validation mlogloss")
        ax.axvline(model.best_iteration, color="red", linestyle="--", label="Best iteration")
        ax.set_xlabel("Boosting Round")
        ax.set_ylabel("mlogloss")
        ax.set_title("XGBoost Learning Curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "learning_curve.png", dpi=150)
        plt.close(fig)
        log.info("Saved learning_curve.png")

    # ── SHAP feature importance ──────────────────────────────────────────────
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, show=False, plot_type="dot", max_display=25)
        plt.tight_layout()
        plt.savefig(out / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Saved shap_summary.png")
    except Exception as e:
        log.warning(f"SHAP failed: {e}")

    # ── Summary banner ───────────────────────────────────────────────────────
    log.info("\n" + "=" * 55)
    log.info("  OUTPUTS")
    log.info("=" * 55)
    for f in sorted(out.iterdir()):
        log.info(f"  {f.name}")
    log.info("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: dict):
    calls  = load_call_transcripts(cfg["market_db"])
    prices = load_prices(cfg["market_db"])

    if cfg["use_sec"]:
        nlp = load_nlp_metrics(cfg["sec_db"], cfg["mda_only"])
    else:
        log.info("SEC NLP features disabled (--no-use_sec) — using call features only")
        nlp = pd.DataFrame()   # empty DataFrame; build_dataset will skip the join

    df = build_dataset(calls, prices, nlp, cfg["days_after"])
    df = df.dropna(subset=["label"])

    feature_cols = get_feature_cols(df)
    df_model     = df[feature_cols + ["label", "ticker", "report_date"]].copy()

    # Drop rows missing more than half their features
    min_valid = int(len(feature_cols) * 0.5)
    df_model  = df_model.dropna(thresh=min_valid + 3)
    log.info(f"Final model dataset: {len(df_model):,} rows")

    if len(df_model) < 100:
        raise ValueError(
            "Too few rows after cleaning. "
            "Check that DB paths are correct and tables contain data."
        )

    X_train, X_test, y_train, y_test, le = split_data(df_model, feature_cols, cfg)
    model = train_model(X_train, X_test, y_train, y_test, cfg)

    out = Path(cfg["output_dir"])
    evaluate(model, X_test, y_test, le, df_model.loc[X_test.index].reset_index(drop=True), out)

    model.save_model(str(out / "xgb_model.ubj"))
    log.info(f"Model saved → {out / 'xgb_model.ubj'}")


def parse_args() -> dict:
    p = argparse.ArgumentParser(
        description="Train XGBoost on earnings call + SEC + price data"
    )
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", default=v, action="store_true")
        else:
            p.add_argument(f"--{k}", default=v, type=type(v))
    return vars(p.parse_args())


if __name__ == "__main__":
    cfg = parse_args()
    log.info(f"Config: {cfg}")
    run(cfg)