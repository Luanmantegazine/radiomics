"""Morphometric tier: CN vs AD, FreeSurfer features only (no age/sex).

- Volumes: normalised by eTIV inside every training fold (inside the Pipeline,
  so it is refit for each inner and outer training split).
    residual     : OLS volume ~ eTIV fitted on CN rows of the training split only;
                   the residual is the feature (default).
    proportional : volume / eTIV.
- Thickness (ThickAvg): raw, never eTIV-normalised.
- z-score fitted inside the training split; left/right kept separate.
- Nested CV grouped by subject (StratifiedGroupKFold), pooled out-of-fold
  predictions, cluster bootstrap by subject for the 95% CIs.
"""
import argparse, json, warnings
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from common import (ETIV, OPTIONAL_VOLUME_FEATURES, THICKNESS_FEATURES, VOLUME_FEATURES)

warnings.filterwarnings("ignore")


class HeadSizeNormalizer(BaseEstimator, TransformerMixin):
    """Normalise volume columns by eTIV; pass thickness columns through raw."""
    def __init__(self, vol_cols, thick_cols, etiv_col=ETIV, method="residual"):
        self.vol_cols, self.thick_cols = vol_cols, thick_cols
        self.etiv_col, self.method = etiv_col, method

    def fit(self, X, y=None):
        if self.method == "residual":
            ctl = np.asarray(y) == 0                       # CN rows of the training split
            e = X.loc[ctl, self.etiv_col].to_numpy(float)
            self.coef_ = {}
            for c in self.vol_cols:
                b, a = np.polyfit(e, X.loc[ctl, c].to_numpy(float), 1)
                self.coef_[c] = (a, b)
        return self

    def transform(self, X):
        e = X[self.etiv_col].to_numpy(float)
        cols = []
        for c in self.vol_cols:
            v = X[c].to_numpy(float)
            if self.method == "residual":
                a, b = self.coef_[c]; cols.append(v - (a + b * e))
            else:
                cols.append(v / e)
        for c in self.thick_cols:
            cols.append(X[c].to_numpy(float))
        return np.column_stack(cols)


def make_models(vol, thick, method, seed):
    def pipe(clf):
        return Pipeline([("norm", HeadSizeNormalizer(vol, thick, method=method)),
                         ("z", StandardScaler()), ("clf", clf)])
    return {
        "ElasticNet-LR": (pipe(LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000,
                                                  class_weight="balanced", random_state=seed)),
                          {"clf__C": [0.01, 0.1, 1, 10], "clf__l1_ratio": [0.1, 0.5, 0.9]}),
        "SVM-RBF": (pipe(SVC(probability=True, class_weight="balanced", random_state=seed)),
                    {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01]}),
        "RandomForest": (pipe(RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                     random_state=seed, n_jobs=-1)),
                         {"clf__max_depth": [2, 4, None], "clf__min_samples_leaf": [3, 8]}),
        "GradBoost": (pipe(HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)),
                      {"clf__max_depth": [2, 3], "clf__learning_rate": [0.05, 0.1],
                       "clf__max_iter": [100]}),
    }


def cluster_bootstrap(y, p, groups, n_boot, seed):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}
    aucs, bacc = [], []
    for _ in range(n_boot):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if len(np.unique(y[s])) < 2: continue
        aucs.append(roc_auc_score(y[s], p[s])); bacc.append(balanced_accuracy_score(y[s], p[s] >= 0.5))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return ci(aucs), ci(bacc)


def calibration(y, p):
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6).fit(z, y)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="morpho_tier.csv")
    ap.add_argument("--method", choices=["residual", "proportional"], default="residual")
    ap.add_argument("--with-optional", action="store_true",
                    help="add lateral ventricles and cortex volume (volumes, eTIV-normalised)")
    ap.add_argument("--outer", type=int, default=5); ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=1, help="repeats of the outer CV (different splits)")
    ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="morpho_results.json")
    a = ap.parse_args()

    vol = VOLUME_FEATURES + (OPTIONAL_VOLUME_FEATURES if a.with_optional else [])
    thick = THICKNESS_FEATURES
    need = vol + thick + [ETIV]                       # explicit whitelist: no age/sex
    df = pd.read_csv(a.data)
    df = df[df.supervised_label.isin(["CN", "AD"])]
    n0 = len(df); df = df.dropna(subset=need).reset_index(drop=True)
    if len(df) < n0: print(f"dropped {n0 - len(df)} sessions with missing features")
    if df.empty: raise SystemExit("No complete rows: fill the fs_* columns first (extract_freesurfer.py).")
    X, y, g = df[need], (df.supervised_label == "AD").astype(int).to_numpy(), df.subject_id.to_numpy()
    print(f"{len(df)} sessions, {len(np.unique(g))} subjects, AD={y.sum()} CN={(1-y).sum()}; "
          f"method={a.method}; features={len(vol)+len(thick)}")

    results = {}
    for name in make_models(vol, thick, a.method, a.seed):
        oof = np.zeros((a.repeats, len(df))); fold_auc = []
        for r in range(a.repeats):
            outer = StratifiedGroupKFold(a.outer, shuffle=True, random_state=a.seed + r)
            for tr, te in outer.split(X, y, g):
                est, grid = make_models(vol, thick, a.method, a.seed)[name]
                inner = StratifiedGroupKFold(a.inner, shuffle=True, random_state=a.seed)
                gs = GridSearchCV(est, grid, scoring="roc_auc", cv=inner, n_jobs=-1)
                gs.fit(X.iloc[tr], y[tr], groups=g[tr])
                oof[r, te] = gs.predict_proba(X.iloc[te])[:, 1]
                fold_auc.append(roc_auc_score(y[te], oof[r, te]) if len(np.unique(y[te])) > 1 else np.nan)
        p = oof.mean(0)
        auc_ci, bacc_ci = cluster_bootstrap(y, p, g, a.boot, a.seed)
        pred = p >= 0.5
        sens = float((pred & (y == 1)).sum() / (y == 1).sum()); spec = float((~pred & (y == 0)).sum() / (y == 0).sum())
        ci_int, ci_slope = calibration(y, p)
        results[name] = {"auc": float(roc_auc_score(y, p)), "auc_ci95": auc_ci,
                         "balanced_acc": float(balanced_accuracy_score(y, pred)), "balanced_acc_ci95": bacc_ci,
                         "sensitivity": sens, "specificity": spec, "brier": float(brier_score_loss(y, p)),
                         "calib_intercept": ci_int, "calib_slope": ci_slope,
                         "fold_auc_mean": float(np.nanmean(fold_auc)), "fold_auc_sd": float(np.nanstd(fold_auc))}
        print(f"{name:14s} AUC {results[name]['auc']:.3f} [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]  "
              f"bAcc {results[name]['balanced_acc']:.3f}  sens {sens:.2f}  spec {spec:.2f}  Brier {results[name]['brier']:.3f}")

    # Interpretation only: Elastic Net refit on all data (not an unbiased performance estimate)
    est, grid = make_models(vol, thick, a.method, a.seed)["ElasticNet-LR"]
    gs = GridSearchCV(est, grid, scoring="roc_auc", cv=StratifiedGroupKFold(a.inner, shuffle=True, random_state=a.seed))
    gs.fit(X, y, groups=g)
    coefs = dict(zip(vol + thick, gs.best_estimator_.named_steps["clf"].coef_[0].round(4).tolist()))
    results["_elasticnet_full_fit_coefs"] = coefs
    json.dump(results, open(a.out, "w"), indent=2); print("coefs:", coefs); print("saved", a.out)

if __name__ == "__main__":
    main()
