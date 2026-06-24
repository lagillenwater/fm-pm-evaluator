"""Per-patient drug-recommendation eval for Soragni -- the clinical metric.

For each organoid we rank its screened drugs by each policy and score the top-1/3/5
recommendation against the organoid's own screen:

  regret@k = best observed viability among the top-k picks minus the panel's best
             (0 = the shortlist contains the patient's actual best drug). Reported
             raw (viability points) and normalized to [0,1] (the recommended
             drug's potency percentile within the panel; panel-size invariant).

  recall@k = among patients with >=1 drug below a response threshold tau, the
             fraction whose top-k shortlist contains an effective (sub-tau) drug.

Two contexts:

  cohort (default): models may use the other patients' screens (leave-one-patient-out).
    drugmean_soragni / expr_pca / stack / biomarker, baseline = drugmean_soragni.

  --screen-free: NO Soragni screens in training at all -- every model is trained on
    GDSC2 only and applied to Soragni's transcriptome (the actual clinical setting:
    predict from the tumor transcriptome, no organoid screen). baseline = gdsc_mean.
    Policies: gdsc_mean (external prior), expr_transfer / stack_transfer (per-drug
    GDSC2 ridge -> Soragni, shared drugs only), bilinear (PharmaFormer-lite:
    expression x drug-structure, full panel), biomarker (prior rules + GDSC2 base).

Personalization diagnostic (Fig-7 analog, after Ahlmann-Eltze et al. 2025): for each
policy, how much does the recommendation actually depend on the patient? cross_patient_rho
is the mean pairwise Spearman of the predicted drug-score vectors across patients --
1.00 means every patient gets the SAME ranking (collapsed to the drug-mean / marginal),
and the observed value is the real level the model would have to reproduce.

Controls (--control), each run against the SAME downstream eval so the metric brackets
itself:

  negative: within-drug permutation of the Soragni response (breaks the patient<->response
    link, preserves each drug's marginal). Any personalization edge must vanish; a model
    whose lift survives this is reading an artifact. Averaged over --n-control draws.

  positive: a planted, low-rank drug x patient interaction on a flat drug-mean, in the
    representation set by --plant-space (expr genes, or the Stack embedding). A model can
    only recover a signal that lives in its own input space, so Stack needs its own plant
    -- an expr-space plant under-tests it. In screen-free the SAME per-drug direction is
    planted in GDSC2 and Soragni so the matching transfer model can recover it: that
    model's regret drops toward oracle and its lift over the prior turns significant,
    proving the apparatus has power and the real null is biological, not a dead pipeline.

  uv run python scripts/per_patient_eval.py
  uv run python scripts/per_patient_eval.py --screen-free --stack-gdsc g.csv --stack-soragni s.csv
  uv run python scripts/per_patient_eval.py --control negative
  uv run python scripts/per_patient_eval.py --screen-free --control positive
  uv run python scripts/per_patient_eval.py --screen-free --control positive --plant-space stack \
      --stack-gdsc g.csv --stack-soragni s.csv
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fmharness.bilinear import bilinear_features
from fmharness.controls import permute_within_drug, plant_interaction
from fmharness.data.loaders import canonicalize_patient_id, load_tranche
from fmharness.evaluation import build_sample_design, cpm_bundle, grouped_cv_predict
from fmharness.probe import make_head
from fmharness.probe.heads import HEADS
from fmharness.probe.kernel import _GAMMAS, _KERNEL_ALPHAS

SEED = 0
KS = (1, 3, 5)
TAUS = (50.0, 70.0)  # Viability_Score (% of vehicle); < tau = an effective drug
ALPHAS = np.logspace(-1.0, 6.0, 8)

# Genomic biomarker -> drug links (mirrors the genomic rows of biomarker_anchored.py).
GENOMIC = [
    {"drug": "Everolimus", "gene": "PIK3CA", "kind": "mut", "direction": "sensitize"},
    {"drug": "Rapamycin", "gene": "PIK3CA", "kind": "mut", "direction": "sensitize"},
    {"drug": "Trametinib", "gene": "NF1", "kind": "mut", "direction": "sensitize"},
    {"drug": "Palbociclib", "gene": "CDK4", "kind": "amp", "direction": "sensitize"},
    {"drug": "Palbociclib", "gene": "RB1", "kind": "del", "direction": "resist"},
    {"drug": "Lenvatinib", "gene": "FGFR4", "kind": "amp", "direction": "sensitize"},
    {"drug": "Dovitinib", "gene": "FGFR4", "kind": "amp", "direction": "sensitize"},
    {"drug": "Olaparib", "gene": "FANCA", "kind": "mut", "direction": "sensitize"},
]


def _wes_alterations(repo: Path) -> dict[str, dict[str, set[str]]]:
    """{kind -> {gene -> set(patient)}} from WES SNV/CNV."""
    snv = pd.read_parquet(repo / "data/raw/soragni/tables/snv.parquet")
    cnv = pd.read_parquet(repo / "data/raw/soragni/tables/cnv.parquet")
    snv = snv[snv["BestEffect_Variant_Classification"].astype(str) != "intron"]
    alt: dict[str, dict[str, set[str]]] = {"mut": {}, "amp": {}, "del": {}}
    snv_genes = snv["BestEffect_Hugo_Symbol"].astype(str)
    for gene, sid in zip(snv_genes, snv["Sample_ID"].astype(str), strict=True):
        alt["mut"].setdefault(gene, set()).add(canonicalize_patient_id(sid))
    for gene, sid, call in zip(
        cnv["Gene"].astype(str),
        cnv["Sample_ID"].astype(str),
        cnv["Pathologist_Call"].astype(str),
        strict=True,
    ):
        p = canonicalize_patient_id(sid)
        if call == "Amplification":
            alt["amp"].setdefault(gene, set()).add(p)
        elif "deletion" in call.lower():
            alt["del"].setdefault(gene, set()).add(p)
    return alt


def _regret_recall(rank: list[str], viab: dict[str, float]) -> dict:
    """regret@k (raw + normalized) and per-tau hit/eligible for one patient ranking."""
    best = min(viab.values())
    worst = max(viab.values())
    rng = (worst - best) or 1.0
    out: dict = {}
    for k in KS:
        if len(viab) <= k and k > 1:
            continue  # top-k trivial when the panel has <= k drugs
        topk_best = min(viab[d] for d in rank[:k])
        out[f"regret_raw@{k}"] = topk_best - best
        out[f"regret_norm@{k}"] = (topk_best - best) / rng
        for tau in TAUS:
            if best < tau:  # an effective drug exists for this patient
                out[f"elig@{k}@{tau}"] = 1
                out[f"hit@{k}@{tau}"] = int(topk_best < tau)
    return out


def _pca(train: pd.DataFrame, test: pd.DataFrame, k: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frozen StandardScaler+PCA fit on train (GDSC2), applied to train and test."""
    pipe = Pipeline([("sc", StandardScaler()), ("pca", PCA(k, random_state=SEED))]).fit(train)
    return (
        pd.DataFrame(pipe.transform(train), index=train.index),
        pd.DataFrame(pipe.transform(test), index=test.index),
    )


def _make_regressor(head: str):
    """Per-drug regressor for the screen-free transfer: linear ridge or RBF kernel.

    The kernel variant mirrors KernelProbe -- standardize the PCA scores, then an
    RBF KernelRidge with (alpha, gamma) chosen by inner CV -- so the head-invariance
    check uses the same nonlinear map here as in the transfer benchmark.
    """
    if head == "kernel":
        return GridSearchCV(
            Pipeline([("sc", StandardScaler()), ("kr", KernelRidge(kernel="rbf"))]),
            {"kr__alpha": list(_KERNEL_ALPHAS), "kr__gamma": list(_GAMMAS)},
            scoring="neg_mean_squared_error",
        )
    return RidgeCV(alphas=ALPHAS)


def _per_drug_transfer(
    z_tr: pd.DataFrame,
    dg: pd.DataFrame,
    z_te: pd.DataFrame,
    panel: dict,
    drug2cid: dict,
    cid2auc: dict,
    big: float,
    head: str = "linear",
) -> tuple[dict, float]:
    """Regress per shared drug on GDSC2 (z -> AUC), predict Soragni. Returns (scores,
    in-distribution held-out Spearman as a not-a-dead-model sanity)."""
    models: dict[str, object] = {}
    indist_t: list[float] = []
    indist_p: list[float] = []
    for cid, g in dg.groupby("drug"):
        y = g.groupby("patient")["y"].mean()
        lines = [ln for ln in y.index if ln in z_tr.index]
        if len(lines) < 10:
            continue
        x = z_tr.loc[lines].to_numpy()
        yv = y.loc[lines].to_numpy(float)
        models[str(cid)] = _make_regressor(head).fit(x, yv)
        # in-dist held-out (5-fold over lines) for the sanity correlation
        for tr, te in GroupKFold(min(5, len(lines))).split(x, yv, groups=lines):
            pred = _make_regressor(head).fit(x[tr], yv[tr]).predict(x[te])
            indist_t.extend(yv[te])
            indist_p.extend(pred)
    out: dict[str, dict[str, float]] = {}
    for p in panel:
        s = {}
        for d in panel[p]:
            cid = str(drug2cid.get(d))
            if cid in models and p in z_te.index:
                s[d] = float(models[cid].predict(z_te.loc[[p]].to_numpy())[0])
            else:
                s[d] = cid2auc.get(cid, big)  # GDSC2 mean if covered-but-unmodeled, else last
        out[p] = s
    rho = float(spearmanr(indist_t, indist_p).statistic) if indist_t else float("nan")
    return out, rho


def _load_fp(repo: Path) -> pd.DataFrame:
    """PubChem CID -> Morgan fingerprint (CoderData drug_descriptors, mapped to CID)."""
    frames = []
    for ds in ("gdscv2", "sarcoma"):
        d = pd.read_csv(repo / f"data/raw/coderdata/{ds}_drug_descriptors.tsv.gz", sep="\t")
        d = d[d["structural_descriptor"] == "morgan fingerprint"]
        dr = pd.read_csv(repo / f"data/raw/coderdata/{ds}_drugs.tsv.gz", sep="\t")
        id2cid = {
            str(i): str(int(c))
            for i, c in zip(dr["improve_drug_id"], dr["pubchem_id"], strict=False)
            if pd.notna(c)
        }
        frames.append(d.assign(cid=d["improve_drug_id"].astype(str).map(id2cid)))
    mf = pd.concat(frames).dropna(subset=["cid"]).drop_duplicates("cid")
    bits = np.array([[int(c) for c in str(v)] for v in mf["descriptor_value"]], dtype=np.float64)
    return pd.DataFrame(bits, index=pd.Index(mf["cid"].astype(str)))


def _bilinear_transfer(
    z_g: pd.DataFrame,
    z_s: pd.DataFrame,
    dg: pd.DataFrame,
    fp: pd.DataFrame,
    panel: dict,
    drug2cid: dict,
    cid2auc: dict,
    big: float,
    kg: int = 20,
) -> tuple[dict, float]:
    """PharmaFormer-lite: ridge on [z, g, z(x)g] over GDSC2 (line x drug), predict
    Soragni (organoid x drug) for every fingerprinted drug. Returns (scores, in-dist
    held-out Spearman)."""
    gdrugs = sorted(set(dg["drug"].astype(str)) & set(fp.index))
    gpca = PCA(min(kg, len(gdrugs) - 1), random_state=SEED).fit(fp.loc[gdrugs])
    gmat = pd.DataFrame(gpca.transform(fp), index=fp.index)

    def feats(z: pd.DataFrame, samples: list[str], cids: list[str]) -> np.ndarray:
        return bilinear_features(z.loc[samples].to_numpy(), gmat.loc[cids].to_numpy())

    tr = dg[dg["drug"].astype(str).isin(gmat.index) & dg["patient"].isin(z_g.index)]
    x_tr = feats(z_g, list(tr["patient"]), list(tr["drug"].astype(str)))
    y_tr = tr["y"].to_numpy(float)
    model = Pipeline([("sc", StandardScaler()), ("ridge", RidgeCV(alphas=ALPHAS))]).fit(x_tr, y_tr)

    # in-dist sanity: held-out GDSC2 lines
    indist_t, indist_p = [], []
    lines = np.array(sorted(tr["patient"].unique()))
    for tri, tei in GroupKFold(min(5, len(lines))).split(
        x_tr, y_tr, groups=tr["patient"].to_numpy()
    ):
        m = Pipeline([("sc", StandardScaler()), ("ridge", RidgeCV(alphas=ALPHAS))]).fit(
            x_tr[tri], y_tr[tri]
        )
        indist_t.extend(y_tr[tei])
        indist_p.extend(m.predict(x_tr[tei]))
    rho = float(spearmanr(indist_t, indist_p).statistic) if indist_t else float("nan")

    out: dict[str, dict[str, float]] = {}
    for p in panel:
        s: dict[str, float] = {}
        valid = [
            (d, str(drug2cid.get(d)))
            for d in panel[p]
            if str(drug2cid.get(d)) in gmat.index and p in z_s.index
        ]
        if valid:
            pred = model.predict(feats(z_s, [p] * len(valid), [c for _, c in valid]))
            s.update({d: float(pr) for (d, _), pr in zip(valid, pred, strict=True)})
        for d in panel[p]:
            s.setdefault(d, cid2auc.get(str(drug2cid.get(d)), big))
        out[p] = s
    return out, rho


def _screenfree_scores(
    x_df: pd.DataFrame,
    dg: pd.DataFrame,
    xg_df: pd.DataFrame,
    panel: dict,
    drug2cid: dict,
    cid2auc: dict,
    alt: dict,
    big: float,
    repo: Path,
    stack_gdsc: str | None,
    stack_soragni: str | None,
    verbose: bool,
    head: str = "linear",
) -> tuple[dict, list[str]]:
    """Build GDSC2-trained (screen-free) policies and the report order."""
    genes = sorted(set(x_df.columns) & set(xg_df.columns))
    z_g, z_s = _pca(np.log1p(xg_df[genes]), np.log1p(x_df[genes]))

    scores: dict[str, dict] = {}
    scores["gdsc_mean"] = {
        p: {d: cid2auc.get(str(drug2cid.get(d)), big) for d in panel[p]} for p in panel
    }
    scores["expr_transfer"], rho_e = _per_drug_transfer(
        z_g, dg, z_s, panel, drug2cid, cid2auc, big, head
    )
    scores["bilinear"], rho_b = _bilinear_transfer(
        z_g, z_s, dg, _load_fp(repo), panel, drug2cid, cid2auc, big
    )
    if verbose:
        print(
            f"  [in-dist sanity] expr_transfer per-drug rho={rho_e:+.3f}  bilinear rho={rho_b:+.3f}"
        )

    order = ["oracle", "gdsc_mean", "expr_transfer"]
    if stack_gdsc and stack_soragni:
        sg = pd.read_csv(stack_gdsc, index_col=0)
        ss = pd.read_csv(stack_soragni, index_col=0)
        sg.index, ss.index = sg.index.astype(str), ss.index.astype(str)
        zsg, zss = _pca(sg, ss)
        scores["stack_transfer"], rho_s = _per_drug_transfer(
            zsg, dg, zss, panel, drug2cid, cid2auc, big, head
        )
        if verbose:
            print(f"  [in-dist sanity] stack_transfer per-drug rho={rho_s:+.3f}")
        order.append("stack_transfer")
    order += ["bilinear", "biomarker", "random"]

    scores["biomarker"] = {}
    for p in panel:
        s = dict(scores["gdsc_mean"][p])
        for bm in GENOMIC:
            if bm["drug"] in s and p in alt[bm["kind"]].get(bm["gene"], set()):
                s[bm["drug"]] += -1e6 if bm["direction"] == "sensitize" else 1e6
        scores["biomarker"][p] = s

    # coverage: mean fraction of each patient's panel a model actually scores (vs falls back)
    if verbose:
        for pol in ("expr_transfer", "stack_transfer", "bilinear"):
            if pol not in scores:
                continue
            cov = np.mean([np.mean([scores[pol][p][d] < big for d in panel[p]]) for p in panel])
            print(f"  [coverage] {pol}: scores {cov:.0%} of panel (rest fall back to GDSC2 mean)")
    return scores, order


def _build_scores(
    args: argparse.Namespace,
    design: pd.DataFrame,
    dg: pd.DataFrame,
    xg_df: pd.DataFrame,
    x_df: pd.DataFrame,
    x_log: pd.DataFrame,
    panel: dict,
    patients: list[str],
    drug2cid: dict,
    cid2auc: dict,
    alt: dict,
    big: float,
    repo: Path,
    verbose: bool,
) -> tuple[dict, list[str]]:
    """All policy scores + report order for the current (possibly control-transformed)
    response. Screen-free trains on GDSC2 only; cohort uses other patients' screens."""
    if args.screen_free:
        return _screenfree_scores(
            x_df,
            dg,
            xg_df,
            panel,
            drug2cid,
            cid2auc,
            alt,
            big,
            repo,
            args.stack_gdsc,
            args.stack_soragni,
            verbose,
            args.head,
        )

    scores: dict[str, dict] = {}
    scores["drugmean_soragni"] = {}
    for p in patients:
        m = design[design["patient"] != p].groupby("drug")["y"].mean().to_dict()
        scores["drugmean_soragni"][p] = {d: m.get(d, np.inf) for d in panel[p]}
    scores["drugmean_gdsc"] = {
        p: {
            d: cid2auc.get(str(drug2cid.get(d)), big + scores["drugmean_soragni"][p][d])
            for d in panel[p]
        }
        for p in patients
    }

    def _frozen(feat: pd.DataFrame) -> tuple[dict, dict]:
        pr = grouped_cv_predict(
            make_head(args.head, n_components=10, per_drug=True, reducer="pca"),
            feat,
            design,
            n_splits=len(patients),
            seed=SEED,
        )
        full = {p: dict(zip(g["drug"], g["y_pred"], strict=True)) for p, g in pr.groupby("patient")}
        nobase = {
            p: dict(zip(g["drug"], g["y_resid"], strict=True)) for p, g in pr.groupby("patient")
        }
        return full, nobase

    scores["expr_pca"], scores["expr_nobase"] = _frozen(x_log)
    if args.stack_soragni:
        es = pd.read_csv(args.stack_soragni, index_col=0)
        es.index = es.index.astype(str)
        scores["stack"], scores["stack_nobase"] = _frozen(es)
    scores["biomarker"] = {}
    for p in patients:
        s = dict(scores["drugmean_soragni"][p])
        for bm in GENOMIC:
            if bm["drug"] in s and p in alt[bm["kind"]].get(bm["gene"], set()):
                s[bm["drug"]] += -1e6 if bm["direction"] == "sensitize" else 1e6
        scores["biomarker"][p] = s
    order = ["oracle", "drugmean_soragni", "drugmean_gdsc", "expr_pca", "expr_nobase"]
    if "stack" in scores:
        order += ["stack", "stack_nobase"]
    order += ["biomarker", "random"]
    return scores, order


def _score(
    scores: dict, panel: dict, patients: list[str], n_rand: int, rng: np.random.Generator
) -> dict[str, list[dict]]:
    """regret/recall record per policy per patient for the current response."""
    recs: dict[str, list[dict]] = {pol: [] for pol in ["oracle", "random", *scores]}
    for p in patients:
        viab = panel[p]
        drugs = list(viab)
        recs["oracle"].append(_regret_recall(sorted(drugs, key=lambda d: viab[d]), viab))
        rr = [_regret_recall(list(rng.permutation(drugs)), viab) for _ in range(n_rand)]
        recs["random"].append({k: float(np.mean([r[k] for r in rr if k in r])) for k in rr[0]})
        for pol, sc in scores.items():
            rank = sorted(drugs, key=lambda d, sc=sc, p=p: sc[p].get(d, np.inf))
            recs[pol].append(_regret_recall(rank, viab))
    return recs


def _personalization(score_by_patient: dict, patients: list[str]) -> tuple[float, float, float]:
    """cross_patient_rho, distinct top-1 count, modal top-1 share for one policy.

    cross_patient_rho = mean pairwise Spearman of the per-patient drug-score vectors
    (shared drugs); 1.0 => every patient gets the same ranking = collapsed to the
    drug-mean. (Pairwise over a 17-organoid cohort -- ~136 pairs, intentionally exact.)
    """
    sb = {p: score_by_patient.get(p, {}) for p in patients}
    rhos: list[float] = []
    for i in range(len(patients)):
        ai = sb[patients[i]]
        for j in range(i + 1, len(patients)):
            bj = sb[patients[j]]
            shared = [d for d in ai if d in bj and np.isfinite(ai[d]) and np.isfinite(bj[d])]
            if len(shared) < 4:
                continue
            va = np.array([ai[d] for d in shared])
            vb = np.array([bj[d] for d in shared])
            if va.std() == 0 or vb.std() == 0:
                rhos.append(1.0)  # constant vectors => identical ranking (collapsed)
                continue
            r = spearmanr(va, vb).statistic
            if np.isfinite(r):
                rhos.append(float(r))
    rho = float(np.mean(rhos)) if rhos else float("nan")
    top1 = []
    for p in patients:
        fin = {d: v for d, v in sb[p].items() if np.isfinite(v)}
        if fin:
            top1.append(min(fin, key=lambda d: fin[d]))
    distinct = float(len(set(top1)))
    modal = float(max(Counter(top1).values()) / len(top1)) if top1 else float("nan")
    return rho, distinct, modal


def _diagnostic(scores: dict, order: list[str], panel: dict, patients: list[str]) -> dict:
    """Personalization row per policy + the observed reference (== oracle ranking)."""
    rows = {"observed": _personalization(panel, patients)}
    for pol in order:
        if pol in scores:
            rows[pol] = _personalization(scores[pol], patients)
    return rows


def _build_response(
    control: str,
    design: pd.DataFrame,
    dg: pd.DataFrame,
    x_log: pd.DataFrame,
    xg_log: pd.DataFrame,
    drug2cid: dict,
    eff_mult: float,
    noise_mult: float,
    plant_rank: int,
    plant_space: str,
    stk: dict,
    screen_free: bool,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (design, dg) with the response transformed for the requested control.
    Only the response (y) changes -- expression is never touched."""
    if control == "none":
        return design, dg
    if control == "negative":
        y = permute_within_drug(design["drug"], design["y"], rng)
        return design.assign(y=y), dg  # break Soragni's patient<->response link only
    # positive: plant a drug x patient interaction on a FLAT drug-mean (the cohort's
    # grand mean). The within-patient ranking is then driven purely by a low-rank,
    # recoverable personalization signal. A model can only recover a signal that lives
    # in (a linear function of) its OWN inputs, so the positive control for Stack must
    # plant in Stack-embedding space (--plant-space stack), not expression -- otherwise
    # a Stack "failure" conflates a poor representation with a signal outside its span.
    use_stack = plant_space == "stack"
    rs = float(
        np.std(design["y"].to_numpy() - design.groupby("drug")["y"].transform("mean").to_numpy())
    )
    if not screen_free:
        emb = (stk["s"] if use_stack else x_log).reindex(design["patient"]).to_numpy()
        gmean = float(design["y"].mean())
        signal = plant_interaction(
            design["drug"],
            pd.Series(np.zeros(len(design))),  # zero response -> flat drug-mean
            emb,
            effect=eff_mult * rs,
            noise_sd=noise_mult * rs,
            rng=rng,
            n_components=plant_rank,
        )
        return design.assign(y=gmean + signal), dg
    zs_src, zg_src = (stk["s"], stk["g"]) if use_stack else (x_log, xg_log)
    ys, yg = _plant_shared_interaction(
        design, dg, zs_src, zg_src, drug2cid, eff_mult, noise_mult, rng, k=plant_rank
    )
    return design.assign(y=ys), dg.assign(y=yg)


def _plant_shared_interaction(
    design: pd.DataFrame,
    dg: pd.DataFrame,
    zs_src: pd.DataFrame,
    zg_src: pd.DataFrame,
    drug2cid: dict,
    eff_mult: float,
    noise_mult: float,
    rng: np.random.Generator,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Plant the SAME per-drug direction in GDSC2 and Soragni (a shared, centered
    drug x patient interaction) on a flat (grand-mean) drug-mean, in whatever
    representation is passed -- shared expression genes or a Stack embedding -- so the
    matching GDSC2-trained transfer model can recover it on Soragni. Directions are
    keyed by PubChem CID; the low-rank PC space is fit on GDSC2 and applied to both.
    Returns (y_soragni, y_gdsc) aligned to design, dg."""
    dims = sorted(set(zs_src.columns) & set(zg_src.columns))
    mg = zg_src[dims].to_numpy()
    ms = zs_src[dims].to_numpy()
    scaler = StandardScaler().fit(mg)
    kk = min(k, mg.shape[0] - 1, len(dims))
    pca = PCA(kk, random_state=SEED).fit(scaler.transform(mg))
    zg = pd.DataFrame(pca.transform(scaler.transform(mg)), index=zg_src.index)
    zs = pd.DataFrame(pca.transform(scaler.transform(ms)), index=zs_src.index)

    g_cids = dg["drug"].astype(str).to_numpy()
    s_cids = np.array([str(drug2cid.get(d)) for d in design["drug"]])
    cids = sorted(set(g_cids) & set(s_cids))
    w = rng.standard_normal((len(cids), kk))
    w -= w.mean(axis=0, keepdims=True)  # center across drugs -> pure interaction
    ci = {c: i for i, c in enumerate(cids)}

    def signal(z: pd.DataFrame, samples: np.ndarray, cid_arr: np.ndarray) -> np.ndarray:
        zz = z.reindex(samples).fillna(0.0).to_numpy()
        idx = np.array([ci.get(c, -1) for c in cid_arr])
        wm = np.zeros((len(cid_arr), kk))
        m = idx >= 0
        wm[m] = w[idx[m]]
        s = np.einsum("ij,ij->i", zz, wm)
        sd = s.std()
        return (s - s.mean()) / (sd if sd > 0 else 1.0)

    sg = signal(zg, dg["patient"].to_numpy(), g_cids)
    ss = signal(zs, design["patient"].to_numpy(), s_cids)
    rs_g = float(np.std(dg["y"].to_numpy() - dg.groupby("drug")["y"].transform("mean").to_numpy()))
    rs_s = float(
        np.std(design["y"].to_numpy() - design.groupby("drug")["y"].transform("mean").to_numpy())
    )
    gmean_g = float(dg["y"].mean())  # flat drug-mean -> ranking is pure interaction
    gmean_s = float(design["y"].mean())
    yg = gmean_g + eff_mult * rs_g * sg + noise_mult * rs_g * rng.standard_normal(len(sg))
    ys = gmean_s + eff_mult * rs_s * ss + noise_mult * rs_s * rng.standard_normal(len(ss))
    return ys, yg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--screen-free", action="store_true", help="train on GDSC2 only, no Soragni screens"
    )
    ap.add_argument("--stack-gdsc", default=None)
    ap.add_argument("--stack-soragni", default=None)
    # Predictive head for the transcriptome policies (expr_pca/stack and the
    # screen-free transfers): "linear" ridge or "kernel" RBF, for head-invariance.
    ap.add_argument("--head", choices=list(HEADS), default="linear")
    ap.add_argument("--n-rand", type=int, default=500)
    ap.add_argument("--control", choices=["none", "negative", "positive"], default="none")
    ap.add_argument(
        "--n-control", type=int, default=0, help="control draws (0 = auto: neg 20, pos 5)"
    )
    ap.add_argument(
        "--plant-effect", type=float, default=2.0, help="planted effect, in within-drug SDs"
    )
    ap.add_argument(
        "--plant-noise", type=float, default=0.5, help="planted noise, in within-drug SDs"
    )
    ap.add_argument(
        "--plant-rank", type=int, default=5, help="PC rank of the planted signal (<= probe's 10)"
    )
    ap.add_argument(
        "--plant-space",
        choices=["expr", "stack"],
        default="expr",
        help="representation to plant the positive signal in (stack requires --stack-*)",
    )
    args = ap.parse_args()
    if (
        args.control == "positive"
        and args.plant_space == "stack"
        and not (args.stack_gdsc and args.stack_soragni)
    ):
        ap.error("--plant-space stack requires --stack-gdsc and --stack-soragni")

    repo = Path(__file__).resolve().parent.parent
    sb = cpm_bundle(load_tranche("sarcoma", repo))
    x_df, design = build_sample_design(sb, "organoid", "viability")  # drug = Soragni name
    x_log = np.log1p(x_df)
    drug2cid = {a.drug_name: a.pubchem_cid for a in sb.drug_assays if a.pubchem_cid}
    alt = _wes_alterations(repo)

    gb = cpm_bundle(load_tranche("gdscv2", repo, cancer_type_filter=["sarcoma"]))
    xg_df, dg = build_sample_design(gb, "all", "auc", drug_key="pubchem_cid")
    xg_log = np.log1p(xg_df)
    patients = sorted(design["patient"].unique())

    # Stack embeddings (if provided): used both as a scored policy and, for the
    # positive control, as the space to plant the signal in (--plant-space stack).
    stk: dict = {}
    if args.stack_gdsc and args.stack_soragni:
        sgf = pd.read_csv(args.stack_gdsc, index_col=0)
        ssf = pd.read_csv(args.stack_soragni, index_col=0)
        sgf.index, ssf.index = sgf.index.astype(str), ssf.index.astype(str)
        stk = {"g": sgf, "s": ssf}

    v = design["y"].to_numpy(float)
    ctx = "SCREEN-FREE (GDSC2-trained)" if args.screen_free else "cohort (leave-one-patient-out)"
    print(f"=== Context: {ctx} ===")
    print(
        f"Viability_Score: n={len(v)} min={v.min():.1f} median={np.median(v):.1f} max={v.max():.1f}"
    )
    panel0 = {p: dict(zip(g["drug"], g["y"], strict=True)) for p, g in design.groupby("patient")}
    npat = len(patients)
    for tau in TAUS:
        elig = sum(min(panel0[p].values()) < tau for p in patients)
        print(f"  tau<{tau:.0f}: {np.mean(v < tau):.0%} of cells; {elig}/{npat} patients treatable")

    n_ctrl = args.n_control or {"none": 1, "negative": 20, "positive": 5}[args.control]
    baseline = "gdsc_mean" if args.screen_free else "drugmean_soragni"
    if args.control != "none":
        print(
            f"\n[control: {args.control}] {n_ctrl} draw(s)"
            + (
                f", planted effect={args.plant_effect}sd noise={args.plant_noise}sd"
                f" in {args.plant_space}-space"
                if args.control == "positive"
                else " (within-drug response permutation)"
            )
        )

    # Run the SAME eval n_ctrl times (1 for the real data), pooling regret/recall
    # records and averaging the personalization diagnostic.
    all_recs: dict[str, list[dict]] = {}
    diag_rows: list[dict] = []
    order: list[str] = []
    for perm in range(n_ctrl):
        crng = np.random.default_rng(1000 + perm)
        design_p, dg_p = _build_response(
            args.control,
            design,
            dg,
            x_log,
            xg_log,
            drug2cid,
            args.plant_effect,
            args.plant_noise,
            args.plant_rank,
            args.plant_space,
            stk,
            args.screen_free,
            crng,
        )
        cid2auc = dg_p.groupby("drug")["y"].mean().to_dict()
        big = max(cid2auc.values()) + 1.0
        panel = {
            p: dict(zip(g["drug"], g["y"], strict=True)) for p, g in design_p.groupby("patient")
        }
        scores, order = _build_scores(
            args,
            design_p,
            dg_p,
            xg_df,
            x_df,
            x_log,
            panel,
            patients,
            drug2cid,
            cid2auc,
            alt,
            big,
            repo,
            verbose=(perm == 0),
        )
        srng = np.random.default_rng(SEED + perm)
        recs = _score(scores, panel, patients, args.n_rand, srng)
        for pol, rows in recs.items():
            all_recs.setdefault(pol, []).extend(rows)
        diag_rows.append(_diagnostic(scores, order, panel, patients))

    def agg(rows: list[dict], key: str) -> float:
        vals = [r[key] for r in rows if key in r]
        return float(np.mean(vals)) if vals else float("nan")

    def recall(rows: list[dict], k: int, tau: float) -> tuple[float, int]:
        elig = [r for r in rows if r.get(f"elig@{k}@{tau}")]
        if not elig:
            return float("nan"), 0
        return float(np.mean([r[f"hit@{k}@{tau}"] for r in elig])), len(elig)

    print("\n=== Normalized regret (0 = picked the patient's best; lower better) ===")
    print(f"{'policy':18s}" + "".join(f"{'@' + str(k):>9}" for k in KS))
    for pol in order:
        print(f"{pol:18s}" + "".join(f"{agg(all_recs[pol], f'regret_norm@{k}'):>9.3f}" for k in KS))

    for tau in TAUS:
        print(f"\n=== recall@k of an effective drug (viability < {tau:.0f}) ===")
        for pol in order:
            cells = "".join(f"{recall(all_recs[pol], k, tau)[0]:>9.2f}" for k in KS)
            print(f"{pol:18s}{cells}{('  (n=' + str(recall(all_recs[pol], 1, tau)[1]) + ')'):>10}")

    print(f"\n=== Lift: paired Wilcoxon of regret_norm@1 vs {baseline} ===")
    base = [r.get("regret_norm@1", np.nan) for r in all_recs[baseline]]
    for pol in [o for o in order if o not in ("oracle", "random", baseline)]:
        diff = np.array([r.get("regret_norm@1", np.nan) for r in all_recs[pol]]) - np.array(base)
        diff = diff[np.isfinite(diff)]
        if np.any(diff != 0):
            pv = float(wilcoxon(diff, alternative="less").pvalue)
            print(f"  {pol:16s} mean delta={diff.mean():+.3f}  (p_less={pv:.3f}, n={len(diff)})")
        else:
            print(f"  {pol:16s} no difference from {baseline}")

    # Personalization diagnostic (Fig-7 analog), averaged over control draws.
    keys = set().union(*[set(d) for d in diag_rows])
    adiag = {
        key: tuple(float(np.mean([d[key][t] for d in diag_rows if key in d])) for t in range(3))
        for key in keys
    }
    print("\n=== Personalization diagnostic (does the recommendation depend on the patient?) ===")
    print("  cross_pt_rho: mean pairwise Spearman of predicted drug-score vectors across patients")
    print(
        "  (1.00 = same ranking for every patient = collapsed to drug-mean; observed = real level)"
    )
    print(f"{'policy':18s}{'cross_pt_rho':>13}{'distinct_top1':>14}{'modal_share':>12}")
    ro, do, mo = adiag["observed"]
    print(f"{'observed/oracle':18s}{ro:>13.2f}{do:>14.1f}{mo:>12.2f}")
    for pol in order:
        if pol in adiag:
            rr, dd, mm = adiag[pol]
            print(f"{pol:18s}{rr:>13.2f}{dd:>14.1f}{mm:>12.2f}")


if __name__ == "__main__":
    main()
