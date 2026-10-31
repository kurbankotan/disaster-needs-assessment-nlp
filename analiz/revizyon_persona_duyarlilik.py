# -*- coding: utf-8 -*-
"""Ana çalıştırmanın (runs/20261016_1116) kayıtlı tahminlerinden yapılan ek analizler; API ve model gerektirmez.

persona      : persona dağılımı, üslup çeşitliliği, persona ve kayıt dilinin metinden öngörülebilirliği,
               örtük anlatım başarımının personaya göre değişimi
coklu ihtiyaç: birden fazla ihtiyaç içeren paylaşımlarda gömme modellerinin tüm ihtiyaçları bulma oranı ve F1 (ilk k tahmin)
ilçe duyarlılığı: ilçe başına paylaşım sayısına göre ilçe doğruluğu (500 tekrarlı örnekleme)
Çıktılar runs/revizyon/results/analiz/ klasörüne yazılır.
"""
import os, sys, json, math, re
import numpy as np, pandas as pd
from collections import Counter
from scipy.stats import chisquare, chi2_contingency
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revizyon_ortak import *

RNG = np.random.default_rng(42)
MAIN = {"TF-IDF (gövde)": "Temel: TF-IDF_stem", "USE (düz)": "USE_plain", "SBERT-mpnet (düz)": "SBERT-mpnet_plain",
        "Word2Vec (lemma)": "Word2Vec_lemma", "Claude Opus 5.5": "BDM_claude-opus-5-5", "Claude Sonnet 5.5": "BDM_claude-sonnet-5-5",
        "Claude Haiku 5.5": "BDM_claude-haiku-5-5", "GPT-5 mini": "BDM_gpt-5-mini", "Gemini 3.8 Flash": "BDM_gemini-3.8-flash"}
LLMS = ["Claude Opus 5.5", "Claude Sonnet 5.5", "Claude Haiku 5.5", "GPT-5 mini", "Gemini 3.8 Flash"]
SUMMARY = {}

d, nl = data()
LAB = d["labels_en"].tolist()
P = {k: preds(v) for k, v in MAIN.items()}
for k, p in P.items():
    assert (p["id"].values == d["id"].values).all(), k
    assert [set(x.split(";")) for x in p["gercek"]] == [set(x) for x in LAB], k

def hit1(top1):
    return np.array([isinstance(t, str) and t in L for t, L in zip(top1, LAB)], dtype=float)

# ---------------------------------------------------------------- persona
def persona():
    vc = d["persona"].value_counts()
    chi = chisquare(vc.values)
    tab = pd.crosstab(d["persona"], d["explicitness"]).reindex(columns=["explicit", "mixed", "implicit"]).fillna(0).astype(int)
    tab["toplam"] = tab.sum(axis=1)
    tab = tab.sort_values("toplam", ascending=False)
    tab.to_csv(os.path.join(OUT, "persona_dagilim.csv"), encoding="utf-8-sig")
    # çeşitlilik ölçütleri (Türkçe özgün metin)
    toks = [re.findall(r"\w+", str(m).lower()) for m in d["Message"]]
    allt = [t for T in toks for t in T]
    bi = [tuple(T[i:i + 2]) for T in toks for i in range(len(T) - 1)]
    openings = [" ".join(T[:2]) for T in toks]
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(sublinear_tf=True).fit_transform(d["Message"].astype(str))
    idx = RNG.choice(len(d), size=(4000, 2))
    idx = idx[idx[:, 0] != idx[:, 1]]
    pair_sim = np.asarray(X[idx[:, 0]].multiply(X[idx[:, 1]]).sum(axis=1)).ravel()
    div = {"distinct1": len(set(allt)) / len(allt), "distinct2": len(set(bi)) / len(bi),
           "benzersiz_acilis_orani": len(set(openings)) / len(openings),
           "en_sik_acilis": Counter(openings).most_common(5),
           "ikili_tfidf_benzerligi_ort": float(pair_sim.mean()), "ikili_tfidf_benzerligi_p95": float(np.percentile(pair_sim, 95))}
    # persona ve kayıt dili metinden öngörülebiliyor mu? (5 katlı çapraz doğrulama)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline, make_union
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    def cv_acc(y, shuffle=False):
        yy = np.array(y)
        if shuffle:
            yy = RNG.permutation(yy)
        pipe = make_pipeline(make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                                        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
                             LogisticRegression(max_iter=2000, C=5.0))
        sc = cross_val_score(pipe, d["Message"].astype(str), yy, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="accuracy")
        return float(sc.mean())
    pred = {}
    for col in ("persona", "register"):
        maj = d[col].value_counts(normalize=True).iloc[0]
        pred[col] = {"siniflar": int(d[col].nunique()), "cv_dogruluk": cv_acc(d[col]), "karisik_etiketle": cv_acc(d[col], True), "cogunluk": float(maj)}
    # örtük paylaşımlarda personaya göre isabet
    imp = (d["explicitness"] == "implicit").values
    rows, het = [], {}
    for name in ["TF-IDF (gövde)", "USE (düz)", "SBERT-mpnet (düz)", "Claude Opus 5.5", "GPT-5 mini", "Gemini 3.8 Flash"]:
        h = hit1(P[name]["top1"])
        g = pd.DataFrame({"persona": d["persona"].values[imp], "h": h[imp]}).groupby("persona")["h"]
        s = g.agg(["mean", "size"])
        for per, r in s.iterrows():
            rows.append({"model": name, "persona": per, "ortuk_n": int(r["size"]), "ortuk_hit1": float(r["mean"])})
        ct = pd.crosstab(d["persona"].values[imp], h[imp])
        c2 = chi2_contingency(ct.values)
        het[name] = {"ortuk_hit1_genel": float(h[imp].mean()), "persona_aralik": [float(s["mean"].min()), float(s["mean"].max())],
                     "ki_kare": float(c2[0]), "sd": int(c2[2]), "p": float(c2[1])}
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "persona_ortuk_isabet.csv"), index=False, encoding="utf-8-sig")
    SUMMARY["persona"] = {"persona_sayisi": int(vc.size), "min": int(vc.min()), "maks": int(vc.max()), "ort": float(vc.mean()),
                             "ss": float(vc.std()), "ki_kare_esit_dagilim": float(chi.statistic), "p": float(chi.pvalue),
                             "cesitlilik": div, "ongorulebilirlik": pred, "ortuk_persona_heterojenlik": het}

# ---------------------------------------------------------------- ilçe duyarlılığı
def district_sensitivity(reps=500, ks=(5, 10, 20, 30, 40)):
    dist = d["district"].values
    groups = pd.Series(range(len(d))).groupby(dist).apply(list).to_dict()
    ref = {g: top_modes(Counter(x for i in idx for x in LAB[i])) for g, idx in groups.items()}   # ilçenin tüm verideki baskın ihtiyacı
    rows = []
    sources = {"Kehanet (gerçek etiketler)": None, **{k: P[k]["top1"].tolist() for k in ["TF-IDF (gövde)", "USE (düz)", "SBERT-mpnet (düz)", "Word2Vec (lemma)", "Claude Opus 5.5", "GPT-5 mini", "Gemini 3.8 Flash"]}}
    for name, top1 in sources.items():
        for k in list(ks) + ["tümü"]:
            accs = []
            for r in range(reps if k != "tümü" else 1):
                ok = []
                for g, idx in groups.items():
                    sel = idx if k == "tümü" else list(RNG.choice(idx, size=min(k, len(idx)), replace=False))
                    if top1 is None:
                        c = Counter(x for i in sel for x in LAB[i])
                    else:
                        c = Counter(top1[i] for i in sel if isinstance(top1[i], str))
                    ok.append(1 if top_modes(c) & ref[g] else 0)
                accs.append(np.mean(ok))
            rows.append({"kaynak": name, "ilce_basina_paylasim": k, "ilce_dogrulugu_ort": float(np.mean(accs)),
                         "ga_alt": float(np.percentile(accs, 2.5)), "ga_ust": float(np.percentile(accs, 97.5))})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, "ilce_duyarlilik.csv"), index=False, encoding="utf-8-sig")
    SUMMARY["ilce_duyarlilik"] = {n: {str(r["ilce_basina_paylasim"]): [round(r["ilce_dogrulugu_ort"], 3), round(r["ga_alt"], 3), round(r["ga_ust"], 3)]
                                        for _, r in g.iterrows()} for n, g in tab.groupby("kaynak")}

# ---------------------------------------------------------------- çoklu ihtiyaç (gömme modelleri, ilk k)
def multi_need():
    n_g = np.array([len(L) for L in LAB])
    multi = n_g >= 2
    rows = []
    for name in ["TF-IDF (gövde)", "USE (düz)", "SBERT-mpnet (düz)", "Word2Vec (lemma)"]:
        p = P[name]
        tops = [[x for x in (a, b, c) if isinstance(x, str)] for a, b, c in zip(p.top1, p.top2, p.top3)]
        allk = np.array([set(L) <= set(T[:len(L)]) for L, T in zip(LAB, tops)], dtype=float)
        # mikro/makro F1: tahmin kümesi ilk |gerçek| tahmin (k bilindiği varsayımı) ve yalnız ilk tahmin
        res = {"model": name, "coklu_paylasim_orani": float(multi.mean()), "tum_ihtiyaclar_ilk_k_coklu": float(allk[multi].mean())}
        for mode in ("ilk1", "ilk_k"):
            tp = Counter(); fp = Counter(); fn = Counter()
            for L, T in zip(LAB, tops):
                pr = set(T[:1] if mode == "ilk1" else T[:len(L)]); gl = set(L)
                for x in pr & gl: tp[x] += 1
                for x in pr - gl: fp[x] += 1
                for x in gl - pr: fn[x] += 1
            TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
            mic = 2 * TP / (2 * TP + FP + FN)
            labs = set(tp) | set(fp) | set(fn)
            mac = np.mean([2 * tp[x] / (2 * tp[x] + fp[x] + fn[x]) if (2 * tp[x] + fp[x] + fn[x]) else 0 for x in labs])
            res[f"mikro_F1_{mode}"] = mic; res[f"makro_F1_{mode}"] = float(mac)
        rows.append(res)
    tab = pd.DataFrame(rows); tab.to_csv(os.path.join(OUT, "coklu_ihtiyac_gomme.csv"), index=False, encoding="utf-8-sig")
    SUMMARY["coklu_ihtiyac_gomme"] = tab.set_index("model").round(4).to_dict(orient="index")

if __name__ == "__main__":
    persona(); district_sensitivity(); multi_need()
    json.dump(SUMMARY, open(os.path.join(OUT, "persona_duyarlilik_ozet.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(SUMMARY, ensure_ascii=False, indent=1, default=float)[:12000])
