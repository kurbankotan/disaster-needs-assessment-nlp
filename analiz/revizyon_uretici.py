# -*- coding: utf-8 -*-
"""D) Üretici x değerlendirici: aynı 617 plandan Claude Opus 5.5, Gemini 3.8 Flash ve GPT-5 mini ile yazılmış metinler.
- BDM ve gömme isabetleri, anlatım biçimine göre kırılım
- Kendi ailesini kayırma (self-preference): değerlendirici ve üretici sabit etkili, plan kümelenmiş doğrusal olasılık modeli
  ve aile bazında fark farkı (paired bootstrap)
- Sözcüksel sızıntı: ihtiyacın İngilizce adının (gövdelenmiş tüm içerik sözcükleri) metinde geçme oranı"""
from revizyon_ortak import *
import unicodedata
import statsmodels.formula.api as smf
from nltk.stem import PorterStemmer

GENS = ["claude", "gemini", "gpt"]
DAT = {g: pd.read_csv(os.path.join(RR, f"alt_{g}_veri.csv"), encoding="utf-8-sig") for g in GENS}
ids = DAT["claude"]["id"].values
for g in GENS:
    DAT[g] = DAT[g].set_index("id").reindex(ids).reset_index()
    assert (DAT[g]["explicitness"].values == DAT["claude"]["explicitness"].values).all()
LABS = [[TR2EN[x] for x in r if isinstance(x, str) and x] for r in DAT["claude"][["label1", "label2", "label3"]].values]
EXPA = DAT["claude"]["explicitness"].values

def skor_alt(g, tag):
    for f in os.listdir(RT):
        if unicodedata.normalize("NFC", f) == unicodedata.normalize("NFC", f"skor_alt_{g}_{tag}.csv"):
            return load_skor(f, ids)
    raise FileNotFoundError(tag)

METHODS = {f"BDM {m}": (lambda g, m=m: f"BDM_{m}_tek_en" + ("_makale_" if g == "claude" else "")) for m in LLM_IDS}
METHODS.update({"TF-IDF (gövde)": lambda g: "Temel_TF-IDF_en_stem_ad", "USE (düz)": lambda g: "USE_en_plain_ad",
                "SBERT-mpnet (düz)": lambda g: "SBERT-mpnet_en_plain_ad", "Qwen3-Emb-4B (ham)": lambda g: "Qwen3-Emb-4B_en_raw_ad",
                "Qwen3-Emb-4B (düz)": lambda g: "Qwen3-Emb-4B_en_plain_ad", "mE5 (düz)": lambda g: "mE5-large-instruct_en_plain_ad",
                "BGE-M3 (düz)": lambda g: "BGE-M3_en_plain_ad", "Word2Vec (lemma)": lambda g: "Word2Vec_en_lemma_ad"})
HIT = {}
for mth, fn in METHODS.items():
    for g in GENS:
        d = skor_alt(g, fn(g))
        HIT[(mth, g)] = hier_hits(d.top1.tolist(), LABS)[0]

# --- tablo: yöntem x üretici x anlatım
rows = []
for mth in METHODS:
    for g in GENS:
        h = HIT[(mth, g)]; lo, hi = boot_ci(h)
        r = {"yontem": mth, "uretici": g, "n": len(h), "hit1": h.mean(), "ga_alt": lo, "ga_ust": hi}
        for e in ("explicit", "mixed", "implicit"):
            r[f"hit1_{e}"] = h[EXPA == e].mean()
        rows.append(r)
TAB = pd.DataFrame(rows); TAB.to_csv(f"{OUT}/uretici_yontem_anlatim.csv", index=False, encoding="utf-8-sig")

# --- BDM ortalaması ile en iyi gömme arasındaki fark (üretici x anlatım)
gap = []
emb = [m for m in METHODS if not m.startswith("BDM")]
for g in GENS:
    for e in ("tümü", "explicit", "mixed", "implicit"):
        m = np.ones(len(ids), bool) if e == "tümü" else EXPA == e
        llm = np.mean([HIT[(f"BDM {x}", g)][m].mean() for x in LLM_IDS])
        best = max(emb, key=lambda k: HIT[(k, g)][m].mean())
        gap.append({"uretici": g, "anlatim": e, "n": int(m.sum()), "bdm_ort": llm, "en_iyi_gomme": best,
                    "en_iyi_gomme_hit1": HIT[(best, g)][m].mean(), "fark": llm - HIT[(best, g)][m].mean(),
                    "tfidf_hit1": HIT[("TF-IDF (gövde)", g)][m].mean()})
GAP = pd.DataFrame(gap); GAP.to_csv(f"{OUT}/uretici_bdm_gomme_farki.csv", index=False, encoding="utf-8-sig")

# --- sözcüksel sızıntı
ps = PorterStemmer(); STOP = {"and", "for", "s", "of", "the", "a", "to", "with", "no", "on", "in", "free"}
LET = re.compile(r"[^a-zA-ZğĞçÇşŞüÜöÖıİ]")
def toks(t):
    return [ps.stem(w) for w in LET.sub(" ", str(t)).lower().split() if w not in STOP]
NEED_STEMS = {n: set(toks(n)) for n in EN_LABELS}
def leak(text, labels):
    tt = set(toks(text))
    return float(np.mean([NEED_STEMS[n] <= tt for n in labels])) if labels else np.nan
lk = []
for g in GENS:
    L = np.array([leak(t, l) for t, l in zip(DAT[g]["English"], LABS)])
    for e in ("tümü", "explicit", "mixed", "implicit"):
        m = np.ones(len(ids), bool) if e == "tümü" else EXPA == e
        lk.append({"uretici": g, "anlatim": e, "n": int(m.sum()), "ad_gecme_orani": float(np.nanmean(L[m])),
                   "ort_kelime": float(DAT[g]["English"][m].str.split().str.len().mean()),
                   "capraz_denetim_gecti": float(DAT[g]["qc_ok"][m].astype(bool).mean())})
# makaledeki 2.000 paylaşımda da (aynı ölçüt)
Lp = np.array([leak(t, l) for t, l in zip(D["English"], LAB)])
for e in ("tümü", "explicit", "mixed", "implicit"):
    m = np.ones(len(D), bool) if e == "tümü" else EXP == e
    lk.append({"uretici": "claude (2.000 paylaşım)", "anlatim": e, "n": int(m.sum()), "ad_gecme_orani": float(np.nanmean(Lp[m])),
               "ort_kelime": float(D["English"][m].str.split().str.len().mean()), "capraz_denetim_gecti": np.nan})
LK = pd.DataFrame(lk); LK.to_csv(f"{OUT}/uretici_sozcuksel_sizinti.csv", index=False, encoding="utf-8-sig")

# --- kendi ailesini kayırma: doğrusal olasılık modeli, plan kümelenmiş SH
long = []
for m in LLM_IDS:
    for g in GENS:
        h = HIT[(f"BDM {m}", g)]
        for i, v in enumerate(h):
            long.append({"plan": int(ids[i]), "degerlendirici": m, "uretici": g, "dogru": v,
                         "ayni_aile": int(FAMILY[m] == g), "anlatim": EXPA[i]})
LONG = pd.DataFrame(long)
ols = smf.ols("dogru ~ C(degerlendirici) + C(uretici) + ayni_aile", data=LONG).fit(cov_type="cluster", cov_kwds={"groups": LONG["plan"]})
ols2 = smf.ols("dogru ~ C(degerlendirici) * C(anlatim) + C(uretici) * C(anlatim) + ayni_aile", data=LONG).fit(
    cov_type="cluster", cov_kwds={"groups": LONG["plan"]})
logit = smf.logit("dogru ~ C(degerlendirici) + C(uretici) + ayni_aile", data=LONG).fit(disp=0, cov_type="cluster", cov_kwds={"groups": LONG["plan"]})
SP = {"lpm_ayni_aile": float(ols.params["ayni_aile"]), "lpm_ga": [float(x) for x in ols.conf_int().loc["ayni_aile"]], "lpm_p": float(ols.pvalues["ayni_aile"]),
      "lpm_anlatim_kontrollu": float(ols2.params["ayni_aile"]), "lpm_anlatim_kontrollu_ga": [float(x) for x in ols2.conf_int().loc["ayni_aile"]],
      "lpm_anlatim_kontrollu_p": float(ols2.pvalues["ayni_aile"]),
      "logit_or": float(np.exp(logit.params["ayni_aile"])), "logit_ga": [float(np.exp(x)) for x in logit.conf_int().loc["ayni_aile"]],
      "logit_p": float(logit.pvalues["ayni_aile"]), "n_gozlem": len(LONG), "n_plan": len(ids)}

# aile bazında fark farkı: (değerlendiricinin kendi ailesinin metninde - diğer metinlerde) - (diğer değerlendiricilerin aynı farkı)
rng = np.random.default_rng(42)
def did(fam, idx):
    own = [m for m in LLM_IDS if FAMILY[m] == fam]; oth = [m for m in LLM_IDS if FAMILY[m] != fam]
    others_g = [g for g in GENS if g != fam]
    a = np.mean([HIT[(f"BDM {m}", fam)][idx].mean() - np.mean([HIT[(f"BDM {m}", g)][idx].mean() for g in others_g]) for m in own])
    b = np.mean([HIT[(f"BDM {m}", fam)][idx].mean() - np.mean([HIT[(f"BDM {m}", g)][idx].mean() for g in others_g]) for m in oth])
    return a - b
DID = {}
for fam in GENS:
    est = did(fam, np.arange(len(ids)))
    bs = [did(fam, rng.integers(0, len(ids), len(ids))) for _ in range(2000)]
    DID[fam] = {"fark_farki": float(est), "ga": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
SP["aile_fark_farki"] = DID

# Claude metinleri: Opus kendi metnini GPT/Gemini metinlerine göre nasıl değerlendiriyor (McNemar)
mc = []
for m in LLM_IDS:
    for g1, g2 in [("claude", "gemini"), ("claude", "gpt"), ("gemini", "gpt")]:
        p, a, b = mcnemar_exact(HIT[(f"BDM {m}", g1)], HIT[(f"BDM {m}", g2)])
        mc.append({"degerlendirici": m, "a": g1, "b": g2, "a_hit1": HIT[(f"BDM {m}", g1)].mean(), "b_hit1": HIT[(f"BDM {m}", g2)].mean(), "p": p})
MC = pd.DataFrame(mc); MC["p_holm"] = holm(MC.p); MC.to_csv(f"{OUT}/uretici_mcnemar.csv", index=False, encoding="utf-8-sig")
jdump(SP, "kendi_ailesini_kayirma.json")

pd.set_option("display.width", 250)
piv = TAB.pivot_table(index="yontem", columns="uretici", values="hit1").round(3); print(piv)
print(TAB.pivot_table(index="yontem", columns="uretici", values="hit1_implicit").round(3))
print(GAP.round(3).to_string(index=False)); print(LK.round(3).to_string(index=False))
print(json.dumps(SP, indent=1, ensure_ascii=False))
print(MC.round(4).to_string(index=False))
