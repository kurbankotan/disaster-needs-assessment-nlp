# -*- coding: utf-8 -*-
"""Duyarlılık ve ek analizler (API ya da model indirmesi gerekmez; kayıtlı tahminleri okur).
1. Kendi ailesini kayırma: üç üreticinin metni de çapraz denetimden geçen planlarda (455) LPM, logit ve aile fark farkı
2. Denetim temelli duyarlılık: ana yöntemlerin anlatım biçimine göre hit@1'i (a) Sonnet denetiminden geçen 1.878 paylaşımda,
   (b) GPT-5 mini çapraz denetiminden geçen/geçemeyen Claude metinlerinde (617 ortak plan)
3. Esnek puan: denetçinin metinde bulduğu fazladan liste ihtiyaçları da doğru sayılırsa hit@1
4. Çok etiket için uygulanabilir karar kuralı: göreli puan eşiği (s_j >= tau * s_1), tau beş katlı çapraz doğrulamayla
5. Uçtan uca ilçe doğruluğu: ilçe metinden çıkarıldığında (üç yöntem x beş yazım koşulu) ilçe düzeyi ihtiyaç doğruluğu
6. Persona başına paylaşım sayıları (Ek tablo için)"""
from revizyon_ortak import *
import pickle, unicodedata
import statsmodels.formula.api as smf

H = pickle.load(open(os.path.join(OUT, "sentetik_hits.pkl"), "rb"))   # revizyon_sentetik.py üretir
RES = {}

# =====================================================================================================
# 1) Kendi ailesini kayırma: temiz planlar
GENS = ["claude", "gemini", "gpt"]
DAT = {g: pd.read_csv(os.path.join(RR, f"alt_{g}_veri.csv"), encoding="utf-8-sig") for g in GENS}
gids = DAT["claude"]["id"].values
for g in GENS:
    DAT[g] = DAT[g].set_index("id").reindex(gids).reset_index()
GLABS = [[TR2EN[x] for x in r if isinstance(x, str) and x] for r in DAT["claude"][["label1", "label2", "label3"]].values]
GEXP = DAT["claude"]["explicitness"].values
CLEAN = (DAT["claude"].qc_ok.astype(bool) & DAT["gemini"].qc_ok.astype(bool) & DAT["gpt"].qc_ok.astype(bool)).values
assert CLEAN.sum() == 455

def skor_alt(g, tag):
    for f in os.listdir(RT):
        if unicodedata.normalize("NFC", f) == unicodedata.normalize("NFC", f"skor_alt_{g}_{tag}.csv"):
            return load_skor(f, gids)
    raise FileNotFoundError(tag)

GHIT = {}
for m in LLM_IDS:
    for g in GENS:
        tag = f"BDM_{m}_tek_en" + ("_makale_" if g == "claude" else "")
        GHIT[(m, g)] = hier_hits(skor_alt(g, tag).top1.tolist(), GLABS)[0]
EMB = {"TF-IDF (gövde)": "Temel_TF-IDF_en_stem_ad", "USE": "USE_en_plain_ad", "Qwen3-Embedding-4B": "Qwen3-Emb-4B_en_plain_ad"}
for k, tag in EMB.items():
    for g in GENS:
        GHIT[(k, g)] = hier_hits(skor_alt(g, tag).top1.tolist(), GLABS)[0]

def self_pref(mask):
    idx = np.where(mask)[0]
    long = [{"plan": int(gids[i]), "degerlendirici": m, "uretici": g, "dogru": GHIT[(m, g)][i], "ayni_aile": int(FAMILY[m] == g)}
            for m in LLM_IDS for g in GENS for i in idx]
    L = pd.DataFrame(long)
    ols = smf.ols("dogru ~ C(degerlendirici) + C(uretici) + ayni_aile", data=L).fit(cov_type="cluster", cov_kwds={"groups": L["plan"]})
    lg = smf.logit("dogru ~ C(degerlendirici) + C(uretici) + ayni_aile", data=L).fit(disp=0, cov_type="cluster", cov_kwds={"groups": L["plan"]})
    rng = np.random.default_rng(42)
    def did(fam, ii):
        own = [m for m in LLM_IDS if FAMILY[m] == fam]; oth = [m for m in LLM_IDS if FAMILY[m] != fam]
        og = [g for g in GENS if g != fam]
        f = lambda ms: np.mean([GHIT[(m, fam)][ii].mean() - np.mean([GHIT[(m, g)][ii].mean() for g in og]) for m in ms])
        return f(own) - f(oth)
    D_ = {}
    for fam in GENS:
        est = did(fam, idx)
        bs = [did(fam, rng.choice(idx, len(idx))) for _ in range(2000)]
        D_[fam] = {"fark_farki": float(est), "ga": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    return {"n_plan": int(len(idx)), "n_gozlem": len(L), "lpm_ayni_aile": float(ols.params["ayni_aile"]),
            "lpm_ga": [float(x) for x in ols.conf_int().loc["ayni_aile"]], "lpm_p": float(ols.pvalues["ayni_aile"]),
            "logit_or": float(np.exp(lg.params["ayni_aile"])), "logit_ga": [float(np.exp(x)) for x in lg.conf_int().loc["ayni_aile"]],
            "logit_p": float(lg.pvalues["ayni_aile"]), "aile_fark_farki": D_,
            "anlatim": {e: int(((GEXP == e) & mask).sum()) for e in ("explicit", "mixed", "implicit")}}

SP = {"tum_617": self_pref(np.ones(len(gids), bool)), "temiz_455": self_pref(CLEAN),
      "ortuk_617": self_pref(GEXP == "implicit"), "ortuk_temiz_455": self_pref(CLEAN & (GEXP == "implicit"))}   # örtük paylaşımlarda kesinlik
# kayıtlı ana analizle tutarlılık
old = json.load(open(os.path.join(OUT, "kendi_ailesini_kayirma.json"), encoding="utf-8"))   # revizyon_uretici.py üretir
assert abs(SP["tum_617"]["lpm_ayni_aile"] - old["lpm_ayni_aile"]) < 1e-9
RES["kendi_ailesini_kayirma"] = SP
rows = []
for k in [*LLM_IDS, *EMB]:
    for g in GENS:
        h = GHIT[(k, g)]
        for nm, msk in [("tum_617", np.ones(len(gids), bool)), ("temiz_455", CLEAN)]:
            rows.append({"yontem": LLM_NAME.get(k, k), "uretici": g, "kume": nm, "n": int(msk.sum()), "hit1": h[msk].mean(),
                         "hit1_acik": h[msk & (GEXP == "explicit")].mean(), "hit1_ortuk": h[msk & (GEXP == "implicit")].mean(),
                         "n_ortuk": int((msk & (GEXP == "implicit")).sum())})
GT = pd.DataFrame(rows); GT.to_csv(os.path.join(OUT, "duyarlilik_uretici_455.csv"), index=False, encoding="utf-8-sig")

# =====================================================================================================
# 2) Denetim temelli duyarlılık (ana çalışmanın 2.000 paylaşımı ve tahminleri)
M6 = [("Temel: TF-IDF", "en/stem/ad", "TF-IDF (gövde)"), ("USE", "en/plain/ad", "USE"), ("SBERT-mpnet", "en/plain/ad", "SBERT-mpnet"),
      ("Qwen3-Emb-4B", "en/plain/ad+tanim", "Qwen3-Embedding-4B + tanım")] + [(f"BDM {m}", "sifir_atis/en", LLM_NAME[m]) for m in LLM_IDS]
SON = D["qc_ok"].astype(bool).values                          # Claude Sonnet 5.5 denetimi (makaledeki)
pos = {i: k for k, i in enumerate(IDS)}
G617 = np.array([pos[i] for i in gids])                      # 617 ortak planın ana veri içindeki sıraları
GPTA = DAT["claude"].qc_ok.astype(bool).values                # GPT-5 mini çapraz denetimi (Claude metinleri)
rows = []
for name, cond, lab in M6:
    h = H[(name, cond)]["item"]
    r = {"yontem": lab}
    for e in ("explicit", "implicit"):
        m = EXP == e
        r[f"tum_{e}"] = h[m].mean(); r[f"sonnet_gecti_{e}"] = h[m & SON].mean(); r[f"sonnet_kaldi_{e}"] = h[m & ~SON].mean()
        hg = h[G617]; mg = GEXP == e
        r[f"g617_{e}"] = hg[mg].mean(); r[f"gpt_gecti_{e}"] = hg[mg & GPTA].mean(); r[f"gpt_kaldi_{e}"] = hg[mg & ~GPTA].mean()
    r["tum"] = h.mean(); r["sonnet_gecti"] = h[SON].mean()
    rows.append(r)
AUD = pd.DataFrame(rows); AUD.to_csv(os.path.join(OUT, "duyarlilik_denetim.csv"), index=False, encoding="utf-8-sig")
RES["denetim_n"] = {"sonnet_gecti": int(SON.sum()), "sonnet_gecti_ortuk": int((SON & (EXP == "implicit")).sum()),
                    "sonnet_gecti_acik": int((SON & (EXP == "explicit")).sum()),
                    "gpt_gecti": int(GPTA.sum()), "gpt_gecti_ortuk": int((GPTA & (GEXP == "implicit")).sum()),
                    "gpt_kaldi_ortuk": int((~GPTA & (GEXP == "implicit")).sum()),
                    "gpt_gecti_acik": int((GPTA & (GEXP == "explicit")).sum()), "gpt_kaldi_acik": int((~GPTA & (GEXP == "explicit")).sum())}
# BDM ortalaması ile en iyi yerel yöntem arasındaki örtük fark, alt kümelere göre
loc = [x[2] for x in M6[:4]]; llm = [LLM_NAME[m] for m in LLM_IDS]
gap = {}
for col in ["tum_implicit", "sonnet_gecti_implicit", "g617_implicit", "gpt_gecti_implicit", "gpt_kaldi_implicit"]:
    a = AUD.set_index("yontem")[col]
    gap[col] = {"bdm_ort": float(a[llm].mean()), "bdm_min": float(a[llm].min()), "en_iyi_yerel": float(a[loc].max()),
                "en_iyi_yerel_ad": a[loc].idxmax(), "fark": float(a[llm].mean() - a[loc].max())}
RES["denetim_ortuk_fark"] = gap
# sıralama korunuyor mu (Spearman, örtük)
from scipy.stats import spearmanr
A_ = AUD.set_index("yontem")
RES["denetim_siralama_spearman_ortuk"] = {
    "sonnet_gecti_vs_tum": float(spearmanr(A_["tum_implicit"], A_["sonnet_gecti_implicit"]).correlation),
    "gpt_gecti_vs_g617": float(spearmanr(A_["g617_implicit"], A_["gpt_gecti_implicit"]).correlation)}

# =====================================================================================================
# 3) Esnek puan: denetçinin bulduğu fazladan liste ihtiyaçları da doğru sayılır
def parse_extra(v):
    if not isinstance(v, str) or not v.strip():
        return []
    return [TR2EN[x.strip()] for x in re.split(r"[;,|]", v) if x.strip() in TR2EN]
EXTRA = [parse_extra(v) for v in D["qc_extra"]]
n_extra = sum(1 for e in EXTRA if e)
LEN = [list(dict.fromkeys(L + e)) for L, e in zip(LAB, EXTRA)]
rows = []
for name, cond, lab in M6:
    t1 = H[(name, cond)]["top1"]
    strict = np.array([isinstance(t, str) and t in L for t, L in zip(t1, LAB)], float)
    len_ = np.array([isinstance(t, str) and t in L for t, L in zip(t1, LEN)], float)
    rows.append({"yontem": lab, "hit1": strict.mean(), "esnek_hit1": len_.mean(), "fark": len_.mean() - strict.mean(),
                 "ortuk_hit1": strict[EXP == "implicit"].mean(), "ortuk_esnek_hit1": len_[EXP == "implicit"].mean()})
LEN_T = pd.DataFrame(rows); LEN_T.to_csv(os.path.join(OUT, "duyarlilik_esnek_puan.csv"), index=False, encoding="utf-8-sig")
RES["esnek_puan_n_fazladan"] = n_extra

# =====================================================================================================
# 4) Çok etiket için uygulanabilir karar kuralı
def multilabel_scores(gold_lists, pred_lists):
    """Defterdeki (NLP_Afet_Revizyon) tanımın aynısı: BDM satırlarıyla karşılaştırılabilir."""
    tp, fp, fn = Counter(), Counter(), Counter(); allf, exact = [], []
    for G_, P_ in zip(gold_lists, pred_lists):
        G_, P_ = set(G_), set(P_ or [])
        for x in P_ & G_: tp[x] += 1
        for x in P_ - G_: fp[x] += 1
        for x in G_ - P_: fn[x] += 1
        allf.append(G_ <= P_); exact.append(G_ == P_)
    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    p = TP / (TP + FP) if TP + FP else 0.0; r = TP / (TP + FN) if TP + FN else 0.0
    f1s = []
    for l in set(tp) | set(fn):
        if tp[l] + fn[l] == 0:
            continue
        pl = tp[l] / (tp[l] + fp[l]) if tp[l] + fp[l] else 0.0; rl = tp[l] / (tp[l] + fn[l])
        f1s.append(2 * pl * rl / (pl + rl) if pl + rl else 0.0)
    return {"mikro_kesinlik": p, "mikro_duyarlilik": r, "mikro_f1": (2 * p * r / (p + r) if p + r else 0.0),
            "makro_f1": float(np.mean(f1s)), "tum_ihtiyaclar_bulundu": float(np.mean(allf)), "tam_kume": float(np.mean(exact)),
            "ort_tahmin_sayisi": float(np.mean([len(P_ or []) for P_ in pred_lists]))}

def ranked_from_skor(d):
    tops = d[["top1", "top2", "top3"]].values.tolist(); sc = d[["s1", "s2", "s3"]].values.astype(float)
    return tops, sc

Ssim = tfidf_stem_sim()                                       # makaledeki TF-IDF (gövde, yalnız ad) benzerlikleri
o = np.argsort(-Ssim, axis=1)[:, :3]
TF_TOPS = [[EN_LABELS[j] for j in row] for row in o]; TF_SC = np.take_along_axis(Ssim, o, axis=1)
assert np.mean([a[0] == b for a, b in zip(TF_TOPS, H[("Temel: TF-IDF", "en/stem/ad")]["top1"])]) > 0.995

def rule(tops, sc, tau):
    out = []
    for T_, s in zip(tops, sc):
        keep = [T_[0]] + [t for t, v in zip(T_[1:], s[1:]) if isinstance(t, str) and s[0] > 0 and v >= tau * s[0]]
        out.append(keep)
    return out

TAUS = np.round(np.arange(0.50, 1.0001, 0.01), 2)
def cv_rule(tops, sc, gold, k=5, seed=42):
    n = len(tops); rng = np.random.default_rng(seed); folds = rng.permutation(n) % k
    pred = [None] * n; chosen = []
    for f in range(k):
        tr = np.where(folds != f)[0]; te = np.where(folds == f)[0]
        best = max(TAUS, key=lambda t: (multilabel_scores([gold[i] for i in tr], rule([tops[i] for i in tr], sc[tr], t))["mikro_f1"], -t))
        chosen.append(float(best))
        for i, p in zip(te, rule([tops[i] for i in te], sc[te], best)):
            pred[i] = p
    return pred, chosen

ML = []
def add_ml(lab, lang, tops, sc):
    pred, ch = cv_rule(tops, sc, LAB)
    r = multilabel_scores(LAB, pred); r.update({"yontem": lab, "dil": lang, "tau_katlar": ch, "tau_ortanca": float(np.median(ch))})
    k1 = multilabel_scores(LAB, [[t[0]] for t in tops]); r["mikro_f1_ilk1"] = k1["mikro_f1"]
    ML.append(r)
add_ml("TF-IDF (gövde)", "en", TF_TOPS, TF_SC)
for lab, tag, lang in [("USE + tanım", "USE_en_plain_ad+tanim", "en"), ("Qwen3-Embedding-4B + tanım", "Qwen3-Emb-4B_en_plain_ad+tanim", "en"),
                       ("Qwen3-Embedding-4B + tanım", "Qwen3-Emb-4B_tr_raw_ad+tanim", "tr"), ("mE5-large-instruct + tanım", "mE5-large-instruct_en_plain_ad+tanim", "en")]:
    d = load_skor(f"skor_sentetik_{tag}", IDS)
    tops, sc = ranked_from_skor(d)
    add_ml(lab, lang, tops, sc)
MLT = pd.DataFrame(ML); MLT.to_csv(os.path.join(OUT, "coklu_etiket_esik_kurali.csv"), index=False, encoding="utf-8-sig")

# =====================================================================================================
# 5) Uçtan uca ilçe doğruluğu
EXT = pd.read_csv(os.path.join(RT, "ilce_cikarimi_sentetik.csv"), encoding="utf-8-sig")
def e2e(extracted, top1):
    true_groups = pd.Series(range(len(IDS))).groupby(DIST).apply(list).to_dict()
    acts = {g: top_modes(Counter(x for i in idx for x in LAB[i])) for g, idx in true_groups.items()}
    ex = pd.Series(list(extracted))
    pg = ex[ex.notna()].index.to_series().groupby(ex[ex.notna()]).apply(list).to_dict()
    ok = []
    for g in acts:
        idx = pg.get(g, [])
        c = Counter(x for i in idx for x in LAB[i]) if top1 is None else Counter(top1[i] for i in idx if isinstance(top1[i], str))
        ok.append(1 if (top_modes(c) & acts[g]) else 0)
    return float(np.mean(ok))
SRC = {"Gerçek etiketler (kehanet)": None, "Claude Opus 5.5": H[("BDM claude-opus-5-5", "sifir_atis/en")]["top1"],
       "TF-IDF (gövde)": H[("Temel: TF-IDF", "en/stem/ad")]["top1"], "USE": H[("USE", "en/plain/ad")]["top1"]}
# bilinen ilçeyle makaledeki değerin yeniden üretimi
for k, t in SRC.items():
    if t is not None:
        assert abs(e2e(DIST, t) - district_accuracy(t)[0]) < 1e-12, k
rows = []
for (boz, yon), g in EXT.groupby(["bozma", "yontem"]):
    g = g.set_index("id").reindex(IDS)
    assert (g.gercek.values == DIST).all()
    ext = g.tahmin.where(g.tahmin.notna(), None).values
    r = {"bozma": boz, "yontem": yon, "cikarim_dogrulugu": float((g.tahmin.values == DIST).mean()),
         "cikarim_yok": float(g.tahmin.isna().mean())}
    for k, t in SRC.items():
        r[k] = e2e(ext, t)
    rows.append(r)
for k, t in SRC.items():
    rows.append({"bozma": "bilinen_ilce", "yontem": "–", "cikarim_dogrulugu": 1.0, "cikarim_yok": 0.0, **{kk: e2e(DIST, tt) for kk, tt in SRC.items()}})
    break
E2E = pd.DataFrame(rows); E2E.to_csv(os.path.join(OUT, "ilce_uctan_uca.csv"), index=False, encoding="utf-8-sig")

# =====================================================================================================
# 6) Persona sayıları
PER = pd.crosstab(D["persona"], D["explicitness"]).reindex(columns=["explicit", "mixed", "implicit"]).fillna(0).astype(int)
PER["toplam"] = PER.sum(axis=1); PER = PER.sort_values(["toplam", "implicit"], ascending=False)
PER.to_csv(os.path.join(OUT, "persona_sayilari.csv"), encoding="utf-8-sig")
RES["persona"] = {"n_persona": int(len(PER)), "min": int(PER.toplam.min()), "max": int(PER.toplam.max()),
                  "ortanca": float(PER.toplam.median()), "ortuk_min": int(PER.implicit.min()), "ortuk_max": int(PER.implicit.max())}

jdump(RES, "duyarlilik_ozet.json")
pd.set_option("display.width", 250)
print(json.dumps(RES, indent=1, ensure_ascii=False, default=float))
print(GT[GT.kume == "temiz_455"].pivot_table(index="yontem", columns="uretici", values=["hit1", "hit1_ortuk"]).round(3))
print(AUD.round(3).to_string(index=False))
print(LEN_T.round(3).to_string(index=False))
print(MLT.drop(columns=["tau_katlar"]).round(3).to_string(index=False))
print(E2E.round(3).to_string(index=False))
