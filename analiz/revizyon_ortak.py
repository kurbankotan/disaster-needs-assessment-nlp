# -*- coding: utf-8 -*-
"""Revizyon analizleri için ortak tanımlar: depodaki çalıştırma klasörlerinin yolları, veri yükleyiciler,
üç düzeyli ihtiyaç hiyerarşisi (madde, yakın anlam grubu, sektör), ölçütler ve istatistiksel testler.

Tüm yollar depo köküne göredir; API anahtarı ya da model indirmesi gerekmez.
  runs/20261016_1116/            makaledeki ana çalıştırma
  runs/20261016_0743_claude_maliyet/  Claude modellerinin ölçülen maliyetleri
  runs/revizyon/                 revizyon deneylerinin çalıştırması (notebooks/NLP_Afet_Revizyon.ipynb)
  runs/revizyon/results/analiz/  bu betiklerin ürettiği türetilmiş tablolar
"""
import os, sys, json, math, re
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from scipy.stats import binomtest

ANALIZ = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ANALIZ)
sys.path.insert(0, ANALIZ)
RUN = os.path.join(ROOT, "runs", "20261016_1116")
REF = os.path.join(ROOT, "runs", "20261016_0743_claude_maliyet")
PRED = os.path.join(RUN, "results", "tahminler")
REV = os.path.join(ROOT, "runs", "revizyon")
REVR = os.path.join(REV, "results")
RT = os.path.join(REVR, "tahminler"); RR = os.path.join(REVR, "revizyon")
REV_DATA = os.path.join(REV, "data")
OUT = os.path.join(REVR, "analiz"); os.makedirs(OUT, exist_ok=True)

from sektor_semasi import ITEM_SECTOR, SECTOR_NAMES

def data():
    d = pd.read_excel(os.path.join(RUN, "results", "veri_seti_sentetik_islenmis.xlsx"))
    nl = pd.read_csv(os.path.join(RUN, "results", "ihtiyac_listesi.csv"))
    tr2en = dict(zip(nl.tr, nl.en))
    d["labels_tr"] = d[["label1", "label2", "label3"]].apply(lambda r: [x for x in r if isinstance(x, str) and x.strip()], axis=1)
    d["labels_en"] = d["labels_tr"].apply(lambda L: list(dict.fromkeys(tr2en[x] for x in L)))
    return d, nl

def preds(tag):
    p = os.path.join(PRED, f"mesaj_sentetik_{tag}.csv")
    if not os.path.exists(p):
        p = p.replace(": ", "_ ")
    return pd.read_csv(p).sort_values("id").reset_index(drop=True)

# Makaledeki ilçe ölçütü (defterdeki district_metric_paper ile aynı)
from collections import Counter
def top_modes(c):
    if not c:
        return set()
    m = max(c.values()); return {k for k, v in c.items() if v == m}

def district_acc(districts, labels, top1):
    acc = {}
    df = pd.DataFrame({"d": districts, "i": range(len(districts))})
    for dname, idx in df.groupby("d").indices.items():
        actual = Counter(x for i in idx for x in labels[i])
        predicted = Counter(top1[i] for i in idx if isinstance(top1[i], str))
        if not actual or not predicted:
            continue
        acc[dname] = 1 if (top_modes(actual) & top_modes(predicted)) else 0
    return acc

D, NL = data()
TR2EN = dict(zip(NL.tr, NL.en)); EN2TR = {v: k for k, v in TR2EN.items()}
EN_LABELS = list(dict.fromkeys(NL.en))
LAB = D["labels_en"].tolist()
EXP = D["explicitness"].values
IDS = D["id"].values
DIST = D["district"].values

# ---------------------------------------------------------------- 3 düzeyli hiyerarşi: madde -> yakın anlam grubu -> sektör
# Yakın anlam grupları sektörlerin içinde tanımlanmıştır (iç içe hiyerarşi); listelenmeyen maddeler tek maddelik gruptur.
GROUPS = {
    "İlaç": ["ilaç", "ağrı kesici", "antibiyotik", "soğuk algınlığı ilacı", "ateş düşürücü", "alerji ilaçları",
             "astım ilacı", "insülin", "kalp ilacı"],
    "İlk yardım malzemesi": ["yarabandı", "bandaj", "ilk yardım kiti"],
    "Kan": ["kan", "kan toplama merkezi"],
    "Genel gıda": ["gıda", "kuru gıda", "konserve", "sıcak yemek", "ekmek"],
    "Özel beslenme": ["glutesiz yiyecek", "tuzsuz-şekersiz yiyecek"],
    "Bebek beslenmesi": ["bebek maması", "biberon"],
    "Hijyen ve temizlik malzemesi": ["sabun", "ıslak mendil", "peçete", "diş fırçası", "diş macunu", "hijyenik ped",
                                     "bebek bezi", "temizlik ürünü"],
    "Barınak": ["çadır", "konteynır"],
    "Uyku ve örtü": ["battaniye", "uyku tulumu"],
    "Giysi": ["yedek giysi", "mont"],
    "Isınma": ["soba", "odun", "elektrikli ısıtıcı"],
    "Mutfak eşyası": ["kap kacak", "mutfak eşyası"],
    "Aydınlatma": ["fener", "el feneri", "aydınlatıcı", "pil"],
    "Güç kaynağı": ["jenaratör", "powerbank"],
}
GROUP_OF_TR = {x: g for g, v in GROUPS.items() for x in v}
for x in NL.tr:
    GROUP_OF_TR.setdefault(x, x)                                   # tek maddelik grup
assert set(GROUP_OF_TR) == set(NL.tr) and len(GROUP_OF_TR) == 66
for g, v in GROUPS.items():                                        # iç içe olma: grup tek bir sektörde
    assert len({ITEM_SECTOR[x] for x in v}) == 1, g
GROUP_EN = {TR2EN[x]: g for x, g in GROUP_OF_TR.items()}
SECTOR_EN = {TR2EN[x]: s for x, s in ITEM_SECTOR.items()}
N_GROUPS = len(set(GROUP_OF_TR.values()))

def hier_hits(top1, labels=None):
    labels = LAB if labels is None else labels
    item = np.array([isinstance(t, str) and t in L for t, L in zip(top1, labels)], float)
    grp = np.array([isinstance(t, str) and GROUP_EN.get(t) in {GROUP_EN[x] for x in L} for t, L in zip(top1, labels)], float)
    sec = np.array([isinstance(t, str) and SECTOR_EN.get(t) in {SECTOR_EN[x] for x in L} for t, L in zip(top1, labels)], float)
    return item, grp, sec

def partial_credit(item, grp, sec, w=(1.0, 0.5, 0.25)):
    return np.where(item == 1, w[0], np.where(grp == 1, w[1], np.where(sec == 1, w[2], 0.0)))

def boot_ci(x, reps=2000, seed=42):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    m = rng.integers(0, len(x), size=(reps, len(x)))
    v = x[m].mean(axis=1)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

def paired_boot_diff(a, b, reps=2000, seed=42):
    a = np.asarray(a, float); b = np.asarray(b, float); rng = np.random.default_rng(seed)
    m = rng.integers(0, len(a), size=(reps, len(a)))
    v = a[m].mean(axis=1) - b[m].mean(axis=1)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

def mcnemar_exact(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    n01 = int((a & ~b).sum()); n10 = int((~a & b).sum())
    if n01 + n10 == 0:
        return 1.0, n01, n10
    return float(binomtest(n01, n01 + n10, 0.5).pvalue), n01, n10

def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj

def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return float("nan"), float("nan")
    r = binomtest(int(k), int(n)).proportion_ci(confidence_level=1 - alpha, method="exact")
    return float(r.low), float(r.high)

def load_skor(name, ids=None):
    p = os.path.join(RT, name if name.endswith(".csv") else name + ".csv")
    d = pd.read_csv(p, encoding="utf-8-sig")
    if ids is not None:
        d = d.set_index("id").reindex(ids).reset_index()
    return d

def paper_top1(tag):
    return preds(tag)["top1"].tolist()

def district_accuracy(top1, labels=None, dist=None):
    labels = LAB if labels is None else labels; dist = DIST if dist is None else dist
    acc = district_acc(list(dist), labels, list(top1))
    return float(np.mean(list(acc.values()))), acc

# Makaledeki BDM maliyetleri (USD / paylaşım) ve gecikmeler
def llm_cost_latency():
    t5 = pd.read_csv(os.path.join(RUN, "results", "tablo5_bdm_sentetik_sayisal.csv")); t5r = pd.read_csv(os.path.join(REF, "results", "tablo5_bdm_sentetik_sayisal.csv"))
    rep = pd.read_csv(os.path.join(RR, "revizyon_tum_sonuclar.csv"))
    rep = rep[(rep.deney == "bdm_tekrar") & rep.kosul.str.startswith("tekrar")]
    out = {}
    for mid in ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "gpt-5-mini", "gemini-3.8-flash"]:
        src = t5r if mid.startswith("claude") else t5
        r = src[src.model == mid].iloc[0]
        out[mid] = {"usd_paylasim": float(r.maliyet_usd) / float(r.n_mesaj), "ms_makale": float(r.ms_mesaj),
                    "ms_tekrar": float(rep[rep.yontem == f"BDM {mid}"].ms_paylasim.mean())}
    return out

LLM_IDS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "gpt-5-mini", "gemini-3.8-flash"]
LLM_NAME = {"claude-opus-5-5": "Claude Opus 5.5", "claude-sonnet-5-5": "Claude Sonnet 5.5", "claude-haiku-5-5": "Claude Haiku 5.5",
            "gpt-5-mini": "GPT-5 mini", "gemini-3.8-flash": "Gemini 3.8 Flash"}
FAMILY = {"claude-opus-5-5": "claude", "claude-sonnet-5-5": "claude", "claude-haiku-5-5": "claude", "gpt-5-mini": "gpt",
          "gemini-3.8-flash": "gemini"}

def llm_top1_paper(mid):
    """Makaledeki sıfır atış tahminleri; GPT-5 mini için protokol tutarlı sürüm (210 paylaşım yeniden)."""
    if mid == "gpt-5-mini":
        d = load_skor("skor_sentetik_BDM_gpt-5-mini_protokol_tutarli", IDS)
        return d["top1"].tolist()
    return paper_top1(f"BDM_{mid}")

def jdump(obj, name):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=float)


# ---------------------------------------------------------------- makaledeki TF-IDF (gövdelenmiş metin, yalnız ihtiyaç adı) benzerlikleri
def tfidf_stem_sim():
    """Makaledeki TF-IDF temel çizgisinin benzerlik matrisini (2.000 paylaşım x 66 ihtiyaç) yeniden hesaplar.
    Defterdeki ön işlemeyle aynıdır; en yüksek puanlı ihtiyaç kayıtlı tahminlerle karşılaştırılarak doğrulanır."""
    from nltk.stem import PorterStemmer
    from sklearn.feature_extraction.text import TfidfVectorizer
    let = re.compile(r"[^a-zA-ZğĞçÇşŞüÜöÖıİ]")
    def plain(t): return " ".join(let.sub(" ", str(t)).replace("İ", "i").lower().split())
    stop_min = {"and", "for", "s", "of", "the", "a", "to", "with", "no", "on", "in"}
    ps = PorterStemmer()
    needs = [" ".join(ps.stem(w) for w in plain(n).split() if w not in stop_min) for n in EN_LABELS]
    msgs = D["text_stem"].fillna("").tolist()
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1).fit(msgs + needs)
    M = vec.transform(msgs).toarray().astype(np.float32); N = vec.transform(needs).toarray().astype(np.float32)
    mn = np.linalg.norm(M, axis=1, keepdims=True); nn = np.linalg.norm(N, axis=1, keepdims=True)
    return (M / (mn + 1e-9)) @ (N / (nn + 1e-9)).T
