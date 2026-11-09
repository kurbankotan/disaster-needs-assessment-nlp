# -*- coding: utf-8 -*-
"""Simüle paylaşımlarda tekrar eden açılış ifadelerinin ("lütfen yardım", "merhaba ben", "please help", "I'm" vb.)
sözcüksel yöntemin (TF-IDF) isabetini etkileyip etkilemediği.

Açılış: paylaşımın ilk iki sözcüğü (\\w+; Türkçede Türkçe, İngilizcede İngilizce küçük harf kuralıyla). İki koşul sınanır:
  tekrar_eden : açılışı en az 5 paylaşımda geçen paylaşımlarda bu iki sözcük metinden çıkarılır (kalıplaşmış açılışlar);
  tumu        : tüm paylaşımlarda ilk iki sözcük çıkarılır (üst sınır sınaması).
Açılış İngilizce TF-IDF için İngilizce metinde, Türkçe TF-IDF için Türkçe özgün metinde belirlenir. Ön işleme makaledekiyle
aynıdır (İngilizce: düz metin ve Porter gövdesi; Türkçe: Snowball Türkçe gövdesi). TF-IDF her koşulda değiştirilmiş paylaşımlar
ve ihtiyaç metinleri üzerinde yeniden kurulur; sıfır vektörlü paylaşım tahminsiz (yanlış) sayılır. Özgün koşulun tahminleri
makaledeki kayıtlı tahminlerle karşılaştırılarak doğrulanır.

Gereksinim: nltk stopwords verisi (nltk.download("stopwords")), inflect, snowballstemmer, scikit-learn.
Çıktı: runs/revizyon/results/analiz/acilis_ablasyonu.csv ve acilis_ablasyonu.json
"""
import os, sys, re
import numpy as np, pandas as pd
from collections import Counter
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revizyon_ortak import D, NL, LAB, EXP, EN_LABELS, TR2EN, RUN, RT, RR, OUT, jdump

import inflect, snowballstemmer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

ESIK = 5                     # "tekrar eden açılış": en az bu kadar paylaşımda geçen iki sözcüklü açılış
B = 2000                     # eşleştirilmiş önyükleme sayısı
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------- makaledeki ön işleme
INF = inflect.engine(); STOP = set(stopwords.words("english")); PS = PorterStemmer()
LET = re.compile(r"[^a-zA-ZğĞçÇşŞüÜöÖıİ]")
LET_TR = re.compile(r"[^a-zA-ZğĞçÇşŞüÜöÖıİâÂîÎûÛ]")
STEM_TR = snowballstemmer.stemmer("turkish")
def n2w(t): return re.sub(r"\d+", lambda m: " " + (INF.number_to_words(int(m.group())) if len(m.group()) < 12 else "") + " ", t)
def plain_en(t): return " ".join(LET.sub(" ", n2w(str(t))).replace("İ", "i").lower().split())
def stem_en(t): return " ".join(PS.stem(w) for w in plain_en(t).split() if w not in STOP)
def tr_lower(s): return str(s).replace("İ", "i").replace("I", "ı").lower()
def stem_tr(t): return " ".join(STEM_TR.stemWords(" ".join(tr_lower(LET_TR.sub(" ", str(t))).split()).split()))

assert (D["English"].map(stem_en) == D["text_stem"].fillna("")).all(), "İngilizce gövdeleme makaledekiyle aynı değil"

DESC = pd.read_csv(os.path.join(RR, "ihtiyac_tanimlari.csv"), encoding="utf-8-sig")
DESC_EN = dict(zip(DESC.ihtiyac_en, DESC.tanim_en))
NEEDS_TR = list(NL.tr)

# ---------------------------------------------------------------- açılışların çıkarılması
WORD = re.compile(r"\w+")
def opening(t, low):
    return " ".join(low(m.group(0)) for m in list(WORD.finditer(str(t)))[:2])
def drop_opening(t):
    ms = list(WORD.finditer(str(t)))
    return str(t)[ms[1].end():] if len(ms) >= 2 else ""

def variants(texts, low):
    ops = [opening(t, low) for t in texts]
    c = Counter(ops)
    rec = np.array([c[o] >= ESIK for o in ops])
    return {"ozgun": list(texts),
            "tekrar_eden": [drop_opening(t) if r else t for t, r in zip(texts, rec)],
            "tumu": [drop_opening(t) for t in texts]}, rec, c

# ---------------------------------------------------------------- TF-IDF
def tfidf_top1(msgs, needs, labels):
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1).fit(list(msgs) + list(needs))
    M = vec.transform(msgs).toarray().astype(np.float32); N = vec.transform(needs).toarray().astype(np.float32)
    mn = np.linalg.norm(M, axis=1, keepdims=True); nn = np.linalg.norm(N, axis=1, keepdims=True)
    S = (M / (mn + 1e-9)) @ (N / (nn + 1e-9)).T
    top = [labels[j] for j in np.argmax(S, axis=1)]
    return [None if z else t for t, z in zip(top, mn[:, 0] == 0)]

def hits(top1):
    return np.array([t is not None and t in L for t, L in zip(top1, LAB)], dtype=float)

def compare(h0, h1):
    b = int(((h0 == 1) & (h1 == 0)).sum()); c = int(((h0 == 0) & (h1 == 1)).sum())
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    idx = RNG.integers(0, len(h0), size=(B, len(h0)))
    diff = (h1[idx] - h0[idx]).mean(axis=1)
    return {"fark": float(h1.mean() - h0.mean()), "ga_alt": float(np.percentile(diff, 2.5)), "ga_ust": float(np.percentile(diff, 97.5)),
            "kaybedilen": b, "kazanilan": c, "mcnemar_p": float(p)}

KOSULLAR = {
    "en_govde_ad": ("en", lambda t: stem_en(t), [stem_en(n) or n.lower() for n in EN_LABELS], list(EN_LABELS)),
    "en_govde_ad+tanim": ("en", lambda t: stem_en(t), [stem_en(f"{n}: {DESC_EN[n]}") or n.lower() for n in EN_LABELS], list(EN_LABELS)),
    "tr_snowball_ad": ("tr", lambda t: stem_tr(t), [stem_tr(n) for n in NEEDS_TR], [TR2EN[n] for n in NEEDS_TR]),
}
def en_lower(s): return str(s).replace("İ", "i").lower()
V = {"en": variants(D["English"].astype(str).tolist(), en_lower), "tr": variants(D["Message"].astype(str).tolist(), tr_lower)}
rows, summary = [], {"esik": ESIK, "acilis": {}}
for lang, (vs, rec, c) in V.items():
    summary["acilis"][lang] = {"tekrar_eden_acilisli_paylasim": int(rec.sum()), "farkli_tekrar_eden_acilis": sum(1 for v in c.values() if v >= ESIK),
                               "en_sik": c.most_common(10),
                               "bos_kalan_tekrar_eden": int(sum(not WORD.search(t) for t in vs["tekrar_eden"])),
                               "bos_kalan_tumu": int(sum(not WORD.search(t) for t in vs["tumu"]))}
for kosul, (lang, prep, needs, labels) in KOSULLAR.items():
    vs, rec, _ = V[lang]
    H = {}
    for v, texts in vs.items():
        top1 = tfidf_top1([prep(t) for t in texts], needs, labels)
        H[v] = hits(top1)
        if v == "ozgun" and kosul == "en_govde_ad":
            ref = pd.read_csv(os.path.join(RUN, "results", "tahminler", "mesaj_sentetik_Temel_ TF-IDF_stem.csv")).set_index("id").reindex(D["id"])
            summary["dogrulama_en_govde_ad"] = float(np.mean([a == b for a, b in zip(top1, ref["top1"].values)]))
        if v == "ozgun" and kosul == "en_govde_ad+tanim":
            ref = pd.read_csv(os.path.join(RT, "mesaj_sentetik_Temel_TF-IDF_en_stem_ad+tanim.csv")).set_index("id").reindex(D["id"])
            summary["dogrulama_en_govde_ad+tanim"] = float(np.mean([a == b for a, b in zip(top1, ref["top1"].values)]))
    for v in ("ozgun", "tekrar_eden", "tumu"):
        r = {"kosul": kosul, "metin": v, "hit1": float(H[v].mean())}
        for e, nm in (("explicit", "acik"), ("mixed", "karma"), ("implicit", "ortuk")):
            r["hit1_" + nm] = float(H[v][EXP == e].mean())
        r["hit1_tekrar_eden_acilisli"] = float(H[v][rec].mean()); r["hit1_diger"] = float(H[v][~rec].mean())
        if v != "ozgun":
            r.update(compare(H["ozgun"], H[v]))
            r["fark_ortuk"] = float(H[v][EXP == "implicit"].mean() - H["ozgun"][EXP == "implicit"].mean())
        rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "acilis_ablasyonu.csv"), index=False, encoding="utf-8-sig")
summary["sonuclar"] = rows
jdump(summary, "acilis_ablasyonu.json")
pd.set_option("display.width", 200)
print(res.round(4).to_string(index=False))
print({k: v for k, v in summary.items() if k.startswith("dogrulama")})
for lang in ("en", "tr"):
    print(lang, {k: v for k, v in summary["acilis"][lang].items() if k != "en_sik"})
