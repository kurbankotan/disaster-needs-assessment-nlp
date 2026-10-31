# -*- coding: utf-8 -*-
"""E) Gerçek veri: kapalı küme isabeti (kesin binom GA), çoğunluk temel çizgisi, sektör isabeti, anlatım biçimi;
Türkçe TF-IDF (çevrimdışı; sentetik ve gerçek Türkçe), sentetik ve gerçek sıralama ilişkisi."""
from revizyon_ortak import *
import unicodedata
from scipy.stats import spearmanr, kendalltau
from sklearn.feature_extraction.text import TfidfVectorizer
import snowballstemmer

# Gerçek paylaşımların altın etiketleri. Tweet metinleri platform kullanım koşulları nedeniyle depoda yoktur: kimlik ve etiket
# dosyası her zaman okunur. Metinli dosya (gercek_veri_altin.csv) yalnızca yerelde varsa Türkçe TF-IDF yeniden hesaplanır;
# yoksa bu yöntemin depodaki paylaşım düzeyi tahminleri (results/analiz/tahmin_gercek_tr_tfidf_tr_*.csv) kullanılır.
G_TEXT = os.path.join(REV_DATA, "gercek_veri_altin.csv")
G_IDS = os.path.join(REV_DATA, "gercek_veri_altin_kimlikler.csv")
HAS_TEXT = os.path.exists(G_TEXT)
G = pd.read_csv(G_TEXT if HAS_TEXT else G_IDS, dtype=str, encoding="utf-8-sig").fillna("")
print("Gerçek veri:", "metinli dosya (Türkçe TF-IDF yeniden hesaplanır)" if HAS_TEXT else "kimlik ve etiket dosyası (Türkçe TF-IDF tahminleri depodan okunur)")
REAL = {}
for k in ("EN", "TR"):
    d = G[G.kume == k].copy().reset_index(drop=True); d["id"] = np.arange(len(d))
    d["labels_tr"] = d.ihtiyaclar_tr.apply(lambda s: [x for x in s.split(";") if x])
    d["labels_en"] = d.labels_tr.apply(lambda L: [TR2EN[x] for x in L])
    REAL["gercek_" + k.lower()] = d
DESC = pd.read_csv(os.path.join(RR, "ihtiyac_tanimlari.csv"), encoding="utf-8-sig")
DESC_TR = dict(zip(DESC.ihtiyac_tr, DESC.tanim_tr)); DESC_EN = dict(zip(DESC.ihtiyac_en, DESC.tanim_en))

def closed(setname):
    d = REAL[setname]; return d[d.labels_en.str.len() > 0].reset_index(drop=True)

def score_file(setname, tag):
    for f in os.listdir(RT):
        if unicodedata.normalize("NFC", f) == unicodedata.normalize("NFC", f"skor_{setname}_{tag}.csv"):
            return pd.read_csv(os.path.join(RT, f), encoding="utf-8-sig")
    return None

rows = []
def add(setname, name, cond, top1_by_id, extra=None):
    c = closed(setname)
    top1 = [top1_by_id.get(i) for i in c.id]
    item, grp, sec = hier_hits(top1, c.labels_en.tolist())
    k = int(item.sum()); n = len(item); lo, hi = clopper_pearson(k, n)
    r = {"veri": setname, "yontem": name, "kosul": cond, "n": n, "dogru": k, "hit1": k / n, "ga_alt": lo, "ga_ust": hi,
         "grup_hit1": grp.mean(), "sektor_hit1": sec.mean()}
    for e, lab in (("acik", "Açık"), ("ortuk", "Örtük")):
        m = (c.anlatim == lab).values
        r[f"n_{e}"] = int(m.sum()); r[f"hit1_{e}"] = item[m].mean() if m.any() else np.nan
    r.update(extra or {})
    rows.append(r)
    return item

# --- Colab çıktıları (makine çevirisi koşulları hariç: gercek_tr + en/...)
HITS_REAL = {}
for f in sorted(os.listdir(RT)):
    fn = unicodedata.normalize("NFC", f)
    m = re.match(r"skor_(gercek_en|gercek_tr)_(.+)\.csv$", fn)
    if not m:
        continue
    setname, tag = m.groups()
    d = pd.read_csv(os.path.join(RT, f), encoding="utf-8-sig")
    mp = dict(zip(d.id, d.top1))
    if tag.startswith("BDM_"):
        mid, cond = re.match(r"BDM_([\w.\-]+?)_(kapali_tek_en|kapali_tek_tr|kapali_tek_en_mt)$", tag).groups()
        if cond == "kapali_tek_en_mt":
            continue
        HITS_REAL[(setname, "BDM " + mid, "sifir_atis")] = add(setname, "BDM " + mid, "sifir_atis", mp)
        continue
    if tag.startswith(("SetFit", "Dondurulmuş")):
        mm = re.match(r"(SetFit-EN|SetFit-TR|Dondurulmuş\+LR-EN|Dondurulmuş\+LR-TR)_(.+)$", tag)
        name, cond = mm.groups()
        if setname == "gercek_tr" and name.endswith("-EN"):
            continue                                            # makine çevirisi üzerinde
        HITS_REAL[(setname, name, cond)] = add(setname, name, cond, mp)
        continue
    mm = re.match(r"(.+?)_(en|tr)_(plain|lemma|stem|raw)_(ad\+tanim|ad)$", tag)
    model, lang, tt, nm = mm.groups()
    if setname == "gercek_tr" and lang == "en" and not model.startswith(("Temel_Çoğunluk", "Temel_Rastgele")):
        continue                                                # makine çevirisi üzerinde
    model = model.replace("Temel_", "Temel: ").replace("_TR_", " (TR)").replace("SBERT-çokdilli", "SBERT çok dilli")
    HITS_REAL[(setname, model, f"{lang}/{tt}/{nm}")] = add(setname, model, f"{lang}/{tt}/{nm}", mp)

# --- Türkçe TF-IDF (çevrimdışı): sentetik 2.000 Türkçe paylaşım ve gerçek Türkçe paylaşımlar
LET = re.compile(r"[^a-zA-ZğĞçÇşŞüÜöÖıİâÂîÎûÛ]")
def tr_lower(s): return str(s).replace("İ", "i").replace("I", "ı").lower()
def plain_tr(t): return " ".join(tr_lower(LET.sub(" ", str(t))).split())
STEM = snowballstemmer.stemmer("turkish")
def stem_tr(t): return " ".join(STEM.stemWords(plain_tr(t).split()))
NEEDS_TR = list(NL.tr)
def tfidf_tr(msgs_raw, mode, need_mode):
    f = plain_tr if mode == "plain" else stem_tr
    msgs = [f(t) for t in msgs_raw]
    needs = [f(n if need_mode == "ad" else f"{n}: {DESC_TR[n]}") for n in NEEDS_TR]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1).fit(msgs + needs)
    M = vec.transform(msgs).toarray(); N = vec.transform(needs).toarray()
    M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9); N /= (np.linalg.norm(N, axis=1, keepdims=True) + 1e-9)
    Sim = M @ N.T
    return [TR2EN[NEEDS_TR[j]] for j in np.argmax(Sim, axis=1)], Sim

tr_syn = []
for mode in ("plain", "stem"):
    for nm in ("ad", "ad+tanim"):
        top1, _ = tfidf_tr(D["Message"].tolist(), mode, nm)
        item, grp, sec = hier_hits(top1)
        lo_, hi_ = boot_ci(item)
        r = {"yontem": "TF-IDF (TR)", "kosul": f"tr/{mode}/{nm}", "hit1": item.mean(), "ga_alt": lo_, "ga_ust": hi_, "sektor_hit1": sec.mean()}
        for e in ("explicit", "mixed", "implicit"):
            r[f"hit1_{e}"] = item[EXP == e].mean()
        r["ilce_dogrulugu"], _ = district_accuracy(top1)
        tr_syn.append(r)
        d = REAL["gercek_tr"]
        fpred = os.path.join(OUT, f"tahmin_gercek_tr_tfidf_tr_{mode}_{nm}.csv")
        if HAS_TEXT:
            top1r, _ = tfidf_tr(d.metin.tolist(), mode, nm)
            pd.DataFrame({"sira": d.sira, "top1": top1r}).to_csv(fpred, index=False, encoding="utf-8-sig")
        else:
            pr = pd.read_csv(fpred, encoding="utf-8-sig")
            assert pr.sira.tolist() == d.sira.tolist(), fpred
            top1r = pr.top1.tolist()
        HITS_REAL[("gercek_tr", "TF-IDF (TR)", f"tr/{mode}/{nm}")] = add("gercek_tr", "TF-IDF (TR)", f"tr/{mode}/{nm}", dict(zip(d.id, top1r)))
TRSYN = pd.DataFrame(tr_syn); TRSYN.to_csv(f"{OUT}/tfidf_turkce_sentetik.csv", index=False, encoding="utf-8-sig")

R = pd.DataFrame(rows); R.to_csv(f"{OUT}/gercek_kapali_kume.csv", index=False, encoding="utf-8-sig")

# --- sentetik ve gerçek sıralama ilişkisi (aynı koşul)
SYN = pd.read_csv(f"{OUT}/sentetik_tum_yontemler.csv", encoding="utf-8-sig")
def syn_hit(name, cond):
    x = SYN[(SYN.yontem == name) & (SYN.kosul == cond)]
    return float(x.hit1.iloc[0]) if len(x) else np.nan
pairs_en = [(f"BDM {m}", "sifir_atis/en", "sifir_atis") for m in LLM_IDS] + [
    ("Temel: TF-IDF", "en/stem/ad", "en/stem/ad"), ("USE", "en/plain/ad", "en/plain/ad"), ("SBERT-mpnet", "en/plain/ad", "en/plain/ad"),
    ("Word2Vec", "en/lemma/ad", "en/lemma/ad"), ("GloVe", "en/stem/ad", "en/stem/ad"), ("FastText", "en/stem/ad", "en/stem/ad"),
    ("BERT", "en/stem/ad", "en/stem/ad"), ("RoBERTa", "en/stem/ad", "en/stem/ad"), ("ALBERT", "en/plain/ad", "en/plain/ad"),
    ("ALBERT-paraphrase", "en/lemma/ad", "en/lemma/ad"), ("BERT-raw", "en/plain/ad", "en/plain/ad"),
    ("Qwen3-Emb-4B", "en/plain/ad+tanim", "en/plain/ad+tanim"), ("mE5-large-instruct", "en/plain/ad+tanim", "en/plain/ad+tanim"),
    ("BGE-M3", "en/plain/ad", "en/plain/ad"), ("Qwen3-Emb-0.6B", "en/plain/ad+tanim", "en/plain/ad+tanim")]
pairs_tr = [(f"BDM {m}", "sifir_atis/tr", "sifir_atis") for m in LLM_IDS] + [
    ("mE5-large-instruct", "tr/plain/ad+tanim", "tr/plain/ad+tanim"), ("BGE-M3", "tr/raw/ad+tanim", "tr/raw/ad+tanim"),
    ("Qwen3-Emb-4B", "tr/raw/ad+tanim", "tr/raw/ad+tanim"), ("SBERT çok dilli (TR)", "tr/plain/ad", "tr/plain/ad"),
    ("BERTurk-STS (TR)", "tr/plain/ad", "tr/plain/ad"), ("Qwen3-Emb-0.6B", "tr/plain/ad+tanim", "tr/plain/ad+tanim")]
RANK = {}
for setname, pairs in (("gercek_en", pairs_en), ("gercek_tr", pairs_tr)):
    xs, ys, names = [], [], []
    for name, cs, cr in pairs:
        a = syn_hit(name, cs); b = R[(R.veri == setname) & (R.yontem == name) & (R.kosul == cr)]
        if np.isnan(a) or b.empty:
            continue
        xs.append(a); ys.append(float(b.hit1.iloc[0])); names.append(f"{name} | {cs}")
    rho = spearmanr(xs, ys); tau = kendalltau(xs, ys)
    # bootstrap (gerçek paylaşımlar yeniden örneklenir)
    c = closed(setname); rng = np.random.default_rng(42); bs = []
    H = {nm: HITS_REAL[(setname, nm.split(" | ")[0], [p[2] for p in pairs if f"{p[0]} | {p[1]}" == nm][0])] for nm in names}
    for _ in range(2000):
        idx = rng.integers(0, len(c), len(c))
        bs.append(spearmanr(xs, [H[nm][idx].mean() for nm in names]).statistic)
    RANK[setname] = {"n_yontem": len(xs), "spearman": float(rho.statistic), "spearman_p": float(rho.pvalue),
                     "spearman_ga": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                     "kendall": float(tau.statistic), "kendall_p": float(tau.pvalue),
                     "yontemler": dict(zip(names, [[round(a, 3), round(b, 3)] for a, b in zip(xs, ys)]))}
jdump(RANK, "siralama_iliskisi.json")

# --- gerçek veride BDM ile en iyi yerel yöntemin eşleştirilmiş karşılaştırması
cmp = []
for setname, loc in (("gercek_en", ("Temel: TF-IDF", "en/stem/ad")), ("gercek_en", ("Qwen3-Emb-4B", "en/plain/ad+tanim")),
                     ("gercek_tr", ("mE5-large-instruct", "tr/raw/ad+tanim")), ("gercek_tr", ("TF-IDF (TR)", "tr/stem/ad"))):
    for m in LLM_IDS:
        a = HITS_REAL[(setname, f"BDM {m}", "sifir_atis")]; b = HITS_REAL[(setname, loc[0], loc[1])]
        p, n01, n10 = mcnemar_exact(a, b)
        cmp.append({"veri": setname, "bdm": m, "yerel": " | ".join(loc), "bdm_hit1": a.mean(), "yerel_hit1": b.mean(), "p": p})
CMP = pd.DataFrame(cmp); CMP.to_csv(f"{OUT}/gercek_bdm_yerel_mcnemar.csv", index=False, encoding="utf-8-sig")

# --- açık küme ve HumAID özetleri (Colab), gerçek veri tanımlayıcıları
RES = pd.read_csv(os.path.join(RR, "revizyon_tum_sonuclar.csv"))
op = RES[(RES.deney == "bdm_gercek") & (RES.kosul == "acik_kume")][["veri", "yontem", "n", "ikili_kesinlik", "ikili_duyarlilik", "ikili_f1", "dp", "yp", "yn", "dn",
                                                                    "ihtiyac_mikro_f1", "ihtiyac_makro_f1", "ihtiyac_tum_ihtiyaclar_bulundu", "sektor_mikro_f1",
                                                                    "sektor_tum_ihtiyaclar_bulundu", "listede_yok_f1"]]
op.to_csv(f"{OUT}/gercek_acik_kume.csv", index=False, encoding="utf-8-sig")
_prev = os.path.join(OUT, "gercek_veri_tanim.json")
_prev = json.load(open(_prev, encoding="utf-8")) if os.path.exists(_prev) else {}
def ort_kelime(k, d):
    """Paylaşım başına ortalama kelime; metin yoksa önceki çıktıdaki değer korunur."""
    if HAS_TEXT:
        return float(d.metin.str.split().str.len().mean())
    return _prev.get(k, {}).get("ort_kelime", float("nan"))
desc = {}
for k, d in REAL.items():
    pos = d[d.somut_ihtiyac == "Evet"]; c = closed(k)
    desc[k] = {"n_altin": len(d), "somut_ihtiyac": len(pos), "listede_ihtiyacli": len(c), "yalniz_listede_olmayan": int(((pos.labels_tr.str.len() == 0)).sum()),
               "listede_olmayan_iceren": int((pos.listede_olmayan != "").sum()), "anlatim_pozitif": pos.anlatim.value_counts().to_dict(),
               "anlatim_kapali": c.anlatim.value_counts().to_dict(), "ihtiyac_sayisi_kapali": Counter(c.labels_tr.str.len()).most_common(),
               "en_sik_ihtiyaclar": Counter(x for L in c.labels_tr for x in L).most_common(6),
               "ort_kelime": ort_kelime(k, d)}
jdump(desc, "gercek_veri_tanim.json")

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
for s in ("gercek_en", "gercek_tr"):
    print("==", s); print(R[R.veri == s].sort_values("hit1", ascending=False)[["yontem", "kosul", "n", "dogru", "hit1", "ga_alt", "ga_ust", "sektor_hit1", "n_acik", "hit1_acik", "n_ortuk", "hit1_ortuk"]].round(3).head(30).to_string(index=False))
print(TRSYN.round(3).to_string(index=False))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "yontemler"} for k, v in RANK.items()}, indent=1))
print(CMP.round(4).to_string(index=False))
print(json.dumps(desc, ensure_ascii=False, indent=1))
