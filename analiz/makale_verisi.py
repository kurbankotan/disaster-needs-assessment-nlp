# -*- coding: utf-8 -*-
"""Sonuç klasöründen (runs/<etiket>) makale için gereken tüm sayıları toplar."""
import os, json, re, glob
import numpy as np, pandas as pd
from collections import Counter

PRICES_USD_PER_MTOK = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50),
                       "gpt-5-mini": (0.25, 2.00), "gemini-3.8-flash": (0.75, 3.75)}
LLM_MIN_COVERAGE = 0.98   # mesajların bu oranından azı yanıtlanmış BDM'ler (kota/limit) makaleye alınmaz
MAIN = ["BERT", "ALBERT", "RoBERTa", "GloVe", "Word2Vec", "USE", "FastText"]
EXTRA = ["SBERT-mpnet", "BERT-raw", "ALBERT-paraphrase"]
TURK = ["SBERT-çokdilli (TR)", "BERTurk-STS (TR)"]
BASE = ["Temel: TF-IDF", "Temel: Rastgele", "Temel: Çoğunluk"]
LLM_ORDER = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
LLM_SHORT = {"claude-opus-5-5": "Claude Opus 5.5", "claude-sonnet-5-5": "Claude Sonnet 5.5", "claude-haiku-5-5": "Claude Haiku 5.5"}
def llm_short(mid):
    if mid in LLM_SHORT:
        return LLM_SHORT[mid]
    if mid.startswith("gpt-"):
        return ("GPT-" + mid[4:]).replace("-mini", " mini").replace("-nano", " nano").replace("-preview", " (önizleme)").replace("-chat", "")
    if mid.startswith("gemini-"):
        t = mid[7:].replace("-flash", " Flash").replace("-pro", " Pro").replace("-lite", " Lite").replace("-preview", " (önizleme)")
        return "Gemini " + t.replace("-", " ")
    return mid
def llm_provider(mid):
    return "Anthropic" if mid.startswith("claude") else ("OpenAI" if mid.startswith("gpt") or mid.startswith("o") else ("Google" if mid.startswith("gemini") else "diğer"))
TT_TR = {"plain": "Düz", "lemma": "Lemma", "stem": "Gövde"}
MODEL_LABEL = {"SBERT-çokdilli (TR)": "SBERT çok dilli (TR)", "BERTurk-STS (TR)": "BERTurk-STS (TR)", "Temel: TF-IDF": "TF-IDF (sözcüksel)",
               "Temel: Rastgele": "Rastgele", "Temel: Çoğunluk": "Çoğunluk"}

from decimal import Decimal, ROUND_HALF_UP
def _round_half_up(x, nd):
    q = Decimal(1).scaleb(-nd)
    return Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)

def f2(x, nd=2):
    """Ondalık virgül, yarım-yukarı yuvarlama, eksi için Unicode eksi işareti."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    v = _round_half_up(abs(float(x)), nd)
    s = f"{v:.{nd}f}".replace(".", ",")
    return ("−" + s) if (float(x) < 0 and v != 0) else s

def pct(x, nd=1):
    return "%" + f"{_round_half_up(float(x) * 100, nd):.{nd}f}".replace(".", ",")


# ---------------------------------------------------------------- Türkçe ek yardımcıları
# Son söylenen sayı sözcüğüne göre: (iyelik, yönelme, bulunma+iyelik, tamlayan)
_WORD_SUF = {
    "sıfır": ("ı", "a", "ında", "ın"), "bir": ("i", "e", "inde", "in"), "iki": ("si", "ye", "sinde", "nin"),
    "üç": ("ü", "e", "ünde", "ün"), "dört": ("ü", "e", "ünde", "ün"), "beş": ("i", "e", "inde", "in"),
    "altı": ("sı", "ya", "sında", "nın"), "yedi": ("si", "ye", "sinde", "nin"), "sekiz": ("i", "e", "inde", "in"),
    "dokuz": ("u", "a", "unda", "un"), "on": ("u", "a", "unda", "un"), "yirmi": ("si", "ye", "sinde", "nin"),
    "otuz": ("u", "a", "unda", "un"), "kırk": ("ı", "a", "ında", "ın"), "elli": ("si", "ye", "sinde", "nin"),
    "altmış": ("ı", "a", "ında", "ın"), "yetmiş": ("i", "e", "inde", "in"), "seksen": ("i", "e", "inde", "in"),
    "doksan": ("ı", "a", "ında", "ın"), "yüz": ("ü", "e", "ünde", "ün"), "bin": ("i", "e", "inde", "in"),
}
_DIGIT = ["sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_TENS = {1: "on", 2: "yirmi", 3: "otuz", 4: "kırk", 5: "elli", 6: "altmış", 7: "yetmiş", 8: "seksen", 9: "doksan"}
_CASE = {"poss": 0, "dat": 1, "loc": 2, "gen": 3}

def _last_word(num_str):
    """'%99,1' -> 'bir'; '2.000' -> 'bin'; '0,70' -> 'sıfır'; '40' -> 'kırk'."""
    s = re.sub(r"[^\d,]", "", str(num_str))
    if "," in s:
        dec = s.split(",")[-1]
        return _DIGIT[int(dec[-1])] if dec else "sıfır"
    if not s:
        return "bir"
    if s.endswith("000"):
        return "bin"
    if s.endswith("00"):
        return "yüz"
    if s.endswith("0"):
        return _TENS[int(s[-2])]
    return _DIGIT[int(s[-1])]

def suf(num_str, case="poss"):
    """Sayı dizgisine kesme işaretiyle ek getirir: suf('%99,1','loc') -> "%99,1'inde"."""
    return f"{num_str}'{_WORD_SUF[_last_word(num_str)][_CASE[case]]}"

def pct_s(x, case="poss", nd=1):
    return suf(pct(x, nd), case)

def f2_s(x, case="poss", nd=2):
    return suf(f2(x, nd), case)

# Model adlarına ek (okunuşa göre): (iyelik/belirtme, yönelme)
_MODEL_SUF = {"USE": ("'yi", "'ye"), "SBERT-mpnet": ("'i", "'e"), "Word2Vec": ("'i", "'e"), "FastText": ("'i", "'e"),
              "GloVe": ("'u", "'a"), "BERT": ("'ü", "'e"), "RoBERTa": ("'yı", "'ya"), "ALBERT": ("'i", "'e"),
              "ALBERT-paraphrase": ("'i", "'e"), "BERT-raw": ("'u", "'a"), "Temel: TF-IDF": ("'yi", "'ye"), "TF-IDF": ("'yi", "'ye"),
              "SBERT-çokdilli (TR)": ("'yi", "'ye"), "BERTurk-STS (TR)": ("'yi", "'ye")}
def model_s(name, case="acc"):
    a, d = _MODEL_SUF.get(name, ("'i", "'e"))
    return name + (a if case == "acc" else d)

def ratio_word(a, b):
    """a/b oranını 'yaklaşık üç kat' biçiminde yuvarlak sözcükle verir."""
    a, b = float(a), float(b)
    if not b:
        return ""
    k = int(round(a / b))
    words = {2: "iki", 3: "üç", 4: "dört", 5: "beş", 6: "altı", 7: "yedi", 8: "sekiz", 9: "dokuz", 10: "on"}
    return words.get(k, str(k))

def p_str(p, nd=3):
    """p değeri: 'p < 0,001' ya da 'p = 0,008'."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "p = ?"
    return "p < 0,001" if p < 0.001 else f"p = {f2(p, nd)}"

def load(run_dir, ref_run_dir=None):
    """ref_run_dir: önbellekten tamamlanan çalıştırmalarda 0 görünen API maliyetleri için referans çalıştırma (gerçek çağrıların yapıldığı)."""
    R = os.path.join(run_dir, "results")
    D = {"run_dir": run_dir, "ref_run_dir": ref_run_dir}
    res = pd.read_csv(os.path.join(R, "sonuclar_tum_konfigurasyonlar.csv"))
    D["res"] = res
    D["syn"] = res[res.veri == "sentetik"].copy()
    def row(df, model, tt):
        r = df[(df.model == model) & (df.metin == tt)]
        return None if r.empty else r.iloc[0]
    D["row"] = row
    def best(df, model):
        r = df[df.model == model]
        return None if r.empty else r.sort_values("hit1", ascending=False).iloc[0]
    D["best"] = best
    # BDM
    D["llm"] = {}; D["llm_incomplete"] = {}
    for s in ("sentetik",):
        p = os.path.join(R, f"tablo5_bdm_{s}_sayisal.csv")
        t = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
        if not t.empty:
            cov = 1 - t["tahminsiz"].fillna(0) / t["n_mesaj"].replace(0, np.nan)
            bad = t[cov.fillna(0) < LLM_MIN_COVERAGE]
            D["llm_incomplete"][s] = {r.model: (int(r.tahminsiz), int(r.n_mesaj)) for r in bad.itertuples()}
            t = t[cov.fillna(0) >= LLM_MIN_COVERAGE].reset_index(drop=True)
        D["llm"][s] = t
    # maliyet: bu çalıştırmada çağrı yapılmadıysa (önbellek) referans çalıştırmadan al
    D["cost_source"] = {}
    for s, t in D["llm"].items():
        if t.empty or "maliyet_usd" not in t:
            continue
        for i, r in t.iterrows():
            if not (r.maliyet_usd > 0):
                ref = None
                if ref_run_dir:
                    pr = os.path.join(ref_run_dir, "results", f"tablo5_bdm_{s}_sayisal.csv")
                    if os.path.exists(pr):
                        tr_ = pd.read_csv(pr); m = tr_[(tr_.model == r.model) & (tr_.maliyet_usd > 0)]
                        if not m.empty:
                            ref = float(m.iloc[0].maliyet_usd) * (r.n_mesaj / float(m.iloc[0].n_mesaj))
                if ref is not None:
                    t.at[i, "maliyet_usd"] = ref; D["cost_source"][(s, r.model)] = "referans çalıştırma"
                else:
                    t.at[i, "maliyet_usd"] = np.nan; D["cost_source"][(s, r.model)] = "yok"
    excluded = set().union(*[set(v) for v in D["llm_incomplete"].values()]) if D["llm_incomplete"] else set()
    D["llm_excluded"] = excluded
    # anlamlılık (eksik BDM satırları çıkarılır)
    D["t8"] = pd.read_csv(os.path.join(R, "tablo8_anlamlilik_en_iyiye_karsi_sentetik.csv"))
    if excluded:
        D["t8"] = D["t8"][~D["t8"]["Karşılaştırılan"].apply(lambda s: any(s.startswith(m + " ") for m in excluded))].reset_index(drop=True)
    D["t9"] = pd.read_csv(os.path.join(R, "tablo9_metin_turu_etkisi_sentetik.csv"))
    D["t3b"] = pd.read_csv(os.path.join(R, "tablo3b_model_ayarlari.csv"))
    D["t6"] = pd.read_csv(os.path.join(R, "tablo6_cikarim_suresi.csv"))
    D["needs"] = pd.read_csv(os.path.join(R, "ihtiyac_listesi.csv"))
    D["en2tr"] = {}
    for _tr, _en in zip(D["needs"]["tr"], D["needs"]["en"]):
        D["en2tr"].setdefault(_en, _tr)
    D["qc"] = json.load(open(os.path.join(run_dir, "logs", "kalite_kontrol_ozet.json"), encoding="utf-8"))
    D["cfg"] = json.load(open(os.path.join(run_dir, "logs", "config.json"), encoding="utf-8"))
    D["profiles"] = json.load(open(os.path.join(run_dir, "logs", "ilce_profilleri.json"), encoding="utf-8"))
    # veri seti
    dpath = glob.glob(os.path.join(run_dir, "data", "sentetik_paylasimlar_*.csv"))[0]
    ds = pd.read_csv(dpath); D["ds"] = ds
    D["n"] = len(ds); D["n_district"] = ds.district.nunique()
    D["per_district"] = ds.groupby("district").size()
    D["n_needs_dist"] = ds.n_needs.value_counts(normalize=True).reindex([1, 2, 3]).fillna(0)
    D["expl_dist"] = ds.explicitness.value_counts(normalize=True).reindex(["explicit", "mixed", "implicit"]).fillna(0)
    D["flags"] = ds[["typos", "no_diacritics", "hashtag", "emoji", "neighbourhood", "numbers"]].mean()
    D["words_tr"] = ds.Message.str.split().str.len().mean(); D["words_en"] = ds.English.str.split().str.len().mean()
    D["chars_tr"] = ds.Message.str.len().mean(); D["chars_en"] = ds.English.str.len().mean()
    D["labels_used"] = pd.concat([ds.label1, ds.label2, ds.label3]).dropna().nunique()
    # günlükten üretim istatistikleri
    log = open(os.path.join(run_dir, "logs", "log.txt"), encoding="utf-8").read()
    m = re.findall(r"Tur (\d): (\d+) kayıt planlandı", log); D["planned"] = sum(int(x[1]) for x in m)
    D["gen_rounds"] = [(int(a), int(b), int(c)) for a, b, c in re.findall(r"Tur (\d): (\d+) kayıt planlandı \((\d+) çağrı\)", log)]
    _acc = re.findall(r"Tur (\d): (\d+) kabul, retler: (\{.*?\})", log)
    D["gen_accepted"] = sum(int(b) for _, b, _ in _acc)
    D["gen_rejected"] = sum(sum(eval(c).values()) for _, _, c in _acc)
    D["gen_missing"] = D["planned"] - D["gen_accepted"] - D["gen_rejected"]
    _ta = re.findall(r"toplam kabul (\d+)", log); D["gen_total_accepted"] = int(_ta[-1]) if _ta else None
    rej = re.findall(r"retler: (\{.*?\})", log)
    tot = Counter()
    for r in rej:
        tot.update(eval(r))
    D["rejects"] = dict(tot)
    dup = re.findall(r"tekrar elenen (\d+)", log); D["dups"] = sum(int(x) for x in dup)
    m = re.search(r"Token kullanımı claude-opus-5-5: (\d+) çağrı.*?tahmini \$([\d.]+)", log); D["gen_calls"], D["gen_cost"] = (int(m.group(1)), float(m.group(2))) if m else (None, None)
    m = re.search(r"Token kullanımı claude-sonnet-5-5: (\d+) çağrı.*?tahmini \$([\d.]+)", log); D["qc_cost"] = float(m.group(2)) if m else None
    m = re.search(r"Üretim bitti: (\d+) kayıt, süre ([\d.]+) dk", log); D["gen_minutes"] = float(m.group(2)) if m else None
    env = open(os.path.join(run_dir, "logs", "ortam.txt"), encoding="utf-8").read()
    D["env"] = dict(l.split(": ", 1) for l in env.splitlines() if ": " in l)
    pe = os.path.join(run_dir, "logs", "ortam_ek.txt")
    if os.path.exists(pe):
        D["env"].update(dict(l.split(": ", 1) for l in open(pe, encoding="utf-8").read().splitlines() if ": " in l))
    # etiket sayıları (her ihtiyacın veri setinde kaç kez etiket olduğu)
    D["label_counts"] = Counter(x for col in ("label1", "label2", "label3") for x in ds[col].dropna())
    # GPT protokol düzeltmesi (tamamlama defteri) yapıldı mı?
    D["gpt_redo_done"] = False
    for pj in glob.glob(os.path.join(run_dir, "logs", "bdm_ek_*.json")):
        try:
            jj = json.load(open(pj, encoding="utf-8"))
            if ["sentetik", "gpt-5-mini"] in jj.get("yeniden_istenen", []) and ["sentetik", "gpt-5-mini"] in jj.get("tamamlanan", []):
                D["gpt_redo_done"] = True
        except Exception:
            pass
    D["fig2_example"] = None
    p2 = os.path.join(run_dir, "results", "sekil2_ornek.json")
    if os.path.exists(p2):
        ex2 = json.load(open(p2, encoding="utf-8"))
        # tutarlılık: örnekteki ilk üç ihtiyaç, boru hattının aynı paylaşım için kaydettiği top-1..top-3 ile aynı olmalı
        ex2["tutarli"] = False
        pf2 = os.path.join(R, "tahminler", "mesaj_sentetik_SBERT-mpnet_plain.csv")
        if os.path.exists(pf2) and ex2.get("top"):
            pr2 = pd.read_csv(pf2).set_index("id")
            if int(ex2["id"]) in pr2.index:
                rr = pr2.loc[int(ex2["id"])]
                ex2["tutarli"] = [x["en"] for x in ex2["top"][:3]] == [rr.top1, rr.top2, rr.top3]
        D["fig2_example"] = ex2
    # açık/örtük: mesaj düzeyinden yeniden hesapla (temel çizgiler dâhil)
    pred_dir = os.path.join(R, "tahminler")
    import unicodedata as _ud
    def _pkey(n):
        """Unicode biçimi (NFC/NFD) ve Windows'ta ':' yerine '_' yazılması fark etmeksizin aynı anahtar."""
        return _ud.normalize("NFC", n).replace(":", "_")
    _PRED = {_pkey(f_): os.path.join(pred_dir, f_) for f_ in os.listdir(pred_dir)} if os.path.isdir(pred_dir) else {}
    def pred_path(name):
        """Tahmin dosyasının diskteki yolu (dosya adı biçim farklarından bağımsız)."""
        return _PRED.get(_pkey(name), os.path.join(pred_dir, name))
    D["pred_path"] = pred_path
    expl = ds.set_index("id")["explicitness"]
    rows = {}
    for f in glob.glob(os.path.join(pred_dir, "mesaj_sentetik_*.csv")):
        import unicodedata
        tag = unicodedata.normalize("NFC", os.path.basename(f)[len("mesaj_sentetik_"):-4]).replace("Temel_ ", "Temel: ")   # Windows'ta ':' -> '_' 
        if tag.startswith("BDM_") and tag[4:] in excluded:
            continue
        pm = pd.read_csv(f)
        if len(pm) != len(ds):
            continue
        h = pm.set_index("id")["hit1"].reindex(expl.index)
        rows[tag] = h.groupby(expl.values).mean().reindex(["explicit", "mixed", "implicit"]).to_dict()
    D["expl_by_tag"] = rows
    D["pred_dir"] = pred_dir
    # BDM'ler arası McNemar
    def mcnemar(a, b):
        from scipy.stats import binomtest
        n10 = int(((a == 1) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
        p = binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if (n10 + n01) else 1.0
        return p, n10, n01
    syn_llm = D["llm"]["sentetik"]
    _prov_rank = {"OpenAI": 0, "Google": 1}
    models = [m for m in LLM_ORDER if m in set(syn_llm.model)] + sorted([m for m in syn_llm.model if m not in LLM_ORDER], key=lambda m: (_prov_rank.get(llm_provider(m), 9), m))
    D["llm_models"] = models
    D["llm_other"] = [m for m in models if not m.startswith("claude")]
    llm_hits = {}
    for mname in models:
        f = pred_path(f"mesaj_sentetik_BDM_{mname}.csv")
        if os.path.exists(f):
            llm_hits[mname] = pd.read_csv(f).set_index("id")["hit1"].reindex(expl.index).values
    D["llm_pairs"] = {}
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            if a in llm_hits and b in llm_hits:
                D["llm_pairs"][(a, b)] = mcnemar(llm_hits[a], llm_hits[b])
    # gecikme/maliyet ölçümü (Tablo 6b): tüm BDM'ler aynı koşullarda
    D["bench"] = {}
    pb = os.path.join(R, "tablo6b_bdm_gecikme_maliyet_olcumu.csv")
    if os.path.exists(pb):
        tb = pd.read_csv(pb)
        for r in tb.to_dict("records"):
            if bool(r.get("Tamam", True)) and r["Model"] in models:
                D["bench"][r["Model"]] = r
    # değerlendirme günlüğünden çağrı başına token (OpenAI / Google): ölçüm yoksa maliyet tahmini için
    D["eval_tokens"] = {}
    for mm in re.finditer(r"sentetik_BDM_([\w.\-]+): .*?\| (\d+) çağrı \| çağrı başına giriş (\d+)(?: \(önbellek \d+\))?, çıktı (\d+) token", log):
        D["eval_tokens"][mm.group(1)] = (int(mm.group(2)), int(mm.group(3)), int(mm.group(4)))
    # en iyi yerel model (temel çizgi dışı)
    emb = D["syn"][~D["syn"].model.isin(BASE)]
    D["best_local"] = emb.sort_values("hit1", ascending=False).iloc[0]
    tf = D["syn"][D["syn"].model == "Temel: TF-IDF"].sort_values("hit1", ascending=False).iloc[0]
    D["best_tfidf"] = tf
    # ilçe haritası çeşitliliği: en iyi yapılandırmalar ve gerçek dağılım
    D["district_div"] = {}
    def _div(tag):
        """İlçe modları eşitlikte küme olarak alınır (Eş. 2 ile ve gerçek dağılımın sayımıyla tutarlı)."""
        f = pred_path(f"mesaj_sentetik_{tag}.csv")
        if not os.path.exists(f):
            return None
        pm = pd.read_csv(f)
        modes = {}
        for dname, g in pm.groupby("district"):
            vc = g.top1.dropna().value_counts()
            modes[dname] = set(vc[vc == vc.max()].index) if len(vc) else set()
        cnt = Counter(x for v in modes.values() for x in v)
        top, top_n = cnt.most_common(1)[0]
        only_n = sum(1 for v in modes.values() if v == {top})
        return {"n_farkli": len(cnt), "en_sik": str(top), "en_sik_n": int(top_n), "en_sik_tek": int(only_n), "n_ilce": int(len(modes))}
    for m in MAIN + EXTRA + TURK + ["Temel: TF-IDF"]:
        b = D["best"](D["syn"], m)
        if b is not None:
            r = _div(f"{m}_{b.metin}")
            if r: D["district_div"][m] = dict(r, metin=b.metin)
    for m in models:
        r = _div(f"BDM_{m}")
        if r: D["district_div"]["BDM_" + m] = r
    f0 = os.path.join(pred_dir, f"ilce_sentetik_{D['best_local'].model}_{D['best_local'].metin}.csv")
    D["true_div"] = len(set().union(*[set(str(s).split(", ")) for s in pd.read_csv(f0).gercek_en_sik])) if os.path.exists(f0) else None
    # açık / örtük alt kümelerde eşleştirilmiş McNemar
    def _hits(tag):
        f = pred_path(f"mesaj_sentetik_{tag}.csv")
        return pd.read_csv(f).set_index("id")["hit1"].reindex(expl.index).fillna(0).values if os.path.exists(f) else None
    D["subset_test"] = {}
    def subset_test(tag_a, tag_b, group):
        key = (tag_a, tag_b, group)
        if key not in D["subset_test"]:
            a, b = _hits(tag_a), _hits(tag_b)
            if a is None or b is None:
                D["subset_test"][key] = None
            else:
                msk = (expl.values == group)
                D["subset_test"][key] = mcnemar(a[msk], b[msk])
        return D["subset_test"][key]
    D["subset_test_fn"] = subset_test
    def pair_test(tag_a, tag_b):
        a, b = _hits(tag_a), _hits(tag_b)
        return None if a is None or b is None else mcnemar(a, b)
    D["pair_test_fn"] = pair_test
    return D

if __name__ == "__main__":
    import sys
    D = load(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print("n", D["n"], "qc", D["qc"], "rejects", D["rejects"], "dups", D["dups"], "best local", D["best_local"].model, D["best_local"].metin, D["best_local"].hit1)
    print("llm pairs", D["llm_pairs"])
    print({k: {kk: round(v, 3) for kk, v in vv.items()} for k, vv in D["expl_by_tag"].items() if "TF-IDF" in k or "USE" in k})
