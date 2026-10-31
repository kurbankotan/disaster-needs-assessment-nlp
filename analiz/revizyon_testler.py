# -*- coding: utf-8 -*-
"""B) Yeni karşılaştırmalar için eşleştirilmiş McNemar tam testleri (aile içinde Holm düzeltmesi) ve fark GA'ları."""
from revizyon_ortak import *
import pickle
H = pickle.load(open(f"{OUT}/sentetik_hits.pkl", "rb"))

def get(name, cond, level="item", mask=None):
    v = H[(name, cond)][level]
    return v if mask is None else v[mask]

QWEN = ("Qwen3-Emb-4B", "en/plain/ad+tanim"); USE = ("USE", "en/plain/ad"); TF = ("Temel: TF-IDF", "en/stem/ad")
OPUS = ("BDM claude-opus-5-5", "sifir_atis/en"); HAIKU = ("BDM claude-haiku-5-5", "sifir_atis/en")
fams = {
    "A_yeni_kodlayici": [
        (QWEN, USE, None), (QWEN, TF, None), (QWEN, HAIKU, None), (QWEN, OPUS, None),
        (("mE5-large-instruct", "en/plain/ad+tanim"), USE, None), (("BGE-M3", "en/plain/ad"), USE, None),
    ],
    "B_tanim_etkisi": [
        (("Temel: TF-IDF", "en/stem/ad+tanim"), TF, None), (QWEN, ("Qwen3-Emb-4B", "en/plain/ad"), None),
        (("mE5-large-instruct", "en/plain/ad+tanim"), ("mE5-large-instruct", "en/plain/ad"), None),
        (("USE", "en/plain/ad+tanim"), USE, None), (("SBERT-mpnet", "en/plain/ad+tanim"), ("SBERT-mpnet", "en/plain/ad"), None),
    ],
    "B2_tanim_etkisi_ortuk": [
        (("Temel: TF-IDF", "en/stem/ad+tanim"), TF, "implicit"), (QWEN, ("Qwen3-Emb-4B", "en/plain/ad"), "implicit"),
        (("USE", "en/plain/ad+tanim"), USE, "implicit"), (("SBERT-mpnet", "en/plain/ad+tanim"), ("SBERT-mpnet", "en/plain/ad"), "implicit"),
        (("mE5-large-instruct", "en/plain/ad+tanim"), ("mE5-large-instruct", "en/plain/ad"), "implicit"),
    ],
    "B3_tanim_etkisi_acik": [
        (("Temel: TF-IDF", "en/stem/ad+tanim"), TF, "explicit"), (QWEN, ("Qwen3-Emb-4B", "en/plain/ad"), "explicit"),
        (("USE", "en/plain/ad+tanim"), USE, "explicit"), (("SBERT-mpnet", "en/plain/ad+tanim"), ("SBERT-mpnet", "en/plain/ad"), "explicit"),
        (("mE5-large-instruct", "en/plain/ad+tanim"), ("mE5-large-instruct", "en/plain/ad"), "explicit"),
    ],
    "C_turkce_ingilizce": [((f"BDM {m}", "sifir_atis/tr"), (f"BDM {m}", "sifir_atis/en"), None) for m in LLM_IDS],
    "D_ortuk_kodlayici": [(QWEN, USE, "implicit"), (QWEN, HAIKU, "implicit"), (QWEN, TF, "implicit")],
    "E_ablasyon": [(("RoBERTa-güncel", "en/plain/ad"), ("RoBERTa", "en/plain/ad"), None), (("RoBERTa", "en/plain/ad"), ("RoBERTa-raw", "en/plain/ad"), None)],
}
rows = []
for fam, comps in fams.items():
    res = []
    for a, b, sub in comps:
        mask = None if sub is None else (EXP == sub)
        x, y = get(*a, mask=mask), get(*b, mask=mask)
        p, n01, n10 = mcnemar_exact(x, y)
        lo, hi = paired_boot_diff(x, y)
        res.append({"aile": fam, "a": f"{a[0]} | {a[1]}", "b": f"{b[0]} | {b[1]}", "alt_kume": sub or "tümü", "n": len(x),
                    "a_hit1": x.mean(), "b_hit1": y.mean(), "fark": x.mean() - y.mean(), "fark_ga_alt": lo, "fark_ga_ust": hi,
                    "a_dogru_b_yanlis": n01, "a_yanlis_b_dogru": n10, "p": p})
    adj = holm([r["p"] for r in res])
    for r, q in zip(res, adj):
        r["p_holm"] = q
    rows += res

# Sektör düzeyinde: en iyi kodlayıcı ile BDM'ler (fark kapanıyor mu?)
res = []
for m in LLM_IDS:
    x, y = get(*QWEN, level="sec"), get(f"BDM {m}", "sifir_atis/en", level="sec")
    p, n01, n10 = mcnemar_exact(x, y); lo, hi = paired_boot_diff(x, y)
    res.append({"aile": "F_sektor_duzeyi", "a": "Qwen3-Emb-4B | en/plain/ad+tanim (sektör)", "b": f"BDM {m} (sektör)", "alt_kume": "tümü", "n": len(x),
                "a_hit1": x.mean(), "b_hit1": y.mean(), "fark": x.mean() - y.mean(), "fark_ga_alt": lo, "fark_ga_ust": hi,
                "a_dogru_b_yanlis": n01, "a_yanlis_b_dogru": n10, "p": p})
for r, q in zip(res, holm([r["p"] for r in res])):
    r["p_holm"] = q
rows += res

T = pd.DataFrame(rows)
T.to_csv(f"{OUT}/anlamlilik_testleri.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 48)
print(T[["aile", "a", "b", "alt_kume", "a_hit1", "b_hit1", "fark", "fark_ga_alt", "fark_ga_ust", "p", "p_holm"]].round(4).to_string(index=False))

# ---------------------------------------------------------------- az örnekli öğrenme: test kümesinde (N = 1.472) eşleştirilmiş karşılaştırmalar
# 66 örnekli BDM istemi ile sıfır atış; SetFit (İngilizce, üç tohum) ile sıfır atış USE, TF-IDF ve Claude Haiku 5.5. Holm düzeltmesi 14 karşılaştırmanın tamamına uygulanır.
def _test_hits(name):
    d = load_skor(f"skor_sentetik_test_{name}").sort_values("id").reset_index(drop=True)
    return d.id.values, np.array([isinstance(t, str) and t in set(g.split(";")) for t, g in zip(d.top1, d.gercek)], float)
az = []
for m in LLM_IDS:
    i1, a = _test_hits(f"BDM_{m}_66_ornekli"); i0, b = _test_hits(f"BDM_{m}_sifir_atis_makale"); assert (i1 == i0).all()
    lo, hi = paired_boot_diff(a, b); p, _, _ = mcnemar_exact(a, b)
    az.append({"karsilastirma": m, "a": a.mean(), "b": b.mean(), "fark": a.mean() - b.mean(), "ga_alt": lo, "ga_ust": hi, "p": p})
_, use0 = _test_hits("USE_sifir_atis_plain"); _, tf0 = _test_hits("Temel_TF-IDF_sifir_atis_stem"); _, haiku0 = _test_hits("BDM_claude-haiku-5-5_sifir_atis_makale")
for s_ in (42, 43, 44):
    _, a = _test_hits(f"SetFit-EN_K8_tohum{s_}")
    for nm, b in (("USE", use0), ("TF-IDF", tf0), ("Haiku0", haiku0)):
        lo, hi = paired_boot_diff(a, b); p, _, _ = mcnemar_exact(a, b)
        az.append({"karsilastirma": f"SetFit-EN s{s_} vs {nm}", "a": a.mean(), "b": b.mean(), "fark": a.mean() - b.mean(), "ga_alt": lo, "ga_ust": hi, "p": p})
AZ = pd.DataFrame(az); AZ["p_holm"] = holm(AZ.p)
AZ.to_csv(f"{OUT}/az_ornekli_testler.csv", index=False, encoding="utf-8-sig")
print(AZ.round(4).to_string(index=False))

