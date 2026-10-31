# -*- coding: utf-8 -*-
"""A) Sentetik veride tüm yöntemler: madde / yakın anlam grubu / sektör düzeyinde isabet, kısmi puan,
anlatım biçimine göre isabet, ilçe doğruluğu (madde ve sektör düzeyinde). Çıktı: out2/sentetik_tum_yontemler.csv"""
from revizyon_ortak import *
import unicodedata

rows = []; HITS = {}

def add(name, cond, top1, top3=None, src=""):
    item, grp, sec = hier_hits(top1)
    pc = partial_credit(item, grp, sec)
    lo, hi = boot_ci(item)
    r = {"yontem": name, "kosul": cond, "kaynak": src, "n": len(top1), "hit1": item.mean(), "hit1_ga_alt": lo, "hit1_ga_ust": hi,
         "grup_hit1": grp.mean(), "sektor_hit1": sec.mean(), "kismi_puan": pc.mean(),
         "hatalarda_ayni_grup": float(((grp == 1) & (item == 0)).sum() / max(1, (item == 0).sum())),
         "hatalarda_ayni_sektor": float(((sec == 1) & (item == 0)).sum() / max(1, (item == 0).sum()))}
    if top3 is not None:
        r["hit3"] = float(np.mean([any(isinstance(t, str) and t in L for t in T) for T, L in zip(top3, LAB)]))
    for e in ("explicit", "mixed", "implicit"):
        m = EXP == e
        r[f"hit1_{e}"] = item[m].mean(); r[f"sektor_hit1_{e}"] = sec[m].mean()
    r["ilce_dogrulugu"], _ = district_accuracy(top1)
    sec_lab = [[SECTOR_EN[x] for x in L] for L in LAB]
    sec_top = [SECTOR_EN.get(t) if isinstance(t, str) else None for t in top1]
    r["ilce_dogrulugu_sektor"], _ = district_accuracy(sec_top, labels=sec_lab)
    rows.append(r); HITS[(name, cond)] = {"item": item, "grp": grp, "sec": sec, "top1": list(top1)}

# --- makaledeki yöntemler (makale çalıştırması)
for f in sorted(os.listdir(PRED)):
    if not f.startswith("mesaj_sentetik_"):
        continue
    tag = unicodedata.normalize("NFC", f[len("mesaj_sentetik_"):-4])
    p = pd.read_csv(os.path.join(PRED, f)).set_index("id").reindex(IDS).reset_index()
    if tag.startswith("BDM_"):
        mid = tag[4:]
        if mid == "gpt-5-mini":
            add("BDM " + mid, "makale(karisik_protokol)", p.top1.tolist(), src="makale")
            d = load_skor("skor_sentetik_BDM_gpt-5-mini_protokol_tutarli", IDS)
            add("BDM " + mid, "sifir_atis/en", d.top1.tolist(), src="revizyon")
        else:
            add("BDM " + mid, "sifir_atis/en", p.top1.tolist(), src="makale")
        continue
    model, tt = tag.rsplit("_", 1)
    model = model.replace("Temel_ ", "Temel: ").replace("SBERT-çokdilli", "SBERT çok dilli")
    top3 = [[x for x in (a, b, c) if isinstance(x, str)] for a, b, c in zip(p.top1, p.top2, p.top3)]
    add(model, f"en/{tt}/ad" if "(TR)" not in model else "tr/plain/ad", p.top1.tolist(), top3, src="makale")

# --- revizyon çalıştırması (yeni kodlayıcılar, tanımlı ihtiyaçlar, ablasyon, Türkçe BDM, tekrarlar)
for f in sorted(os.listdir(RT)):
    if not f.startswith("skor_sentetik_") or f.startswith("skor_sentetik_test_"):
        continue
    tag = unicodedata.normalize("NFC", f[len("skor_sentetik_"):-4])
    if tag.endswith("_coklu") or "protokol" in tag:
        continue
    d = load_skor(f, IDS)
    top3 = [[x for x in (a, b, c) if isinstance(x, str)] for a, b, c in zip(d.top1, d.top2, d.top3)]
    if tag.startswith("BDM_"):
        mid, cond = re.match(r"BDM_([\w.\-]+?)_(tr_tek|tekrar\d)$", tag).groups()
        add("BDM " + mid, {"tr_tek": "sifir_atis/tr"}.get(cond, cond), d.top1.tolist(), src="revizyon")
        continue
    m = re.match(r"(.+?)_(en|tr)_(plain|lemma|stem|raw)_(ad\+tanim|ad)$", tag)
    model, lang, tt, nm = m.groups()
    model = model.replace("Temel_", "Temel: ").replace("_TR_", " (TR)").replace("SBERT-çokdilli", "SBERT çok dilli")
    add(model, f"{lang}/{tt}/{nm}", d.top1.tolist(), top3, src="revizyon")

T = pd.DataFrame(rows)
T.to_csv(f"{OUT}/sentetik_tum_yontemler.csv", index=False, encoding="utf-8-sig")
import pickle
pickle.dump(HITS, open(f"{OUT}/sentetik_hits.pkl", "wb"))
pd.DataFrame([{"ihtiyac_tr": x, "ihtiyac_en": TR2EN[x], "yakin_anlam_grubu": GROUP_OF_TR[x], "sektor": ITEM_SECTOR[x]}
              for x in NL.tr]).to_csv(f"{OUT}/hiyerarsi_semasi.csv", index=False, encoding="utf-8-sig")

# Makaledeki değerlerle tutarlılık denetimi
chk = {("USE", "en/plain/ad"): 0.712, ("Temel: TF-IDF", "en/stem/ad"): 0.712, ("BDM claude-opus-5-5", "sifir_atis/en"): 0.943,
       ("Qwen3-Emb-4B", "en/plain/ad+tanim"): 0.812, ("BDM gpt-5-mini", "sifir_atis/en"): 0.9115, ("BDM claude-opus-5-5", "sifir_atis/tr"): 0.9445}
for k, v in chk.items():
    got = T[(T.yontem == k[0]) & (T.kosul == k[1])].hit1.iloc[0]
    assert abs(got - v) < 6e-4, (k, got, v)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
show = T.sort_values("hit1", ascending=False)[["yontem", "kosul", "hit1", "grup_hit1", "sektor_hit1", "kismi_puan", "hatalarda_ayni_grup",
                                               "hatalarda_ayni_sektor", "hit1_explicit", "hit1_mixed", "hit1_implicit", "ilce_dogrulugu", "ilce_dogrulugu_sektor"]]
print(show.round(3).head(60).to_string(index=False))
print(len(T), "satır")
