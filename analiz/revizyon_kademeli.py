# -*- coding: utf-8 -*-
"""C) Kademeli tasarımın sınanması: ilk kademe yerel yöntem (TF-IDF ya da Qwen3-Emb-4B + tanım),
güveni düşük paylaşımlar BDM'ye. Eşik çapraz doğrulamayla (5 kat x 20 tekrar) eğitim katlarında seçilir, test katında uygulanır.
Güven ölçütleri: en yüksek benzerlik (s1) ve ilk iki benzerlik farkı (s1 - s2)."""
from revizyon_ortak import *
import pickle, time
from sklearn.model_selection import KFold

H = pickle.load(open(f"{OUT}/sentetik_hits.pkl", "rb"))
CL = llm_cost_latency()
CL["gpt-5-mini"]["usd_paylasim"] = 0.08986625 / 1790      # makale çalıştırmasında 179 çağrı (1.790 paylaşım) için ölçülen maliyet

# ---- ilk kademeler
Ssim = tfidf_stem_sim()                                      # makaledeki TF-IDF (gövde, yalnız ad) benzerlikleri
srt = np.sort(Ssim, axis=1)[:, ::-1]
tf_top1 = [EN_LABELS[i] for i in np.argmax(Ssim, axis=1)]
assert np.mean([a == b for a, b in zip(tf_top1, H[("Temel: TF-IDF", "en/stem/ad")]["top1"])]) > 0.995
FIRST = {"TF-IDF (gövde, ad)": {"top1": H[("Temel: TF-IDF", "en/stem/ad")]["top1"], "s1": srt[:, 0], "s2": srt[:, 1]}}
for nm, f in [("TF-IDF (gövde, ad+tanım)", "skor_sentetik_Temel_TF-IDF_en_stem_ad+tanim"),
              ("Qwen3-Emb-4B (düz, ad+tanım)", "skor_sentetik_Qwen3-Emb-4B_en_plain_ad+tanim")]:
    d = load_skor(f, IDS)
    FIRST[nm] = {"top1": d.top1.tolist(), "s1": d.s1.values, "s2": d.s2.values}
for nm, v in FIRST.items():
    v["hit"] = hier_hits(v["top1"])[0]; v["marj"] = v["s1"] - v["s2"]

# İlk kademe gecikmesi (ms/paylaşım): TF-IDF bu makinede ölçülür; Qwen3-Emb-4B Colab A100 ölçümü (toplu ve tek)
from sklearn.feature_extraction.text import TfidfVectorizer
msgs = D["text_stem"].fillna("").tolist()
vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(msgs)
t0 = time.perf_counter()
for i in range(500):
    vec.transform([msgs[i]])
tf_ms_bu_makine = (time.perf_counter() - t0) / 500 * 1000
# İlk kademe süresi, sıfır atışlı yerel yöntemlerin aynı işlemcide ölçüldüğü çalıştırmadan alınır (notebooks/NLP_Afet_CPU_Gecikme.ipynb,
# runs/revizyon/results/revizyon2/cpu_gecikme.json; TF-IDF tek paylaşım süresinin ortancası). Bu dosya yoksa önceki sürümde analiz
# makinesinde ölçülen değer kullanılır. Süre donanıma bağlı olduğundan bu makinedeki ölçüm yalnızca bilgi olarak yazdırılır.
CPU_JSON = os.path.join(REVR, "revizyon2", "cpu_gecikme.json")
if os.path.exists(CPU_JSON):
    _cpu = json.load(open(CPU_JSON, encoding="utf-8"))
    TF_MS_MAKALE = float(_cpu["modeller"]["TF-IDF (gövde)"]["tek_ms_ortanca"]); _kaynak = f"aynı işlemci ölçümü ({_cpu['donanim']['islemci']})"
else:
    TF_MS_MAKALE = 0.3440349980000974; _kaynak = "önceki analiz makinesi ölçümü"
tf_ms = TF_MS_MAKALE
print(f"TF-IDF dönüşümü bu makinede {tf_ms_bu_makine:.3f} ms/paylaşım; tablolarda {_kaynak} {TF_MS_MAKALE:.3f} ms kullanılır.")
lat = pd.read_csv(os.path.join(RR, "yeni_kodlayici_gecikme.csv"))
q4 = lat[lat.Model == "Qwen3-Emb-4B"].iloc[0]
FIRST_MS = {"TF-IDF (gövde, ad)": tf_ms, "TF-IDF (gövde, ad+tanım)": tf_ms, "Qwen3-Emb-4B (düz, ad+tanım)": float(q4.tek_paylasim_ms_cuda)}

LLM_HIT = {m: hier_hits(llm_top1_paper(m))[0] for m in LLM_IDS}
LLM_TOP1 = {m: llm_top1_paper(m) for m in LLM_IDS}
GRID = np.round(np.arange(0, 1.0001, 0.01), 2)

def route_by_share(conf, share):
    if share <= 0:
        return np.zeros(len(conf), bool)
    thr = np.quantile(conf, share)
    return conf < thr

TIE = np.random.default_rng(7).random(len(IDS))
def route_by_rank(conf, share):
    """Betimsel eğri için: güveni en düşük ceil(share*N) paylaşım (eşitlikler sabit rastgele sırayla bozulur)."""
    k = int(np.ceil(share * len(conf) - 1e-9)); r = np.zeros(len(conf), bool)
    if k > 0:
        r[np.lexsort((TIE, conf))[:k]] = True
    return r

def curve(first, m, conf):
    out = []
    for q in GRID:
        r = route_by_rank(conf, q)
        out.append(np.where(r, LLM_HIT[m], first["hit"]).mean())
    return np.array(out)

# ---- betimsel eğriler (tüm veri)
rows = []
for fn, F in FIRST.items():
    for cn in ("s1", "marj"):
        for m in LLM_IDS:
            acc = curve(F, m, F[cn])
            for q, a in zip(GRID, acc):
                rows.append({"ilk_kademe": fn, "guven": cn, "bdm": m, "bdm_payi": q, "hit1": a,
                             "usd_1000": q * CL[m]["usd_paylasim"] * 1000})
CURVE = pd.DataFrame(rows); CURVE.to_csv(f"{OUT}/kademeli_egriler.csv", index=False, encoding="utf-8-sig")

# ---- çapraz doğrulama
def cv_eval(F, m, cn, policy, reps=20, k=5, delta=0.01, budget=0.30):
    conf = F[cn]; res = []; preds0 = None
    for rep in range(reps):
        kf = KFold(n_splits=k, shuffle=True, random_state=rep)
        routed = np.zeros(len(conf), bool)
        for tr, te in kf.split(conf):
            if policy == "hedef":            # eğitim katında BDM tek başına isabetinin en fazla delta altında kalan en küçük pay
                ctr = conf[tr]; llm_tr = LLM_HIT[m][tr]; f_tr = F["hit"][tr]
                target = llm_tr.mean() - delta; q_sel = 1.0
                for q in GRID:
                    r = route_by_share(ctr, q)
                    if np.where(r, llm_tr, f_tr).mean() >= target:
                        q_sel = q; break
            elif policy == "en_iyi":         # eğitim katında isabeti en yüksek pay (eşitlikte en küçük pay)
                ctr = conf[tr]; llm_tr = LLM_HIT[m][tr]; f_tr = F["hit"][tr]
                accs = [np.where(route_by_share(ctr, q), llm_tr, f_tr).mean() for q in GRID]
                q_sel = float(GRID[int(np.argmax(np.round(accs, 10)))])
            else:                            # sabit bütçe: eğitim katındaki güven dağılımının alt %budget'ı
                ctr = conf[tr]; q_sel = budget
            thr = np.quantile(ctr, q_sel) if q_sel > 0 else -np.inf
            routed[te] = conf[te] < thr
        hit = np.where(routed, LLM_HIT[m], F["hit"])
        top1 = [l if r else f for r, l, f in zip(routed, LLM_TOP1[m], F["top1"])]
        da, _ = district_accuracy(top1)
        res.append({"hit1": hit.mean(), "pay": routed.mean(), "ilce": da,
                    "ortuk_payi_yonlenen": float((EXP[routed] != "explicit").mean()) if routed.any() else np.nan,
                    "yonlenen_ortuk_kapsami": float(routed[EXP == "implicit"].mean())})
        if rep == 0:
            preds0 = (hit, routed)
    R = pd.DataFrame(res)
    hit0, routed0 = preds0
    p_llm, a, b = mcnemar_exact(hit0, LLM_HIT[m]); p_first, _, _ = mcnemar_exact(hit0, F["hit"])
    lo, hi = paired_boot_diff(hit0, LLM_HIT[m])
    return {"ilk_kademe": F["name"], "guven": cn, "bdm": m, "politika": policy,
            "hit1_ort": R.hit1.mean(), "hit1_ss": R.hit1.std(), "hit1_min": R.hit1.min(), "hit1_max": R.hit1.max(),
            "bdm_payi_ort": R.pay.mean(), "bdm_payi_min": R.pay.min(), "bdm_payi_max": R.pay.max(),
            "ilce_dogrulugu_ort": R.ilce.mean(), "yonlenenlerde_acik_olmayan_payi": R.ortuk_payi_yonlenen.mean(),
            "ortuklerin_yonlenen_payi": R.yonlenen_ortuk_kapsami.mean(),
            "usd_1000": R.pay.mean() * CL[m]["usd_paylasim"] * 1000, "bdm_tek_usd_1000": CL[m]["usd_paylasim"] * 1000,
            "ms_paylasim": FIRST_MS[F["name"]] + R.pay.mean() * CL[m]["ms_tekrar"], "bdm_tek_ms": CL[m]["ms_tekrar"],
            "bdm_tek_hit1": LLM_HIT[m].mean(), "ilk_kademe_hit1": F["hit"].mean(),
            "fark_bdm_tek": hit0.mean() - LLM_HIT[m].mean(), "fark_ga_alt": lo, "fark_ga_ust": hi,
            "p_bdm_tek": p_llm, "p_ilk_kademe": p_first}

for fn, F in FIRST.items():
    F["name"] = fn
cv_rows = []
for fn, F in FIRST.items():
    for cn in ("s1", "marj"):
        for m in LLM_IDS:
            for pol in ("hedef", "butce30", "en_iyi"):
                cv_rows.append(cv_eval(F, m, cn, pol))
CV = pd.DataFrame(cv_rows)
CV.to_csv(f"{OUT}/kademeli_capraz_dogrulama.csv", index=False, encoding="utf-8-sig")

# ---- kehanet karşılaştırması: açık olmayan (karma + örtük) paylaşımların tümü BDM'ye (uygulanamaz üst sınır)
orc = []
for fn, F in FIRST.items():
    r = EXP != "explicit"
    for m in LLM_IDS:
        hit = np.where(r, LLM_HIT[m], F["hit"])
        orc.append({"ilk_kademe": fn, "bdm": m, "bdm_payi": r.mean(), "hit1": hit.mean(), "bdm_tek_hit1": LLM_HIT[m].mean(),
                    "usd_1000": r.mean() * CL[m]["usd_paylasim"] * 1000})
ORC = pd.DataFrame(orc); ORC.to_csv(f"{OUT}/kademeli_anlatim_kehaneti.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 30)
print("TF-IDF ms/paylaşım:", round(tf_ms, 3), "| Qwen3-4B tek (CUDA):", FIRST_MS["Qwen3-Emb-4B (düz, ad+tanım)"])
print(CV[["ilk_kademe", "guven", "bdm", "politika", "hit1_ort", "hit1_ss", "bdm_payi_ort", "ilce_dogrulugu_ort", "usd_1000", "bdm_tek_usd_1000",
          "ms_paylasim", "bdm_tek_hit1", "fark_bdm_tek", "fark_ga_alt", "fark_ga_ust", "p_bdm_tek", "yonlenenlerde_acik_olmayan_payi", "ortuklerin_yonlenen_payi"]].round(4).to_string(index=False))
print(ORC.round(4).to_string(index=False))
json.dump({k: {kk: (vv if not isinstance(vv, (np.floating,)) else float(vv)) for kk, vv in v.items()} for k, v in CL.items()},
          open(f"{OUT}/bdm_maliyet_gecikme.json", "w"), indent=1)
