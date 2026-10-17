# -*- coding: utf-8 -*-
"""Makale şekilleri (Türkçe etiket, ondalık virgül, Times uyumlu Liberation Serif, siyah metin, 300 dpi, şekil içinde başlık yok).
Şekiller makaleye yerleştirildikleri genişlikte çizilir: büyük şekiller 15 cm (WL), küçük şekiller 8,5 cm (WS)."""
import os, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from makale_verisi import MAIN, EXTRA, TURK, BASE, LLM_ORDER, LLM_SHORT, llm_short, TT_TR, MODEL_LABEL, f2

plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 9, "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "legend.fontsize": 9, "mathtext.fontset": "stix", "text.color": "black", "axes.labelcolor": "black",
                     "xtick.color": "black", "ytick.color": "black", "axes.spines.top": False, "axes.spines.right": False})
WL, WS = 15 / 2.54, 8.5 / 2.54   # inç: büyük ve küçük şekil genişliği
TRD = FuncFormatter(lambda v, pos: f"{v:.2f}".replace(".", ","))
PAL = {"plain": "#0072B2", "lemma": "#E69F00", "stem": "#009E73"}
GROUP_COL = {"main": "#0072B2", "extra": "#56B4E9", "turk": "#009E73", "base": "#999999", "llm": "#D55E00"}
DISTRICT_COORDS = {
    "Adalar": (40.8741659, 29.1293251), "Arnavutköy": (41.184471, 28.7412446), "Ataşehir": (40.9929379, 29.1135187),
    "Avcılar": (40.9799389, 28.7216689), "Bağcılar": (41.034547, 28.8567558), "Bahçelievler": (41.0030495, 28.8657701),
    "Bakırköy": (40.9782585, 28.8744461), "Başakşehir": (41.1075794, 28.7950711), "Bayrampaşa": (41.0345549, 28.9118417),
    "Beşiktaş": (41.0428465, 29.0075283), "Beykoz": (41.1343001, 29.0920378), "Beylikdüzü": (41.0037541, 28.6372583),
    "Beyoğlu": (41.0284233, 28.9736808), "Büyükçekmece": (41.021654, 28.579757), "Çatalca": (41.1436804, 28.4605154),
    "Çekmeköy": (41.0351579, 29.1739149), "Esenler": (41.0376175, 28.8824519), "Esenyurt": (41.0342862, 28.6801113),
    "Eyüpsultan": (41.0478358, 28.9327383), "Fatih": (41.0192846, 28.9479296), "Gaziosmanpaşa": (41.0578305, 28.9122452),
    "Güngören": (41.0252832, 28.8726498), "Kadıköy": (40.9912955, 29.0245631), "Kağıthane": (41.0796544, 28.9731198),
    "Kartal": (40.88858, 29.1856536), "Küçükçekmece": (40.9918737, 28.7711956), "Maltepe": (40.9247539, 29.1310782),
    "Pendik": (40.8768715, 29.2349672), "Sancaktepe": (40.9905196, 29.2288624), "Sarıyer": (41.1685803, 29.0572623),
    "Silivri": (41.0742476, 28.2481709), "Sultanbeyli": (40.9670242, 29.2671314), "Sultangazi": (41.1043344, 28.8614367),
    "Şile": (41.1744067, 29.6125216), "Şişli": (41.0637891, 28.9831642), "Tuzla": (40.8161732, 29.3034194),
    "Ümraniye": (41.0256362, 29.0963049), "Üsküdar": (41.0265498, 29.0151321), "Zeytinburnu": (40.9898653, 28.9037467),
}

def group_of(m):
    if m in MAIN: return "main"
    if m in EXTRA: return "extra"
    if m in TURK: return "turk"
    return "base"

def fig_hit1(D, out):
    syn = D["syn"]; models = [m for m in MAIN + EXTRA if m in syn.model.values]
    x = np.arange(len(models)); w = 0.26
    fig, ax = plt.subplots(figsize=(WL, 3.6))
    for k, tt in enumerate(["plain", "lemma", "stem"]):
        s = syn[syn.metin == tt].set_index("model").reindex(models)
        err = np.vstack([(s.hit1 - s.hit1_ga_alt).clip(lower=0), (s.hit1_ga_ust - s.hit1).clip(lower=0)])
        ax.bar(x + (k - 1) * w, s.hit1, w, yerr=err, capsize=1.5, color=PAL[tt], label=TT_TR[tt], error_kw={"lw": 0.6})
    tf = D["best_tfidf"]; op = D["llm"]["sentetik"].set_index("model").loc["claude-opus-5-5"]
    ax.axhline(tf.hit1, color="#555555", ls="--", lw=0.9, label=f"TF-IDF ({TT_TR[tf.metin].lower()}) {f2(tf.hit1)}")
    ax.axhline(op.hit1, color="#D55E00", ls=":", lw=1.1, label=f"Claude Opus 5.5 {f2(op.hit1)}")
    rnd = syn[syn.model == "Temel: Rastgele"].hit1.iloc[0]
    ax.axhline(rnd, color="#999999", ls="-.", lw=0.8, label=f"Rastgele {f2(rnd)}")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha="right"); ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(TRD); ax.set_ylabel("Mesaj düzeyi isabet (hit@1)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.36))
    fig.set_size_inches(WL, 4.0); fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)

def fig_all(D, out):
    syn = D["syn"]; rows = []
    for m in MAIN + EXTRA + TURK + BASE:
        r = D["best"](syn, m)
        if r is None: continue
        lab = MODEL_LABEL.get(m, m) + (f"\n({TT_TR[r.metin].lower()})" if m not in ("Temel: Rastgele", "Temel: Çoğunluk") else "")
        rows.append((lab, r.hit1, r.hit1_ga_alt, r.hit1_ga_ust, GROUP_COL[group_of(m)]))
    _L = D["llm"]["sentetik"].set_index("model")
    for m in D["llm_models"]:
        r = _L.loc[m]
        rows.append((llm_short(m) + "\n(BDM)", r.hit1, r.hit1_ga_alt, r.hit1_ga_ust, GROUP_COL["llm"]))
    fig, ax = plt.subplots(figsize=(WL, 4.2))
    vals = np.array([r[1] for r in rows]); lo = np.array([r[2] for r in rows]); hi = np.array([r[3] for r in rows])
    ax.bar(range(len(rows)), vals, yerr=np.vstack([np.clip(vals - lo, 0, None), np.clip(hi - vals, 0, None)]), capsize=1.5,
           color=[r[4] for r in rows], error_kw={"lw": 0.6})
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f2(v), ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], fontsize=8.5, rotation=90)
    ax.set_ylim(0, 1.15); ax.yaxis.set_major_formatter(TRD); ax.set_ylabel("hit@1 (en iyi metin türü)"); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COL[g]) for g in ("main", "extra", "turk", "base", "llm")]
    ax.legend(handles, ["Özgün yedi model", "Ek modeller", "Türkçe modeller", "Temel çizgiler", "BDM'ler"], frameon=False, ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)

def fig_explicit(D, out, models=("BERT", "RoBERTa", "GloVe", "Word2Vec", "FastText", "USE", "SBERT-mpnet", "ALBERT-paraphrase", "SBERT-çokdilli (TR)", "Temel: TF-IDF")):
    syn = D["syn"]; rows = []
    for m in models:
        r = D["best"](syn, m)
        if r is None: continue
        tag = f"{m}_{r.metin}"
        if tag in D["expl_by_tag"]:
            rows.append((MODEL_LABEL.get(m, m) + f" ({TT_TR[r.metin].lower()})", D["expl_by_tag"][tag]))
    for mname in D["llm_models"]:
        tag = f"BDM_{mname}"
        if tag in D["expl_by_tag"]:
            rows.append((llm_short(mname) + " (BDM)", D["expl_by_tag"][tag]))
    x = np.arange(len(rows)); w = 0.27
    fig, ax = plt.subplots(figsize=(WL, 3.9))
    for k, (g, lab, col) in enumerate((("explicit", "Açık", "#0072B2"), ("mixed", "Karma", "#E69F00"), ("implicit", "Örtük", "#009E73"))):
        ax.bar(x + (k - 1) * w, [r[1][g] for r in rows], w, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=40, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.yaxis.set_major_formatter(TRD); ax.set_ylabel("hit@1")
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.04))
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    return rows

def load_district_polys(run_dir):
    """Çalıştırma klasöründe ilçe sınırları (data/geo/istanbul_ilceler.geojson) varsa {ilçe: [halka dizileri]} verir."""
    import json, glob as _glob
    cands = _glob.glob(os.path.join(run_dir, "data", "geo", "*.geojson")) + _glob.glob(os.path.join(run_dir, "**", "istanbul_ilceler.geojson"), recursive=True)
    for p in cands:
        try:
            gj = json.load(open(p, encoding="utf-8"))
            polys = {}
            for f in gj["features"]:
                g = f["geometry"]; rings = g["coordinates"] if g["type"] == "Polygon" else [r for poly in g["coordinates"] for r in poly]
                polys[f["properties"].get("ilce") or f["properties"].get("shapeName")] = [np.array(r)[:, :2] for r in rings if len(r) >= 3]
            if len(polys) >= 30:
                return polys, gj.get("source", "")
        except Exception:
            continue
    return None, ""

def fig_maps(D, out, panels):
    """2x2 harita: ilçe rengi = ilçe ölçütüne göre doğru (yeşil) / yanlış (turuncu).
    Sınır verisi varsa ilçe poligonları doldurulur; yoksa işaret boyutu = ilçe içi hit@1 olan nokta işaretleri çizilir."""
    from matplotlib.patches import Polygon as MplPolygon
    pred_dir = D["pred_dir"]
    polys, geo_src = load_district_polys(D["run_dir"])
    D["map_mode"] = "poligon" if polys else "nokta"; D["map_source"] = geo_src
    OK, BAD = "#2E9E6B", "#E07B39"
    fig, axes = plt.subplots(2, 2, figsize=(WL, 4.3 if polys else 3.6))
    axes = axes.reshape(-1)
    for ax, (title, fname) in zip(axes, panels):
        df = pd.read_csv(os.path.join(pred_dir, fname)).set_index("İlçe")
        for d, (lat, lon) in DISTRICT_COORDS.items():
            has = d in df.index
            ok = bool(df.loc[d, "dogru_makale_olcutu"] == 1) if has else None
            col = (OK if ok else BAD) if has else "#F2F2F2"
            if polys and d in polys:
                area = 0.0
                for ring in polys[d]:
                    ax.add_patch(MplPolygon(ring, closed=True, facecolor=col, edgecolor="#555555", linewidth=0.3, zorder=2))
                    x, y = ring[:, 0], ring[:, 1]; area += 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            elif polys:   # sınırı bulunamayan ilçe (ör. Adalar): nokta işareti
                ax.scatter(lon, lat, s=18, color=col, edgecolor="#555555", linewidth=0.3, zorder=3)
            else:
                h = df.loc[d, "hit1_ort"] if has else float("nan")
                ax.scatter(lon, lat, s=(16 + 60 * (0 if pd.isna(h) else h)) if has else 10, color=col, edgecolor="black" if has else "grey",
                           linewidth=0.35, zorder=3, alpha=0.95)
                if (has and not ok) or d in ("Silivri", "Çatalca", "Şile", "Adalar", "Beykoz", "Sarıyer", "Tuzla", "Arnavutköy"):
                    ax.annotate(d, (lon, lat), fontsize=3.4, xytext=(2, 1.5), textcoords="offset points", zorder=4,
                                color="#222222" if (has and not ok) else "#777777")
        acc = df["dogru_makale_olcutu"].mean()
        ax.set_title(f"{title} (ilçe doğruluğu {f2(acc)})", fontsize=9, pad=3)
        if polys:
            allpts = np.vstack([r for rings in polys.values() for r in rings])
            ax.set_xlim(allpts[:, 0].min() - 0.02, allpts[:, 0].max() + 0.02); ax.set_ylim(min(allpts[:, 1].min(), 40.84) - 0.02, allpts[:, 1].max() + 0.02)
        else:
            ax.set_xlim(28.15, 29.72); ax.set_ylim(40.76, 41.26)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(0.5)
        ax.set_aspect(1 / math.cos(math.radians(41)))
    if polys:
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=OK, edgecolor="#555555", linewidth=0.3, label="En sık tahmin, en sık gerçek ihtiyaçla örtüşüyor"),
                   plt.Rectangle((0, 0), 1, 1, facecolor=BAD, edgecolor="#555555", linewidth=0.3, label="Örtüşmüyor")]
    else:
        handles = [plt.Line2D([], [], marker="o", color=OK, linestyle="", markersize=6, markeredgecolor="black", markeredgewidth=0.3, label="En sık tahmin, en sık gerçek ihtiyaçla örtüşüyor"),
                   plt.Line2D([], [], marker="o", color=BAD, linestyle="", markersize=6, markeredgecolor="black", markeredgewidth=0.3, label="Örtüşmüyor (ilçe adı yazılı)"),
                   plt.Line2D([], [], marker="o", color="white", linestyle="", markersize=4, markeredgecolor="black", markeredgewidth=0.3, label="İşaret boyutu: ilçe içi hit@1")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0), handletextpad=0.4, columnspacing=1.2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.10, wspace=0.04, hspace=0.24)
    fig.savefig(out, dpi=300); plt.close(fig)

def fig_framework(D, out):
    """Çalışmanın genel akış şeması (matplotlib; tüm ölçüler punto, metin kutuya göre sarılır)."""
    import textwrap
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    n = f"{D['n']:,}".replace(",", ".")
    llm_names = [llm_short(m) for m in D["llm_models"]]
    llms = ", ".join(nm.replace(" ", "\u00a0") for nm in llm_names[:5]) + (" ve diğerleri" if len(llm_names) > 5 else "")   # model adları satır sonunda bölünmez
    FS, LH, PAD, CW = 9.0, 11.8, 6.0, 4.35   # punto: yazı, satır yüksekliği, iç boşluk, ortalama karakter genişliği
    W = WL * 72                               # 15 cm
    GAP = 16.0                                # kutular arası düşey boşluk (ok payı)
    BW = (W - 22) / 2                          # iki sütunlu satırlarda kutu genişliği
    def wrap(text, w):
        """Metni kutu genişliğine göre sarar; başlık (ilk parça) kalın yazıldığı için daha dar sarılır.
        Dönüş: satırlar ve başlığa ait satır sayısı (başlık sarılsa da tüm satırları kalın yazılır)."""
        cap = max(12, int((w - 2 * PAD) / CW))
        lines, ntitle = [], 1
        for k, part in enumerate(text.split("\n")):
            wl = textwrap.wrap(part, int(cap / 1.08) if k == 0 else cap) or [""]
            if k == 0:
                ntitle = len(wl)
            lines.extend(wl)
        return lines, ntitle
    rows = [
        [dict(x=0, w=W, fc="#D6E2F0", bold=True, text=f"Etiket öncelikli veri üretimi (Claude Opus 5.5)\nilçe + 1-3 ihtiyaç + anlatım özellikleri (açık / karma / örtük, persona, kayıt dili, yazım hatası) → Türkçe ve İngilizce paylaşım")],
        [dict(x=0, w=W, fc="#D6E2F0", bold=True, text=f"Otomatik doğrulama ve tekrar eleme → bağımsız model denetimi (Claude Sonnet 5.5)\n{n} paylaşım, {D['n_district']} ilçe, {len(D['needs'])} maddelik ihtiyaç listesi")],
        [dict(x=0, w=BW, fc="#E8EEF7", bold=False, text="Ön işleme (İngilizce sürüm)\ndüz / lemmalanmış / gövdelenmiş metin; aynı işlem ihtiyaç adlarına da uygulanır"),
         dict(x=W - BW, w=BW, fc="#E8EEF7", bold=False, text=f"Büyük dil modeli değerlendirmesi (sıfır\u00a0atış)\n{llms}\nJSON şemasıyla listeden tek ihtiyaç")],
        [dict(x=0, w=BW, fc="#E8EEF7", bold=False, text="12 dil temsil modeli + 3 temel çizgi\nBERT, ALBERT, RoBERTa, GloVe, Word2Vec, USE, FastText; SBERT-mpnet, BERT-raw, ALBERT-paraphrase; çok dilli SBERT, BERTurk-STS (Türkçe metin); TF-IDF, rastgele, çoğunluk → kosinüs benzerliği, top-1"),
         None],
        [dict(x=0, w=BW, fc="#E2F0E8", bold=False, text="Ölçütler\nhit@1, hit@3, ilçe doğruluğu; %95 önyükleme güven aralıkları; McNemar + Holm; açık / örtük anlatım ayrımı; gecikme (CPU, GPU, API) ve maliyet"),
         dict(x=W - BW, w=BW, fc="#E2F0E8", bold=False, text="İlçe düzeyinde ihtiyaç haritaları\nher ilçe için en sık tespit edilen ihtiyaç; gerçek dağılımla karşılaştırma; karar destek çıktısı")],
    ]
    # yükseklikler
    for row in rows:
        hmax = 0
        for b in row:
            if b is None:
                continue
            b["lines"], b["nt"] = wrap(b["text"], b["w"]); b["h"] = len(b["lines"]) * LH + 2 * PAD; hmax = max(hmax, b["h"])
        for b in row:
            if b is not None:
                b["h"] = hmax
        row_h = hmax
        for b in row:
            if b is not None:
                b["row_h"] = row_h
    H = sum(next(b for b in row if b is not None)["row_h"] for row in rows) + GAP * (len(rows) - 1)
    fig = plt.figure(figsize=(W / 72, H / 72)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    def draw(b, y):
        ax.add_patch(FancyBboxPatch((b["x"] + 1, y + 1), b["w"] - 2, b["h"] - 2, boxstyle="round,pad=0,rounding_size=5", fc=b["fc"], ec="#1F4E79", lw=0.8))
        nl = len(b["lines"]); ytxt = y + b["h"] / 2 + (nl - 1) * LH / 2   # ilk satırın merkezi
        for i, ln in enumerate(b["lines"]):
            ax.text(b["x"] + b["w"] / 2, ytxt - i * LH, ln, ha="center", va="center", fontsize=FS,
                    fontweight="bold" if i < b["nt"] else "normal", color="black")
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=0.8, color="#1F4E79", shrinkA=0, shrinkB=0))
    ytop = H; ys = []
    for row in rows:
        rh = next(b for b in row if b is not None)["row_h"]; y = ytop - rh; ys.append((y, rh))
        for b in row:
            if b is not None:
                draw(b, y)
        ytop = y - GAP
    # oklar
    (y0, h0), (y1, h1), (y2, h2), (y3, h3), (y4, h4) = ys
    arrow(W / 2, y0, W / 2, y1 + h1)                       # üretim → doğrulama
    arrow(W / 2 - 60, y1, BW / 2 + 20, y2 + h2)            # doğrulama → ön işleme
    arrow(W / 2 + 60, y1, W - BW / 2 - 20, y2 + h2)        # doğrulama → BDM
    arrow(BW / 2, y2, BW / 2, y3 + h3)                     # ön işleme → modeller
    arrow(BW / 2, y3, BW / 2, y4 + h4)                     # modeller → ölçütler
    arrow(W - BW / 2, y2, W - BW / 2, y4 + h4)             # BDM → haritalar
    arrow(BW, y4 + h4 / 2, W - BW, y4 + h4 / 2)            # ölçütler → haritalar
    fig.savefig(out, dpi=300); plt.close(fig)

NEED_DISPLAY_TR_FIG = {"jenaratör": "jeneratör", "glutesiz yiyecek": "glutensiz yiyecek", "konteynır": "konteyner"}

def fig_cosine(D, out):
    """Şekil 2: (a) kosinüs benzerliği şeması; (b) veri setinden bir paylaşım için gerçek model puanlarıyla ilk beş ihtiyaç (varsa).
    Dönen: (b) için kullanılan örnek sözlüğü ya da None."""
    import textwrap
    ex = D.get("fig2_example")
    has_b = bool(ex and ex.get("top") and ex.get("tutarli") is True)   # boru hattıyla tutarlı değilse yalnız (a) paneli
    fig, axes = plt.subplots(1, 2 if has_b else 1, figsize=(WL, 3.2) if has_b else (WS, 3.0), gridspec_kw={"width_ratios": [1, 1.25]} if has_b else None)
    ax = axes[0] if has_b else axes
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.annotate("", xy=(0.82, 0.48), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#1F4E79", lw=1.6))
    ax.annotate("", xy=(0.52, 0.86), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=1.6))
    ax.text(0.84, 0.48, "A (paylaşım\ngömmesi)", fontsize=9, color="black", va="center")
    ax.text(0.56, 0.86, "B (ihtiyaç\ngömmesi)", fontsize=9, color="black", ha="left", va="center")
    from matplotlib.patches import Arc
    import math as _m
    a1 = _m.degrees(_m.atan2(0.48, 0.82)); a2 = _m.degrees(_m.atan2(0.86, 0.52))
    ax.add_patch(Arc((0, 0), 0.42, 0.42, theta1=a1, theta2=a2, color="black", lw=0.9))
    ax.text(0.2 * _m.cos(_m.radians((a1 + a2) / 2)) + 0.03, 0.2 * _m.sin(_m.radians((a1 + a2) / 2)) + 0.03, "θ", fontsize=10)
    ax.text(0.02, 0.98, "sim(A,B) = cos(θ)", fontsize=10, style="italic", va="top")
    ax.text(0.98, 0.04, "Küçük açı → yüksek benzerlik", fontsize=9, ha="right", va="bottom")
    if has_b:
        ax.set_title("(a) Gömme uzayında kosinüs benzerliği", fontsize=9)
        bx = axes[1]
        top = ex["top"][:5]
        names = [NEED_DISPLAY_TR_FIG.get(x["tr"], x["tr"]) for x in top][::-1]; vals = [float(x["skor"]) for x in top][::-1]
        cols = ["#B7CDE3"] * (len(vals) - 1) + ["#1F4E79"]
        bx.barh(range(len(vals)), vals, color=cols, height=0.6)
        for i_, v_ in enumerate(vals):
            bx.text(v_ + 0.01, i_, f2(v_), va="center", fontsize=9)
        bx.set_yticks(range(len(vals))); bx.set_yticklabels(names, fontsize=9)
        bx.set_xlim(0, 1); bx.xaxis.set_major_formatter(TRD); bx.tick_params(axis="x", labelsize=7)
        bx.set_xlabel("Kosinüs benzerliği", fontsize=9)
        quote = "\n".join(textwrap.wrap("“" + ex["metin_en"] + "”", 40))
        bx.set_title(f"(b) Örnek: {ex['ilce']} paylaşımı ({ex['model_label']})\n{quote}", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)
    return ex if has_b else None

def fig_transformer(out):
    """Şekil 3: Transformer kodlayıcı ve kod çözücü yığınlarının yazar tarafından çizilen şeması (Vaswani vd. 2017'de tanımlanan yapıya göre)."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    FS, BW_, BH, G, HD = 9, 160, 19, 7, 20          # punto, kutu genişliği/yüksekliği, kutu arası boşluk, yığın başlığı yüksekliği
    W_ = WL                                          # 15 cm
    WP = W_ * 72
    xl = 20; xr = WP - 20 - BW_
    AN = "Artık bağlantı + katman normalizasyonu"
    enc = ["Çok başlı öz-dikkat", AN, "Konuma göre ileri beslemeli ağ", AN]
    dec = ["Maskeli çok başlı öz-dikkat", AN, "Kodlayıcı-kod çözücü dikkati", AN, "Konuma göre ileri beslemeli ağ", AN]
    y_in, h_in = 6, BH + 8
    y0 = y_in + h_in + 22
    stack_h = lambda n: n * (BH + G) - G
    dec_top = y0 + stack_h(len(dec)); enc_top = y0 + stack_h(len(enc))
    yo = dec_top + 8 + HD + 18
    H_ = (yo + BH + 6 + 14) / 72
    fig = plt.figure(figsize=(W_, H_)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, WP); ax.set_ylim(0, H_ * 72); ax.axis("off")
    def box(x, y, w, h, text, fc, ec="#333333", lw=0.7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=4", fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=FS, color="black")
    def arr(x1, y1, x2, y2, rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=0.8, color="#333333",
                                     shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))
    def stack(x, items, label, fc, ec):
        top = y0 + stack_h(len(items))
        ax.add_patch(FancyBboxPatch((x - 8, y0 - 8), BW_ + 16, stack_h(len(items)) + 16 + HD, boxstyle="round,pad=0,rounding_size=6", fc=fc, ec=ec, lw=0.9))
        for k, t in enumerate(items):
            box(x, y0 + k * (BH + G), BW_, BH, t, "white")
            if k:
                arr(x + BW_ / 2, y0 + k * (BH + G) - G, x + BW_ / 2, y0 + k * (BH + G))
        ax.text(x + BW_ / 2, top + 8 + HD / 2 - 4, label, ha="center", va="center", fontsize=FS, fontweight="bold", color="black")
        return top + 8 + HD                            # kabın üst kenarı
    box(xl, y_in, BW_, h_in, "Girdi gömmesi +\nkonum kodlaması", "#F2F2F2")
    box(xr, y_in, BW_, h_in, "Çıktı gömmesi (bir konum kaydırılmış)\n+ konum kodlaması", "#F2F2F2")
    enc_box_top = stack(xl, enc, "Kodlayıcı katmanı (× N)", "#DCE6F2", "#1F4E79")
    dec_box_top = stack(xr, dec, "Kod çözücü katmanı (× N)", "#E3F1E8", "#2E7D4F")
    arr(xl + BW_ / 2, y_in + h_in, xl + BW_ / 2, y0 - 8)
    arr(xr + BW_ / 2, y_in + h_in, xr + BW_ / 2, y0 - 8)
    # kodlayıcı çıktısı -> kodlayıcı-kod çözücü dikkati
    yc = y0 + 2 * (BH + G) + BH / 2
    arr(xl + BW_ + 8, enc_top - BH / 2, xr - 8, yc, rad=-0.3)
    ax.text((xl + BW_ + 8 + xr - 8) / 2, enc_top + 2, "kodlayıcı\nçıktısı", ha="center", va="bottom", fontsize=FS, color="black")
    # çıkış
    box(xr, yo, BW_, BH, "Doğrusal katman + softmax", "#F2F2F2")
    arr(xr + BW_ / 2, dec_box_top, xr + BW_ / 2, yo)
    ax.text(xr + BW_ / 2, yo + BH + 4, "Çıktı olasılıkları", ha="center", va="bottom", fontsize=FS, color="black")
    # BERT notu
    ax.text(xl + BW_ / 2, enc_box_top + 10, "BERT yalnızca kodlayıcı\nyığınını kullanır", ha="center", va="bottom", fontsize=FS, style="italic", color="black")
    fig.savefig(out, dpi=300); plt.close(fig)

def make_all(D, outdir):
    os.makedirs(outdir, exist_ok=True)
    fig_framework(D, os.path.join(outdir, "fig_framework.png"))
    fig_transformer(os.path.join(outdir, "fig_transformer.png"))
    D["fig2_used"] = fig_cosine(D, os.path.join(outdir, "fig_cosine.png"))
    fig_hit1(D, os.path.join(outdir, "fig_hit1.png"))
    fig_all(D, os.path.join(outdir, "fig_all.png"))
    expl_rows = fig_explicit(D, os.path.join(outdir, "fig_explicit.png"))
    panels = []
    for m in (D["best_local"].model, "SBERT-mpnet", "Word2Vec"):
        r = D["best"](D["syn"], m)
        panels.append((f"{m} / {TT_TR[r.metin]}", f"ilce_sentetik_{m}_{r.metin}.csv"))
    panels.append(("Claude Opus 5.5 (BDM)", "ilce_sentetik_BDM_claude-opus-5-5.csv"))
    fig_maps(D, os.path.join(outdir, "fig_maps.png"), panels)
    return expl_rows, panels

if __name__ == "__main__":
    import sys
    from makale_verisi import load
    D = load(sys.argv[1], sys.argv[3] if len(sys.argv) > 3 else None); make_all(D, sys.argv[2]); print("ok")
