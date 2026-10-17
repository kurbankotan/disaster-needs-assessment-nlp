# -*- coding: utf-8 -*-
"""Şekillerin Carlito yazı tipli düzenini ve İngilizce/Türkçe grafiksel özetleri depodaki çalıştırma çıktılarından yeniden üretir.

Üretilen dosyalar: genel çerçeve, hit@1, genel karşılaştırma ve anlatım biçimi şekilleri, ayrı paneller hâlinde dört ilçe
haritası ve İngilizce/Türkçe grafiksel özetler. API anahtarı ya da model indirmesi gerekmez; yalnızca runs/ altındaki tablolar,
tahmin dosyaları ve ilçe sınırları okunur. Yazılar Calibri ile aynı ölçülere sahip, açık lisanslı Carlito yazı tipindedir;
sistemde yoksa matplotlib başka bir yazı tipine geçer (sayılar değişmez, yalnızca görünüm değişir).

Kullanım (depo kök klasöründen):
    python analiz/sekiller_carlito.py                 # şekiller: makale_sekilleri_carlito_yeniden/
    python analiz/sekiller_carlito.py <çıktı_klasörü>
"""
import os, sys, math, shutil, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import numpy as np, pandas as pd                      # noqa: E402
import matplotlib                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import Polygon as MplPolygon, FancyBboxPatch, FancyArrowPatch   # noqa: E402
from makale_verisi import load, f2                    # noqa: E402
from makale_sekilleri import make_all, DISTRICT_COORDS, load_district_polys            # noqa: E402

RUN = os.path.join(ROOT, "runs", "20261016_1116")                     # makalede raporlanan çalıştırma
REF = os.path.join(ROOT, "runs", "20261016_0743_claude_maliyet")      # Claude çağrılarının gerçek maliyetleri
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "makale_sekilleri_carlito_yeniden")
FONT = "Carlito"

OK, BAD = "#2E9E6B", "#E07B39"
CM = 1 / 2.54

# ------------------------------------------------------------------ ilçe haritaları (her panel ayrı dosya)
def _draw_map(ax, D, fname, polys, lw=0.3):
    df = pd.read_csv(os.path.join(D["pred_dir"], fname)).set_index("İlçe")
    for d, (lat, lon) in DISTRICT_COORDS.items():
        has = d in df.index
        ok = bool(df.loc[d, "dogru_makale_olcutu"] == 1) if has else None
        col = (OK if ok else BAD) if has else "#F2F2F2"
        if polys and d in polys:
            for ring in polys[d]:
                ax.add_patch(MplPolygon(ring, closed=True, facecolor=col, edgecolor="#555555", linewidth=lw, zorder=2))
        else:
            ax.scatter(lon, lat, s=14, color=col, edgecolor="#555555", linewidth=lw, zorder=3)
    allpts = np.vstack([r for rings in polys.values() for r in rings])
    ax.set_xlim(allpts[:, 0].min() - 0.02, allpts[:, 0].max() + 0.02)
    ax.set_ylim(min(allpts[:, 1].min(), 40.84) - 0.02, allpts[:, 1].max() + 0.02)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5)
    ax.set_aspect(1 / math.cos(math.radians(41)))
    return float(df["dogru_makale_olcutu"].mean())

def map_panels(D, outdir, panels, width_cm=7.2):
    """Her paneli ayrı PNG olarak çizer (başlıksız; model adı ve ilçe doğruluğu şekil yazısında verilir)."""
    polys, _ = load_district_polys(D["run_dir"])
    assert polys, "ilçe sınırları bulunamadı"
    out = []
    for k, (title, fname) in enumerate(panels):
        fig = plt.figure(figsize=(width_cm * CM, width_cm * CM * 0.62))
        ax = fig.add_axes([0.005, 0.01, 0.99, 0.98])
        acc = _draw_map(ax, D, fname, polys)
        p = os.path.join(outdir, "ilce_haritasi_" + fname[len("ilce_sentetik_"):-len(".csv")] + ".png")
        fig.savefig(p, dpi=400); plt.close(fig)
        out.append((title, p, acc))
    return out

# ------------------------------------------------------------------ grafiksel özet
TXT = {
    "en": dict(posts="Label-first\nsimulated posts", posts2="{n} posts · {nd} districts\n66-item need list\nTurkish / English",
               emb="12 language representation\nmodels + TF-IDF", emb2="cosine similarity,\ntop-scoring need",
               llm="5 large language models", llm2="zero-shot; answer\nlimited to the list",
               map="District-level\nneed map", mapnote="USE: district accuracy {acc}",
               chart="Message-level hit@1 by phrasing", expl="Explicit", impl="Implicit",
               overall="Overall hit@1: USE {use} · TF-IDF {tf} · LLMs {lo}–{hi}"),
    "tr": dict(posts="Etiket öncelikli\nsimüle paylaşımlar", posts2="{n} paylaşım · {nd} ilçe\n66 maddelik ihtiyaç listesi\nTürkçe / İngilizce",
               emb="12 dil temsil modeli\n+ TF-IDF", emb2="kosinüs benzerliği,\nen yüksek puanlı ihtiyaç",
               llm="5 büyük dil modeli", llm2="sıfır atış; yanıt\nlisteyle sınırlı",
               map="İlçe düzeyinde\nihtiyaç haritası", mapnote="USE: ilçe doğruluğu {acc}",
               chart="Anlatım biçimine göre mesaj düzeyi isabet (hit@1)", expl="Açık", impl="Örtük",
               overall="Genel hit@1: USE {use} · TF-IDF {tf} · BDM’ler {lo}–{hi}"),
}
GROUPS = ("TF-IDF", "USE", "SBERT-mpnet", "Claude Opus 5.5", "Gemini 3.8 Flash")

def graphical_abstract(D, lang, out, width_cm=10.0, height_cm=7.8):
    T = TXT[lang]
    dec = (lambda x: f2(x)) if lang == "tr" else (lambda x: f2(x).replace(",", "."))
    n_s = f"{D['n']:,}".replace(",", "." if lang == "tr" else ",")
    plt.rcParams.update({"font.family": FONT})
    W, H = width_cm * CM, height_cm * CM
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 70); ax.axis("off")
    PT = 70 / (height_cm / 2.54 * 72)          # 1 punto kaç veri birimi
    def stack(xc, yc, parts):
        """parts: [(metin, punto, kalın)]; satırları dikeyde ortalanmış blok olarak yazar."""
        lines = [(ln, fs, b) for txt, fs, b in parts for ln in txt.split("\n")]
        hs = [fs * 1.22 * PT for _, fs, _ in lines]; gap = 0.9
        tot = sum(hs) + gap * (len(parts) - 1)
        y = yc + tot / 2; prev = None
        for (ln, fs, b), h in zip(lines, hs):
            key = (fs, b)
            if prev is not None and key != prev:
                y -= gap
            ax.text(xc, y - h / 2, ln, ha="center", va="center", fontsize=fs, fontweight="bold" if b else "normal", color="black")
            y -= h; prev = key
    def box(x, y, w, h, fc, title, sub, fs=6.6, fs2=5.9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4", fc=fc, ec="#1F4E79", lw=0.7))
        stack(x + w / 2, y + h / 2, [(title, fs, True), (sub, fs2, False)])
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=6.5, lw=0.8, color="#1F4E79", shrinkA=0, shrinkB=0))
    # üst bölüm: veri -> iki yol -> ilçe haritası
    box(0.5, 41.5, 27.5, 28.0, "#D6E2F0", T["posts"], T["posts2"].format(n=n_s, nd=D["n_district"]))
    box(32.5, 56.5, 33.5, 13.0, "#E8EEF7", T["emb"], T["emb2"], fs=6.1, fs2=5.7)
    box(32.5, 41.5, 33.5, 13.0, "#FBE5D6", T["llm"], T["llm2"], fs=6.1, fs2=5.7)
    arrow(28.0, 59.0, 32.5, 63.0); arrow(28.0, 52.0, 32.5, 48.0)
    polys, _ = load_district_polys(D["run_dir"])
    bl = D["best_local"]
    stack(85.0, 66.0, [(T["map"], 6.3, True)])
    axm = fig.add_axes([0.705, 0.635, 0.29, 0.20])
    acc = _draw_map(axm, D, f"ilce_sentetik_{bl.model}_{bl.metin}.csv", polys, lw=0.2)
    for sp in axm.spines.values():
        sp.set_linewidth(0.4)
    stack(85.0, 42.8, [(T["mapnote"].format(acc=dec(acc)), 5.7, False)])
    arrow(66.0, 63.0, 70.2, 57.5); arrow(66.0, 48.0, 70.2, 51.5)
    # alt bölüm: anlatım biçimine göre isabet
    e = D["expl_by_tag"]
    tf = D["best_tfidf"]; sb = D["best"](D["syn"], "SBERT-mpnet")
    keys = [f"Temel: TF-IDF_{tf.metin}", f"{bl.model}_{bl.metin}", f"SBERT-mpnet_{sb.metin}", "BDM_claude-opus-5-5", "BDM_gemini-3.8-flash"]
    for k_ in keys:
        assert k_ in e, k_
    ex_v = [e[k_]["explicit"] for k_ in keys]; im_v = [e[k_]["implicit"] for k_ in keys]
    L = D["llm"]["sentetik"]
    ax.text(0.8, 37.0, T["chart"], ha="left", va="center", fontsize=6.3, fontweight="bold")
    ax.text(0.8, 33.4, T["overall"].format(use=dec(bl.hit1), tf=dec(tf.hit1), lo=dec(L.hit1.min()), hi=dec(L.hit1.max())), ha="left", va="center", fontsize=5.8)
    axb = fig.add_axes([0.065, 0.075, 0.925, 0.36])
    x = np.arange(len(keys)); w = 0.36
    b1 = axb.bar(x - w / 2, ex_v, w, color="#0072B2", label=T["expl"])
    b2 = axb.bar(x + w / 2, im_v, w, color="#009E73", label=T["impl"])
    for bars in (b1, b2):
        for r in bars:
            axb.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.03, dec(r.get_height()), ha="center", va="bottom", fontsize=5.6)
    axb.set_xticks(x); axb.set_xticklabels(GROUPS, fontsize=5.9)
    axb.set_ylim(0, 1.16); axb.set_yticks([0, 0.5, 1.0]); axb.set_yticklabels([dec(v) for v in (0, 0.5, 1.0)], fontsize=5.6)
    for s_ in ("top", "right"):
        axb.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"):
        axb.spines[s_].set_linewidth(0.5)
    axb.tick_params(length=2, width=0.5, pad=1.5)
    fig.legend(handles=[b1, b2], labels=[T["expl"], T["impl"]], frameon=False, fontsize=5.9, ncol=2, loc="center right",
               bbox_to_anchor=(0.995, 0.505), handlelength=1.0, handleheight=0.7, columnspacing=0.9, handletextpad=0.4)
    fig.savefig(out, dpi=600); plt.close(fig)
    return out

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    D = load(RUN, REF)
    plt.rcParams.update({"font.family": FONT})
    # Genel çerçeve, hit@1, genel karşılaştırma ve anlatım biçimi: makale_sekilleri.py ile aynı çizimler (yalnızca yazı tipi Carlito)
    tmp = tempfile.mkdtemp()
    try:
        _, panels = make_all(D, tmp)
        for src, dst in (("fig_framework.png", "genel_cerceve.png"), ("fig_hit1.png", "hit1_model_metin_turu.png"),
                         ("fig_all.png", "genel_karsilastirma.png"), ("fig_explicit.png", "anlatim_bicimi.png")):
            shutil.copyfile(os.path.join(tmp, src), os.path.join(OUT, dst))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # İlçe haritaları (dört panel)
    for title, path, acc in map_panels(D, OUT, panels):
        print(f"İlçe haritası: {title}, ilçe doğruluğu {f2(acc)} -> {os.path.basename(path)}")
    # Grafiksel özetler
    for lg in ("en", "tr"):
        graphical_abstract(D, lg, os.path.join(OUT, f"grafiksel_ozet_{lg}.png"))
    print(f"Şekiller yazıldı: {OUT}")
