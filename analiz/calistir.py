# -*- coding: utf-8 -*-
"""Makaledeki şekilleri ve temel sayıları depodaki çalıştırma çıktılarından yeniden üretir.

API anahtarı ya da model indirmesi gerekmez; yalnızca runs/ altındaki tablolar ve tahmin dosyaları okunur.

Kullanım (depo kök klasöründen):
    python analiz/calistir.py                 # şekiller: makale_sekilleri_yeniden/
    python analiz/calistir.py <çıktı_klasörü>
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from makale_verisi import load, f2          # noqa: E402
from makale_sekilleri import make_all       # noqa: E402

RUN = os.path.join(ROOT, "runs", "20261016_1116")                     # makalede raporlanan çalıştırma
REF = os.path.join(ROOT, "runs", "20261016_0743_claude_maliyet")      # Claude çağrılarının gerçek maliyetleri
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "makale_sekilleri_yeniden")

D = load(RUN, REF)
make_all(D, OUT)

bl = D["best_local"]
llm = D["llm"]["sentetik"]
ozet = {
    "paylasim_sayisi": int(D["n"]),
    "ilce_sayisi": int(D["n_district"]),
    "en_iyi_gomme_modeli": {"model": bl.model, "metin_turu": bl.metin, "hit1": f2(bl.hit1)},
    "bdm_hit1": {r.model: f2(r.hit1) for r in llm.itertuples()} if "hit1" in llm.columns else {},
}
print(json.dumps(ozet, ensure_ascii=False, indent=1))
print(f"Şekiller yazıldı: {OUT}")
