"""
Elle hazirlanmis test kumeleri icin toplu tahmin + hata raporu.

Egitim/gold gibi resmi bir set degil -- kullanicinin (Gemini ile) uretip
elle kategorize ettigi cumleleri MEVCUT modelle hizlica tarayip nerede
zayif oldugunu gormek icin. jsonl formati: {"metin": ..., "kategori": ...}

Kullanim:
    python -m src.toplu_test data/raw/manuel_test_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from src import config as C
from src.evaluate import model_yukle, tahmin_et
from src.train import cihaz_sec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dosya")
    args = ap.parse_args()

    satirlar = []
    with open(args.dosya, encoding="utf-8") as f:
        for satir in f:
            satir = satir.strip()
            if satir:
                satirlar.append(json.loads(satir))

    print(f"{len(satirlar)} kayit yuklendi. Model yukleniyor...")
    cihaz = cihaz_sec()
    model, tokenizer, _ = model_yukle(cihaz)

    girdi = [{"metin": r["metin"], "label": "0"} for r in satirlar]
    tahminler, guvenler, _ = tahmin_et(model, tokenizer, girdi, cihaz)

    dogru = 0
    kategori_dogru = Counter()
    kategori_toplam = Counter()
    karisan = defaultdict(Counter)
    hatalar = []

    for r, t_idx, g in zip(satirlar, tahminler, guvenler):
        gercek = r["kategori"]
        tahmin = C.ID2LABEL[t_idx]
        kategori_toplam[gercek] += 1
        if tahmin == gercek:
            dogru += 1
            kategori_dogru[gercek] += 1
        else:
            karisan[gercek][tahmin] += 1
            hatalar.append((r["metin"], gercek, tahmin, g))

    print(f"\nGenel doğruluk: {dogru}/{len(satirlar)} = {dogru / len(satirlar):.1%}\n")

    print("Kategori bazında:")
    for k in C.CATEGORY_KEYS:
        toplam = kategori_toplam.get(k, 0)
        if toplam == 0:
            continue
        d = kategori_dogru.get(k, 0)
        print(f"  {C.DISPLAY_NAME[k]:<28} {d}/{toplam} = {d / toplam:.1%}")

    if hatalar:
        print(f"\n{len(hatalar)} hata:\n")
        for metin, gercek, tahmin, g in sorted(hatalar, key=lambda x: x[3]):
            print(f"  [{C.DISPLAY_NAME[gercek]} -> {C.DISPLAY_NAME[tahmin]} ({g:.1%})] {metin}")

        print("\nEn çok karışan çiftler:")
        ciftler = Counter()
        for gercek, karsi in karisan.items():
            for tahmin, n in karsi.items():
                ciftler[(gercek, tahmin)] += n
        for (gercek, tahmin), n in ciftler.most_common(10):
            print(f"  {C.DISPLAY_NAME[gercek]} -> {C.DISPLAY_NAME[tahmin]}: {n}")


if __name__ == "__main__":
    main()
