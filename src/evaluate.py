"""
Adim 4b — Degerlendirme: iki test seti, confusion matrix, guven dagilimi.

IKI AYRI TEST raporlanir ve aradaki FARK asil bulgudur:
  test.csv       cogaltilmis veriden ayrilmis; train ile ayni dagilimdan gelir
  gold_test.csv  bagimsiz uretilmis, elle gozden gecirilmis, few-shot'ta hic
                 kullanilmamis; zor ve sinirda ornekler icerir

Ikisi yakinsa sentetik veri gercekci demektir. Gold belirgin dusukse model
cogaltmanin kendine ozgu kaliplarini ezberlemis demektir. Tek test setiyle bu
ayrim yapilamaz -- "veri gercekci mi" elestirisine verilecek olculebilir cevap
bu farktir.

Ayrica:
  - sinif bazli F1 ve confusion matrix (her iki set icin)
  - EN COK KARISAN KATEGORI CIFTLERI -- taksonomi sinir sorunlarini gorunur
    kilar (bkz. CLAUDE.md "Guncel Acik Noktalar")
  - stil bazli dogruluk: model hangi yazim stilinde zorlaniyor
  - guven dagilimi: CONFIDENCE_THRESHOLD'u kalibre etmek icin

Kullanim:
    python -m src.evaluate
    python -m src.evaluate --hatalari-goster
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import config as C
from src.train import BildirimVeriseti, cihaz_sec, csv_oku


# ---------------------------------------------------------------------------
# Model yukleme
# ---------------------------------------------------------------------------

def model_yukle(cihaz):
    if not (C.MODEL_DIR / "egitim_ozeti.json").exists():
        raise SystemExit(
            f"HATA: egitilmis model yok ({C.MODEL_DIR}).\n"
            f"Once calistir: python -m src.train"
        )
    ozet = json.loads((C.MODEL_DIR / "egitim_ozeti.json").read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(C.MODEL_DIR)
    if ozet.get("lora"):
        from peft import PeftConfig, PeftModel
        pc = PeftConfig.from_pretrained(C.MODEL_DIR)
        taban = AutoModelForSequenceClassification.from_pretrained(
            pc.base_model_name_or_path,
            num_labels=C.NUM_LABELS,
            id2label=C.ID2LABEL,
            label2id=C.LABEL2ID,
        )
        model = PeftModel.from_pretrained(taban, C.MODEL_DIR)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(C.MODEL_DIR)

    model.to(cihaz).eval()
    return model, tokenizer, ozet


def tahmin_et(model, tokenizer, satirlar, cihaz):
    """(tahminler, guvenler, tum_olasiliklar) doner."""
    ds = BildirimVeriseti(satirlar, tokenizer, C.MAX_LENGTH)
    dl = DataLoader(ds, batch_size=C.BATCH_SIZE)
    tahminler, guvenler, olasiliklar = [], [], []
    with torch.no_grad():
        for parti in dl:
            girdi = {k: v.to(cihaz) for k, v in parti.items() if k != "labels"}
            logit = model(**girdi).logits
            olasi = torch.softmax(logit, dim=-1).cpu()
            guven, tahmin = olasi.max(dim=-1)
            tahminler.extend(tahmin.tolist())
            guvenler.extend(guven.tolist())
            olasiliklar.extend(olasi.tolist())
    return tahminler, guvenler, olasiliklar


# ---------------------------------------------------------------------------
# Raporlama
# ---------------------------------------------------------------------------

def kisa_ad(kategori_key: str, genislik: int = 12) -> str:
    return C.DISPLAY_NAME[kategori_key][:genislik]


def matris_yazdir(gercek, tahmin) -> None:
    cm = confusion_matrix(gercek, tahmin, labels=range(C.NUM_LABELS))
    basliklar = [kisa_ad(k, 6) for k in C.CATEGORY_KEYS]
    print(f"\n{'gercek \\ tahmin':<22}" + "".join(f"{b:>8}" for b in basliklar))
    for i, k in enumerate(C.CATEGORY_KEYS):
        satir = "".join(
            f"{cm[i][j]:>8}" if i != j else f"{'[' + str(cm[i][j]) + ']':>8}"
            for j in range(C.NUM_LABELS)
        )
        print(f"{kisa_ad(k, 20):<22}{satir}")
    print("  (kosegen [] = dogru tahmin)")
    return cm


def karisan_ciftler(cm, ust: int = 6) -> list[tuple[str, str, int]]:
    """En cok karisan (gercek, tahmin) ciftleri -- taksonomi sinir sorunlari."""
    ciftler = []
    for i, gercek in enumerate(C.CATEGORY_KEYS):
        for j, tahmin in enumerate(C.CATEGORY_KEYS):
            if i != j and cm[i][j] > 0:
                ciftler.append((gercek, tahmin, int(cm[i][j])))
    ciftler.sort(key=lambda x: -x[2])
    return ciftler[:ust]


def ikincil_kategori_analizi(gercek, tahmin, olasiliklar) -> dict:
    """Ikincil kategori mekanizmasinin olculmesi.

    Taksonomi sinir sorunlarina kural yazmak yerine modelin kendi olasilik
    dagilimini kullaniyoruz: marj (top1-top2) kucukse model iki kategori
    arasinda kararsiz demektir ve ikisini birden dondurmek dogru davranis.
    Burada o mekanizmanin ne kadar ise yaradigi olculuyor.
    """
    olasi = np.array(olasiliklar)
    gercek_np = np.array(gercek)
    sirali = np.sort(olasi, axis=1)
    marj = sirali[:, -1] - sirali[:, -2]
    top2 = np.argsort(olasi, axis=1)[:, -2:]

    dogru = np.array(tahmin) == gercek_np
    top2_dogru = np.array([g in t2 for g, t2 in zip(gercek_np, top2)])
    isaretli = marj < C.MARGIN_THRESHOLD

    return {
        "top1_accuracy": float(dogru.mean()),
        "top2_accuracy": float(top2_dogru.mean()),
        "marj_dogru": float(marj[dogru].mean()),
        "marj_yanlis": float(marj[~dogru].mean()) if (~dogru).any() else None,
        "cift_kategorili": int(isaretli.sum()),
        "kurtarilan_hata": int((isaretli & ~dogru & top2_dogru).sum()),
        "toplam_hata": int((~dogru).sum()),
        "bosuna_isaretlenen": int((isaretli & dogru).sum()),
    }


def set_degerlendir(ad: str, satirlar, model, tokenizer, cihaz, hatalari_goster: bool):
    gercek = [int(r["label"]) for r in satirlar]
    tahmin, guven, olasiliklar = tahmin_et(model, tokenizer, satirlar, cihaz)

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(gercek, tahmin)
    macro = f1_score(gercek, tahmin, average="macro", zero_division=0)
    sinif_f1 = f1_score(gercek, tahmin, average=None, labels=range(C.NUM_LABELS),
                        zero_division=0)

    print(f"\n{'=' * 78}\n{ad}  —  {len(satirlar)} kayit\n{'=' * 78}")
    print(f"accuracy   {acc:.4f}   (hedef {C.TARGET_ACCURACY})   "
          f"{'GECTI' if acc >= C.TARGET_ACCURACY else 'KALDI'}")
    print(f"macro F1   {macro:.4f}   (hedef {C.TARGET_MACRO_F1})   "
          f"{'GECTI' if macro >= C.TARGET_MACRO_F1 else 'KALDI'}")
    en_dusuk = sinif_f1.min()
    print(f"en dusuk sinif F1  {en_dusuk:.4f}   (hedef {C.MIN_PER_CLASS_F1})   "
          f"{'GECTI' if en_dusuk >= C.MIN_PER_CLASS_F1 else 'KALDI'}")

    print(f"\n{'kategori':<28} {'F1':>7} {'destek':>7}")
    destek = Counter(gercek)
    for i, k in enumerate(C.CATEGORY_KEYS):
        uyari = "  <-- hedefin altinda" if sinif_f1[i] < C.MIN_PER_CLASS_F1 else ""
        print(f"{C.DISPLAY_NAME[k]:<28} {sinif_f1[i]:>7.4f} {destek[i]:>7}{uyari}")

    cm = matris_yazdir(gercek, tahmin)

    ciftler = karisan_ciftler(cm)
    if ciftler:
        print("\nEN COK KARISAN CIFTLER (gercek -> tahmin):")
        for g, t, n in ciftler:
            print(f"  {n:>3}x  {C.DISPLAY_NAME[g]:<26} -> {C.DISPLAY_NAME[t]}")

    # Stil bazli: model hangi yazim stilinde zorlaniyor?
    if satirlar and satirlar[0].get("stil"):
        stil_dogru: dict[str, list[int]] = defaultdict(list)
        for r, g, t in zip(satirlar, gercek, tahmin):
            stil_dogru[r["stil"]].append(1 if g == t else 0)
        print("\nSTIL BAZLI DOGRULUK:")
        for stil in C.STYLE_KEYS:
            v = stil_dogru.get(stil, [])
            if v:
                print(f"  {stil:<16} {sum(v) / len(v):.4f}  ({sum(v)}/{len(v)})")

    # Guven dagilimi -- CONFIDENCE_THRESHOLD kalibrasyonu icin
    guven_np = np.array(guven)
    dogru_mask = np.array([g == t for g, t in zip(gercek, tahmin)])
    print(f"\nGUVEN DAGILIMI (CONFIDENCE_THRESHOLD={C.CONFIDENCE_THRESHOLD} "
          f"kalibrasyonu icin):")
    print(f"  dogru tahminlerde ortalama guven  {guven_np[dogru_mask].mean():.4f}")
    if (~dogru_mask).any():
        print(f"  YANLIS tahminlerde ortalama guven {guven_np[~dogru_mask].mean():.4f}")
    for esik in (0.50, 0.60, 0.70, 0.80, 0.90):
        altinda = guven_np < esik
        if altinda.any():
            yakalanan = (altinda & ~dogru_mask).sum()
            print(f"  esik {esik:.2f}: {altinda.sum():>3} kayit dusuk guvenli "
                  f"({yakalanan}/{(~dogru_mask).sum()} hatayi yakalar, "
                  f"{(altinda & dogru_mask).sum()} dogruyu bosuna isaretler)")
        else:
            print(f"  esik {esik:.2f}:   0 kayit dusuk guvenli")

    ik = ikincil_kategori_analizi(gercek, tahmin, olasiliklar)
    print(f"\nIKINCIL KATEGORI (MARGIN_THRESHOLD={C.MARGIN_THRESHOLD}):")
    print(f"  top-1 dogruluk {ik['top1_accuracy']:.4f}  ->  "
          f"TOP-2 dogruluk {ik['top2_accuracy']:.4f}")
    print(f"  marj ortalama: dogru tahminlerde {ik['marj_dogru']:.3f}, "
          f"yanlislarda {ik['marj_yanlis']:.3f}" if ik["marj_yanlis"] is not None
          else f"  marj ortalama: {ik['marj_dogru']:.3f}")
    print(f"  {ik['cift_kategorili']} kayit cift kategorili donerdi "
          f"(%{100 * ik['cift_kategorili'] / len(satirlar):.1f} trafik): "
          f"{ik['kurtarilan_hata']}/{ik['toplam_hata']} hatada dogru cevap "
          f"ikinci etikette, {ik['bosuna_isaretlenen']} dogru bosuna isaretlenir")

    if hatalari_goster:
        print("\nHATALI TAHMINLER:")
        for r, g, t, gv in zip(satirlar, gercek, tahmin, guven):
            if g != t:
                print(f"  guven {gv:.2f} | {C.DISPLAY_NAME[C.ID2LABEL[g]]} "
                      f"-> {C.DISPLAY_NAME[C.ID2LABEL[t]]}")
                print(f"        [{r.get('stil', '?')}] {r['metin']}")

    return {"ad": ad, "n": len(satirlar), "accuracy": acc, "macro_f1": macro,
            "min_sinif_f1": float(en_dusuk),
            "sinif_f1": {k: float(sinif_f1[i]) for i, k in enumerate(C.CATEGORY_KEYS)},
            "karisan_ciftler": [(g, t, n) for g, t, n in ciftler],
            "ikincil_kategori": ik}


def main() -> None:
    ap = argparse.ArgumentParser(description="Adim 4b -- degerlendirme")
    ap.add_argument("--hatalari-goster", action="store_true",
                    help="yanlis tahmin edilen bildirimleri tek tek yazdir")
    ap.add_argument("--device", choices=["mps", "cpu"])
    args = ap.parse_args()

    cihaz = cihaz_sec(args.device)
    model, tokenizer, ozet = model_yukle(cihaz)
    print(f"cihaz: {cihaz} | LoRA: {ozet.get('lora')} | "
          f"en iyi epoch: {ozet.get('en_iyi_epoch')}")

    sonuclar = []
    for ad, path in (("TEST (cogaltilmis dagilim)", C.TEST_FILE),
                     ("GOLD TEST (bagimsiz, elle gozden gecirilmis)", C.GOLD_TEST_FILE)):
        sonuclar.append(
            set_degerlendir(ad, csv_oku(path), model, tokenizer, cihaz,
                            args.hatalari_goster)
        )

    # Asil bulgu: iki setin FARKI
    t, g = sonuclar[0], sonuclar[1]
    print(f"\n{'=' * 78}\nTEST vs GOLD — sentetik veri ne kadar gercekci?\n{'=' * 78}")
    print(f"{'metrik':<16} {'test':>9} {'gold':>9} {'fark':>9}")
    for anahtar, etiket in (("accuracy", "accuracy"), ("macro_f1", "macro F1")):
        fark = g[anahtar] - t[anahtar]
        print(f"{etiket:<16} {t[anahtar]:>9.4f} {g[anahtar]:>9.4f} {fark:>+9.4f}")
    fark = g["macro_f1"] - t["macro_f1"]
    if fark > -0.05:
        print("\nYORUM: gold skoru test'e cok yakin. Sentetik veri gercek "
              "bildirimlere yakin duruyor; model kaliplari degil sinifi ogrenmis.")
    elif fark > -0.15:
        print("\nYORUM: gold skoru olcülü sekilde dusuk. Beklenen bir fark — "
              "gold bilerek daha zor ve sinirda ornekler iceriyor.")
    else:
        print("\nYORUM: gold skoru belirgin dusuk. Model cogaltmanin kendine "
              "ozgu kaliplarini ogrenmis olabilir; veri cesitliligi artirilmali.")

    (C.MODEL_DIR / "degerlendirme.json").write_text(
        json.dumps(sonuclar, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n-> {C.MODEL_DIR / 'degerlendirme.json'}")


if __name__ == "__main__":
    main()
