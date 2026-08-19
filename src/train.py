"""
Adim 4a — Model egitimi: BERTurk + LoRA.

dbmdz/bert-base-turkish-cased uzerine PEFT/LoRA ile siniflandirma basligi
egitir. Tam fine-tuning yerine LoRA secildi cunku 1280 ornekle 110M parametreyi
tamamen egitmek asiri ogrenmeye acik; LoRA sadece query/value projeksiyonlarina
dusuk ranklı adaptor ekleyip egitilen parametre sayisini ~%1'e indiriyor.

Apple Silicon notu: MPS backend kullanilir. MPS'te bilinen kisitlar var
(bazi operatorler desteklenmiyor, fp16 sorunlu) -- bu yuzden fp32'de egitiyoruz
ve sorun cikarsa CPU'ya dusuyoruz.

Kullanim:
    python -m src.train
    python -m src.train --epochs 8 --no-lora
    python -m src.train --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import config as C


# ---------------------------------------------------------------------------
# Yeniden uretilebilirlik
# ---------------------------------------------------------------------------

def tohum_ek(seed: int = C.SEED) -> None:
    """Ayni tohumla ayni sonuc. Rapor icin kritik: 'bu skoru nasil aldin'
    sorusunun cevabi tekrar calistirilabilir olmali."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def cihaz_sec(tercih: str | None = None) -> torch.device:
    if tercih:
        return torch.device(tercih)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Veri
# ---------------------------------------------------------------------------

class BildirimVeriseti(Dataset):
    """Tokenize edilmis arıza bildirimleri."""

    def __init__(self, satirlar: list[dict], tokenizer, max_length: int):
        self.metinler = [r["metin"] for r in satirlar]
        self.etiketler = [int(r["label"]) for r in satirlar]
        self.enc = tokenizer(
            self.metinler,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.etiketler)

    def __getitem__(self, i: int) -> dict:
        return {
            "input_ids": self.enc["input_ids"][i],
            "attention_mask": self.enc["attention_mask"][i],
            "labels": torch.tensor(self.etiketler[i], dtype=torch.long),
        }


def csv_oku(path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"HATA: {path} yok. Once calistir: python -m src.preprocess"
        )
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


TR_HARFLER = set("çğıöşüÇĞİÖŞÜ")


def ascii_cogalt(satirlar: list[dict]) -> list[dict]:
    """Aksan iceren kayitlarin ASCII'ye katlanmis kopyalarini ekler.

    Gerekcesi config.AUGMENT_ASCII_FOLD'da olculmus haliyle duruyor: aksan
    kaybolunca dogruluk 6.4 puan dusuyor cunku BERTurk tokenizer'i kelimeyi
    parcaliyor ("asansör" 1 parca, "asansor" 3 parca). Modele iki yazimin da
    ayni sinifa ait oldugunu ogretiyoruz.

    Sadece aksan ICEREN kayitlar cogaltilir; digerlerinin ASCII hali zaten
    kendisiyle ayni olurdu ve birebir tekrar eklemek ogrenmeye katki yapmaz.
    """
    from src.review import _strip_diacritics

    ek = [
        {**r, "metin": _strip_diacritics(r["metin"]), "kaynak": "ascii_fold"}
        for r in satirlar
        if set(r["metin"]) & TR_HARFLER
    ]
    return satirlar + ek


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def model_kur(use_lora: bool):
    model = AutoModelForSequenceClassification.from_pretrained(
        C.BASE_MODEL,
        num_labels=C.NUM_LABELS,
        id2label=C.ID2LABEL,
        label2id=C.LABEL2ID,
    )

    if not use_lora:
        return model, sum(p.numel() for p in model.parameters() if p.requires_grad)

    from peft import LoraConfig, TaskType, get_peft_model

    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=C.LORA_R,
        lora_alpha=C.LORA_ALPHA,
        lora_dropout=C.LORA_DROPOUT,
        target_modules=C.LORA_TARGET_MODULES,
        # Siniflandirma basligi (classifier) rastgele baslatiliyor, LoRA
        # adaptorleri disinda o da egitilmeli -- yoksa model hicbir sey
        # ogrenemez. PEFT bunu modules_to_save ile yapiyor.
        modules_to_save=["classifier"],
    )
    model = get_peft_model(model, lora)
    egitilen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, egitilen


# ---------------------------------------------------------------------------
# Egitim dongusu
#
# HuggingFace Trainer yerine elle dongu yaziyoruz: Trainer accelerate uzerinden
# MPS'te zaman zaman dtype/device surprizleri cikariyor ve ne oldugunu gormek
# zorlasiyor. Bu boyutta bir is (1280 ornek, 5 epoch) icin elle dongu hem
# seffaf hem yeterli.
# ---------------------------------------------------------------------------

def degerlendir(model, yukleyici, cihaz) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    tahminler, gercekler = [], []
    with torch.no_grad():
        for parti in yukleyici:
            girdi = {k: v.to(cihaz) for k, v in parti.items() if k != "labels"}
            cikti = model(**girdi)
            tahminler.extend(cikti.logits.argmax(dim=-1).cpu().tolist())
            gercekler.extend(parti["labels"].tolist())
    acc = accuracy_score(gercekler, tahminler)
    f1 = f1_score(gercekler, tahminler, average="macro", zero_division=0)
    return acc, f1, tahminler, gercekler


def egit(args) -> None:
    tohum_ek()
    cihaz = cihaz_sec(args.device)
    print(f"cihaz: {cihaz}  |  model: {C.BASE_MODEL}  |  LoRA: {not args.no_lora}")

    tokenizer = AutoTokenizer.from_pretrained(C.BASE_MODEL)
    train_satir = csv_oku(C.TRAIN_FILE)
    val_satir = csv_oku(C.VAL_FILE)

    cogalt = C.AUGMENT_ASCII_FOLD and not args.no_augment
    if cogalt:
        onceki = len(train_satir)
        train_satir = ascii_cogalt(train_satir)
        print(f"train {onceki} -> {len(train_satir)} "
              f"(+{len(train_satir) - onceki} ASCII katlanmis kopya) | "
              f"val {len(val_satir)}")
    else:
        print(f"train {len(train_satir)} | val {len(val_satir)}  (cogaltma KAPALI)")

    train_ds = BildirimVeriseti(train_satir, tokenizer, C.MAX_LENGTH)
    val_ds = BildirimVeriseti(val_satir, tokenizer, C.MAX_LENGTH)
    train_dl = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=C.BATCH_SIZE)

    model, egitilen = model_kur(not args.no_lora)
    toplam = sum(p.numel() for p in model.parameters())
    print(f"parametre: {toplam:,} toplam | {egitilen:,} egitilebilir "
          f"(%{100 * egitilen / toplam:.2f})")
    model.to(cihaz)

    epochs = args.epochs or C.NUM_EPOCHS
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr or C.LEARNING_RATE,
        weight_decay=C.WEIGHT_DECAY,
    )
    toplam_adim = len(train_dl) * epochs
    plan = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=args.lr or C.LEARNING_RATE,
        total_steps=toplam_adim,
        pct_start=C.WARMUP_RATIO,
        anneal_strategy="linear",
    )

    print(f"\n{'epoch':>5} {'train kayip':>12} {'val acc':>9} {'val f1':>8} {'sure':>7}")
    en_iyi_f1, en_iyi_epoch = -1.0, -1
    gecmis = []

    for epoch in range(1, epochs + 1):
        model.train()
        t0, kayip_top = time.time(), 0.0
        for parti in train_dl:
            parti = {k: v.to(cihaz) for k, v in parti.items()}
            cikti = model(**parti)
            cikti.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optim.step()
            plan.step()
            optim.zero_grad()
            kayip_top += cikti.loss.item()

        kayip = kayip_top / len(train_dl)
        acc, f1, _, _ = degerlendir(model, val_dl, cihaz)
        sure = time.time() - t0
        gecmis.append({"epoch": epoch, "kayip": kayip, "val_acc": acc, "val_f1": f1})

        isaret = ""
        if f1 > en_iyi_f1:
            en_iyi_f1, en_iyi_epoch = f1, epoch
            model.save_pretrained(C.MODEL_DIR)
            tokenizer.save_pretrained(C.MODEL_DIR)
            isaret = "  <- kaydedildi"
        print(f"{epoch:>5} {kayip:>12.4f} {acc:>9.4f} {f1:>8.4f} {sure:>6.1f}s{isaret}")

    # En iyi val F1'i veren epoch kaydedilir, sonuncusu degil: son epoch
    # genelde asiri ogrenmis olur ve test skoru dusuk cikar.
    print(f"\nen iyi: epoch {en_iyi_epoch}, val macro-F1 {en_iyi_f1:.4f}")
    print(f"model kaydedildi: {C.MODEL_DIR}")

    ozet = {
        "base_model": C.BASE_MODEL,
        "lora": not args.no_lora,
        "epochs": epochs,
        "batch_size": C.BATCH_SIZE,
        "learning_rate": args.lr or C.LEARNING_RATE,
        "max_length": C.MAX_LENGTH,
        "seed": C.SEED,
        "cihaz": str(cihaz),
        "ascii_cogaltma": cogalt,
        "train_kayit": len(train_satir),
        "egitilebilir_parametre": egitilen,
        "toplam_parametre": toplam,
        "en_iyi_epoch": en_iyi_epoch,
        "en_iyi_val_f1": en_iyi_f1,
        "gecmis": gecmis,
    }
    (C.MODEL_DIR / "egitim_ozeti.json").write_text(
        json.dumps(ozet, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nSONRAKI ADIM: python -m src.evaluate")


def main() -> None:
    ap = argparse.ArgumentParser(description="Adim 4a -- BERTurk + LoRA egitimi")
    ap.add_argument("--epochs", type=int, help=f"varsayilan {C.NUM_EPOCHS}")
    ap.add_argument("--lr", type=float, help=f"varsayilan {C.LEARNING_RATE}")
    ap.add_argument("--no-lora", action="store_true", help="tam fine-tuning yap")
    ap.add_argument("--no-augment", action="store_true",
                    help="ASCII katlanmis kopyalari EKLEME (kiyas icin)")
    ap.add_argument("--device", choices=["mps", "cpu"], help="varsayilan: varsa mps")
    egit(ap.parse_args())


if __name__ == "__main__":
    main()
