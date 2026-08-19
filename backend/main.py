"""
Adim 5 — FastAPI servisi.

Egitilmis BERTurk + LoRA modelini REST API olarak sunar. Model surec
basladiginda BIR KEZ yuklenir ve bellekte tutulur; her istekte yeniden
yuklemek 2-3 saniye surerdi.

Uc mekanizma birlikte calisir:
  1. kategori + guven        -> normal tahmin
  2. low_confidence          -> guven CONFIDENCE_THRESHOLD (0.70) altindaysa
  3. ikincil_kategori        -> marj (top1-top2) MARGIN_THRESHOLD (0.40)
                                altindaysa; model iki kategori arasinda
                                kararsizsa ikisini birden bildirir

(2) ve (3) farkli seyler: (2) "model emin degil", (3) "model HANGI iki secenek
arasinda kararsiz". Bir bildirim ikisini birden tetikleyebilir veya sadece
birini. Olculdu (Adim 4): top-1 dogruluk 0.913/0.925 iken top-2 0.963/0.975 --
yani model yanildiginda dogru cevap cogu zaman ikinci sirada.

Calistirma:
    ./venv/bin/uvicorn backend.main:app --reload --port 8000
    (veya) python -m backend.main
"""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import config as C

# ---------------------------------------------------------------------------
# Model durumu
#
# Modul seviyesinde tek bir sozluk: uygulama acilirken doldurulur, isteklerde
# okunur. PyTorch cikarimi bloklayici; FastAPI async oldugu icin es zamanli
# isteklerde ayni model nesnesine dokunulmasin diye kilit kullaniyoruz.
# (Prototip icin yeterli; gercek yukte birden fazla worker/kuyruk gerekir.)
# ---------------------------------------------------------------------------

DURUM: dict = {"model": None, "tokenizer": None, "cihaz": None, "ozet": None}
KILIT = threading.Lock()


@asynccontextmanager
async def yasam_dongusu(app: FastAPI):
    from src.evaluate import model_yukle
    from src.train import cihaz_sec

    baslangic = time.perf_counter()
    cihaz = cihaz_sec()
    model, tokenizer, ozet = model_yukle(cihaz)
    DURUM.update(model=model, tokenizer=tokenizer, cihaz=cihaz, ozet=ozet)
    print(f"[backend] model yuklendi ({time.perf_counter() - baslangic:.1f} sn) "
          f"| cihaz={cihaz} | LoRA={ozet.get('lora')} "
          f"| en iyi epoch={ozet.get('en_iyi_epoch')}")
    yield
    DURUM.clear()


app = FastAPI(
    title="Metro İstanbul Arıza Tespit Sınıflandırıcı",
    description="Serbest metinli arıza bildirimlerini bakım ekibine yönlendirir.",
    version="1.0.0",
    lifespan=yasam_dongusu,
)

# Arayuz (Adim 6) ayri bir gelistirme sunucusunda calisacak, tarayici
# cross-origin istegi engellemesin diye. Prototip oldugu icin genis birakildi;
# kuruma entegrasyonda daraltilmali.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Semalar
# ---------------------------------------------------------------------------

class TahminIstegi(BaseModel):
    metin: str = Field(..., description="Arıza bildirimi metni",
                       examples=["Yürüyen merdiven durdu 2. peron"])


class KategoriOlasilik(BaseModel):
    kategori: str
    ad: str
    olasilik: float
    renk: str


class TahminYaniti(BaseModel):
    kategori: str
    ad: str
    renk: str
    guven: float

    low_confidence: bool
    low_confidence_mesaji: str | None = None

    ikincil_kategori: str | None = None
    ikincil_ad: str | None = None
    ikincil_olasilik: float | None = None
    ikincil_mesaji: str | None = None

    marj: float
    dagilim: list[KategoriOlasilik]
    yanit_suresi_ms: float


# ---------------------------------------------------------------------------
# Cikarim
# ---------------------------------------------------------------------------

def tahmin_yap(metin: str) -> dict:
    model, tokenizer, cihaz = DURUM["model"], DURUM["tokenizer"], DURUM["cihaz"]

    girdi = tokenizer(
        metin,
        truncation=True,
        padding="max_length",
        max_length=C.MAX_LENGTH,
        return_tensors="pt",
    )
    girdi = {k: v.to(cihaz) for k, v in girdi.items()}

    with KILIT, torch.no_grad():
        logit = model(**girdi).logits
    olasi = torch.softmax(logit, dim=-1)[0].cpu()

    sirali = torch.argsort(olasi, descending=True)
    birinci, ikinci = int(sirali[0]), int(sirali[1])
    guven = float(olasi[birinci])
    marj = guven - float(olasi[ikinci])

    dagilim = sorted(
        (
            {
                "kategori": k,
                "ad": C.DISPLAY_NAME[k],
                "olasilik": float(olasi[i]),
                "renk": C.CATEGORY_COLOR[k],
            }
            for i, k in enumerate(C.CATEGORY_KEYS)
        ),
        key=lambda d: -d["olasilik"],
    )

    birincil_key = C.ID2LABEL[birinci]
    sonuc = {
        "kategori": birincil_key,
        "ad": C.DISPLAY_NAME[birincil_key],
        "renk": C.CATEGORY_COLOR[birincil_key],
        "guven": guven,
        "low_confidence": guven < C.CONFIDENCE_THRESHOLD,
        "low_confidence_mesaji": None,
        "ikincil_kategori": None,
        "ikincil_ad": None,
        "ikincil_olasilik": None,
        "ikincil_mesaji": None,
        "marj": marj,
        "dagilim": dagilim,
    }

    if sonuc["low_confidence"]:
        sonuc["low_confidence_mesaji"] = C.LOW_CONFIDENCE_MESSAGE

    # Marj kucukse model iki kategori arasinda kararsiz demektir. Taksonomide
    # gercekten ortusen bildirimler var (orn. "acil tahliye anonsu duyulmuyor"
    # hem guvenlik hem operasyon kapsaminda) -- tek cevaba zorlamak yerine
    # ikisini birden bildiriyoruz.
    if marj < C.MARGIN_THRESHOLD:
        ikincil_key = C.ID2LABEL[ikinci]
        sonuc.update(
            ikincil_kategori=ikincil_key,
            ikincil_ad=C.DISPLAY_NAME[ikincil_key],
            ikincil_olasilik=float(olasi[ikinci]),
            ikincil_mesaji=C.SECONDARY_CATEGORY_MESSAGE,
        )

    return sonuc


# ---------------------------------------------------------------------------
# Uc noktalar
# ---------------------------------------------------------------------------

@app.get("/health")
def saglik():
    """Model yuklu mu, hangi cihazda, hangi egitimden."""
    ozet = DURUM.get("ozet") or {}
    return {
        "durum": "hazir" if DURUM.get("model") is not None else "model yuklenmedi",
        "cihaz": str(DURUM.get("cihaz")),
        "taban_model": ozet.get("base_model"),
        "lora": ozet.get("lora"),
        "en_iyi_epoch": ozet.get("en_iyi_epoch"),
        "en_iyi_val_f1": ozet.get("en_iyi_val_f1"),
        "confidence_threshold": C.CONFIDENCE_THRESHOLD,
        "margin_threshold": C.MARGIN_THRESHOLD,
    }


@app.get("/kategoriler")
def kategoriler():
    """Arayuzun etiket/renk/kapsam gostermesi icin taksonomi."""
    return [
        {
            "kategori": k,
            "ad": C.DISPLAY_NAME[k],
            "renk": C.CATEGORY_COLOR[k],
            "kapsam": C.CATEGORIES[k]["scope"],
        }
        for k in C.CATEGORY_KEYS
    ]


@app.get("/ornekler")
def ornekler(adet: int = 8):
    """Arayuzdeki 'tek tikla doldur' listesi.

    Kaynak gold.jsonl: elle gozden gecirilmis, EGITIME HIC GIRMEMIS kayitlar.
    Egitim verisinden ornek gostermek demoyu oldugundan iyi gosterirdi.
    """
    import json
    import random

    if not C.GOLD_FILE.exists():
        return []
    kayitlar = [json.loads(s) for s in C.GOLD_FILE.open(encoding="utf-8") if s.strip()]
    rng = random.Random(C.SEED)
    # Her kategoriden en fazla bir ornek -- liste tek bir kategoriye yigilmasin
    kategoriye_gore: dict[str, list[dict]] = {}
    for r in kayitlar:
        kategoriye_gore.setdefault(r["kategori"], []).append(r)
    secim = [rng.choice(v) for v in kategoriye_gore.values()]
    rng.shuffle(secim)
    return [{"metin": r["metin"], "stil": r["stil"]} for r in secim[:adet]]


@app.post("/predict", response_model=TahminYaniti)
def predict(istek: TahminIstegi):
    if DURUM.get("model") is None:
        raise HTTPException(503, "Model henüz yüklenmedi.")

    metin = " ".join(istek.metin.split())
    if not metin:
        raise HTTPException(400, "Metin boş olamaz.")
    if len(metin) > C.MAX_CHARS:
        raise HTTPException(
            400, f"Metin çok uzun ({len(metin)} karakter, en fazla {C.MAX_CHARS})."
        )

    baslangic = time.perf_counter()
    sonuc = tahmin_yap(metin)
    sonuc["yanit_suresi_ms"] = (time.perf_counter() - baslangic) * 1000
    return sonuc


def main() -> None:
    import uvicorn
    uvicorn.run("backend.main:app", host=C.API_HOST, port=C.API_PORT, reload=False)


if __name__ == "__main__":
    main()
