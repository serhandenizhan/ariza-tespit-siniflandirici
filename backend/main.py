"""
Adim 5 — FastAPI servisi.

Egitilmis BERTurk + LoRA modelini REST API olarak sunar. Model surec
basladiginda BIR KEZ yuklenir ve bellekte tutulur; her istekte yeniden
yuklemek 2-3 saniye surerdi.

API ALAN ADLARI INGILIZCE. Projenin geri kalani (config, degisken adlari,
dokumantasyon) Turkce ama dis dunyaya acilan sozlesme REST konvansiyonuna
uyuyor. Kategori KEY'leri (arac_tren, istasyon_mekanik...) zaten ASCII, aynen
korunuyor; her yanitta insan-okunur `label` ve arayuz icin `color` da var.

Uc mekanizma birlikte calisir:
  1. category + confidence   -> normal tahmin
  2. low_confidence          -> guven CONFIDENCE_THRESHOLD (0.70) altindaysa
  3. secondary_category      -> marj (top1-top2) MARGIN_THRESHOLD (0.40)
                                altindaysa; model iki kategori arasinda
                                kararsizsa ikisini birden bildirir
  manual_review = (1) veya (2) tetiklendiyse true -- operatore "bu bildirime
  insan baksin" diyen tek alan.

(2) ve (3) farkli seyler: (2) "model emin degil", (3) "model HANGI iki secenek
arasinda kararsiz". Olculdu (Adim 4): top-1 dogruluk 0.913/0.925 iken top-2
0.963/0.975 -- model yanildiginda dogru cevap cogu zaman ikinci sirada.

Calistirma:
    ./venv/bin/uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import random
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

STATE: dict = {"model": None, "tokenizer": None, "device": None, "meta": None}
LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.evaluate import model_yukle
    from src.train import cihaz_sec

    t0 = time.perf_counter()
    device = cihaz_sec()
    model, tokenizer, meta = model_yukle(device)
    STATE.update(model=model, tokenizer=tokenizer, device=device, meta=meta,
                 load_seconds=time.perf_counter() - t0)
    print(f"[backend] model yuklendi ({STATE['load_seconds']:.1f} sn) "
          f"| device={device} | LoRA={meta.get('lora')} "
          f"| best epoch={meta.get('en_iyi_epoch')}")
    yield
    STATE.clear()


app = FastAPI(
    title="Metro İstanbul Arıza Tespit Sınıflandırıcı",
    description=(
        "Serbest metinli arıza bildirimlerini 8 bakım kategorisinden birine "
        "yönlendirir. Düşük güven ve sınırda bildirim durumlarını ayrı ayrı "
        "işaretler."
    ),
    version="1.0.0",
    lifespan=lifespan,
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
# Semalar — hepsi response_model olarak baglanir ki OpenAPI/Swagger dokumante
# olsun (aksi halde Swagger "string" gosteriyor).
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str = Field(..., description="Arıza bildirimi metni",
                      examples=["Yürüyen merdiven durdu 2. peron"])


class PredictResponse(BaseModel):
    category: str = Field(..., description="Kategori anahtarı, örn. istasyon_mekanik")
    label: str = Field(..., description="İnsan okunur ad, örn. İstasyon Mekanik")
    color: str = Field(..., description="Arayüz rozeti için renk kodu")
    confidence: float

    probabilities: dict[str, float] = Field(
        ..., description="Tüm kategoriler için olasılık (kategori anahtarı -> olasılık)"
    )

    low_confidence: bool = Field(..., description="confidence < CONFIDENCE_THRESHOLD")
    manual_review: bool = Field(
        ..., description="low_confidence veya secondary_category varsa true"
    )
    manual_review_message: str | None = None

    secondary_category: str | None = None
    secondary_label: str | None = None
    secondary_confidence: float | None = None
    secondary_message: str | None = None

    margin: float = Field(..., description="top1 - top2 olasılık farkı")
    response_time_ms: float


class CategoryInfo(BaseModel):
    category: str
    label: str
    color: str
    scope: str
    excludes: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class ModelInfo(BaseModel):
    base_model: str
    lora: bool
    trainable_params: int
    total_params: int
    best_epoch: int
    best_val_macro_f1: float
    epochs: int
    learning_rate: float
    batch_size: int
    max_length: int
    seed: int
    ascii_augmentation: bool
    train_records: int
    device: str
    load_seconds: float
    confidence_threshold: float
    margin_threshold: float
    num_labels: int


class ExampleItem(BaseModel):
    text: str
    style: str


# ---------------------------------------------------------------------------
# Cikarim
# ---------------------------------------------------------------------------

def run_prediction(text: str) -> dict:
    model, tokenizer, device = STATE["model"], STATE["tokenizer"], STATE["device"]

    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=C.MAX_LENGTH,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with LOCK, torch.no_grad():
        logits = model(**encoded).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu()

    order = torch.argsort(probs, descending=True)
    first, second = int(order[0]), int(order[1])
    confidence = float(probs[first])
    margin = confidence - float(probs[second])

    primary = C.ID2LABEL[first]
    low_confidence = confidence < C.CONFIDENCE_THRESHOLD
    borderline = margin < C.MARGIN_THRESHOLD

    result = {
        "category": primary,
        "label": C.DISPLAY_NAME[primary],
        "color": C.CATEGORY_COLOR[primary],
        "confidence": confidence,
        "probabilities": {
            k: float(probs[i]) for i, k in enumerate(C.CATEGORY_KEYS)
        },
        "low_confidence": low_confidence,
        # Operatore tek bir "insan baksin" sinyali: model ya emin degil, ya da
        # iki kategori arasinda kararsiz. Ikisi de manuel incelemeyi gerektirir.
        "manual_review": low_confidence or borderline,
        "manual_review_message": C.LOW_CONFIDENCE_MESSAGE if low_confidence else None,
        "secondary_category": None,
        "secondary_label": None,
        "secondary_confidence": None,
        "secondary_message": None,
        "margin": margin,
    }

    # Marj kucukse model iki kategori arasinda kararsiz demektir. Taksonomide
    # gercekten ortusen bildirimler var (orn. "acil tahliye anonsu duyulmuyor"
    # hem guvenlik hem operasyon kapsaminda) -- tek cevaba zorlamak yerine
    # ikisini birden bildiriyoruz.
    if borderline:
        secondary = C.ID2LABEL[second]
        result.update(
            secondary_category=secondary,
            secondary_label=C.DISPLAY_NAME[secondary],
            secondary_confidence=float(probs[second]),
            secondary_message=C.SECONDARY_CATEGORY_MESSAGE,
        )

    return result


# ---------------------------------------------------------------------------
# Uc noktalar
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    """Servis ayakta mi, model yuklendi mi. Yuk dengeleyici/izleme icin."""
    loaded = STATE.get("model") is not None
    return {
        "status": "ok" if loaded else "loading",
        "model_loaded": loaded,
        "device": str(STATE.get("device")),
    }


@app.get("/model-info", response_model=ModelInfo, tags=["system"])
def model_info():
    """Hangi model, hangi egitim, hangi esikler. Raporlanabilirlik icin:
    bir tahminin hangi model surumunden geldigi izlenebilmeli."""
    if STATE.get("model") is None:
        raise HTTPException(503, "Model henüz yüklenmedi.")
    meta = STATE["meta"]
    return {
        "base_model": meta["base_model"],
        "lora": meta["lora"],
        "trainable_params": meta["egitilebilir_parametre"],
        "total_params": meta["toplam_parametre"],
        "best_epoch": meta["en_iyi_epoch"],
        "best_val_macro_f1": meta["en_iyi_val_f1"],
        "epochs": meta["epochs"],
        "learning_rate": meta["learning_rate"],
        "batch_size": meta["batch_size"],
        "max_length": meta["max_length"],
        "seed": meta["seed"],
        "ascii_augmentation": meta.get("ascii_cogaltma", False),
        "train_records": meta.get("train_kayit", 0),
        "device": str(STATE["device"]),
        "load_seconds": STATE.get("load_seconds", 0.0),
        "confidence_threshold": C.CONFIDENCE_THRESHOLD,
        "margin_threshold": C.MARGIN_THRESHOLD,
        "num_labels": C.NUM_LABELS,
    }


@app.get("/categories", response_model=list[CategoryInfo], tags=["taxonomy"])
def categories():
    """Taksonomi: arayuzun etiket/renk/kapsam gosterebilmesi icin."""
    return [
        {
            "category": k,
            "label": C.DISPLAY_NAME[k],
            "color": C.CATEGORY_COLOR[k],
            "scope": C.CATEGORIES[k]["scope"],
            "excludes": C.CATEGORIES[k]["exclude"],
        }
        for k in C.CATEGORY_KEYS
    ]


@app.get("/examples", response_model=list[ExampleItem], tags=["taxonomy"])
def examples(count: int = 8):
    """Arayuzdeki 'tek tikla doldur' listesi.

    Kaynak gold.jsonl: elle gozden gecirilmis, EGITIME HIC GIRMEMIS kayitlar.
    Egitim verisinden ornek gostermek demoyu oldugundan iyi gosterirdi.
    """
    if not C.GOLD_FILE.exists():
        return []
    records = [json.loads(s) for s in C.GOLD_FILE.open(encoding="utf-8") if s.strip()]
    rng = random.Random(C.SEED)
    # Her kategoriden en fazla bir ornek -- liste tek kategoriye yigilmasin
    by_category: dict[str, list[dict]] = {}
    for r in records:
        by_category.setdefault(r["kategori"], []).append(r)
    picked = [rng.choice(v) for v in by_category.values()]
    rng.shuffle(picked)
    return [{"text": r["metin"], "style": r["stil"]} for r in picked[:count]]


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest):
    if STATE.get("model") is None:
        raise HTTPException(503, "Model henüz yüklenmedi.")

    text = " ".join(request.text.split())
    if not text:
        raise HTTPException(400, "Metin boş olamaz.")
    if len(text) > C.MAX_CHARS:
        raise HTTPException(
            400, f"Metin çok uzun ({len(text)} karakter, en fazla {C.MAX_CHARS})."
        )

    t0 = time.perf_counter()
    result = run_prediction(text)
    result["response_time_ms"] = (time.perf_counter() - t0) * 1000
    return result


def main() -> None:
    import uvicorn
    uvicorn.run("backend.main:app", host=C.API_HOST, port=C.API_PORT, reload=False)


if __name__ == "__main__":
    main()
