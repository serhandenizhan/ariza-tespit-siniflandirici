"""
Backend entegrasyon testleri.

FastAPI'nin TestClient'i uygulamayi gercek bir HTTP istemcisiyle ayaga kaldirir
ve lifespan'i calistirir -- yani MODEL GERCEKTEN YUKLENIR. Bu bir birim testi
degil, uctan uca entegrasyon testi: tokenizer, LoRA adaptoru, esikler ve
yanit semasi birlikte dogrulanir.

Model yuklemesi birkac saniye surdugu icin istemci modul kapsaminda bir kez
kuruluyor (fixture scope="module").

Calistirma:
    ./venv/bin/pytest tests/ -v
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src import config as C


@pytest.fixture(scope="module")
def client():
    # 'with' sart: lifespan (model yukleme) ancak boyle calisir.
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Saglik ve meta
# ---------------------------------------------------------------------------

def test_health_modeli_yuklu_bildirir(client):
    r = client.get("/health")
    assert r.status_code == 200
    veri = r.json()
    assert veri["status"] == "ok"
    assert veri["model_loaded"] is True


def test_model_info_egitim_bilgisini_dondurur(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    veri = r.json()
    assert veri["base_model"] == C.BASE_MODEL
    assert veri["num_labels"] == C.NUM_LABELS
    assert veri["confidence_threshold"] == C.CONFIDENCE_THRESHOLD
    assert veri["margin_threshold"] == C.MARGIN_THRESHOLD
    # LoRA'nin asil iddiasi: parametrelerin cok kucuk bir kismi egitiliyor
    assert veri["trainable_params"] < veri["total_params"] * 0.02


def test_categories_taksonominin_tamamini_dondurur(client):
    r = client.get("/categories")
    assert r.status_code == 200
    veri = r.json()
    assert len(veri) == C.NUM_LABELS
    assert {d["category"] for d in veri} == set(C.CATEGORY_KEYS)
    for d in veri:
        assert d["label"] and d["color"].startswith("#") and d["scope"]


def test_examples_gold_setinden_gelir_ve_egitimde_yok(client):
    """Ornekler gold'dan gelmeli: egitim verisinden ornek gostermek demoyu
    oldugundan iyi gosterirdi."""
    import csv
    import json

    r = client.get("/examples?count=5")
    assert r.status_code == 200
    ornekler = r.json()
    assert 0 < len(ornekler) <= 5

    gold = {json.loads(s)["metin"]
            for s in C.GOLD_FILE.open(encoding="utf-8") if s.strip()}
    with C.TRAIN_FILE.open(encoding="utf-8") as f:
        train = {satir["metin"] for satir in csv.DictReader(f)}

    for o in ornekler:
        assert o["text"] in gold, "ornek gold setinden gelmeli"
        assert o["text"] not in train, "ornek egitim verisinde OLMAMALI"


# ---------------------------------------------------------------------------
# /predict — sozlesme
# ---------------------------------------------------------------------------

def test_predict_temel_sozlesme(client):
    r = client.post("/predict", json={"text": "Yürüyen merdiven durdu 2. peron"})
    assert r.status_code == 200
    veri = r.json()

    assert veri["category"] in C.CATEGORY_KEYS
    assert veri["label"] == C.DISPLAY_NAME[veri["category"]]
    assert 0.0 <= veri["confidence"] <= 1.0
    assert veri["response_time_ms"] > 0

    # olasilik dagilimi: tum kategoriler, toplami 1
    assert set(veri["probabilities"]) == set(C.CATEGORY_KEYS)
    assert abs(sum(veri["probabilities"].values()) - 1.0) < 1e-4
    # en yuksek olasilikli kategori, dondurulen kategori olmali
    assert max(veri["probabilities"], key=veri["probabilities"].get) == veri["category"]


def test_predict_esikler_tutarli(client):
    """low_confidence, secondary_category ve manual_review alanlarinin
    birbiriyle ve config esikleriyle tutarli olmasi."""
    for metin in ("Yürüyen merdiven durdu 2. peron",
                  "Peronda su birikintisi var",
                  "asansor calismiyo kadikoy ust kat"):
        veri = client.post("/predict", json={"text": metin}).json()

        assert veri["low_confidence"] == (veri["confidence"] < C.CONFIDENCE_THRESHOLD)
        sinirda = veri["margin"] < C.MARGIN_THRESHOLD
        assert (veri["secondary_category"] is not None) == sinirda
        assert veri["manual_review"] == (veri["low_confidence"] or sinirda)

        if veri["secondary_category"] is not None:
            assert veri["secondary_category"] != veri["category"]
            assert veri["secondary_confidence"] <= veri["confidence"]


def test_predict_bilinen_bildirimi_dogru_siniflandirir(client):
    """Regresyon testi: acik bir mekanik ariza mekanik cikmali. Model
    bozulursa (yanlis checkpoint, bozuk adapter) bu test duser."""
    veri = client.post("/predict",
                       json={"text": "Yürüyen merdiven durdu 2. peron"}).json()
    assert veri["category"] == "istasyon_mekanik"
    assert veri["confidence"] > 0.8


def test_predict_aksansiz_metni_de_dogru_siniflandirir(client):
    """Adim 4'teki ASCII cogaltmasinin regresyon testi: aksan dusmus metin
    aksanlisiyla ayni kategoriye gitmeli."""
    aksanli = client.post("/predict",
                          json={"text": "Asansör çalışmıyor Kadıköy üst kat"}).json()
    aksansiz = client.post("/predict",
                           json={"text": "asansor calismiyo kadikoy ust kat"}).json()
    assert aksanli["category"] == aksansiz["category"] == "istasyon_mekanik"


# ---------------------------------------------------------------------------
# /predict — hatali girdi
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metin", ["", "   ", "\n\t "])
def test_predict_bos_metin_400(client, metin):
    r = client.post("/predict", json={"text": metin})
    assert r.status_code == 400


def test_predict_cok_uzun_metin_400(client):
    r = client.post("/predict", json={"text": "a " * (C.MAX_CHARS)})
    assert r.status_code == 400
    assert "uzun" in r.json()["detail"].lower()


def test_predict_eksik_alan_422(client):
    assert client.post("/predict", json={}).status_code == 422


def test_predict_yanlis_tip_422(client):
    assert client.post("/predict", json={"text": 12345}).status_code == 422


# ---------------------------------------------------------------------------
# Model yukleme davranisi
# ---------------------------------------------------------------------------

def test_model_istek_basina_yeniden_yuklenmiyor(client):
    """Basari kriteri: model process basina BIR KEZ yuklenir.

    Modul seviyesindeki nesne kimligi (id) istekler arasinda degismemeli.
    Ayrica ilk istekten sonraki cagrilar belirgin sekilde hizli olmali.
    """
    from backend.main import STATE

    ilk_kimlik = id(STATE["model"])
    client.post("/predict", json={"text": "Turnike kolu kırıldı"})
    client.post("/predict", json={"text": "Peron aydınlatması söndü"})
    assert id(STATE["model"]) == ilk_kimlik, "model yeniden yuklenmis!"


def test_ardisik_istekler_hizli(client):
    """Isinma sonrasi cikarim milisaniyeler mertebesinde olmali."""
    client.post("/predict", json={"text": "ısınma"})  # ilk cagri
    sureler = [
        client.post("/predict", json={"text": f"Turnike {i} bozuk"})
        .json()["response_time_ms"]
        for i in range(5)
    ]
    assert max(sureler) < 500, f"cikarim cok yavas: {sureler}"


# ---------------------------------------------------------------------------
# OpenAPI dokumantasyonu
# ---------------------------------------------------------------------------

def test_openapi_tum_uc_noktalari_dokumante_ediyor(client):
    """Swagger'da 'string' gosterilmesinin sebebi eksik response_model'di --
    bu testin amaci o gerilemeyi yakalamak."""
    sema = client.get("/openapi.json").json()
    for yol in ("/predict", "/health", "/model-info", "/categories", "/examples"):
        assert yol in sema["paths"], f"{yol} OpenAPI'de yok"

    for yol, yontem in (("/predict", "post"), ("/health", "get"),
                        ("/model-info", "get"), ("/categories", "get"),
                        ("/examples", "get")):
        icerik = sema["paths"][yol][yontem]["responses"]["200"]["content"]
        assert icerik["application/json"]["schema"], f"{yol} yanit semasi bos"


# ---------------------------------------------------------------------------
# Yapisal cikarim (Adim 7)
# ---------------------------------------------------------------------------

def test_predict_yapisal_alanlari_dondurur(client):
    """Spec'teki cikti bicimi: category + line + station + equipment + symptom."""
    veri = client.post(
        "/predict", json={"text": "M4 Ünalan'da yürüyen merdiven çok ses yapıyor"}
    ).json()
    assert veri["category"] == "istasyon_mekanik"
    assert veri["line"] == "M4"
    assert veri["station"] == "Ünalan"
    assert veri["equipment"] == "yürüyen merdiven"
    assert veri["symptom"] == "anormal ses"


def test_predict_bulunamayan_alan_none_doner(client):
    """Hat kodu bildirimlerin ~%6'sinda geciyor; yoksa None donmeli, uydurmamali."""
    veri = client.post("/predict", json={"text": "Turnike kolu kırık"}).json()
    assert veri["line"] is None
    assert veri["station"] is None
    assert veri["equipment"] == "turnike kolu"


def test_yapisal_cikarim_aksana_duyarsiz():
    """Ayni ariza, aksanli ve aksansiz yazim -> ayni yapisal alanlar."""
    from src.extract import cikar

    a = cikar("Asansör çalışmıyor Kadıköy")
    b = cikar("asansor calismiyor kadikoy")
    assert a["equipment"] == b["equipment"] == "asansör"
    assert a["station"] == b["station"] == "Kadıköy"
    assert a["symptom"] == b["symptom"] == "çalışmıyor"


def test_hat_kodu_ekipman_etiketiyle_karismiyor():
    """'T3 trensformatörü'ndeki T3 hat kodu DEGIL, ekipman etiketi."""
    from src.extract import cikar

    veri = cikar("Üsküdar şubesinde T3 trensformatörü aşırı ısı uyarısı verdi.")
    assert veri["line"] is None
    assert veri["equipment"] == "trafo"
