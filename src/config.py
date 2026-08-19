"""
Metro Istanbul Ariza Tespit Siniflandirici -- merkezi konfigurasyon.

Bu dosya projenin TEK dogruluk kaynagidir. Kategori taksonomisi, yollar,
hiperparametreler ve threshold degeri sadece burada tanimlanir; diger tum
moduller (generate_seed, generate_data, preprocess, train, evaluate, backend)
buradan import eder.

NOT: Bu dosyadaki Turkce metinler (kategori aciklamalari, kurallar, istasyon
adlari) dogru aksanlarla yazilir -- bunlar LLM prompt'larina birebir enjekte
ediliyor, ASCII yazim modele yanlis stil sinyali verir. Python UTF-8 kaynak
dosyalarinda Turkce karakterlerin hicbir teknik sakincasi yoktur; ASCII
kisitlamasi sadece ReportLab PDF font render sorunuyla ilgiliydi, kaynak
koda genellenmemeliydi.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Yollar
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
SEED_DIR = DATA_DIR / "seed"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "model"

SEED_FILE = SEED_DIR / "seed.jsonl"        # few-shot yemi olarak kullanilir
GOLD_FILE = SEED_DIR / "gold.jsonl"        # few-shot'ta ASLA kullanilmaz, saf test
RAW_FILE = RAW_DIR / "amplified.jsonl"     # Ollama ciktisi (ham)
CLEAN_FILE = PROCESSED_DIR / "clean.csv"   # temizlenmis + dedup

TRAIN_FILE = PROCESSED_DIR / "train.csv"
VAL_FILE = PROCESSED_DIR / "val.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"
GOLD_TEST_FILE = PROCESSED_DIR / "gold_test.csv"

for _d in (SEED_DIR, RAW_DIR, PROCESSED_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Kategori taksonomisi
#
# Ayrim ilkesi: kategori, bildirimin hangi BAKIM EKIBINE yonlendirilecegini
# belirtir. Arizanin nesnesi degil, sorumlusu belirleyicidir.
# ---------------------------------------------------------------------------

CATEGORIES = {
    "arac_tren": {
        "display": "Araç / Tren",
        "color": "#2563eb",
        "scope": (
            "Trenin üzerindeki her şey: vagon kapısı, fren sistemi, klima, "
            "çer/motor, kabin ekipmanı, tekerlek, koltuk, vagon içi aydınlatma, "
            "vagon içi anons cihazı, makinist kabini"
        ),
        "exclude": (
            "Perondaki peron kapısı (PSD) 'istasyon_mekanik' kategorisine girer. "
            "Trenin gecikmesi 'yolcu_operasyon' kategorisine girer."
        ),
    },
    "istasyon_mekanik": {
        "display": "İstasyon Mekanik",
        "color": "#0891b2",
        "scope": (
            "İstasyondaki hareketli/mekanik ekipman: yürüyen merdiven, asansör, "
            "peron kapısı (PSD), turnikenin FİZİKSEL arızası (kol dönmüyor, "
            "kapak takılı, gövde hasarlı), otomatik giriş kapıları, bariyerler"
        ),
        "exclude": (
            "Turnikenin kart okumaması 'yazilim_sistem' kategorisine girer. "
            "Ekipmanın elektriksiz kalması 'elektrik_enerji' kategorisine girer."
        ),
    },
    "elektrik_enerji": {
        "display": "Elektrik / Enerji",
        "color": "#f59e0b",
        "scope": (
            "Enerji besleme ve aydınlatma: istasyon aydınlatması, elektrik "
            "kesintisi, jeneratör, UPS, elektrik panosu, katener hattı, üçüncü "
            "ray, trafo, kablo arızası, sigorta atması"
        ),
        "exclude": (
            "Cihazın enerjisi varken yazılımsal hata vermesi 'yazilim_sistem' "
            "kategorisine girer."
        ),
    },
    "yazilim_sistem": {
        "display": "Yazılım / Sistem / Bilet",
        "color": "#7c3aed",
        "scope": (
            "Yazılım, ağ ve biletleme: bilet satış otomatı, İstanbulkart "
            "okuyucu YAZILIMI, yolcu bilgilendirme ekranları (PID), sunucu, "
            "ağ/internet kesintisi, uygulama donması, hata mesajı, veritabanı, "
            "SCADA arayüzü"
        ),
        "exclude": (
            "Ekranın fiziksel kırılması 'istasyon_mekanik', ekranın sönmesi/"
            "enerjisiz kalması 'elektrik_enerji' kategorisine girer."
        ),
    },
    "guvenlik_emniyet": {
        "display": "Güvenlik / Emniyet",
        "color": "#dc2626",
        "scope": (
            "Güvenlik ve can emniyeti: CCTV/kamera sistemi, yangın algılama ve "
            "söndürme, acil durum butonu, acil çıkış, yetkisiz giriş, turnikeden "
            "atlama, şüpheli paket, güvenlik ihlali, anons ile tahliye"
        ),
        "exclude": (
            "Kameranın elektriğinin gitmesi 'elektrik_enerji' kategorisine girer."
        ),
    },
    "altyapi_insaat": {
        "display": "Altyapı / İnşaat",
        "color": "#78716c",
        "scope": (
            "Sabit yapı ve inşaat: su sızıntısı, tavan/duvar/zemin hasarı, "
            "çatlak, tünel yapısı, drenaj, kanalizasyon, ray hattının yapısal "
            "durumu, merdiven basamağı kırılması, korkuluk, fayans"
        ),
        "exclude": (
            "Tren içi zemin hasarı 'arac_tren' kategorisine girer. "
            "Sızıntı kaynaklı kirlilik değil, sızıntının kendisi buraya girer."
        ),
    },
    "yolcu_operasyon": {
        "display": "Yolcu / Operasyon",
        "color": "#059669",
        "scope": (
            "Sefer ve yolcu yönetimi: sefer gecikmesi, sefer iptali, seferlerin "
            "seyreltilmesi, anons yapılmaması/yanlış anons, peron yoğunluğu, "
            "kayıp eşya, yolcu yönlendirme, personel eksikliği, tarife sorunu"
        ),
        "exclude": (
            "Gecikmenin teknik sebebi ayrı bir bildirim olarak geldiyse o "
            "bildirim kendi teknik kategorisine girer."
        ),
    },
    "temizlik_cevre": {
        "display": "Temizlik / Çevre",
        "color": "#65a30d",
        "scope": (
            "Temizlik ve çevresel koşullar: kirlilik, çöp birikmesi, koku, "
            "tuvalet temizliği, döküntü, buzlanma, kaygan zemin, haşere/"
            "kemirgen, kış şartları (tuzlama), grafiti"
        ),
        "exclude": (
            "Sızıntı kaynaklı ıslaklık 'altyapi_insaat' kategorisine girer "
            "(kaynak sorunu temizlik değil)."
        ),
    },
}

CATEGORY_KEYS = list(CATEGORIES.keys())
NUM_LABELS = len(CATEGORY_KEYS)

LABEL2ID = {k: i for i, k in enumerate(CATEGORY_KEYS)}
ID2LABEL = {i: k for k, i in LABEL2ID.items()}
DISPLAY_NAME = {k: v["display"] for k, v in CATEGORIES.items()}
CATEGORY_COLOR = {k: v["color"] for k, v in CATEGORIES.items()}


# ---------------------------------------------------------------------------
# Veri uretim ayarlari
# ---------------------------------------------------------------------------

SEED_PER_CATEGORY = 12      # few-shot yemi
GOLD_PER_CATEGORY = 10      # bozulmamis test seti
TARGET_PER_CATEGORY = 200   # nihai egitim verisi (cogaltma sonrasi)

MIN_CHARS = 8               # bundan kisa bildirimler atilir
MAX_CHARS = 300             # bundan uzun bildirimler atilir
# Benzerlik esikleri. IKI AYRI DEGER, cunku esik iki farkli isi goruyor ve
# hata maliyetleri simetrik degil (19 Agu 2026 kalibrasyonu):
#
#   URETIM (generate_data/generate_seed -- yeni kaydi reddet):
#     yanlis reddetme -> iyi bir cumle bosa gider, kota harcanir
#     kacirma         -> veri biraz tekrarli olur
#     Maliyetler dengeli, 0.85 uygun.
#
#   BOLME (preprocess -- kumeleme):
#     yanlis birlestirme -> iki kayit ayni bolmeye duser, kucuk cesitlilik kaybi
#     kacirma            -> ayni cumlenin varyasyonu hem train hem test'e duser,
#                           METRIK SISER (sahte yuksek dogruluk)
#     Kacirmanin bedeli cok daha agir, o yuzden daha agresif: 0.80.
#
# Kalibrasyon olcumu (1600 kayit, ayni kategori icinde):
#   0.85 -> 1 cift birlesti | 0.82 -> 8 | 0.80 -> 23 | 0.78 -> 39 (kumeler 5'e
#   zincirlenmeye basliyor) | 0.75 -> 88 (fazla agresif)
# 0.80-0.85 bandinda gercek anlamsal kopyalar oldugu elle dogrulandi, orn:
#   "Taksim istasyonunda 4 numarali vagondaki yolcu anons cihazi ses vermiyor."
#   "Yolcu anons cihazi ses vermiyor 4. vagon"                        (0.843)
# Ayni bantta gercekten FARKLI arizalar da var (olcut sozcuksel, anlamsal degil):
#   "makinist kabini sag tarafi ayna kirik" / "makinist kabini saati durmus"
#                                                                     (0.804)
# 0.80'de kume boyutu 2'de kaliyor (zincirleme yok), bedel 23 kayit / 1600.
NEAR_DUP_THRESHOLD = 0.85   # uretimde yeni kayit reddi
CLUSTER_THRESHOLD = 0.80    # preprocess'te split oncesi kumeleme
NEAR_DUP_JACCARD = 0.55     # kelime kumesi ortusme esigi (SequenceMatcher yaninda)

# Bildirimlerin yazim stilleri. Gercek personel her zaman duzgun yazmaz.
STYLE_VARIANTS = {
    "standart": (
        "Düzgün yazılmış, kurallı tam cümle, noktalama doğru. Resmi bildirim "
        "dili. UZUNLUK: 8-18 kelime."
    ),
    "devrik": (
        "Acele yazılmış, KISA ve eksiltili. Yüklem başa gelebilir, özne veya "
        "tamlayan düşebilir, noktalama eksik olabilir. UZUNLUK: 4-9 kelime. "
        "DOĞRU örnek: 'Yürüyen merdiven durdu 2. peron'. "
        "DOĞRU örnek: 'Kapı açılmıyor A2 vagon'. "
        "YANLIŞ örnek (bunu ÜRETME, çok uzun ve edebi devrik): "
        "'Kırılmış dönme kolu Mecidiyeköy ana bilet holü 5 numaralı "
        "turnikenin, fiziki geçişe engel duruyor.'"
    ),
    "yazim_yanlisi": (
        "Türkçe karakter eksikliği (ç, ğ, ı, ö, ş, ü yerine ASCII), harf "
        "düşmesi, klavye hatası. Anlam anlaşılır kalmalı. UZUNLUK: 5-14 kelime. "
        "Örnek: 'asansor calismiyo', 'elektirik kesildi 3. perondaki'."
    ),
    "cok_kisa": (
        "Telgraf tarzı, sadece ekipman + belirti. UZUNLUK: 3-6 kelime. "
        "Örnek: 'Turnike 3 bozuk', 'Peronda su var'."
    ),
}

STYLE_KEYS = list(STYLE_VARIANTS.keys())

# Prompt'lara enjekte edilecek slot degerleri -- cesitliligi prompt seviyesinde
# zorlamak icin. Cogaltma asamasinda her cagrida rastgele secilir.
SLOT_VALUES = {
    "istasyon": [
        "Yenikapı", "Taksim", "Şişhane", "Levent", "Kadıköy", "Mecidiyeköy",
        "Vezneciler", "Kartal", "Ataköy", "Bağcılar", "Gayrettepe", "Hacıosman",
        "Uzunçayır", "Bostancı", "Kirazlı", "Esenler", "Sanayi Mahallesi",
        "Şişli", "Üsküdar", "Topkapı", "Şirinevler",
    ],
    "konum": [
        "1. peron", "2. peron", "kuzey giriş", "güney çıkış", "gişe önü",
        "alt kat", "üst kat", "koridor", "bilet holü", "tünel ağzı",
        "personel kapısı", "asansör önü", "turnike bölgesi",
    ],
    "zaman": [
        "sabah vardiyasında", "gece vardiyasında", "akşam yoğun saatte",
        "öğle saatlerinde", "servis başlangıcında", "servis sonunda",
        "hafta sonu", "az önce", "bu sabah 07:20'de",
    ],
    "aciliyet": [
        "acil müdahale gerekiyor", "yolcu güvenliği riskli",
        "sefer aksıyor", "şimdilik idare ediliyor",
        "tekrarlayan bir arıza", "ilk defa oluyor", "bilgi amaçlı",
    ],
}


# ---------------------------------------------------------------------------
# LLM saglayici ayarlari
# ---------------------------------------------------------------------------

# Seed ve gold uretimi icin: "gemini" | "claude" | "groq" | "openrouter"
# NOT (13 Agu 2026): karsilastirma sonucu OpenRouter/Nemotron 3 Ultra kazandi
# (seed %17, gold %9 isaretli -- Gemini/Groq'tan daha iyi, ustelik ucretsiz).
SEED_PROVIDER = "openrouter"

GEMINI_MODEL = "gemini-2.5-flash"        # AI Studio'da guncel id'yi teyit et
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

# Groq: kredi karti gerektirmez, OpenAI-uyumlu API, genis ucretsiz kota
# (30 istek/dk, 14.400 istek/gun civari -- modele gore degisir).
# NOT (13 Agu 2026): llama-3.3-70b-versatile bu proje icin YETERSIZ cikti --
# coklu kisit (kategori + stil + uzunluk + kod orani) icin talimat takibi
# zayif. Config'teki Turkce metnin ASCII olmasi da katkida bulunmus olabilir
# (simdi duzeltildi) -- yeniden denenmeli.
GROQ_MODEL = "llama-3.3-70b-versatile"

# OpenRouter: kredi karti istemeyen ikinci ucretsiz secenek (gunde 50 istek).
# Katalog surekli degisiyor, sadece acik agirlikli modeller ucretsiz.
# NOT (13 Agu 2026): canli listede gpt-oss-120b yoktu (sadece 20b), en genis
# olceki secenek NVIDIA Nemotron 3 Ultra (550B toplam / 55B aktif param) --
# talimat takibinde guclu (IFBench 81.7), coklu dil destekli. Turkce'de
# ozel olcum yok, ama olcek avantaji var.
OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

LLM_TEMPERATURE = 1.0       # cesitlilik istiyoruz, dogruluk degil
LLM_MAX_RETRIES = 3


# --- Cogaltma (Adim 2b) ------------------------------------------------------
# Karar (18 Agu 2026): HIBRIT strateji. OpenRouter/Nemotron kaliteyi belirledi
# ama ucretsiz katman ~50 istek/gun; 1600 ornek icin bu yetmiyor. Bu yuzden
# birincil saglayici OpenRouter, KALICI hata (kota/429/401) gelince kalan is
# otomatik olarak yerel Ollama'ya devrediliyor. Boylece kaliteli modelden
# alabildigimiz kadar aliyoruz, kalanini bedelsiz yerelde tamamliyoruz.
# Uretilen her kaydin 'kaynak' alani hangi modelden geldigini tasir, boylece
# iki modelin katkisi sonradan ayrilabilir (rapor icin de kullanisli).
AMPLIFY_PROVIDER = "hybrid"     # "hybrid" | "openrouter" | "ollama"

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:14b"
OLLAMA_TIMEOUT = 300            # 14B model Apple Silicon'da yavas olabilir

# Ollama'nin varsayilan baglam penceresi 2048 token. Cogaltma prompt'u
# (kategori kapsami + stil tanimi + few-shot + "bunlari tekrarlama" listesi)
# tek basina bunun buyuk kismini yiyor, cikartiya yer kalmiyor: model 25 ornek
# istenmesine ragmen 1-2 ornek dondurup kesiliyordu. Acikca genisletiyoruz.
# qwen2.5:14b 32768 token destekliyor, 8192 hem bol hem 16GB RAM'de guvenli.
OLLAMA_NUM_CTX = 8192
OLLAMA_NUM_PREDICT = 4096

# Her LLM cagrisinda tek (kategori, stil) ikilisi icin kac ornek istenecegi.
# Tek cagrida tek stil istemek uzunluk kuralina uyumu ciddi artiriyor: model
# ayni anda 4 farkli uzunluk araligini yonetmek zorunda kalmiyor.
#
# NOT (18 Agu 2026): 25'ten 40'a cikarildi. Baglayici kisit ornek sayisi degil
# CAGRI sayisi: OpenRouter ucretsiz katmani ~50 istek/gun ve 25'lik partilerle
# 1600 ornek 64 cagri gerektiriyordu -- yani son ~350 kayit zorunlu olarak
# Ollama'ya kaliyordu. 40'lik partiyle ~40 cagri yetiyor ve veri tek gunde
# tamamen Nemotron'dan gelebiliyor. Olcum: Nemotron 40 kaydi 4 cagride,
# %5 isaretli oranla uretti; qwen2.5:14b ayni isi 7 cagride %18 isaretli
# oranla ve uydurma istasyon adlariyla yapti.
AMPLIFY_BATCH_SIZE = 40

# Cogaltmada few-shot olarak kac seed ornegi gosterilecek ve modele "bunlari
# tekrar etme" diye kac mevcut ornek hatirlatilacagi.
AMPLIFY_FEWSHOT_N = 6
AMPLIFY_AVOID_N = 12


# ---------------------------------------------------------------------------
# Egitim hiperparametreleri
# ---------------------------------------------------------------------------

BASE_MODEL = "dbmdz/bert-base-turkish-cased"

MAX_LENGTH = 64             # ariza bildirimleri kisa; 64 token fazlasiyla yeter
NUM_EPOCHS = 12
BATCH_SIZE = 16

# DIKKAT (19 Agu 2026): burada onceden 2e-5 yaziyordu ve model OGRENMIYORDU.
# 2e-5, BERT'i TAM fine-tuning ederken kullanilan standart degerdir; biz LoRA
# kullaniyoruz. LoRA'da parametrelerin sadece %0.54'u (595.976) egitiliyor,
# adaptorler sifirdan basliyor ve siniflandirma basligi rastgele baslatiliyor
# -- bu kadar kucuk bir ogrenme hiziyla agirliklar anlamli mesafe kat edemiyor.
# Olculdu (5 epoch, val macro-F1):
#     2e-5 -> 0.134   (kayip 2.146 -> 2.073; rastgele seviye ln(8)=2.079)
#     1e-4 -> 0.393
#     3e-4 -> 0.850
#     5e-4 -> 0.875
# 15 epoch'ta: 5e-4 -> 0.930 (epoch 13), 1e-3 -> 0.938 (epoch 11).
# Ikisi arasindaki fark val setinde ~1 ornek (n=160), yani gurultu icinde.
# 5e-4 secildi: ayni sonucu daha yumusak bir egriyle veriyor.
# Ders: hiperparametreyi literaturden kopyalamak yetmiyor, EGITIM YONTEMINE
# gore ayarlamak gerekiyor.
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Egitimde aksansiz (ASCII'ye katlanmis) kopyalar da eklensin mi?
#
# NEDEN (19 Agu 2026, nedensel olarak olculdu): test+gold'daki aksan iceren
# 173 kaydin aksanlari kaldirilip yeniden tahmin edildi -- ICERIK AYNI, sadece
# ç/ğ/ı/ö/ş/ü duruyor:
#     orijinal      157/173 = 0.9075
#     ASCII katlanmis 146/173 = 0.8439      -> 6.4 puan dusus
# Mekanizma BERTurk tokenizer'inda gorunuyor:
#     "asansör" -> 1 parca  ['asansör']
#     "asansor" -> 3 parca  ['asa', '##ns', '##or']
# Aksan dusunce kelime anlamsiz alt-parcalara boluuyor.
#
# Gercek hayatta personel Ingilizce klavyeyle yazip aksan dusurebiliyor
# (config'in kendi "Onemli tasarim karari" notu da bunu soyluyor), yani bu
# dayaniklilik sus degil gereklilik.
#
# Cozum: train'deki aksanli kayitlarin ASCII kopyalari egitime eklenir. Model
# "güvenlik" ile "guvenlik"in ayni sey oldugunu ogrenir. Bedava (API yok).
# SIZINTI RISKI YOK: preprocess'teki kumeleme zaten aksan-duyarsiz calisiyor
# (review.normalize aksanlari kaldiriyor), yani bir train kaydinin ASCII
# kopyasi test'teki bir kayitla eslesiyorsa o ikisi zaten ayni kumededir.
AUGMENT_ASCII_FOLD = True

# PEFT / LoRA
USE_LORA = True
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["query", "value"]

# Basari kriterleri
TARGET_ACCURACY = 0.85
TARGET_MACRO_F1 = 0.82
MIN_PER_CLASS_F1 = 0.75


# ---------------------------------------------------------------------------
# Servis ayarlari
# ---------------------------------------------------------------------------

# Bu esigin altinda kategori kesin atanmaz, arayuzde manuel inceleme uyarisi
# cikar.
#
# KALIBRE EDILDI (19 Agu 2026) — k-fold OUT-OF-FOLD tahminlerle, 1280 kayit /
# 102 hata uzerinden. Bkz. src/calibrate.py ve model/kalibrasyon.json.
#
# Neden OOF: esigi test'e bakarak secmek test setini karar surecine sokar;
# val.csv ise epoch seciminde kullanildigi icin model orada FAZLA EMIN
# (olculdu: yanlis tahminlerde ort. guven val 0.892 / test 0.758 / OOF 0.773)
# ve zaten sadece 9 hata iceriyor -- o kadar az hatayla esik secmek gurultuyu
# kalibrasyon sanmaktir. OOF ile hata sayisi 102'ye cikti.
#
# Tarama (OOF, 102 hata):
#   esik  trafik  yakalanan  bosuna  precision  recall
#   0.60   3.6%     25/102     21      0.543    0.245
#   0.70   5.9%     37/102     38      0.493    0.363   <- eski deger
#   0.75   7.3%     47/102     47      0.500    0.461   <- SECILEN
#   0.80   8.9%     49/102     65      0.430    0.480
# 0.75, 0.70'i domine ediyor: ayni precision ama 10 hata daha yakaliyor.
# 0.80'de precision cokuyor -- diz noktasi 0.75.
#
# Ikinci gerekce reliability diagram'dan: model 0.80-0.95 bandinda belirgin
# FAZLA EMIN (0.856 ortalama guven -> 0.686 gercek dogruluk). O bantta guven
# degeri gorundugu kadar guvenilir degil, esigi yukseltmek bunu telafi ediyor.
# (0.95+ bandinda -- verinin %83'u -- model iyi kalibre: 0.994 -> 0.975.)
CONFIDENCE_THRESHOLD = 0.75
LOW_CONFIDENCE_MESSAGE = "Düşük Güven: Manuel İnceleme Önerilir"

# --- Ikincil kategori (taksonomi sinir sorunlarina genel cozum) --------------
# Bazi bildirimler GERCEKTEN iki kategoriye birden girer. Somut ornek:
# "Acil tahliye anonsu yogun saatlerde peronda net duyulamiyor." -- config'in
# kendi metnine gore guvenlik_emniyet ("anons ile tahliye") ve yolcu_operasyon
# ("anons yapilmamasi/yanlis anons") kapsamlarinin IKISINE de giriyor.
#
# Bunu taksonomiye sinir kurallari yazarak cozmek olceklenmiyor: 8 kategoride
# 28 cift var ve gercek veriye gecince bugun bilmedigimiz yenileri cikacak.
# Bunun yerine modelin ZATEN urettigi bilgiyi kullaniyoruz: marj (top1-top2)
# kucukse model iki kategori arasinda kararsiz demektir.
#
# Olculdu: top-1 dogruluk 0.913/0.925 iken TOP-2 dogruluk 0.963/0.975.
#
# KALIBRE EDILDI (19 Agu 2026) — k-fold OOF, 102 hata:
#   marj  trafik  kurtarilan  bosuna  oran
#   0.20   2.5%     12/102      17    0.71
#   0.30   4.1%     20/102      25    0.80   <- SECILEN (tepe)
#   0.40   5.2%     24/102      34    0.71   <- eski deger
#   0.50   6.6%     30/102      44    0.68
#
# DUZELTME NOTU: 0.40 daha once GOLD setinin 8 hatasina bakarak secilmis ve
# "kurtarma/bosuna orani 4.0" diye kaydedilmisti. 102 hatali OOF tabaninda
# gercek oran 0.80 cikti -- yani o olcum tamamen gurultuydu. Az orneklemle
# yapilan kalibrasyonun ne kadar yanildigina dair somut bir ders.
MARGIN_THRESHOLD = 0.30
SECONDARY_CATEGORY_MESSAGE = "Sınırda Bildirim: İkinci Kategori de Değerlendirilmeli"

API_HOST = "0.0.0.0"
API_PORT = 8000
