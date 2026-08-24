import { useEffect, useState } from "react";
import {
  kategoriIstatistikGetir,
  kategorileriGetir,
  modelBilgisiGetir,
  ornekleriGetir,
  tahminEt,
} from "./api";
import SonucKarti from "./components/SonucKarti";
import KategoriGrafik from "./components/KategoriGrafik";
import "./App.css";

const MAKS_KARAKTER = 300; // backend'deki C.MAX_CHARS ile ayni

export default function App() {
  const [metin, setMetin] = useState("");
  const [sonuc, setSonuc] = useState(null);
  const [hata, setHata] = useState(null);
  const [yukleniyor, setYukleniyor] = useState(false);

  const [kategoriler, setKategoriler] = useState({});
  const [ornekler, setOrnekler] = useState([]);
  const [modelBilgi, setModelBilgi] = useState(null);
  const [taksonomiAcik, setTaksonomiAcik] = useState(false);
  const [kategoriIstatistik, setKategoriIstatistik] = useState([]);

  const istatistigiTazele = () =>
    kategoriIstatistikGetir().then(setKategoriIstatistik).catch(() => {});

  // Acilista sabit verileri cek. Backend kapaliysa hata gosterilir ama
  // uygulama cokmez -- kullanici ne yapmasi gerektigini gorur.
  useEffect(() => {
    Promise.all([kategorileriGetir(), ornekleriGetir(8), modelBilgisiGetir()])
      .then(([kats, orns, bilgi]) => {
        setKategoriler(Object.fromEntries(kats.map((k) => [k.category, k])));
        setOrnekler(orns);
        setModelBilgi(bilgi);
      })
      .catch((e) => setHata(e.message));
    istatistigiTazele();
  }, []);

  // Ortam isigi sonucun kategorisiyle renkleniyor (bkz. App.css body::before)
  useEffect(() => {
    document.documentElement.style.setProperty(
      "--vurgu",
      sonuc?.color ?? "#3b82f6"
    );
  }, [sonuc]);

  async function analizEt(e) {
    e?.preventDefault();
    const temiz = metin.trim();
    if (!temiz || yukleniyor || temiz.length > MAKS_KARAKTER) return;

    setYukleniyor(true);
    setHata(null);
    try {
      setSonuc(await tahminEt(temiz));
      // Bu bildirim de loglandigi icin grafik hemen guncellensin -- "cumleler
      // eklendikce surekli guncellenen tablo" istegi budur.
      istatistigiTazele();
    } catch (e) {
      setHata(e.message);
      setSonuc(null);
    } finally {
      setYukleniyor(false);
    }
  }

  // Enter = gonder, Shift+Enter = alt satir
  function tusaBas(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      analizEt();
    }
  }

  const cokUzun = metin.length > MAKS_KARAKTER;

  return (
    <div className="uygulama">
      <header className="ust">
        <div>
          <h1>Arıza Tespit Sınıflandırıcı</h1>
          <p className="alt-baslik">
            Serbest metinli arıza bildirimini ilgili bakım ekibine yönlendirir
          </p>
        </div>
        {modelBilgi && (
          <div className="model-cip" title="Aktif model ve eğitim">
            <b>{modelBilgi.base_model.split("/").pop()}</b>
            <span>
              {modelBilgi.lora ? "LoRA" : "tam fine-tune"} · val F1{" "}
              {modelBilgi.best_val_macro_f1.toFixed(3)}
            </span>
          </div>
        )}
      </header>

      <form className="giris" onSubmit={analizEt}>
        <textarea
          value={metin}
          onChange={(e) => setMetin(e.target.value)}
          onKeyDown={tusaBas}
          placeholder="Örn: Yürüyen merdiven durdu 2. peron"
          rows={3}
          aria-label="Arıza bildirimi"
        />
        <div className="giris-alt">
          <span>
            <span className={cokUzun ? "sayac asim" : "sayac"}>
              {metin.length} / {MAKS_KARAKTER}
            </span>
            <span className="kisayol">↵</span>
          </span>
          <button
            type="submit"
            className="analiz-btn"
            disabled={!metin.trim() || yukleniyor || cokUzun}
          >
            {yukleniyor && <span className="donuyor" aria-hidden="true" />}
            {yukleniyor ? "Analiz ediliyor" : "Analiz Et"}
          </button>
        </div>
      </form>

      {hata && (
        <div className="hata-kutu" role="alert">
          {hata}
        </div>
      )}

      {sonuc && (
        <SonucKarti
          // key: her yeni tahminde bilesen YENIDEN kurulsun -- yoksa React
          // ayni orneği koruyup icerideki onayDurumu state'ini bir onceki
          // tahminden tasir (gercek bulunan hata, elle test ederken cikti).
          key={sonuc.log_id + sonuc.response_time_ms}
          sonuc={sonuc}
          kategoriler={kategoriler}
          onDogrulandi={istatistigiTazele}
        />
      )}

      {kategoriIstatistik.length > 0 && <KategoriGrafik veri={kategoriIstatistik} />}

      {ornekler.length > 0 && (
        <section className="ornekler">
          <h3 className="bolum-baslik">Örnek bildirimler</h3>
          <p className="ornek-not">
            Bu örnekler modelin eğitiminde <b>hiç kullanılmamış</b>, bağımsız
            bir kaynaktan üretilmiş test setinden geliyor.
          </p>
          <div className="ornek-liste">
            {ornekler.map((o) => (
              <button
                key={o.text}
                className="ornek"
                onClick={() => {
                  setMetin(o.text);
                  setSonuc(null);
                }}
                title={`stil: ${o.style}`}
              >
                {o.text}
              </button>
            ))}
          </div>
        </section>
      )}

      {Object.keys(kategoriler).length > 0 && (
        <section className="taksonomi">
          <button
            className="taksonomi-ac"
            onClick={() => setTaksonomiAcik((v) => !v)}
            aria-expanded={taksonomiAcik}
          >
            <span className={taksonomiAcik ? "ok acik" : "ok"}>▶</span>
            Kategoriler ve kapsamları ({Object.keys(kategoriler).length})
          </button>
          {taksonomiAcik && (
            <dl>
              {Object.values(kategoriler).map((k) => (
                <div key={k.category} className="taksonomi-satir">
                  <dt>
                    <span className="nokta" style={{ "--k": k.color }} />
                    {k.label}
                  </dt>
                  <dd>{k.scope}</dd>
                </div>
              ))}
            </dl>
          )}
        </section>
      )}
    </div>
  );
}
