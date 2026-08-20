import { useState } from "react";
import { tahminiDogrula } from "../api";

// Tahmin sonucu: birincil kategori, guven, uyarilar, benzerlik ve olasilik dagilimi.

const yuzde = (x) => `${(x * 100).toFixed(1)}%`;

export default function SonucKarti({ sonuc, kategoriler, onDogrulandi }) {
  const [onayDurumu, setOnayDurumu] = useState(null); // null | "dogru" | "yanlis" | "duzeltiliyor"
  const [gonderiliyor, setGonderiliyor] = useState(false);

  // Dagilimi olasiliga gore sirala; backend sozluk donduruyor.
  const dagilim = Object.entries(sonuc.probabilities)
    .map(([anahtar, olasilik]) => ({
      anahtar,
      olasilik,
      ad: kategoriler[anahtar]?.label ?? anahtar,
      renk: kategoriler[anahtar]?.color ?? "#7a828f",
    }))
    .sort((a, b) => b.olasilik - a.olasilik);

  const dogrulanabilir = sonuc.log_id > 0 && onayDurumu === null;

  async function gonder(dogru, duzeltilmisKategori = null) {
    setGonderiliyor(true);
    try {
      await tahminiDogrula(sonuc.log_id, dogru, duzeltilmisKategori);
      setOnayDurumu(dogru ? "dogru" : "yanlis");
      onDogrulandi?.();
    } catch {
      /* sessizce yut -- onay ikincil bir eylem, akisi kesmemeli */
    } finally {
      setGonderiliyor(false);
    }
  }

  return (
    <section className="sonuc" style={{ "--vurgu": sonuc.color }} aria-live="polite">
      <header className="sonuc-baslik">
        <span className="rozet" style={{ "--k": sonuc.color }}>
          <span className="nokta" />
          {sonuc.label}
        </span>

        {sonuc.secondary_category && (
          <>
            <span className="arti" aria-hidden="true">+</span>
            <span
              className="rozet rozet-ikincil"
              style={{ "--k": kategoriler[sonuc.secondary_category]?.color }}
              title="Sınırda bildirim — ikinci ekip de değerlendirmeli"
            >
              {sonuc.secondary_label} · {yuzde(sonuc.secondary_confidence)}
            </span>
          </>
        )}
      </header>

      <div className="guven">
        <div className="guven-ust">
          <span className="guven-etiket">Güven</span>
          <span className="guven-deger">{yuzde(sonuc.confidence)}</span>
        </div>
        <div className="ray">
          <div
            className="dolgu"
            style={{ width: yuzde(sonuc.confidence), "--k": sonuc.color }}
          />
        </div>
      </div>

      {sonuc.manual_review && (
        <div className="uyari">
          <div className="uyari-baslik">
            <span aria-hidden="true">⚠</span> Manuel İnceleme Önerilir
          </div>
          <ul>
            {sonuc.low_confidence && (
              <li>
                Güven eşiğin altında (<b>{yuzde(sonuc.confidence)}</b>) — model
                bu bildirimden emin değil.
              </li>
            )}
            {sonuc.secondary_category && (
              <li>
                Model <b>{sonuc.label}</b> ile <b>{sonuc.secondary_label}</b>{" "}
                arasında kararsız (fark <b>{yuzde(sonuc.margin)}</b>). Bu
                bildirim gerçekten iki ekibi birden ilgilendiriyor olabilir.
              </li>
            )}
          </ul>
        </div>
      )}

      {/* Onay: kullanicinin isaretledigi kayitlar SADECE bunlar /logs/export
          ile disari alinip elle egitime katilabilir -- bkz. backend/src/db.py */}
      <div className="onay">
        {onayDurumu === null && (
          <>
            <span className="onay-soru">Bu tahmin doğru mu?</span>
            <button
              className="onay-btn onay-dogru"
              disabled={!dogrulanabilir || gonderiliyor}
              onClick={() => gonder(true)}
            >
              ✓ Doğru
            </button>
            <button
              className="onay-btn onay-yanlis"
              disabled={!dogrulanabilir || gonderiliyor}
              onClick={() => setOnayDurumu("duzeltiliyor")}
            >
              ✕ Yanlış
            </button>
          </>
        )}
        {onayDurumu === "duzeltiliyor" && (
          <>
            <span className="onay-soru">Doğrusu:</span>
            <select
              className="onay-secim"
              disabled={gonderiliyor}
              defaultValue=""
              onChange={(e) => e.target.value && gonder(false, e.target.value)}
            >
              <option value="" disabled>seçin…</option>
              {Object.values(kategoriler)
                .filter((k) => k.category !== sonuc.category)
                .map((k) => (
                  <option key={k.category} value={k.category}>{k.label}</option>
                ))}
            </select>
            <button
              className="onay-btn onay-vazgec"
              onClick={() => gonder(false)}
              disabled={gonderiliyor}
              title="Kategori belirtmeden sadece 'yanlış' olarak işaretle"
            >
              atla
            </button>
          </>
        )}
        {onayDurumu === "dogru" && <span className="onay-tesekkur">✓ Teşekkürler, kaydedildi.</span>}
        {onayDurumu === "yanlis" && <span className="onay-tesekkur">✓ Düzeltme kaydedildi.</span>}
        {sonuc.log_id <= 0 && onayDurumu === null && (
          <span className="onay-soru onay-devre-disi">
            (bu istek tekrar olduğu için loglanmadı)
          </span>
        )}
      </div>

      {sonuc.similar.total_found > 0 && (
        <div className="benzer">
          <h3 className="bolum-baslik">
            Benzer Kayıtlar
            <span className="benzer-sayi">
              {" "}
              — {sonuc.similar.total_found} kayıt bulundu
              {sonuc.similar.shown < sonuc.similar.total_found &&
                ` (ilk ${sonuc.similar.shown} gösteriliyor)`}
            </span>
          </h3>
          {sonuc.similar.distribution.map((d) => (
            <div className="dagilim-satir" key={d.category}>
              <span className="dagilim-ad">{d.label}</span>
              <div className="ray ince">
                <div
                  className="dolgu"
                  style={{ width: yuzde(d.ratio), "--k": d.color }}
                />
              </div>
              <span className="dagilim-deger">
                {d.count} · {yuzde(d.ratio)}
              </span>
            </div>
          ))}
        </div>
      )}

      <div className="dagilim">
        <h3 className="bolum-baslik">Tüm kategoriler</h3>
        {dagilim.map((d, i) => (
          <div
            className={`dagilim-satir${i === 0 ? " birincil" : ""}`}
            key={d.anahtar}
          >
            <span className="dagilim-ad">{d.ad}</span>
            <div className="ray ince">
              <div
                className="dolgu"
                style={{
                  width: yuzde(d.olasilik),
                  "--k": d.renk,
                  animationDelay: `${60 + i * 45}ms`,
                }}
              />
            </div>
            <span className="dagilim-deger">{yuzde(d.olasilik)}</span>
          </div>
        ))}
      </div>

      <footer className="sonuc-alt">
        <span>Yanıt süresi {sonuc.response_time_ms.toFixed(1)} ms</span>
        <span>top-2 farkı {yuzde(sonuc.margin)}</span>
      </footer>
    </section>
  );
}
