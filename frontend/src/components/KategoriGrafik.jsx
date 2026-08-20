import { useState } from "react";

// Kategori dagilim grafigi: gecmis havuz + canli eklenen bildirimler birlikte.
// Her bar iki segmentten olusur -- SOLUK kisim gecmis havuz, PARLAK kisim
// canli (API'ye yazilan) kayitlar. Boylece "bu cumleler eklendikce guncellenen
// tablo" istegi tek bir grafikte hem toplami hem buyume kaynagini gosteriyor.

const yuzde = (x) => `${(x * 100).toFixed(1)}%`;

export default function KategoriGrafik({ veri }) {
  const [ipucu, setIpucu] = useState(null); // {x, y, icerik} | null

  if (!veri.length) return null;

  const enBuyuk = Math.max(...veri.map((d) => d.count), 1);
  const sirali = [...veri].sort((a, b) => b.count - a.count);
  const toplamCanli = veri.reduce((s, d) => s + d.live_count, 0);

  return (
    <section className="grafik">
      <div className="grafik-baslik-satir">
        <h3 className="bolum-baslik">Arıza Kategori Dağılımı</h3>
        <div className="grafik-lejant">
          <span><i className="lejant-nokta lejant-toplam" /> toplam</span>
          <span><i className="lejant-nokta lejant-canli" /> canlı eklenen ({toplamCanli})</span>
        </div>
      </div>

      <div className="grafik-govde">
        {sirali.map((d) => {
          const gecmisOran = (d.count - d.live_count) / enBuyuk;
          const canliOran = d.live_count / enBuyuk;
          return (
            <div
              className="grafik-satir"
              key={d.category}
              onMouseEnter={(e) =>
                setIpucu({
                  y: e.currentTarget.offsetTop,
                  icerik: `${d.label}: ${d.count} kayıt (${yuzde(d.ratio)})` +
                    (d.live_count ? ` · ${d.live_count} canlı` : ""),
                })
              }
              onMouseLeave={() => setIpucu(null)}
            >
              <span className="grafik-ad">{d.label}</span>
              <div className="grafik-ray">
                <div
                  className="grafik-dolgu grafik-dolgu-gecmis"
                  style={{ width: yuzde(gecmisOran), "--k": d.color }}
                />
                {d.live_count > 0 && (
                  <div
                    className="grafik-dolgu grafik-dolgu-canli"
                    style={{
                      width: yuzde(canliOran),
                      left: yuzde(gecmisOran),
                      "--k": d.color,
                    }}
                  />
                )}
              </div>
              <span className="grafik-deger">{d.count}</span>
            </div>
          );
        })}
      </div>

      {ipucu && (
        <div className="grafik-ipucu" style={{ top: ipucu.y }}>
          {ipucu.icerik}
        </div>
      )}
    </section>
  );
}
