// Backend istemcisi.
//
// Taban adres derleme zamaninda VITE_API_URL ile degistirilebilir; varsayilan
// gelistirme sunucusu. Backend ayri portta calistigi icin CORS backend
// tarafinda aciktir (bkz. backend/main.py notu).

const TABAN = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

async function istek(yol, secenekler) {
  let yanit;
  try {
    yanit = await fetch(`${TABAN}${yol}`, secenekler);
  } catch {
    // Aglara ulasilamiyor: backend kapali olmasi en olasi sebep. Kullaniciya
    // "bilinmeyen hata" demek yerine ne yapmasi gerektigini soyluyoruz.
    throw new Error(
      `Servise ulaşılamıyor (${TABAN}). Backend çalışıyor mu?\n` +
        `Başlatmak için: ./venv/bin/uvicorn backend.main:app --port 8000`
    );
  }

  if (!yanit.ok) {
    let detay = `Sunucu hatası (${yanit.status})`;
    try {
      const govde = await yanit.json();
      if (govde.detail) {
        // FastAPI dogrulama hatalarinda detail bir dizi olabilir
        detay = Array.isArray(govde.detail)
          ? govde.detail.map((d) => d.msg).join(", ")
          : govde.detail;
      }
    } catch {
      /* govde JSON degilse varsayilan mesaj kalir */
    }
    throw new Error(detay);
  }

  return yanit.json();
}

export const tahminEt = (text) =>
  istek("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

export const kategorileriGetir = () => istek("/categories");
export const ornekleriGetir = (count = 8) => istek(`/examples?count=${count}`);
export const modelBilgisiGetir = () => istek("/model-info");
