"""Akciğer nodülü projesinin görüntü işleme fonksiyonları.

Defterlerde (notebooks/) kullanılan fonksiyonların toplu, yeniden kullanılabilir hali.
Girdi: LUNA25 nodül bloğunun orta kesiti, HU pencereleme sonrası 0-1 aralığında (128x128).

Örnek:
    import numpy as np
    from goruntu_isleme import pencere, bolutle_v2, ozellik_cikar
    k = pencere(hu_kesiti, merkez=-600, genislik=1500)
    maske, bilgi = bolutle_v2(k, ayrinti=True)
    print(ozellik_cikar(k, maske))
"""
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu, sobel
from skimage.measure import regionprops
from skimage.morphology import disk
from skimage.segmentation import watershed

PIKSEL_MM = 0.625                      # LUNA25 bloklarında düzlem içi piksel boyutu
YY, XX = np.mgrid[:128, :128]
R2 = (YY - 64) ** 2 + (XX - 64) ** 2   # merkeze uzaklığın karesi (nodül merkezdedir)
KENAR = np.zeros((128, 128), bool)
KENAR[:2, :] = KENAR[-2:, :] = KENAR[:, :2] = KENAR[:, -2:] = True


# ------------------------------------------------------------------ ön işleme
def pencere(v, merkez=-600, genislik=1500):
    """HU pencereleme = kontrast germe (MATLAB: imadjust). Sonuç 0-1 aralığında."""
    alt, ust = merkez - genislik / 2, merkez + genislik / 2
    return (np.clip(v, alt, ust) - alt) / (ust - alt)


def hu2k(hu, merkez=-600, genislik=1500):
    """Bir HU değerinin pencerelenmiş (0-1) karşılığı."""
    return (hu - (merkez - genislik / 2)) / genislik


KATI = hu2k(-400)     # katı doku / akciğer havası sınırı
BUZLU = hu2k(-750)    # buzlu cam (yarı katı) nodüller için alt eşik


# ------------------------------------------------------------------ morfoloji
def morf(A, islem, r):
    """Kenarı tekrarlayan dolguyla morfolojik işlem (MATLAB: padarray(..., 'replicate')).
    Sıfır dolgu görüntü kenarında sahte yapı üretiyordu."""
    p = r + 1
    return islem(np.pad(A, p, mode="edge"), structure=disk(r))[p:-p, p:-p]


def kapama_hizli(A, r):
    """Disk(r) ile kapama, uzaklık dönüşümüyle: kapama = tümleyenin açılması.
    binary_closing ile piksel piksel aynı sonuç, yaklaşık 12 kat hızlı."""
    p = r + 2
    B = ~np.pad(A, p, mode="edge")                    # tümleyen
    merkez = ndimage.distance_transform_edt(B) > r    # diskin tamamen sığdığı merkezler
    acilmis = ndimage.distance_transform_edt(~merkez) <= r
    return ~acilmis[p:-p, p:-p]


# ------------------------------------------------------------------ bölütleme v1
def bolutle_v1(k):
    """İlk yöntem: bölgesel Otsu + açma/kapama + delik doldurma + kenar işaretli watershed."""
    esik = threshold_otsu(k[32:96, 32:96])
    acik = morf(k > esik, ndimage.binary_opening, 1)
    kapali = ndimage.binary_fill_holes(morf(acik, ndimage.binary_closing, 2))
    if not kapali.any():
        return kapali
    isaret = np.zeros(k.shape, int)
    isaret[KENAR & kapali] = 2
    yy, xx = np.nonzero(kapali)
    j = np.argmin((yy - 64) ** 2 + (xx - 64) ** 2)
    isaret[yy[j], xx[j]] = 1
    return watershed(-ndimage.distance_transform_edt(kapali), isaret, mask=kapali) == 1


# ------------------------------------------------------------------ bölütleme v2
def bolutle_tek(k, esik=KATI, maks_r=40, ayrinti=False):
    """Uyarlamalı 'yuvarlanan top': akciğer maskesine, tohumu (nodülü) içine alan
    en küçük diskle kapama uygulanır; duvara yapışık nodül duvardan ayrılır."""
    bos = np.zeros(k.shape, bool)
    doku = k > esik
    hava = ndimage.binary_opening(~doku, structure=disk(1))
    lab, n = ndimage.label(hava)
    bilgi = {"r": None, "doku": doku, "akc": bos, "ic": bos, "tohum": None}
    if n == 0:
        return (bos, bilgi) if ayrinti else bos
    alan = ndimage.sum(hava, lab, range(1, n + 1))
    kenarda = [l for l in np.unique(lab[KENAR & hava]) if l > 0 and alan[l - 1] >= 200]
    akc_kenar = np.isin(lab, kenarda)                       # kenara uzanan hava = akciğer
    akc_hepsi = np.isin(lab, 1 + np.where(alan >= 200)[0])  # büyük tüm hava bölgeleri
    bilgi["akc"] = akc_kenar if akc_kenar.any() else akc_hepsi
    aday = doku & (R2 <= 36)                                # tohum: merkeze en yakın doku
    if not aday.any():
        return (bos, bilgi) if ayrinti else bos
    yy, xx = np.nonzero(aday)
    j = np.argmin(R2[yy, xx]); ty, tx = yy[j], xx[j]
    bilgi["tohum"] = (ty, tx)
    r_sec, ic = None, np.ones_like(doku)
    for akc in (akc_kenar, akc_hepsi):
        if not akc.any():
            continue
        for r in range(2, maks_r + 1, 2):                   # en küçük yeterli disk
            aday_ic = ndimage.binary_fill_holes(kapama_hizli(akc, r))
            if aday_ic[ty, tx]:
                r_sec, ic = r, aday_ic
                bilgi["akc"] = akc
                break
        if r_sec is not None:
            break
    bilgi["r"], bilgi["ic"] = r_sec, ic
    nod = ndimage.binary_fill_holes(doku & ic)              # kavite içi hava nodüle dahil
    nod = morf(nod, ndimage.binary_opening, 1)
    lab2, _ = ndimage.label(nod)
    if lab2[ty, tx] == 0:
        yy, xx = np.nonzero(nod)
        if len(yy) == 0:
            return (bos, bilgi) if ayrinti else bos
        j = np.argmin(R2[yy, xx]); ty, tx = yy[j], xx[j]
    bil = lab2 == lab2[ty, tx]
    # damarlardan ayırma: uzaklık dönüşümü üzerinde watershed
    dist = ndimage.distance_transform_edt(bil)
    isaret = np.zeros(k.shape, int)
    isaret[ty, tx] = 1
    cevre = bil & (R2 <= 100)
    if cevre.any():
        cy, cx = np.unravel_index(np.argmax(np.where(cevre, dist, -1)), dist.shape)
        isaret[cy, cx] = 1
        rn = dist[cy, cx]
    else:
        rn = dist[ty, tx]
    isaret[bil & (((YY - ty) ** 2 + (XX - tx) ** 2) > (3 * rn + 6) ** 2)] = 2
    isaret[KENAR & bil] = 2
    sonuc = ndimage.binary_fill_holes(watershed(-dist, isaret, mask=bil) == 1)
    return (sonuc, bilgi) if ayrinti else sonuc


def bolutle_v2(k, ayrinti=False):
    """v2: önce katı doku eşiği (−400 HU); maske çok küçükse buzlu cam eşiği (−750 HU)."""
    m, bilgi = bolutle_tek(k, KATI, ayrinti=True)
    bilgi["tur"] = "katı"
    if m.sum() < 8:
        m2, b2 = bolutle_tek(k, BUZLU, ayrinti=True)
        if m2.sum() > m.sum():
            m, bilgi = m2, b2
            bilgi["tur"] = "buzlu cam"
    return (m, bilgi) if ayrinti else m


# ------------------------------------------------------------------ özellikler
OZ_ADLAR = ["alan_mm2", "esdeger_cap_mm", "dairesellik", "doluluk", "eksantriklik",
            "ort_hu", "std_hu", "kalsifikasyon_orani", "kenar_gradyani"]


def ozellik_cikar(k, m):
    """Maskeden 9 özellik: boyut, şekil, yoğunluk ve kenar (Sobel)."""
    if m.sum() == 0:
        return dict.fromkeys(OZ_ADLAR, np.nan)
    p = regionprops(m.astype(np.uint8))[0]
    alan = p.area * PIKSEL_MM ** 2
    cevre = p.perimeter_crofton * PIKSEL_MM          # Crofton: küçük nesnede daha doğru çevre
    hu = 1500 * k[m] - 1350                          # pencerenin tersi
    sinir = m & ~ndimage.binary_erosion(m)           # β(A) = A − (A ⊖ B)
    return {
        "alan_mm2": alan,
        "esdeger_cap_mm": 2 * np.sqrt(alan / np.pi),
        "dairesellik": min(1.0, 4 * np.pi * alan / cevre ** 2) if cevre > 0 else np.nan,
        "doluluk": p.solidity,
        "eksantriklik": p.eccentricity,
        "ort_hu": hu.mean(),
        "std_hu": hu.std(),
        "kalsifikasyon_orani": (k[m] >= 1.0).mean(),
        "kenar_gradyani": sobel(k)[sinir].mean(),
    }
