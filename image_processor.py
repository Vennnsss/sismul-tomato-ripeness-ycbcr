import cv2
import numpy as np

def preprocess_tomato(image_path, target_height=360):
    # 1. Akuisisi & Spatial Downscaling
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan di: {image_path}")

    h, w, _ = img.shape
    aspect_ratio = w / h
    target_width = int(target_height * aspect_ratio)
    downscaled_img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # 2. Transformasi Ruang Warna BGR ke YCbCr
    ycbcr_img = cv2.cvtColor(downscaled_img, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycbcr_img)

    return downscaled_img, y_channel, cb_channel, cr_channel

def apply_luminance_masking(downscaled_img, y_channel, cb_channel, cr_channel, low_thresh=35, high_thresh=220):
    # 3. Luminance Masking: menyaring area shadow (< 35) dan glare (> 220)
    valid_mask = (y_channel >= low_thresh) & (y_channel <= high_thresh)
    
    valid_cb = cb_channel[valid_mask]
    valid_cr = cr_channel[valid_mask]
    
    masked_preview = cv2.bitwise_and(downscaled_img, downscaled_img, mask=valid_mask.astype(np.uint8) * 255)
    
    return valid_mask, valid_cb, valid_cr, masked_preview

def simulate_lossy_and_calculate_mse(downscaled_img, orig_cb, orig_cr, valid_mask, lossy_quality=20):
    # 4. Simulasi transmisi kompresi rendah (lossy) dalam memori
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), lossy_quality]
    _, encimg = cv2.imencode('.jpg', downscaled_img, encode_param)
    lossy_img = cv2.imdecode(encimg, 1)

    # Konversi citra terkompresi ke YCbCr
    lossy_ycbcr = cv2.cvtColor(lossy_img, cv2.COLOR_BGR2YCrCb)
    _, lossy_cr, lossy_cb = cv2.split(lossy_ycbcr)

    # Hitung MSE pada kanal Cb dan Cr (hanya pada piksel valid hasil luminance masking)
    mse_cb = np.mean((orig_cb[valid_mask].astype(np.float64) - lossy_cb[valid_mask].astype(np.float64)) ** 2)
    mse_cr = np.mean((orig_cr[valid_mask].astype(np.float64) - lossy_cr[valid_mask].astype(np.float64)) ** 2)
    total_chroma_mse = (mse_cb + mse_cr) / 2.0

    return lossy_img, mse_cb, mse_cr, total_chroma_mse

def classify_ripeness_and_shelflife(mean_cr, mean_cb):
    # 5. Klasifikasi Kematangan & Pemodelan Regresi Sederhana untuk Shelf-life
    # Nilai netral kromatisitas YCbCr adalah 128. Semakin tinggi Cr > 128, semakin dominan pigmen merah (likopen).
    if mean_cr >= 150:
        status = "Matang (Ripe)"
        # Estimasi sisa umur simpan: 2 - 4 hari sebelum overripe
        estimated_days = max(1, round(4 - ((mean_cr - 150) * 0.1)))
    elif mean_cr >= 132:
        status = "Mengkal (Turning/Breaker)"
        # Estimasi sisa umur simpan: 5 - 8 hari
        estimated_days = max(4, round(8 - ((mean_cr - 132) * 0.15)))
    else:
        status = "Mentah (Green)"
        # Estimasi sisa umur simpan: 9 - 14 hari
        estimated_days = 12

    return status, estimated_days

def run_full_pipeline(image_path, target_height=360, lossy_quality=20):
    resized, y, cb, cr = preprocess_tomato(image_path, target_height)
    mask, clean_cb, clean_cr, preview = apply_luminance_masking(resized, y, cb, cr)
    lossy_img, mse_cb, mse_cr, total_mse = simulate_lossy_and_calculate_mse(resized, cb, cr, mask, lossy_quality)
    
    mean_cr = float(np.mean(clean_cr))
    mean_cb = float(np.mean(clean_cb))
    status, shelf_life = classify_ripeness_and_shelflife(mean_cr, mean_cb)

    # 6. Output terstruktur siap kirim ke Web Dashboard (JSON/Dict)
    result = {
        "resolusi": f"{resized.shape[1]}x{resized.shape[0]}",
        "kualitas_kompresi_lossy": f"{lossy_quality}%",
        "mean_cb": round(mean_cb, 2),
        "mean_cr": round(mean_cr, 2),
        "mse_cb": round(float(mse_cb), 4),
        "mse_cr": round(float(mse_cr), 4),
        "total_degradasi_mse": round(float(total_mse), 4),
        "status_kematangan": status,
        "estimasi_shelf_life_hari": shelf_life
    }
    return result, preview, lossy_img

if __name__ == "__main__":
    sample_file = "sampel-tomat.jpg"
    try:
        hasil, _, _ = run_full_pipeline(sample_file, target_height=360, lossy_quality=20)
        print("=== HASIL ANALISIS PIPELINE TOMAT ===")
        for key, val in hasil.items():
            print(f"{key}: {val}")
    except Exception as e:
        print(f"Error: {e}")