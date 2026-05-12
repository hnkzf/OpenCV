import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

repo_url = "https://github.com/hnkzf/OpenCV.git"
repo_path = "OpenCV_local"
if not os.path.exists(repo_path):
    subprocess.run(["git", "clone", repo_url, repo_path])
    
BASE_DIR = os.getcwd()
gitar_dir = os.path.join(repo_path, "Guitar")
save_dir = os.path.join(BASE_DIR, "Hasil_Jadi")
os.makedirs(save_dir, exist_ok=True)

if os.path.exists(gitar_dir):
    files = [f for f in os.listdir(gitar_dir) if f.lower().endswith('.jpg')]

    for filename in files:
        img_path = os.path.join(gitar_dir, filename)

        img_bgr = cv2.imread(img_path)

        if img_bgr is None:
            print(f"Could not read {filename}")
            continue

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        gauss_processed = cv2.GaussianBlur(img_gray, (3,3), 0)
        sobel_res = cv2.Sobel(gauss_processed, cv2.CV_64F, 1, 1, ksize=3)
        laplace_res = cv2.Laplacian(gauss_processed, cv2.CV_64F)
        canny_res = cv2.Canny(gauss_processed, 100, 200)

        test_noise = np.random.normal(0, 25, img_bgr.shape).astype(np.uint8)
        noisy_res_bgr = cv2.add(img_bgr, test_noise)

        base_name = os.path.splitext(filename)[0]
        outputs = {
            "original": img_bgr,
            "gaussian": gauss_processed,
            "sobel": cv2.normalize(sobel_res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            "laplacian": cv2.normalize(laplace_res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            "canny": canny_res,
            "noise": noisy_res_bgr
        }

        for suffix, image_data in outputs.items():
            save_path = os.path.join(save_dir, f"{base_name}_{suffix}.jpg")
            cv2.imwrite(save_path, image_data)

        titles = ["Original", "Gaussian (Gray)", "Sobel", "Laplacian", "Canny", "Noisy (Color)"]

        img_list = [
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
            gauss_processed,
            outputs["sobel"],
            outputs["laplacian"],
            canny_res,
            cv2.cvtColor(noisy_res_bgr, cv2.COLOR_BGR2RGB)
        ]

        plt.figure(figsize=(12, 8))
        for i in range(6):
            plt.subplot(2, 3, i+1)
            if len(img_list[i].shape) == 3:
                plt.imshow(img_list[i])
            else:
                plt.imshow(img_list[i], cmap='gray')
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{base_name}_visualisasi.png"))
        plt.close()

    print(f"All processed images (Original in Color) saved in: {save_dir}")
else:
    print(f"Directory {gitar_dir} not found.")