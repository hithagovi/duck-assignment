import cv2
import numpy as np

# -----------------------------
# 1. Load image
# -----------------------------
img = cv2.imread("full_duck.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img_rgb.shape

X = img_rgb.reshape(-1, 3).astype(np.float64)

# -----------------------------
# 2. Load training samples
# -----------------------------
duck_pixels = np.load("duck_pixels.npy")        # shape (N1, 3)
nonduck_pixels = np.load("non_duck_pixels.npy") # shape (N0, 3)

# -----------------------------
# 3. Gaussian MLE estimation
# -----------------------------
def estimate_gaussian(X):
    mu = np.mean(X, axis=0)
    cov = np.cov(X.T) + 1e-6 * np.eye(3)  # regularization
    return mu, cov

mu_duck, cov_duck = estimate_gaussian(duck_pixels)
mu_nonduck, cov_nonduck = estimate_gaussian(nonduck_pixels)

# -----------------------------
# 4. Log Gaussian likelihood
# -----------------------------
def log_gaussian(x, mu, cov):
    inv = np.linalg.inv(cov)
    diff = x - mu
    term = np.sum(diff @ inv * diff, axis=1)
    logdet = np.log(np.linalg.det(cov))
    return -0.5 * (term + logdet + 3 * np.log(2 * np.pi))

log_p_duck = log_gaussian(X, mu_duck, cov_duck)
log_p_nonduck = log_gaussian(X, mu_nonduck, cov_nonduck)

# Equal priors → compare likelihoods
duck_mask = log_p_duck > log_p_nonduck
mask = duck_mask.reshape(h, w).astype(np.uint8) * 255

# -----------------------------
# 5. Morphological cleanup
# -----------------------------
kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# -----------------------------
# 6. Remove large land regions
# -----------------------------
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

clean_mask = np.zeros_like(mask)

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]

    # Ducks are small objects
    if 10 < area < 800:
        clean_mask[labels == i] = 255

# -----------------------------
# 7. Apply mask
# -----------------------------
output = np.zeros_like(img_rgb)
output[clean_mask == 255] = img_rgb[clean_mask == 255]

cv2.imwrite("duck_only_output.png", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

print("✅ Duck extraction complete. Saved as duck_only_output.png")
