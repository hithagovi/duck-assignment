import cv2
import numpy as np

# -------------------------------
# Globals
# -------------------------------
duck_pixels = []
nonduck_pixels = []
mode = "duck"

# -------------------------------
# Mouse callback
# -------------------------------
def mouse_callback(event, x, y, flags, param):
    global duck_pixels, nonduck_pixels

    if event == cv2.EVENT_LBUTTONDOWN:
        # IMPORTANT: collect from ORIGINAL image, not enhanced
        pixel = original[y, x]

        if mode == "duck":
            duck_pixels.append(pixel)
            print("Duck:", pixel)
        else:
            nonduck_pixels.append(pixel)
            print("Non-duck:", pixel)

# -------------------------------
# Load image
# -------------------------------
original = cv2.imread("full_duck.jpg")

# Resize for usability
scale = 0.4
original = cv2.resize(original, None, fx=scale, fy=scale)

# -------------------------------
# Create ENHANCED view
# -------------------------------
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# Boost white objects (ducks)
enhanced = cv2.equalizeHist(gray)

# Convert back to BGR for display
enhanced_display = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# -------------------------------
# Window setup
# -------------------------------
cv2.namedWindow("Duck Pixel Collector (Enhanced)", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Duck Pixel Collector (Enhanced)", mouse_callback)

print("Controls:")
print(" d → duck pixels")
print(" n → non-duck pixels")
print(" q → save & quit")

# -------------------------------
# Main loop
# -------------------------------
while True:
    display = enhanced_display.copy()

    color = (0, 255, 0) if mode == "duck" else (0, 0, 255)
    cv2.putText(display, f"MODE: {mode.upper()}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, color, 3)

    cv2.imshow("Duck Pixel Collector (Enhanced)", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('d'):
        mode = "duck"
    elif key == ord('n'):
        mode = "nonduck"
    elif key == ord('q'):
        break

cv2.destroyAllWindows()

# -------------------------------
# Save data
# -------------------------------
duck_pixels = np.array(duck_pixels)
nonduck_pixels = np.array(nonduck_pixels)

np.save("duck_pixels.npy", duck_pixels)
np.save("non_duck_pixels.npy", nonduck_pixels)

print("\nSaved training data:")
print("Duck pixels:", len(duck_pixels))
print("Non-duck pixels:", len(nonduck_pixels))
