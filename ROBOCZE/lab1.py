import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 3, figsize=(10, 10))

# Zadanie 1
x = np.linspace(0, 4 * np.pi, 40)
y = np.sin(x)
ax[0, 0].plot(x, y, c='red')
ax[0, 0].set_title('funkcja sinus')
ax[0, 0].grid()

img = y[:, np.newaxis] * y[np.newaxis, :]
ax[0, 1].imshow(img, cmap='magma')
ax[0, 1].set_title('min: %.3f, max: %.3f' % (np.min(img), np.max(img)))

img_n = img - np.min(img)
img_n = img_n / np.max(img_n)
ax[0, 2].imshow(img_n, cmap='magma')
ax[0, 2].set_title('min: %.3f, max: %.3f' % (np.min(img_n), np.max(img_n)))

# Zadanie 2
for i, bits in enumerate([2, 4, 8]):
    max_val = 2**bits - 1
    d = np.rint(img_n * max_val).astype(int)
    ax[1, i].imshow(d, cmap='magma')
    ax[1, i].set_title('min: %d, max: %d' % (np.min(d), np.max(d)))

# Zadanie 3
noisy = img_n + np.random.normal(size=img_n.shape)
ax[2, 0].imshow(noisy, cmap='magma')
ax[2, 0].set_title('n=1')

for i, n in enumerate([50, 1000]):
    noise = np.random.normal(size=(n, *img_n.shape))
    noisy_imgs = img_n + noise
    avg = np.mean(noisy_imgs, axis=0)
    ax[2, i + 1].imshow(avg, cmap='magma')
    ax[2, i + 1].set_title('n=%d' % n)

plt.tight_layout()
plt.show()