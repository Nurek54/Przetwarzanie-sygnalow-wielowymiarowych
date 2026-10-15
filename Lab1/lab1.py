import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(3,3, figsize=(10,10))

#Zadanie 1
x = np.linspace(0, 4 * np.pi, 40)
y = np.sin(x)

axs[0,0].plot(x, y, c='red')
axs[0,0].set_title('funkcja sinus')
axs[0,0].grid()

img = y[:, np.newaxis] * y[np.newaxis, :]
axs[0,1].imshow(img, cmap='magma')
axs[0,1].set_title('min: %.3f, max: %.3f' % (np.min(img), np.max(img)))

img_n = img - np.min(img)
img_n = img_n / np.max(img_n)

axs[0,2].imshow(img_n, cmap='magma')
axs[0,2].set_title('min: %.3f, max: %.3f' % (np.min(img_n), np.max(img_n)))

#Zadanie 2
for i, bits in enumerate([2, 4, 8]):
    max_value = 2**bits - 1
    d = np.rint(img_n * max_value).astype(int)
    axs[1,i].imshow(d, cmap='magma')
    axs[1,i].set_title('min: %d, max: %d' % (np.min(d), np.max(d)))

#Zadanie 3
nosiy = img_n + np.random.normal(size = img_n.shape)
axs[2,0].imshow(nosiy, cmap='magma')
axs[2,0].set_title('n = 1')

for i, n in enumerate([50, 1000]):
    noise = np.random.normal(size=(n, *img_n.shape))
    noisy_img = img_n + noise
    avg = np.mean(noisy_img, axis = 0)
    axs[2,i+1].imshow(avg, cmap='magma')
    axs[2,i+1].set_title('n = %d' % n)

plt.tight_layout()
plt.show()