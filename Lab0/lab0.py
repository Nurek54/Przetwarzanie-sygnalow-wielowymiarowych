import numpy as np
import matplotlib.pyplot as plt

mono = np.zeros((30, 30), dtype=int)

mono[10:20, 10:20] = 1
mono[15:25, 5:15] = 2

print(mono)

fig, ax = plt.subplots(2, 2, figsize=(7, 7))

ax[0, 0].imshow(mono)
ax[0, 0].set_title('obraz monochromatyczny')

ax[0, 1].imshow(mono, cmap='binary')
ax[0, 1].set_title('obraz monochromatyczny')

plt.tight_layout()
plt.savefig('zad2.png')

color = np.zeros((30, 30, 3))

color[15:25, 5:15, 0] = 1
color[10:20, 10:20, 1] = 1
color[5:15, 15:25, 2] = 1

ax[1, 0].imshow(color)
ax[1, 0].set_title('obraz barwny')

negative = 1 - color

ax[1, 1].imshow(negative)
ax[1, 1].set_title('negatyw')

plt.tight_layout()
plt.savefig('zad3.png')