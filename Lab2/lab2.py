import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import warp, AffineTransform
from scipy.spatial.distance import cdist

# ---------- Zadanie 1 ----------
fig, ax = plt.subplots(3, 2, figsize=(10, 7))

img = data.chelsea()            # jak nie dziala, to: img = data.cat()
ax[0, 0].imshow(img)
ax[0, 0].set_title('Chelsea')

mono = np.mean(img, axis=2)     # srednia z 3 kanalow -> obraz szary
male = mono[::8, ::8]           # co osmy piksel w obu osiach
ax[0, 1].imshow(male, cmap='binary_r')
ax[0, 1].set_title('Zmniejszenie + Mono')

kat = np.radians(15)            # 15 stopni na radiany

rotacja = np.array([[ np.cos(kat), np.sin(kat), 0],
                    [-np.sin(kat), np.cos(kat), 0],
                    [ 0,           0,           1]])

pochylenie = np.array([[1, 0.5, 0],
                       [0, 1,   0],
                       [0, 0,   1]])

obraz_rot = warp(male, AffineTransform(matrix=rotacja))
obraz_poch = warp(male, AffineTransform(matrix=pochylenie))

ax[1, 0].imshow(obraz_rot, cmap='binary_r')
ax[1, 0].set_title('Rotacja')
ax[1, 1].imshow(obraz_poch, cmap='binary_r')
ax[1, 1].set_title('Pochylenie')

plt.tight_layout()
plt.savefig('zad1.png')

# ---------- Zadanie 2 ----------
fig, ax = plt.subplots(1, 2, figsize=(9, 6), sharex=True, sharey=True)

wsp = []
for i in range(male.shape[0]):
    for j in range(male.shape[1]):
        wsp.append([i, j])
wsp = np.array(wsp)             # ksztalt (2166, 2)

kolory = male.flatten()         # obraz 2D -> jeden dlugi wektor

ax[0].scatter(wsp[:, 0], wsp[:, 1], c=kolory, cmap='binary_r')

jedynki = np.ones(len(wsp))
wsp3 = np.column_stack((wsp, jedynki))   # ksztalt (2166, 3)

nowe = (rotacja @ wsp3.T).T     # A x [x, y, 1] dla kazdego punktu

ax[1].scatter(nowe[:, 0], nowe[:, 1], c=kolory, cmap='binary_r')

plt.tight_layout()
plt.savefig('zad2.png')

# ---------- Zadanie 3 ----------
fig, ax = plt.subplots(1, 3, figsize=(12, 6))

ax[0].scatter(wsp[:, 0], wsp[:, 1], c=kolory, cmap='binary_r')

indeksy = np.random.choice(len(wsp), 1000, replace=False)
wsp_los = wsp[indeksy]
kolory_los = kolory[indeksy]

ax[1].scatter(wsp_los[:, 0], wsp_los[:, 1], c=kolory_los, cmap='binary_r')

nowe_x = np.linspace(0, male.shape[0] - 1, 300)
nowe_y = np.linspace(0, male.shape[1] - 1, 400)

xs = []
ys = []
wartosci = []
for x in nowe_x:
    for y in nowe_y:
        odl = cdist([[x, y]], wsp_los)      # odleglosc do 1000 punktow
        najblizszy = np.argmin(odl)         # indeks najblizszego
        xs.append(x)
        ys.append(y)
        wartosci.append(kolory_los[najblizszy])

ax[2].scatter(xs, ys, c=wartosci, cmap='binary_r', s=1)

plt.tight_layout()
plt.savefig('zad3.png')
plt.show()