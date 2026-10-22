import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import warp, AffineTransform
from scipy.spatial.distance import cdist

# Zadanie 1
fig1, ax1 = plt.subplots(2, 2, figsize=(10, 7))

obraz = data.chelsea()
ax1[0,0].imshow(obraz)
ax1[0,0].set_title('Chelsea')

mono = np.mean(obraz, axis=2)

male = mono[::8,::8]

ax1[0,1].imshow(male, cmap='binary_r')
ax1[0,1].set_title('Zmnniejszony + Mono')

kat = np.radians(15)
c = np.cos(kat)
s = np.sin(kat)

rotacja = np.array([[ c, s, 0],
                    [-s, c, 0],
                    [ 0, 0, 1]])

pochylenie = np.array([[1, -0.5, 0],
                       [0, 1, 0],
                       [ 0, 0, 1]])

transformacja_rotacja = AffineTransform(matrix = rotacja)
transformacja_pochylenie = AffineTransform(matrix = pochylenie)

obraz_rotacja = warp(male, rotacja)
obraz_pochylenie = warp(male, pochylenie)

ax1[1,0].imshow(obraz_rotacja, cmap='binary_r')
ax1[1,0].set_title('Rotacja')
ax1[1,1].imshow(obraz_pochylenie, cmap='binary_r')
ax1[1,1].set_title('Pochylenie')

plt.tight_layout()
plt.savefig('zad1.png')

# Zadanie 2

fig2, ax2 = plt.subplots(1, 2, figsize=(9, 6), sharex=True, sharey=True)

wiersze = male.shape[0]
kolumny = male.shape[1]

wsp = []
for i in range(wiersze):
    for j in range(kolumny):
        wsp.append([i, j])
wsp = np.array(wsp)

kolory = male.flatten()

x = wsp[:, 0]
y = wsp[:, 1]

ax2[0].scatter(x,y, c = kolory, cmap='binary_r')
#ax2[0].set_title('test')

jedynki = np.ones(len(wsp))
wsp3 = np.column_stack((wsp, jedynki))

nowe = []
for punkt in wsp3:
    nowy_punkt = rotacja @ punkt
    nowe.append(nowy_punkt)
nowe = np.array(nowe)

# print("Po dodaniu jedynek:")
# print(nowe)

nowe_x = nowe[:, 0]
nowe_y = nowe[:, 1]

ax2[1].scatter(nowe_x, nowe_y, c=kolory, cmap='binary_r')

plt.tight_layout()
plt.savefig('zad2.png')

# Zadanie 3
fig3, ax3 = plt.subplots(1, 3)

ax3[0].scatter(x, y, c=kolory, cmap='binary_r')

ile_punktow = len(wsp)
indeksy = np.random.choice(ile_punktow, 1000, replace=False)

wspolrzedne_losowych = wsp[indeksy]
kolory_losowych = kolory[indeksy]

ax3[1].scatter(wspolrzedne_losowych[:, 0], wspolrzedne_losowych[:, 1], c=kolory_losowych, cmap='binary_r')

siatka_x = np.linspace(0, wiersze - 1, 300)
siatka_y = np.linspace(0, kolumny - 1, 400)

wynik_x = []
wynik_y = []
wynik_kolory = []

for px in siatka_x:
    for py in siatka_y:
        odlegolosc = cdist([[px, py]], wspolrzedne_losowych)
        najblizszy = np.argmin(odlegolosc)
        wynik_x.append(px)
        wynik_y.append(py)

        wynik_kolory.append(kolory_losowych[najblizszy])

ax3[2].scatter(wynik_x, wynik_y, c=wynik_kolory, cmap='binary_r')

plt.tight_layout()
plt.savefig('zad3.png')