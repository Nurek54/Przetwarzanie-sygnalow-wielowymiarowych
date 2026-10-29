import numpy as np
import matplotlib.pyplot as plt
from skimage import data

D = 8
L = 2 ** D

def przeskaluj(wektor):
    wektor = wektor - np.min(wektor)
    wektor = wektor / np.max(wektor)
    wektor = wektor * (L-1)
    return np.round(wektor).astype(int)

# Zadanie 1
fig1, ax1 = plt.subplots(6, 3, figsize=(12, 8))

obraz = data.chelsea()

tozsamosc = np.arange(L)

negacja = (L - 1) - tozsamosc

progowa = np.zeros(L, dtype=int)
progowa[50:150] = L-1

x = np.linspace(0, 2*np.pi, L)
sinus = przeskaluj(np.sin(x))

gamma1 = przeskaluj(tozsamosc ** 0.3)
gamma2 = przeskaluj(tozsamosc ** 3)

luty = [tozsamosc, negacja, progowa, sinus, gamma1, gamma2]
kolory = ['red', 'green', 'blue']

for i in range(6):
    lut = luty[i]
    pomocnicza = np.arange(len(lut))
    ax1[i, 0].scatter(pomocnicza, lut, c='black', s = 2)
    # ax1[i, 0].plot(lut, c='black')
    nowy = lut[obraz]
    ax1[i, 1].imshow(nowy)

    for k in range(3):
        kanal = nowy[:, :, k]
        wartosci, ile = np.unique(kanal, return_counts=True)
        prawd = ile / kanal.size
        ax1[i, 2].scatter(wartosci, prawd, c=kolory[k], s=2)
    ax1[i, 2].set_xlim(0, L - 1)

plt.tight_layout()
plt.savefig('zad1.png')

# Zadanie 2
fig2, ax2 = plt.subplots(2, 3, figsize=(12, 8))

ksiezyc = data.moon()

ax2[0, 0].imshow(ksiezyc, cmap='gray')
ax2[0, 0].set_title("Obraz oryginalny")

wartosci, ile = np.unique(ksiezyc, return_counts=True)
prawd = np.zeros(L)
for n in range(len(wartosci)):
    prawd[wartosci[n]] = ile[n] / ksiezyc.size

ax2[0,1].bar(range(L), prawd, color='black')
ax2[0,1].set_title("Histogram")

dystrybuanta =  np.cumsum(prawd)

ax2[0,2].scatter(range(L), dystrybuanta, c = 'black', s = 2)
ax2[0,2].set_title("Dystrybuanta")

lut = np.round(dystrybuanta * (L - 1)).astype(int)

ax2[1, 0].scatter(range(L), lut, c='black', s=2)
ax2[1, 0].set_title('LUT')

wyrownany = lut[ksiezyc]

ax2[1, 1].imshow(wyrownany, cmap='gray')
ax2[1, 1].set_title('Obraz po wyrownaniu')

wartosci, ile = np.unique(wyrownany, return_counts=True)
prawd = np.zeros(L)
for n in range(len(wartosci)):
    prawd[wartosci[n]] = ile[n] / wyrownany.size

ax2[1, 2].bar(range(L), prawd, color = 'black')
ax2[1, 2].set_title('Histogram po wyrownaniu')

plt.tight_layout()
plt.savefig('zad2.png')
