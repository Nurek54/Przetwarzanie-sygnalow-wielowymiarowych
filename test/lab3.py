import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from scipy.stats import entropy

D = 8           # glebia bitowa
L = 2 ** D      # liczba mozliwych wartosci: 256


# przeskalowanie dowolnego wektora do zakresu 0..L-1 i zamiana na liczby calkowite
def przeskaluj(wektor):
    wektor = wektor - np.min(wektor)          # najmniejsza wartosc -> 0
    wektor = wektor / np.max(wektor)          # najwieksza wartosc -> 1
    wektor = wektor * (L - 1)                 # zakres 0..255
    return np.round(wektor).astype(int)       # kwantyzacja


# Zadanie 1
fig1, ax1 = plt.subplots(6, 3, figsize=(12, 18))

obraz = data.chelsea()      # jak nie dziala, to: obraz = data.cat()

# 1) tozsamosc: 0, 1, 2, ..., 255
tozsamosc = np.arange(L)

# 2) negacja: 255, 254, ..., 0
negacja = (L - 1) - tozsamosc

# 3) progowa: 255 od 50 do 150 probki, reszta 0
progowa = np.zeros(L, dtype=int)
progowa[50:150] = L - 1

# 4) sinus od 0 do 2pi
x = np.linspace(0, 2 * np.pi, L)
sinus = przeskaluj(np.sin(x))

# 5) i 6) korekta gamma: tozsamosc do potegi gamma
gamma1 = przeskaluj(tozsamosc ** 0.3)
gamma2 = przeskaluj(tozsamosc ** 3)

luty = [tozsamosc, negacja, progowa, sinus, gamma1, gamma2]
kolory = ['red', 'green', 'blue']

for i in range(6):
    lut = luty[i]

    # kolumna 1: wykres LUT
    ax1[i, 0].plot(lut, c='black')

    # kolumna 2: obraz po przeksztalceniu
    nowy = lut[obraz]           # kazdy piksel zamieniony wartoscia z LUT
    ax1[i, 1].imshow(nowy)

    # kolumna 3: histogram kazdego kanalu
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
ax2[0, 0].set_title('Obraz oryginalny')

# histogram: prawdopodobienstwo kazdej wartosci 0..255
wartosci, ile = np.unique(ksiezyc, return_counts=True)
prawd = np.zeros(L)
for n in range(len(wartosci)):
    prawd[wartosci[n]] = ile[n] / ksiezyc.size

ax2[0, 1].bar(range(L), prawd)
ax2[0, 1].set_title('Histogram')

# dystrybuanta: suma kumulatywna
dystrybuanta = np.cumsum(prawd)

ax2[0, 2].scatter(range(L), dystrybuanta, c='black', s=2)
ax2[0, 2].set_title('Dystrybuanta')

# LUT: dystrybuanta (0..1) przeskalowana do 0..255
lut = np.round(dystrybuanta * (L - 1)).astype(int)

ax2[1, 0].scatter(range(L), lut, c='black', s=2)
ax2[1, 0].set_title('LUT')

# przeksztalcenie obrazu
wyrownany = lut[ksiezyc]

ax2[1, 1].imshow(wyrownany, cmap='gray')
ax2[1, 1].set_title('Obraz po wyrownaniu')

# histogram po przeksztalceniu
wartosci, ile = np.unique(wyrownany, return_counts=True)
prawd = np.zeros(L)
for n in range(len(wartosci)):
    prawd[wartosci[n]] = ile[n] / wyrownany.size

ax2[1, 2].bar(range(L), prawd)
ax2[1, 2].set_title('Histogram po wyrownaniu')

plt.tight_layout()
plt.savefig('zad2.png')

# Zadanie 3

# LUT z sinusa probkowanego od start do koniec
def lut_sinus(start, koniec):
    x = np.linspace(start, koniec, L)
    return przeskaluj(np.sin(x))


wyniki = np.zeros((10, 10, 10))
konce = np.linspace(0, 7 * np.pi, 10)

for i in range(10):
    lut_r = lut_sinus(-0.5 * np.pi, konce[i])
    kanal_r = lut_r[ksiezyc]

    for j in range(10):
        lut_g = lut_sinus(-0.75 * np.pi, konce[j])
        kanal_g = lut_g[ksiezyc]

        for k in range(10):
            lut_b = lut_sinus(-np.pi, konce[k])
            kanal_b = lut_b[ksiezyc]

            # skladamy 3 kanaly w obraz barwny
            kolorowy = np.zeros((512, 512, 3), dtype=int)
            kolorowy[:, :, 0] = kanal_r
            kolorowy[:, :, 1] = kanal_g
            kolorowy[:, :, 2] = kanal_b

            # entropia: obraz -> wektor -> prawdopodobienstwa
            wektor = kolorowy.flatten()
            wartosci, ile = np.unique(wektor, return_counts=True)
            prawd = ile / wektor.size
            wyniki[i, j, k] = entropy(prawd)

# szukamy kombinacji z najwieksza entropia
najlepsze = np.argwhere(wyniki == np.max(wyniki))[0]
i = najlepsze[0]
j = najlepsze[1]
k = najlepsze[2]
print('najlepsze i, j, k:', i, j, k, 'entropia:', np.max(wyniki))

# odtwarzamy najlepszy obraz
lut_r = lut_sinus(-0.5 * np.pi, konce[i])
lut_g = lut_sinus(-0.75 * np.pi, konce[j])
lut_b = lut_sinus(-np.pi, konce[k])

kolorowy = np.zeros((512, 512, 3), dtype=int)
kolorowy[:, :, 0] = lut_r[ksiezyc]
kolorowy[:, :, 1] = lut_g[ksiezyc]
kolorowy[:, :, 2] = lut_b[ksiezyc]

fig3, ax3 = plt.subplots(1, 3, figsize=(15, 5))

ax3[0].plot(lut_r, c='red')
ax3[0].plot(lut_g, c='green')
ax3[0].plot(lut_b, c='blue')
ax3[0].set_title('Funkcje transformacji')

ax3[1].imshow(kolorowy)
ax3[1].set_title('Obraz po transformacji')

for c in range(3):
    kanal = kolorowy[:, :, c]
    wartosci, ile = np.unique(kanal, return_counts=True)
    prawd = ile / kanal.size
    ax3[2].bar(wartosci, prawd, color=kolory[c], alpha=0.5)
ax3[2].set_title('Histogramy')

plt.tight_layout()
plt.savefig('zad3.png')
plt.show()
