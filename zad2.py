import matplotlib.pyplot as plt
import pandas as pd
import itertools
import math

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


#funkcje oraz generowanie wykresów takie samo jak w ostatnim zadaniu
def korelacja_pearsona(x, y):
    n = len(x) #liczba punktów pomiarowych
    sr_x = sum(x) / n
    sr_y = sum(y) / n #srednie arytemtyczne dla obu wartosci
    licznik = sum((xi - sr_x) * (yi - sr_y) for xi, yi in zip(x, y)) #suma iloczynow odchylen punktow od sredniej
    mianownik = math.sqrt(sum((xi - sr_x)**2 for xi in x) * sum((yi - sr_y)**2 for yi in y)) #iloczyn sum kwadratów odchyleń pod pierwiastkiem
    return licznik / mianownik if mianownik != 0 else 0.0

def regresja_liniowa(x, y): #obliczanie wwspolczynnikow
    n = len(x)
    sr_x = sum(x) / n
    sr_y = sum(y) / n
    licznik = sum((xi - sr_x) * (yi - sr_y) for xi, yi in zip(x, y)) # to amo co w korelacji
    mianownik = sum((xi - sr_x)**2 for xi in x) #oblicza sumę kwadratów odchyleń zmiennej x od jej średniej
    a = licznik / mianownik #wspolczynnik a
    b = sr_y - a * sr_x #wspolczynnik b
    return a, b

kolumny = ["długośc działki kielicha", "szerokość działki kielicha", "długość płatka", "szerokość płatka"] #6kombiacji
dane = pd.read_csv("data2.csv", sep =",", names = kolumny)

fig1, axes1 = plt.subplots(nrows=3, ncols=2, figsize=(12, 15)) #ustawienie pól dla wszystkich wykresów
axes1 = axes1.flatten() #przekształca macierz wykresów w jednowymiarową liste
plt.subplots_adjust(hspace=0.45, wspace=0.3)
pary = list(itertools.combinations(kolumny, 2)) #wypisanie wszystkich możliwych par

kmeans_3 = KMeans(n_clusters=3) #tworzenie algorytmu, dla 3 grup
x = dane.values #tworzenie macierzy (zmiana tyoy z dataframe na czyste dane)
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
labels_3 = kmeans_3.fit_predict(x_scaled)

for idx, (cecha_x, cecha_y) in enumerate(pary):
    ax = axes1[idx]
    x = dane[cecha_x]
    y = dane[cecha_y]
    r = korelacja_pearsona(x, y)
    a, b = regresja_liniowa(x, y)
    ax.scatter(x, y, c = labels_3, edgecolors='black', alpha=0.6, label='Pomiary') #rysowanie punktów danych, o przezroczystosci 0.6
    ax.set_title(f"{cecha_x} vs {cecha_y}", fontsize=11, fontweight='bold')
    ax.set_xlabel(f"{cecha_x} (cm)", fontsize=9)
    ax.set_ylabel(f"{cecha_y} (cm)", fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.6) #siatka półprzezroczysta dla łatwiejszego odczytania danych

print(" ")
print(f" WCSS = {kmeans_3.inertia_}")
plt.savefig("wykresy.png", dpi=300, bbox_inches='tight')
plt.show()









