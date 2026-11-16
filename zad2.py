import matplotlib.pyplot as plt
import pandas as pd
import itertools

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


#funkcje oraz generowanie wykresów takie samo jak w ostatnim zadaniu
kolumny = ["długośc działki kielicha", "szerokość działki kielicha", "długość płatka", "szerokość płatka"] #6kombiacji
dane = pd.read_csv("data2.csv", sep =",", names = kolumny)

fig1, axes1 = plt.subplots(nrows=3, ncols=2, figsize=(12, 15)) #ustawienie pól dla wszystkich wykresów
axes1 = axes1.flatten() #przekształca macierz wykresów w jednowymiarową liste
plt.subplots_adjust(hspace=0.45, wspace=0.3)
pary = list(itertools.combinations(kolumny, 2)) #wypisanie wszystkich możliwych par

kmeans_3 = KMeans(n_clusters=3, random_state=42) #tworzenie algorytmu, dla 3 grup

x = dane.values #tworzenie macierzy (zmiana tyoy z dataframe na czyste dane)
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
labels_3 = kmeans_3.fit_predict(x_scaled)

for idx, (cecha_x, cecha_y) in enumerate(pary):
    ax = axes1[idx]
    x = dane[cecha_x]
    y = dane[cecha_y]
    ax.scatter(x, y, c = labels_3, edgecolors='black', alpha=0.6, label='Pomiary') #rysowanie punktów danych, o przezroczystosci 0.6
    ax.set_title(f"{cecha_x} vs {cecha_y}", fontsize=11, fontweight='bold')
    ax.set_xlabel(f"{cecha_x} (cm)", fontsize=9)
    ax.set_ylabel(f"{cecha_y} (cm)", fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.6) #siatka półprzezroczysta dla łatwiejszego odczytania danych

plt.savefig("wykresy.png", dpi=300, bbox_inches='tight')
plt.show()

k_wartosci = list(range(2, 11))
wcss_tablica = []

print("\nWyniki grupowania dla k od 2 do 10:")
for k in k_wartosci:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x_scaled)

    wcss = round(kmeans.inertia_, 4)
    iteracje = kmeans.n_iter_
    print(f"k = {k:2d} | WCSS: {wcss:8.4f} | Liczba iteracji: {iteracje}")
    wcss_tablica.append(wcss)

plt.figure(figsize=(8, 5))
plt.plot(k_wartosci, wcss_tablica, marker='o', linestyle='-', color='b')
plt.title("Wartość WCSS w zależności od liczby klastrów k", fontsize=12, fontweight='bold')
plt.xlabel("Liczba klastrów (k)", fontsize=10)
plt.ylabel("WCSS", fontsize=10)
plt.xticks(k_wartosci)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("wykres_wcss.png", dpi=300, bbox_inches='tight')
plt.show()












