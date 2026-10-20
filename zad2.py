import sklearn as sk
import pandas as pd

kolumny = ["długośc działki kielicha", "szerokość działki kielicha", "długość płatka", "szerokość płatka"]
dane = pd.read_csv("data2.csv", names = kolumny)

print(dane)






