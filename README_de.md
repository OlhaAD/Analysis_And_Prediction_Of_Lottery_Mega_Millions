# Analyse und Vorhersage der Häufigkeit von Lottozahlen "Mega Millions" mit maschinellen Lernmodellen

## Ziele und Aufgaben
### Ziele
Das Hauptziel dieses Forschungsprojekts besteht darin, historische Lotto-Daten zu analysieren, um die zukünftige Häufigkeit von Lottozahlen unter Verwendung fortschrittlicher maschineller Lernmodelle, insbesondere Long Short-Term Memory (LSTM)-Netzwerken, vorherzusagen. Das Projekt zielt darauf ab, einen umfassenden Vergleich zwischen der traditionellen Trendanalyse und der Vorhersage auf Basis von Deep Learning zu bieten.

### Aufgaben
1. **Datensammlung und -vorverarbeitung**
   - Sammlung und Vorverarbeitung historischer Lotto-Daten mit Fokus auf die Häufigkeit der gezogenen Zahlen im Laufe der Zeit.
   - Umwandlung der Daten in Formate, die für die Analyse und das Training von maschinellen Lernmodellen geeignet sind.
   
2. **Datenvisualisierung**
   - Visualisierung der historischen Verteilungen der Häufigkeit der gezogenen Zahlen zur Erkennung von Trends und Mustern.
   - Erstellung verschiedener Diagramme und Heatmaps zur effektiven Darstellung der Daten.

3. **Datenanalyse**

   - Erstellen von Q-Q-Diagrammen zur Überprüfung der Normalverteilung der Ziehfrequenzen.
   - Durchführung des Shapiro-Wilk-Tests zur statistischen Bewertung der Normalität der Daten.

4. **Trendanalyse**
   - Durchführung einer Trendanalyse zur Identifizierung signifikanter Zahlen basierend auf historischen Daten.
   - Berechnung von Steigungen, Achsenabschnitten und p-Werten für jede Lottozahl zur Bestimmung ihrer statistischen Signifikanz.
   
5. **Entwicklung des polynomiellen Regressionsmodells**
   - Entwicklung und Training von Modellen der polynomiellen Regression zur Vorhersage der Häufigkeit des Auftretens von Zahlen in der Zukunft.
   - Optimierung des Modells durch Verwendung verschiedener Polynomgrade zur Verbesserung der Vorhersagegenauigkeit.

6. **Modellbewertung**
   - Bewertung der Leistung des polynomiellen Regressionsmodells anhand von Metriken wie dem mittleren absoluten Fehler (MAE) und dem Root Mean Squared Error (RMSE).
   - Vergleich der Leistung des polynomiellen Regressionsmodells mit der traditionellen Trendanalyse.

7. **Vorhersage und Vergleich**
   - Vorhersage der Häufigkeit des Auftretens von Zahlen basierend auf ihren vorherigen Trends.
   - Normalisierung der Daten, um die Ergebnisse als Wahrscheinlichkeiten für die Wahl jeder Zahl in einem bestimmten Jahr zu interpretieren.
   - Vorhersage der Häufigkeit des Auftretens von Zahlen für zukünftige Jahre unter Verwendung des trainierten polynomiellen Regressionsmodells.
   - Vergleich der Prognosen des polynomiellen Regressionsmodells mit den Ergebnissen der Trendanalyse zur Identifizierung von Ähnlichkeiten und Unterschieden.

## Detaillierte Methodik
### Werkzeuge und Bibliotheken
Dieses Projekt verwendete die folgenden Werkzeuge und Bibliotheken:
- **Python:** Die Hauptprogrammiersprache, die für die Datenanalyse und das Erstellen von Modellen verwendet wurde.

- **Pandas:** Eine Bibliothek zur Datenmanipulation und -analyse. Verwendet für die Arbeit mit Dataframes.

- **NumPy:** Eine Bibliothek für die Arbeit mit Arrays und numerische Berechnungen.

- **Seaborn:** Eine Visualisierungsbibliothek, die auf matplotlib basiert. Verwendet für die Erstellung statistischer Diagramme.

- **Matplotlib:** Die Hauptbibliothek für die Erstellung von Diagrammen und Visualisierungen.

- **SciPy:** Eine Bibliothek für wissenschaftliche und technische Berechnungen. Verwendet für die Durchführung statistischer Operationen.

- **Scikit-learn:** Eine maschinelle Lernbibliothek in Python. Verwendet für die Datenvorverarbeitung und Bewertung von Modellen.
   - MinMaxScaler: Zum Skalieren der Daten.
   - mean_absolute_error, mean_squared_error: Metriken zur Modellbewertung.

- **TensorFlow und Keras:** Ein Framework und eine High-Level-Bibliothek zum Erstellen und Trainieren von neuronalen Netzwerken.
   - Sequential: Verwendet zum Erstellen des neuronalen Netzwerks.
   - LSTM: Langzeit-Kurzzeitspeicher-Schicht für rekurrente neuronale Netzwerke.
   - Dense: Vollständig verbundene Schicht im neuronalen Netzwerk.
   - Dropout: Regularisierungsschicht zur Vermeidung von Überanpassung.
  
### Datensammlung und -vorverarbeitung
- **Datenquelle:** Die Daten für dieses Projekt stammen von https://catalog.data.gov/dataset/lottery-mega-millions-winning-numbers-beginning-2002

- **Datenbankstruktur:**
Die Mega Millions Lotterie-Datenbank enthält Informationen über die Gewinnzahlen, einschließlich Mega Ball und Multiplier, zusammen mit Ziehungsdaten. Insgesamt umfasst die Datenbank 2290 Zeilen mit folgenden Details:
   - Draw Date: Das Datum der Lotterieziehung, Datentyp - object.
   - Winning Numbers: Die in der Lotterie gezogenen Zahlen, Datentyp - object.
   - Mega Ball: Eine separate Nummer, die aus einem anderen Satz von Kugeln gezogen wird, Datentyp - int64.
   - Multiplier: Eine optionale Funktion, die die Gewinnbeträge erhöht, Datentyp - float64.

   | Draw Date   | Winning Numbers  | Mega Ball | Multiplier |
   |-------------|------------------|-----------|------------|
   | 09/25/2020  | 20 36 37 48 67   | 16        | 2.0        |
   | 09/29/2020  | 14 39 43 44 67   | 19        | 3.0        |
   | 10/02/2020  | 09 38 47 49 68   | 25        | 2.0        |
   | 10/06/2020  | 15 16 18 39 59   | 17        | 3.0        |
   | 10/09/2020  | 05 11 25 27 64   | 13        | 2.0        |
  
**Transformation und Bereinigung zur Analyse und Visualisierung:**
- Aufteilen der Spalte "Winning Numbers" in fünf separate Spalten.
- Umwandlung der Spalte "Draw Date" in das Datumsformat zur einfacheren Bearbeitung der Daten.
- Entfernen der Spalte "Multiplier", da diese für die Analyse nicht benötigt wird.
- Umwandlung der Nummern in ein numerisches Format für eine korrekte Verarbeitung.

```python
df[['Number1', 'Number2', 'Number3', 'Number4', 'Number5']] = df['Winning Numbers'].str.split(expand=True)
df.drop('Winning Numbers', axis=1, inplace=True)
df.drop('Multiplier', axis=1, inplace=True)
columns_to_convert = ['Number1', 'Number2', 'Number3', 'Number4', 'Number5', 'Mega Ball']
df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, downcast='integer', errors='coerce')
df['Draw Date'] = pd.to_datetime(df['Draw Date'])
```
**Normalisierung und Skalierung für das Training des LSTM-Modells:**
- Skalierung der numerischen Daten auf einen Bereich von 0 bis 1 zur Verbesserung der Modellleistung.
- Umwandlung der Daten in ein Format, das den Anforderungen an den LSTM-Input entspricht.
   
### Datenvisualisierung
**Heatmap der Häufigkeit der Gewinnzahlen nach Jahr**
![HeatMapFrequencyOfWinningNumbersWithYears](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfWinningNumbersYears.png)
Aus dieser Visualisierung geht hervor, dass der Datensatz gemäß den Änderungen der Spielregeln in drei Zeitintervalle unterteilt werden sollte, um die Auswirkungen dieser Änderungen auf die Trends bei der Auswahl der Zahlen zu minimieren. Die Zeitintervalle umfassen:

   - **2006-2012:** Eine Periode, in der die Zahlen von 1 bis 56 reichten. Dieses Segment bietet Stabilität und ermöglicht eine Trendanalyse ohne den Einfluss von Änderungen im Zahlensatz.
   - **2014-2017:** Eine Periode, in der die Zahlen bis 75 reichten, was die Untersuchung des Verhaltens der Spieler mit einer erhöhten Anzahl von Zahlen ermöglicht.
   - **2018-2023:** Der aktuellste Zeitraum mit Zahlen bis 70, der auf neue Trends und Änderungen in den Spielerstrategien hinweist.
  
Die separate Betrachtung jedes Zeitraums ermöglicht eine genauere Bewertung der Beliebtheit und Trends bei der Zahlenauswahl im Laufe der Zeit und minimiert Verzerrungen, die durch Regeländerungen im Spiel verursacht werden. Dieser Ansatz bildet die Grundlage für zuverlässigere prädiktive Modellierungen und strategische Planungen im Kontext der Lotterie.

---------------------------------------------------------------------------------------------------------------------------
**Beschreibung der Visualisierungen für die 56-Zahlen-Lotterie**

Zur Analyse der Häufigkeit der Gewinnzahlen in der 56-Zahlen-Lotterie wurde ein horizontales Balkendiagramm erstellt. Das Balkendiagramm zeigt, wie oft jede Zahl über den gesamten betrachteten Zeitraum aufgetreten ist. Diese Visualisierung hilft dabei, die Zahlen zu identifizieren, die am häufigsten aufgetreten sind, was für weitere Analysen und Prognosen nützlich sein kann. Zusätzlich wurde eine Heatmap erstellt, um die Häufigkeit des Auftretens jeder Zahl über verschiedene Jahre in der 56-Zahlen-Lotterie darzustellen.

![FrequencyOfWinningNumbersLoto56](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfWinningNumbers56.png)

![HeatMapFrequencyOfWinninNumbersLoto56](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfWinningNumbersYears56.png)

Ein Histogramm wurde auch für die Mega Ball-Zahlen in der 56-Zahlen-Lotterie erstellt. Der Mega Ball ist eine separate Zahl, die aus einem weiteren Satz Bälle besteht, nämlich 46 Bällen. Diese Visualisierung hilft dabei, die Mega Ball-Zahlen zu identifizieren, die am häufigsten aufgetreten sind. Die Heatmap visualisiert die Häufigkeit des Auftretens des Mega Ball aufgeschlüsselt nach Jahr.

![FrequencyOfWinningMegaBallsLoto56](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfMegaBall56nSort.png)

![HeatMapFrequencyOfWinningMegaBallsLoto56](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfMegaBallByYears56nHM.png)

![HistFrequencyOfWinningMegaBallsLoto56](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/HistFrequencyOfMegaBall56n.png)

---------------------------------------------------------------------------------------------------------------------------
**Beschreibung der Visualisierungen für die 75-Zahlen-Lotterie**

Zur Analyse der Häufigkeit der Gewinnzahlen in der 75-Zahlen-Lotterie wurde ein horizontales Balkendiagramm erstellt. Das Balkendiagramm zeigt, wie oft jede Zahl über den gesamten betrachteten Zeitraum aufgetreten ist. Diese Visualisierung hilft dabei, die Zahlen zu identifizieren, die am häufigsten aufgetreten sind, was für weitere Analysen und Prognosen nützlich sein kann. Zusätzlich wurde eine Heatmap erstellt, um die Häufigkeit des Auftretens jeder Zahl über verschiedene Jahre in der 75-Zahlen-Lotterie darzustellen.

![FrequencyOfWinningNumbersLoto75](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfWinningNumbers75.png)

![HeatMapFrequencyOfWinninNumbersLoto75](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfWinningNumbersYears75.png)

Ein Histogramm wurde auch für die Mega Ball-Zahlen in der 75-Zahlen-Lotterie erstellt. Der Mega Ball ist eine separate Zahl, die aus einem weiteren Satz Bälle besteht, nämlich 15 Bällen. Diese Visualisierung hilft dabei, die Mega Ball-Zahlen zu identifizieren, die am häufigsten aufgetreten sind. Die Heatmap visualisiert die Häufigkeit des Auftretens des Mega Ball aufgeschlüsselt nach Jahr.

![FrequencyOfWinningMegaBallsLoto75](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfMegaBall75nSort.png)

![HeatMapFrequencyOfWinningMegaBallsLoto75](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfMegaBallByYears75nHM.png)

![HistFrequencyOfWinningMegaBallsLoto75](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/HistFrequencyOfMegaBall75n.png)

---------------------------------------------------------------------------------------------------------------------------

**Beschreibung der Visualisierungen für die 70-Zahlen-Lotterie**

Zur Analyse der Häufigkeit der Gewinnzahlen in der 70-Zahlen-Lotterie wurde ein horizontales Balkendiagramm erstellt. Das Balkendiagramm zeigt, wie oft jede Zahl über den gesamten betrachteten Zeitraum aufgetreten ist. Diese Visualisierung hilft dabei, die Zahlen zu identifizieren, die am häufigsten aufgetreten sind, was für weitere Analysen und Prognosen nützlich sein kann. Zusätzlich wurde eine Heatmap erstellt, um die Häufigkeit des Auftretens jeder Zahl über verschiedene Jahre in der 70-Zahlen-Lotterie darzustellen.

![FrequencyOfWinningNumbersLoto70](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfWinningNumbers70.png)

![HeatMapFrequencyOfWinninNumbersLoto70](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfWinningNumbersYears70.png)

Ein Histogramm wurde auch für die Mega Ball-Zahlen in der 70-Zahlen-Lotterie erstellt. Der Mega Ball ist eine separate Zahl, die aus einem weiteren Satz Bälle besteht, nämlich 25 Bällen. Diese Visualisierung hilft dabei, die Mega Ball-Zahlen zu identifizieren, die am häufigsten aufgetreten sind. Die Heatmap visualisiert die Häufigkeit des Auftretens des Mega Ball aufgeschlüsselt nach Jahr.

![FrequencyOfWinningMegaBallsLoto70](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfMegaBall70nSort.png)

![HeatMapFrequencyOfWinningMegaBallsLoto70](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/FrequencyOfMegaBallByYears70nHM.png)

![HistFrequencyOfWinningMegaBallsLoto70](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/HistFrequencyOfMegaBall70n.png)

---------------------------------------------------------------------------------------------------------------------------

### Datenanalyse

Um die Normalität der Verteilung der Gewinnfrequenzen in der Lotterie zu überprüfen, wurden Q-Q-Diagramme erstellt und der Shapiro-Wilk-Test durchgeführt.

#### Q-Q-Diagramme

Das Q-Q-Diagramm vergleicht die Quantile der Frequenzverteilung des Datensatzes mit einer theoretischen Normalverteilung. Die Tatsache, dass die meisten unserer Datenpunkte entlang einer geraden Linie liegen (mit einigen Abweichungen an den Enden), deutet darauf hin, dass die Häufigkeit des Auftretens von Zahlen ungefähr einer Normalverteilung folgt.

- **Q-Q-Diagramm für den Zeitraum 2006-2012 für die 56-Zahlen-Lotterie:**

![Q-Q Plot for 2006-2012](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/QQPlotFrequency_for_2006-2012.png)

- **Q-Q-Diagramm für den Zeitraum 2014-2017 für die 75-Zahlen-Lotterie:**

![Q-Q Plot for 2014-2017](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/QQPlotFrequency_for_2014-2017.png)

- **Q-Q-Diagramm für den Zeitraum 2018-2023 für die 70-Zahlen-Lotterie:**

![Q-Q Plot for 2018-2023](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/QQPlotFrequency_for_2018-2023.png)

#### Shapiro-Wilk-Test

Der Shapiro-Wilk-Test überprüft weiter die Normalität der Daten. Das Ergebnis des Tests liefert einen p-Wert, der verwendet werden kann, um zu bestimmen, ob die Nullhypothese, dass die Daten normalverteilt sind, verworfen werden sollte. Ein hoher p-Wert (typischerweise größer als 0,05) bedeutet, dass die Nullhypothese der Normalität nicht verworfen werden kann, was darauf hinweist, dass die Daten für bestimmte statistische Tests und Konfidenzintervalle, die Normalität voraussetzen, als normal verteilt betrachtet werden können.

Ergebnisse des Shapiro-Wilk-Tests:

- **2006-2012 für die 56-Zahlen-Lotterie:**
  - statistic: 0.9777
  - p-Wert: 0.3827

- **2014-2017 für die 75-Zahlen-Lotterie:**
  - statistic: 0.9804
  - p-Wert: 0.2958

- **2018-2023 für die 70-Zahlen-Lotterie:**
  - statistic: 0.9760
  - p-Wert: 0.1988

Basierend auf den Ergebnissen des Shapiro-Wilk-Tests können wir schließen, dass die Verteilung der Gewinnfrequenzen für alle betrachteten Zeiträume ungefähr normal ist, da die p-Werte in allen Fällen 0,05 übersteigen.

### Trendanalyse
Für die Trendanalyse macht es keinen Sinn, Lotterien mit 56 und 75 Zahlen zu betrachten, da diese bis 2013 durchgeführt wurden. Es ist sinnvoller, sich auf die Analyse der aktuellen Lotterie mit 70 Zahlen zu konzentrieren.

#### Datenaufbereitung
   - **DataFrame:** Der DataFrame heatmap_df70n wird verwendet, wobei jede Spalte eine separate Zahl darstellt und die Zeilen die Häufigkeit des Auftretens dieser Zahl in verschiedenen Zeiträumen darstellen können.
   - **Hinzufügen einer Konstante:** Für jede Zahl wird ein Array X mit einer Konstante und einem Index (Zeit) erstellt, wodurch das Modell den Intercept berücksichtigen kann.
   - **Regressionsmodell:** Für jede Zahl wird ein OLS (Ordinary Least Squares) Modell erstellt, wobei die abhängige Variable y die Häufigkeit der Zahl und die unabhängigen Variablen die Zeit sind.
   - **Speicherung der Ergebnisse:** Die Ergebnisse des Modells, einschließlich des Steigungswertes (slope), des Intercepts und des p-Wertes, werden im Wörterbuch trends gespeichert.
   - **Umwandlung der Ergebnisse in DataFrame:** Das Wörterbuch trends wird in einen DataFrame trends_df umgewandelt, um die Analyse und Visualisierung der Daten zu erleichtern.
Dieser Ansatz ermöglicht es, die Trends in der Häufigkeit jeder Zahl im Laufe der Zeit zu analysieren und festzustellen, welche Zahlen beliebter oder weniger beliebt werden.

```python
trends = {}
for column in heatmap_df70n.columns:
    # Проверяем, является ли столбец числовым
    if pd.api.types.is_numeric_dtype(heatmap_df70n[column]):
        X = sm.add_constant(np.arange(len(heatmap_df70n)))  # Zeit
        y = heatmap_df70n[column].astype(float).values  # Häufigkeit des Vorkommens der Zahl
        model = sm.OLS(y, X)
        results = model.fit()
        # Speichern die Ergebnisse für jede Zahl
        trends[column] = {'Slope': results.params[1], 'Intercept': results.params[0], 'P-value': results.pvalues[1]}

# Konvertieren eines Wörterbuchs in einen DataFrame zur Analyse
trends_df = pd.DataFrame(trends).T
# Legen Sie Optionen fest, um alle Zeilen anzuzeigen
pd.set_option('display.max_rows', None)
print(trends_df)
```
Diese Ergebnisse der linearen Regression für jede Lotterienummer zeigen den Steigungskoeffizienten Slope und den p-Wert. Der Steigungskoeffizient gibt die Richtung des Trends an (ein positiver Wert bedeutet eine Zunahme der Häufigkeit, ein negativer Wert bedeutet eine Abnahme), und der p-Wert gibt die statistische Signifikanz dieses Trends an.

Zahlen mit signifikant negativen Trends und niedrigen p-Werten (zum Beispiel Nummer 14 mit einem Slope von -0,942857 und einem P-Wert von 0,031797) zeigen eine statistisch signifikante Abnahme der Häufigkeit im Laufe der Zeit. Zahlen mit signifikant positiven Trends und niedrigen p-Werten (zum Beispiel Nummer 18 mit einem Slope von 2,028571 und einem P-Wert von 0,037760) zeigen eine statistisch signifikante Zunahme der Häufigkeit.

### Entwicklung eines Polynomregressionsmodells
Die Polynomregression ist eine Methode der Regressionsanalyse, die verwendet wird, um Beziehungen zwischen einer abhängigen Variablen und einer oder mehreren unabhängigen Variablen zu modellieren, indem die unabhängigen Variablen auf verschiedene Potenzen (Grad des Polynoms) angehoben werden. Im Gegensatz zur linearen Regression kann die Polynomregression nichtlineare Beziehungen in den Daten erfassen, was sie nützlich für die Vorhersage komplexer Trends macht.

#### Entwicklungsschritte des Modells
- **Bestimmung der signifikanten Nummern:**
Basierend auf der Trendanalyse werden Nummern mit einem p-Wert von weniger als 0,3 ausgewählt. Dies ermöglicht es, sich auf Nummern zu konzentrieren, die einen statistisch signifikanten Einfluss haben.

```python
significant_numbers = trends_df70n[trends_df70n['P-value'] < 0.3].index.tolist()
significant_numbers_str = list(map(str, significant_numbers))
```
- **Skalierung der Daten:**
Die Daten werden im Bereich von 0 bis 1 normalisiert, um die Korrektheit des Modells sicherzustellen und die Genauigkeit der Vorhersagen zu verbessern.

```python
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(heatmap_df70n)
```
- **Vorbereitung der Daten für die Polynomregression:**
   - Die Daten aus den Jahren 2018-2022 werden zum Training des Modells verwendet.
   - Das Jahr 2023 wird zur Modellüberprüfung verwendet.
   - Vorhersagen werden für die Jahre 2023-2027 gemacht.

```python
years = np.array(heatmap_df70n.index).reshape(-1, 1)
train_years = np.array([2018, 2019, 2020, 2021, 2022]).reshape(-1, 1)
test_year = np.array([2023]).reshape(-1, 1)
future_years = np.array([2023, 2024, 2025, 2026, 2027]).reshape(-1, 1)
```
- **Erstellung und Training des Modells:**
   - Für jede signifikante Nummer werden polynomiale Merkmale erstellt.
   - Das Modell wird auf den Daten der Jahre 2018-2022 trainiert.
   - Vorhersagen für das Testjahr (2023) sowie für die zukünftigen Jahre (2024-2027) werden gemacht.

```python
degree = 2  # Степень полинома
predictions = {}

for column in heatmap_df70n.columns:
    poly = PolynomialFeatures(degree)
    train_years_poly = poly.fit_transform(train_years)
    test_year_poly = poly.transform(test_year)
    future_years_poly = poly.transform(future_years)
    
    model = LinearRegression()
    model.fit(train_years_poly, heatmap_df70n.loc[train_years.flatten(), column])
    
    test_predict = model.predict(test_year_poly)
    heatmap_df70n.loc[2023, column] = test_predict[0]
    
    future_predict = model.predict(future_years_poly)
    predictions[column] = future_predict
```
- **Transformation und Wiederherstellung der Skalen der Daten:**
Die Ergebnisse werden in ein DataFrame umgewandelt und die Skalen der Daten zur Interpretation der Ergebnisse wiederhergestellt.

```python
poly_predictions = pd.DataFrame(predictions, index=[2023, 2024, 2025, 2026, 2027])
poly_predictions = scaler.inverse_transform(poly_predictions)
poly_predictions = pd.DataFrame(poly_predictions, columns=heatmap_df70n.columns, index=[2023, 2024, 2025, 2026, 2027])
```
- **Verifizierung und Extraktion von Daten für signifikante Nummern:**
   - Sicherstellen, dass Indizes und Spalten im richtigen Format vorliegen.
   - Überprüfung, welche Nummern in beiden DataFrames vorhanden sind.
   - Extraktion der Daten für signifikante Nummern.

```python
probability_df70n.columns = probability_df70n.columns.astype(str)
poly_predictions.columns = poly_predictions.columns.astype(str)
heatmap_df70n.columns = heatmap_df70n.columns.astype(str)

available_trend_numbers = [num for num in significant_numbers_str if num in probability_df70n.columns]
available_poly_numbers = [num for num in significant_numbers_str if num in poly_predictions.columns]

if not available_trend_numbers or not available_poly_numbers:
    print("No available trend or poly numbers found.")
else:
    trend_predictions_significant = probability_df70n.loc[:, available_trend_numbers]
    poly_predictions_significant = poly_predictions.loc[:, available_poly_numbers]
```
### Modellbewertung
- **Metriken:** MAE und RMSE für beide Datensätze.
- **Vergleich:** Vergleich der Fehler zur Bestimmung der besten Modellkonfiguration.
- 
### Vorhersage und Vergleich
#### Prognose der Häufigkeit von Zahlen
Die Prognose der Häufigkeit von Lotteriezahlen basiert auf ihren bisherigen Trends (Steigung und Achsenabschnitt, die durch lineare Regression bestimmt werden). Für jede signifikante Zahl (d.h. Zahlen mit einem p-Wert kleiner als 0,3) wird die vorhergesagte Häufigkeit basierend auf der Gleichung der Trendlinie berechnet, wobei der Achsenabschnitt den Anfangswert darstellt und die Steigung die Änderungsrate der Häufigkeit über die Jahre angibt. Dann werden für jedes Jahr im angegebenen Bereich die vorhergesagten Werte berechnet. Der resultierende DataFrame predicted_df70n wird normalisiert, sodass die Werte in jedem Jahr auf 1 summiert werden, was es ermöglicht, die Ergebnisse als Wahrscheinlichkeiten für die Auswahl jeder Zahl in diesem Jahr zu interpretieren.

```python
# Zu prognostizierende Jahre
years = np.arange(2024, 2028)
# Konvertieren eines Wörterbuchs in einen DataFrame mit den richtigen Spalten 
trends_df70n = pd.DataFrame.from_dict(trends, orient='index', columns=['Intercept', 'Slope', 'P-value'])
# Bestimmung des Schwellenwerts  p < 0,3
significant_numbers = trends_df70n[trends_df70n['P-value'] < 0.3]
predicted_frequencies = {}
for index, row in significant_numbers.iterrows():
    predicted_frequencies[index] = row['Intercept'] + row['Slope'] * (years - 2023)
predicted_df70n = pd.DataFrame(predicted_frequencies, index=years)
# Normalisierende Werte
probability_df70n = predicted_df70n.div(predicted_df70n.sum(axis=1), axis=0)
print(probability_df70n)
# Visualisirung
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
for i, year in enumerate(years):
    row = i // 2
    col = i % 2
    probability_df70n.loc[year].plot(kind='bar', ax=axs[row, col], color='indigo')
    axs[row, col].set_title(f'Predicted probabilities for {year} year')
    axs[row, col].set_xlabel('Number')
    axs[row, col].set_ylabel('Probability')

plt.tight_layout()
plt.show()
# Zahlen mit deutlich positiven Trends und niedrigen p-Werten
selected_numbers_df70n = trends_df70n[(trends_df70n['P-value'] < 0.3) & (trends_df70n['Slope'] > 0)]
print("Selected Numbers with Positive Trends and Low P-Values", selected_numbers_df70n.index.tolist())
```

Basierend auf diesen Daten wurden Zahlen mit positiven Trends und niedrigen p-Werten ausgewählt:
```python
Selected Numbers with Positive Trends and Low P-Values: [3, 8, 15, 18, 19, 21, 36, 45, 47, 50, 51, 52, 55, 61]
```

**Visualisierungen der prognostizierten Wahrscheinlichkeiten für 2024–2027:**

![PredictedProbabilitiesTrends70Numbers](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/PredictedProbabilitiesFor2024_2027.png)

#### Vorhersage für Mega Ball
Ein ähnlicher Prozess wurde für Mega Ball-Zahlen befolgt. Als Ergebnis wurden die folgenden Mega Ball-Zahlen mit positiven Trends und niedrigen p-Werten ausgewählt:

```python
Selected Mega Balls with Positive Trends and Low P-Values: [12, 13, 18, 24, 25]
```

**Visualisierungen der prognostizierten Mega Ball-Wahrscheinlichkeiten für 2024–2027:**

![PredictedProbabilitiesTrendsMegaBall](https://github.com/OlhaAD/Analysis_And_Prediction_Of_Lottery_Mega_Millions_Python/blob/main/visualizations/PredictedProbabilitiesMegaBall.png)

- **Vergleich mit der Trendanalyse:** Visualisierung und Vergleich der prognostizierten Häufigkeiten des LSTM-Modells und der Trendanalyse.
  
## Fazit
Das Projekt zielt darauf ab, fortschrittliche Techniken des maschinellen Lernens zur Verbesserung der Genauigkeit bei der Vorhersage der Häufigkeit von Lottozahlen einzusetzen. Durch den Vergleich der traditionellen Trendanalyse mit Deep-Learning-Modellen soll die effektivste Methode zur Vorhersage von Lottoergebnissen ermittelt werden. Die präsentierten Ergebnisse und Visualisierungen bieten wertvolle Einblicke sowohl für Forscher als auch für Lottoliebhaber.
