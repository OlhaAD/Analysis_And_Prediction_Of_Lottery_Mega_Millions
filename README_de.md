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
   
3. **Trendanalyse**
- Durchführung einer Trendanalyse zur Identifizierung signifikanter Zahlen basierend auf historischen Daten.
- Berechnung von Steigungen, Achsenabschnitten und p-Werten für jede Lottozahl zur Bestimmung ihrer statistischen Signifikanz.
   
4. **Entwicklung eines LSTM-Modells**
- Entwicklung und Training von LSTM-Modellen zur Vorhersage der Häufigkeit der gezogenen Zahlen in der Zukunft.
- Optimierung des Modells unter Verwendung verschiedener Hyperparameter wie der Rückkopplungsperiode (look-back), um die Genauigkeit der Vorhersagen zu erhöhen.

5. **Modellbewertung**
- Bewertung der Leistung des LSTM-Modells unter Verwendung von Metriken wie dem mittleren absoluten Fehler (MAE) und dem Root Mean Squared Error (RMSE).
- Vergleich der Leistung des LSTM-Modells mit der traditionellen Trendanalyse.
- 
6. **Vorhersage und Vergleich**
- Vorhersage der Häufigkeit der gezogenen Zahlen für zukünftige Jahre unter Verwendung des trainierten LSTM-Modells.
- Vergleich der Vorhersagen des LSTM-Modells mit den Ergebnissen der Trendanalyse zur Identifizierung von Gemeinsamkeiten und Unterschieden.

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
### Trendanalyse
- **Statistische Analyse:** Berechnung von Steigungen, Achsenabschnitten und p-Werten zur Erkennung von Trends.
- **Signifikanzschwelle:** Verwendung eines p-Werts < 0,3 zur Auswahl signifikanter Zahlen für die Trendanalyse.
  
### Entwicklung eines LSTM-Modells
- **Modellarchitektur:**
  1. LSTM-Schichten mit Dropout zur Regularisierung.
  2. Ausgangsschicht Dense zur Vorhersage der Häufigkeiten.
- **Hyperparameter:**
  1. Rückkopplungsperiode (z.B. 2, 3, 4 Jahre)
  2. Batch-Größe und Anzahl der Epochen.
- **Training und Testen:**
  1. Aufteilung der Daten in Trainings- und Testdatensätze.
  2. Training des Modells und Bewertung seiner Leistung auf beiden Datensätzen.
     
### Modellbewertung
- **Metriken:** MAE und RMSE für beide Datensätze.
- **Vergleich:** Vergleich der Fehler zur Bestimmung der besten Modellkonfiguration.
- 
### Vorhersage und Vergleich
- **Vorhersage der Zukunft:** Vorhersage der Häufigkeit der gezogenen Zahlen für die Jahre 2024, 2025, 2026 und 2027.
- **Vergleich mit der Trendanalyse:** Visualisierung und Vergleich der prognostizierten Häufigkeiten des LSTM-Modells und der Trendanalyse.
  
## Fazit
Das Projekt zielt darauf ab, fortschrittliche Techniken des maschinellen Lernens zur Verbesserung der Genauigkeit bei der Vorhersage der Häufigkeit von Lottozahlen einzusetzen. Durch den Vergleich der traditionellen Trendanalyse mit Deep-Learning-Modellen soll die effektivste Methode zur Vorhersage von Lottoergebnissen ermittelt werden. Die präsentierten Ergebnisse und Visualisierungen bieten wertvolle Einblicke sowohl für Forscher als auch für Lottoliebhaber.
