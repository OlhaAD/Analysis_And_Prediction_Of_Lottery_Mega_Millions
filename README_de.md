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
### Datensammlung und -vorverarbeitung
- **Datenquelle:** Historische Lotto-Daten, einschließlich der Häufigkeit der gezogenen Gewinnzahlen.
- **Vorverarbeitungsschritte:**
1. Normalisierung und Skalierung der Daten für das Training des LSTM-Modells.
2. Umwandlung der Daten in ein Format, das den Eingabeanforderungen des LSTM entspricht.
   
### Datenvisualisierung
- **Histogramm:** Zur Darstellung der Verteilung der Häufigkeit der gezogenen Zahlen.
- **Heatmaps:** Zur Visualisierung der Häufigkeit der gezogenen Gewinnzahlen nach Jahren.
- **Balkendiagramme:** Zum Vergleich der prognostizierten Häufigkeiten verschiedener Modelle.
  
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
