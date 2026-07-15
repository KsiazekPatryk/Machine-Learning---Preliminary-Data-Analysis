# Preliminary Data Analysis — Linking Writing Processes to Writing Quality

## English

### Overview
This notebook (`1) Preliminary-Data-Analysis.ipynb`) performs an exploratory/preliminary data analysis (EDA) for the Kaggle competition **["Linking Writing Processes to Writing Quality"](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/overview)**. The goal of the competition is to predict essay writing quality scores based on typing/editing behavior logged during the writing process (keystrokes, mouse actions, timings, etc.).

### Datasets Used
- `train_logs.csv` – event-level logs of writing actions (keystrokes, mouse clicks, pauses, etc.) for the training essays.
- `test_logs.csv` – same type of logs for the test essays (no target scores).
- `train_scores.csv` – target essay quality scores corresponding to `train_logs.csv` essay IDs.
- `sample_submission.csv` – example submission format for the competition.

### Steps Performed in the Notebook

1. **Configuration**
   - Installed and imported required libraries: `pandas`, `numpy`, `matplotlib`, and `ydata-profiling` (for automated profiling reports).

2. **Data Loading**
   - Loaded all four CSV files (`train_logs`, `test_logs`, `train_scores`, `sample_submission`) into pandas DataFrames.

3. **Data Size Check**
   - Inspected the shape (rows × columns) of each DataFrame to confirm the data matches expectations.
   - Displayed `sample_submission` and `test_df` to understand their structure.

4. **Manual Review of Examples**
   - Used `.head()` and `.tail()` to manually inspect the first and last rows of the training log data.

5. **Checking Column Types**
   - Generated an automated `ydata-profiling` report (`ProfileReport`) to review column types (numeric, boolean, categorical, text), distributions, and general data quality in one place.

6. **Unique Keys**
   - Counted the number of unique essay `id` values (`n_id`).
   - Checked for duplicate rows based on the combination of `id` and `event_id`.

7. **Missing Values**
   - Identified rows containing any null/missing values (`rows_with_null`) for further inspection.

8. **Duplicates**
   - Discussed strategy for checking repeated keys/rows (confirmed via the profiling report that there are no full duplicate rows).

9. **Feature Distribution Analysis**
   - **ID**: Confirmed ~2471 unique 8-character essay identifiers.
   - **Event ID**: Computed the maximum `event_id` per essay (i.e., total number of logged events), examined summary statistics (`.describe()`) and plotted a histogram.
   - **Down Time / Up Time / Action Time**: Explained what these columns represent (key/mouse press time, release time, and duration). Sorted `action_time` values and computed percentiles (0–99th), then plotted a percentile curve to visualize the distribution and spot long-tail/outlier behavior.
   - **Activity**: Counted value frequencies of the `activity` column (type of change: e.g., Input, Remove/Cut, Nonproduction, Move, etc.). Calculated the proportion of "Move" type activities relative to total essays, and visualized the top 5 activity types as a pie chart.
   - **Down Event / Up Event**: Displayed the full frequency counts of key/button names pressed and released.
   - **Text Change**: Analyzed the most frequent and least frequent text change values, computed the length of each text change string, and visualized the distribution of these lengths with a histogram. Sorted the data by a new `text len` column to inspect the longest text changes (potential outliers, e.g., paste actions).
   - **Word Count**: Computed the maximum (final) cumulative word count per essay (`max_word_count_by_id`), reviewed summary statistics, and plotted a histogram of essay lengths (used to spot very short/very long essays).

10. **Outliers**
    - Summarized candidate outliers found during the analysis: very long action durations, very short essays, and very long text changes (e.g., pasted text blocks).

11. **Relationships Between Features**
    - Verified that `action_time` is mathematically consistent with `up_time - down_time` for all rows.
    - Compared `down_event` and `up_event` to measure how often the pressed and released keys/buttons match vs. differ.

12. **Merging Tables**
    - Checked for overlap between `train` and `test` essay IDs (expected to be disjoint).
    - Cross-referenced `train_scores` with the `id`s present in `train_logs` to confirm scores exist for all training essays.

13. **Consistency Between Tables**
    - Verified that `train_logs` and `test_logs` share the exact same set of columns.

14. **Class Distribution (Target Variable)**
    - Reviewed unique score values, plotted a histogram of the score distribution, and computed summary statistics (`.describe()`) to understand the shape/skew of the target variable ahead of modeling.

### Purpose
This preliminary analysis lays the groundwork for feature engineering and model building by:
- Validating data integrity (no unexpected duplicates, consistent schema between train/test).
- Identifying data quality issues (missing values, outliers).
- Understanding the distributions of key behavioral features (timing, activity type, word count).
- Understanding the distribution of the target variable (essay score) to inform modeling choices.

---

## Polski

### Opis ogólny
Ten notatnik (`1) Preliminary-Data-Analysis.ipynb`) zawiera wstępną analizę eksploracyjną danych (EDA) dla konkursu Kaggle **["Linking Writing Processes to Writing Quality"](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/overview)**. Celem konkursu jest przewidywanie oceny jakości napisanego eseju na podstawie zachowań zarejestrowanych podczas pisania i edycji tekstu (naciśnięcia klawiszy, akcje myszy, czasy trwania itd.).

### Wykorzystane zbiory danych
- `train_logs.csv` – szczegółowe logi zdarzeń (naciśnięcia klawiszy, kliknięcia myszy, pauzy itp.) dla esejów treningowych.
- `test_logs.csv` – logi tego samego typu dla esejów testowych (bez ocen docelowych).
- `train_scores.csv` – docelowe oceny jakości esejów odpowiadające identyfikatorom `id` z pliku `train_logs.csv`.
- `sample_submission.csv` – przykładowy format zgłoszenia wyników w konkursie.

### Kroki wykonane w notatniku

1. **Konfiguracja**
   - Zainstalowano i zaimportowano wymagane biblioteki: `pandas`, `numpy`, `matplotlib` oraz `ydata-profiling` (do automatycznych raportów profilujących).

2. **Wczytanie danych**
   - Wczytano wszystkie cztery pliki CSV (`train_logs`, `test_logs`, `train_scores`, `sample_submission`) do ramek danych pandas.

3. **Sprawdzenie rozmiaru danych**
   - Sprawdzono kształt (liczba wierszy × kolumn) każdej ramki danych, aby potwierdzić zgodność z oczekiwaniami.
   - Wyświetlono `sample_submission` oraz `test_df`, aby zrozumieć ich strukturę.

4. **Ręczny przegląd przykładów**
   - Użyto `.head()` i `.tail()` do ręcznego przejrzenia pierwszych i ostatnich wierszy danych logów treningowych.

5. **Sprawdzenie typów kolumn**
   - Wygenerowano automatyczny raport `ydata-profiling` (`ProfileReport`), aby jednym rzutem oka przejrzeć typy kolumn (liczbowe, logiczne, kategoryczne, tekstowe), rozkłady oraz ogólną jakość danych.

6. **Unikalne klucze**
   - Policzono liczbę unikalnych wartości `id` esejów (`n_id`).
   - Sprawdzono duplikaty wierszy na podstawie kombinacji `id` i `event_id`.

7. **Wartości brakujące**
   - Zidentyfikowano wiersze zawierające jakiekolwiek wartości puste/brakujące (`rows_with_null`) w celu dalszej analizy.

8. **Duplikaty**
   - Omówiono strategię sprawdzania powtarzających się kluczy/wierszy (raport profilujący potwierdził brak w pełni zduplikowanych wierszy).

9. **Analiza rozkładu cech**
   - **ID**: Potwierdzono ok. 2471 unikalnych, 8-znakowych identyfikatorów esejów.
   - **Event ID**: Obliczono maksymalną wartość `event_id` dla każdego eseju (czyli całkowitą liczbę zarejestrowanych zdarzeń), sprawdzono statystyki opisowe (`.describe()`) i narysowano histogram.
   - **Down Time / Up Time / Action Time**: Wyjaśniono znaczenie tych kolumn (czas naciśnięcia klawisza/przycisku myszy, czas zwolnienia oraz czas trwania akcji). Posortowano wartości `action_time` i obliczono percentyle (0–99), a następnie narysowano krzywą percentylową w celu wizualizacji rozkładu i wykrycia wartości odstających (długi ogon rozkładu).
   - **Activity**: Policzono częstości występowania wartości w kolumnie `activity` (typ zmiany: np. Input, Remove/Cut, Nonproduction, Move itd.). Obliczono udział aktywności typu „Move” względem wszystkich esejów oraz zwizualizowano 5 najczęstszych typów aktywności na wykresie kołowym.
   - **Down Event / Up Event**: Wyświetlono pełne zestawienie częstości nazw klawiszy/przycisków naciskanych i zwalnianych.
   - **Text Change**: Przeanalizowano najczęstsze i najrzadsze wartości zmian tekstu, obliczono długość każdego ciągu zmiany tekstu i zwizualizowano rozkład tych długości za pomocą histogramu. Posortowano dane według nowej kolumny `text len`, aby przejrzeć najdłuższe zmiany tekstu (potencjalne wartości odstające, np. wklejanie tekstu).
   - **Word Count**: Obliczono maksymalną (końcową) skumulowaną liczbę słów dla każdego eseju (`max_word_count_by_id`), sprawdzono statystyki opisowe i narysowano histogram długości esejów (do wykrywania bardzo krótkich/długich esejów).

10. **Wartości odstające**
    - Podsumowano potencjalne wartości odstające wykryte podczas analizy: bardzo długie czasy trwania akcji, bardzo krótkie eseje oraz bardzo długie zmiany tekstu (np. wklejone bloki tekstu).

11. **Zależności między cechami**
    - Zweryfikowano, że wartość `action_time` jest matematycznie spójna z `up_time - down_time` dla wszystkich wierszy.
    - Porównano `down_event` i `up_event`, aby ocenić, jak często naciśnięty i zwolniony klawisz/przycisk są takie same, a jak często się różnią.

12. **Łączenie tabel**
    - Sprawdzono część wspólną identyfikatorów `id` między zbiorem treningowym a testowym (oczekiwany brak wspólnych elementów).
    - Zestawiono `train_scores` z identyfikatorami `id` występującymi w `train_logs`, aby potwierdzić, że oceny istnieją dla wszystkich esejów treningowych.

13. **Spójność między tabelami**
    - Zweryfikowano, że `train_logs` i `test_logs` mają dokładnie taki sam zestaw kolumn.

14. **Rozkład klas (zmienna docelowa)**
    - Sprawdzono unikalne wartości ocen, narysowano histogram rozkładu ocen oraz obliczono statystyki opisowe (`.describe()`), aby zrozumieć kształt/skośność zmiennej docelowej przed budową modelu.

### Cel
Ta wstępna analiza stanowi podstawę do dalszego inżynierowania cech i budowy modelu poprzez:
- Weryfikację integralności danych (brak nieoczekiwanych duplikatów, spójny schemat między zbiorem treningowym i testowym).
- Identyfikację problemów z jakością danych (braki danych, wartości odstające).
- Zrozumienie rozkładów kluczowych cech behawioralnych (czasy, typ aktywności, liczba słów).
- Zrozumienie rozkładu zmiennej docelowej (ocena eseju), co pomaga w podejmowaniu decyzji modelowych.
