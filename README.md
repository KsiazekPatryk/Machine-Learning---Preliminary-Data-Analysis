# Preliminary Data Analysis - Linking Writing Processes to Writing Quality

## English

### Overview
This project contains exploratory data analysis and early feature preparation for the Kaggle competition **["Linking Writing Processes to Writing Quality"](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/overview)**. The goal of the competition is to predict essay writing quality scores based on typing/editing behavior logged during the writing process (keystrokes, mouse actions, timings, pauses, text edits, etc.).

The work is split across two notebooks:
- `1) Preliminary-Data-Analysis.ipynb` - exploratory data analysis (EDA), data quality checks, feature distribution review, and target score analysis.
- `2) Future Preparation for the Model.ipynb` - initial feature engineering from event-level logs, generation of aggregated essay-level features, profiling of engineered data, and an 80/20 train/test split of score rows.

### Datasets Used
- `train_logs.csv` - event-level logs of writing actions for 2,471 training essays; 8,405,898 rows and 11 columns.
- `test_logs.csv` - event-level logs for the competition test essays; 6 rows and 11 columns in the local sample file.
- `train_scores.csv` - target essay quality scores; 2,471 rows and 2 columns (`id`, `score`).
- `sample_submission.csv` - example submission format for the competition.
- `grouped_train_logs.csv` - generated essay-level feature table created from `train_logs.csv`; 2,471 rows and 24 columns.
- `X_train_scores.csv` - generated score split for model preparation; 1,976 rows plus a saved index column.
- `X_test_scores.csv` - generated score split for model preparation; 495 rows plus a saved index column.

### Preliminary Analysis Notebook

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

### Feature Preparation Notebook
The second notebook builds the first version of an essay-level modeling table from the raw event logs.

Main steps currently implemented:

1. **Data Loading and Split Preparation**
   - Loads `train_logs.csv` with pandas.
   - Contains an early 80/20 split of log rows; the current saved `X_train_scores.csv` and `X_test_scores.csv` files are later overwritten by the score-table split described below.

2. **Essay-Level Aggregation**
   - Groups `train_logs` by essay `id`.
   - Creates base features such as final word count, maximum `up_time`, minimum `down_time`, number of events, and writing duration.
   - Computes `chars_per_essay` from `Input` activities.
   - Computes `chars_per_minute` from character count and writing duration.

3. **Pause Features**
   - Sorts events by `id` and `down_time`.
   - Computes pause time between events.
   - Creates pause-related features including total pause time, pause percentage of writing duration, and pauses per word.

4. **Activity Features**
   - Computes counts and text-change lengths for selected activity types: `Remove/Cut`, `Nonproduction`, `Replace`, and `Paste`.
   - Adds per-word ratios for removals and replacements.

5. **Generated Modeling Table**
   - Fills missing engineered feature values with 0.
   - Saves the resulting essay-level table to `grouped_train_logs.csv`.

6. **Profiling and Score Split**
   - Uses `ydata-profiling` to inspect the engineered feature table.
   - Loads `train_scores.csv`.
   - Creates a stratified 80/20 split of the score table into `X_train_scores.csv` and `X_test_scores.csv`.

7. **Worksheet Verification Checks**
   - Verifies whether `word_count` is cumulative or reflects the current essay state after each event.
   - Confirms that `word_count` decreases in all 2,471 essays, mainly during `Remove/Cut` and `Replace` activities, so it is not a cumulative counter of all words ever typed.
   - Checks `down_time` and `up_time` ordered by `id` and `event_id`.
   - Reports 2,213 essay IDs where `down_time` or `up_time` actually decreases between consecutive events (`diff < 0`). This is the worksheet answer for the second question.
   - Keeps the stricter/non-strict comparison (`diff <= 0`) only as additional context; that variant gives 2,268 IDs because it also counts equal timestamps.

### Purpose
This project currently lays the groundwork for feature engineering and model building by:
- Validating data integrity (no unexpected duplicates, consistent schema between train/test).
- Identifying data quality issues (missing values, outliers).
- Understanding the distributions of key behavioral features (timing, activity type, word count).
- Understanding the distribution of the target variable (essay score) to inform modeling choices.
- Producing an initial aggregated feature table at essay level for future model experiments.
- Verifying worksheet assumptions about `word_count` and event timing behavior directly from `train_logs.csv`.

---

## Polski

### Opis ogólny
Ten projekt zawiera wstępną analizę eksploracyjną danych oraz pierwsze przygotowanie cech dla konkursu Kaggle **["Linking Writing Processes to Writing Quality"](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/overview)**. Celem konkursu jest przewidywanie oceny jakości napisanego eseju na podstawie zachowań zarejestrowanych podczas pisania i edycji tekstu (naciśnięcia klawiszy, akcje myszy, czasy trwania, pauzy, edycje tekstu itd.).

Praca jest podzielona na dwa notatniki:
- `1) Preliminary-Data-Analysis.ipynb` - eksploracyjna analiza danych (EDA), sprawdzenie jakości danych, analiza rozkładów cech i analiza zmiennej docelowej.
- `2) Future Preparation for the Model.ipynb` - pierwsze inżynierowanie cech z logów zdarzeń, generowanie cech na poziomie eseju, profilowanie przygotowanych danych oraz podział ocen na zbiór treningowy i testowy w proporcji 80/20.

### Wykorzystane zbiory danych
- `train_logs.csv` - szczegółowe logi zdarzeń dla 2 471 esejów treningowych; 8 405 898 wierszy i 11 kolumn.
- `test_logs.csv` - logi zdarzeń dla zbioru testowego konkursu; lokalny przykładowy plik ma 6 wierszy i 11 kolumn.
- `train_scores.csv` - docelowe oceny jakości esejów; 2 471 wierszy i 2 kolumny (`id`, `score`).
- `sample_submission.csv` - przykładowy format zgłoszenia wyników w konkursie.
- `grouped_train_logs.csv` - wygenerowana tabela cech na poziomie eseju utworzona z `train_logs.csv`; 2 471 wierszy i 24 kolumny.
- `X_train_scores.csv` - wygenerowany treningowy podział tabeli ocen; 1 976 wierszy oraz zapisana kolumna indeksu.
- `X_test_scores.csv` - wygenerowany testowy podział tabeli ocen; 495 wierszy oraz zapisana kolumna indeksu.

### Notatnik analizy wstępnej

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

### Notatnik przygotowania cech
Drugi notatnik buduje pierwszą wersję tabeli modelowej na poziomie eseju na podstawie surowych logów zdarzeń.

Aktualnie zaimplementowane kroki:

1. **Wczytanie danych i przygotowanie podziału**
   - Wczytuje `train_logs.csv` za pomocą pandas.
   - Zawiera wczesny podział wierszy logów w proporcji 80/20; aktualne zapisane pliki `X_train_scores.csv` i `X_test_scores.csv` są później nadpisywane przez podział tabeli ocen opisany niżej.

2. **Agregacja na poziomie eseju**
   - Grupuje `train_logs` po identyfikatorze eseju `id`.
   - Tworzy podstawowe cechy: końcową liczbę słów, maksymalny `up_time`, minimalny `down_time`, liczbę zdarzeń oraz czas pisania.
   - Oblicza `chars_per_essay` na podstawie aktywności typu `Input`.
   - Oblicza `chars_per_minute` na podstawie liczby znaków i czasu pisania.

3. **Cechy pauz**
   - Sortuje zdarzenia według `id` i `down_time`.
   - Oblicza czas pauzy między kolejnymi zdarzeniami.
   - Tworzy cechy dotyczące pauz, między innymi całkowity czas pauz, udział pauz w czasie pisania i liczbę pauz względem liczby słów.

4. **Cechy aktywności**
   - Oblicza liczby zdarzeń i długości zmian tekstu dla wybranych typów aktywności: `Remove/Cut`, `Nonproduction`, `Replace` i `Paste`.
   - Dodaje wskaźniki względem liczby słów dla usunięć i zamian.

5. **Wygenerowana tabela modelowa**
   - Uzupełnia brakujące wartości cech wartością 0.
   - Zapisuje końcową tabelę cech do `grouped_train_logs.csv`.

6. **Profilowanie i podział ocen**
   - Używa `ydata-profiling` do sprawdzenia wygenerowanej tabeli cech.
   - Wczytuje `train_scores.csv`.
   - Tworzy stratyfikowany podział tabeli ocen w proporcji 80/20 do plików `X_train_scores.csv` i `X_test_scores.csv`.

7. **Weryfikacje do arkusza**
   - Sprawdza, czy `word_count` jest licznikiem kumulatywnym, czy odzwierciedla bieżący stan eseju po każdym zdarzeniu.
   - Potwierdza, że `word_count` spada we wszystkich 2 471 esejach, głównie przy aktywnościach `Remove/Cut` i `Replace`, więc nie jest licznikiem wszystkich kiedykolwiek wpisanych słów.
   - Sprawdza `down_time` i `up_time` po posortowaniu danych według `id` oraz `event_id`.
   - Wskazuje 2 213 identyfikatorów esejów, dla których `down_time` albo `up_time` faktycznie spada między kolejnymi zdarzeniami (`diff < 0`). To jest poprawna odpowiedź do drugiego pytania w arkuszu.
   - Zostawia wariant porównawczy (`diff <= 0`) tylko jako kontekst; daje on 2 268 ID, ponieważ dolicza także równe znaczniki czasu.

### Cel
Ten projekt stanowi obecnie podstawę do dalszego inżynierowania cech i budowy modelu poprzez:
- Weryfikację integralności danych (brak nieoczekiwanych duplikatów, spójny schemat między zbiorem treningowym i testowym).
- Identyfikację problemów z jakością danych (braki danych, wartości odstające).
- Zrozumienie rozkładów kluczowych cech behawioralnych (czasy, typ aktywności, liczba słów).
- Zrozumienie rozkładu zmiennej docelowej (ocena eseju), co pomaga w podejmowaniu decyzji modelowych.
- Wygenerowanie pierwszej zagregowanej tabeli cech na poziomie eseju do przyszłych eksperymentów modelowych.
- Zweryfikowanie założeń z arkusza dotyczących `word_count` oraz zachowania czasów zdarzeń bezpośrednio na podstawie `train_logs.csv`.
