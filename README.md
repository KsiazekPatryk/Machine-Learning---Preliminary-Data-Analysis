# Preliminary Data Analysis - Linking Writing Processes to Writing Quality

## English

### Overview
This project contains exploratory data analysis and early feature preparation for the Kaggle competition **["Linking Writing Processes to Writing Quality"](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/overview)**. The goal of the competition is to predict essay writing quality scores based on typing/editing behavior logged during the writing process (keystrokes, mouse actions, timings, pauses, text edits, etc.).

The essay-quality workflow is organized into twelve notebooks, moving from exploratory analysis and feature preparation to model experiments, ensembling, hyperparameter tuning, explainability, and a Kaggle submission pipeline:
- `1) Preliminary-Data-Analysis.ipynb` - exploratory data analysis (EDA), data quality checks, feature distribution review, and target score analysis.
- `2) Future Preparation for the Model.ipynb` - initial feature engineering from event-level logs, generation of aggregated essay-level features, profiling of engineered data, and an 80/20 train/test split of score rows.
- `3) Model Selection Based on Data. Benchmark. Linear Regression.ipynb` - preparation of modeling data, constant-value benchmarking, Linear Regression evaluation, coefficient analysis, and feature normalization.
- `4) Neural Networks Template.ipynb` - an `MLPRegressor` template with feature scaling, early stopping, and loss/validation-curve inspection. It currently expects `train_data.csv` and `test_data.csv`, which are not present in this repository, so it is not yet wired to the generated local modeling files.
- `5)SVM.ipynb` - SVC classification (scores above 3 versus other scores) and SVR/LinearSVR regression experiments using scaled essay-level features.
- `6)Tree-Based-Models.ipynb` - decision-tree depth experiments, tree feature importance, and `HistGradientBoostingRegressor` evaluation with permutation importance.
- `9)Plan and Generation of New Features.ipynb` - rebuilds the essay-level features and adds advanced ones: pause thresholds, per-word ratios, `perc_time_*` features (time elapsed at word-count milestones), and writing-burst features. The result is saved to `grouped_train_logs_advanced.csv`.
- `10)Training XGBoost on the New Features and Prediction Confidence.ipynb` - trains a native XGBoost model on the advanced features, plots feature importance, and analyses prediction confidence by comparing errors across `writing_time_no_pause` thresholds.
- `11) Model Ensembling.ipynb` - compares `LinearRegression`, `MLPRegressor`, and `RandomForestRegressor`, then combines them in a `VotingRegressor`.
- `12)Hyperparameter Optimization.ipynb` - tunes a `RandomForestRegressor` with Hyperopt (TPE, 50 evaluations, 5-fold cross-validated MSE) and inspects the search trials.
- `13) Explainability with SHAP.ipynb` - repeats the XGBoost model and error analysis from notebook 10, then explains the model with SHAP: summary plots on the training and test sets, a bar plot of mean absolute SHAP values, and a force plot for a single test essay.
- `14) Preparing and Publishing a Model on Kaggle.ipynb` - a self-contained pipeline for the competition submission: it generates the essay-level features from raw logs, trains the scaled `VotingRegressor` ensemble (`LinearRegression`, `MLPRegressor`, `RandomForestRegressor`) on all training essays, predicts scores for the test logs, and writes `submission.csv`. Data paths work both on Kaggle (`/kaggle/input/...`) and locally.

The repository also contains two separate notebooks that are not part of the essay-quality workflow:
- `7) Natural Language Processing.ipynb` - compares bag-of-words, TF-IDF, and Word2Vec-based text representations, then tests a Hugging Face transformer sentiment pipeline.
- `8)Building a Recommender System.ipynb` - builds a movie recommender on the MovieLens `ml-latest-small` dataset: movie and user features, negative sampling, a logistic-regression watch-probability model, and mean average precision evaluation against a random baseline.

### Datasets Used
- `train_logs.csv` - event-level logs of writing actions for 2,471 training essays; 8,405,898 rows and 11 columns.
- `test_logs.csv` - event-level logs for the competition test essays; 6 rows and 11 columns in the local sample file.
- `train_scores.csv` - target essay quality scores; 2,471 rows and 2 columns (`id`, `score`).
- `sample_submission.csv` - example submission format for the competition.
- `grouped_train_logs.csv` - generated essay-level feature table created from `train_logs.csv`; 2,471 rows and 22 columns (including `id`).
- `grouped_train_logs_advanced.csv` - generated essay-level table with the advanced features from notebook 9; 2,471 rows and 39 columns (including `id` and a saved index column). It is the input of notebooks 10-13.
- `X_train_scores.csv` - generated score split for model preparation; 1,976 rows plus a saved index column.
- `X_test_scores.csv` - generated score split for model preparation; 495 rows plus a saved index column.

`X_test_scores.csv` is the internal validation split used to evaluate models; it is separate from the competition's `test_logs.csv`. The tree-based notebook uses scikit-learn's `HistGradientBoostingRegressor`; notebooks 10 and 13 use the native XGBoost library, which requires a working system OpenMP runtime (`libomp`).

The NLP notebook uses the labeled text datasets `train.csv` (27,481 rows) and `test.csv` (4,815 rows). These are separate from the essay-writing competition datasets listed above. Its transformer section uses the default binary DistilBERT SST-2 sentiment pipeline; on the current saved run, the reported accuracy is approximately `0.5028`. The pipeline predicts only positive or negative sentiment, while the dataset also includes neutral labels, so this score is not a three-class evaluation. The repository also contains `training.1600000.processed.noemoticon.csv` and `testdata.manual.2009.06.14.csv`, which no notebook currently uses.

The recommender notebook uses the MovieLens `ml-latest-small/` folder (`ratings.csv` with 100,836 ratings, `movies.csv`, `tags.csv`, and `links.csv`).

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

### Model Selection and Linear Regression Notebook
The third notebook moves the project from feature preparation toward model evaluation.

Current steps include:

1. **Modeling Data Preparation**
   - Loads the aggregated essay-level features and the training/test score splits.
   - Merges the feature table with the score data using the essay `id`.
   - Removes identifiers, saved index columns, and selected features excluded from the initial experiments.
   - Separates the feature matrix from the target `score` variable.

2. **Constant-Value Benchmark**
   - Builds a baseline with `DummyRegressor` using the mean prediction strategy.
   - Evaluates the baseline with Root Mean Squared Error (RMSE).
   - Calculates prediction variance and bias to better understand baseline behavior.

3. **Linear Regression**
   - Adds a reusable `train_and_evaluate` function for fitting models and comparing training and test RMSE.
   - Trains an initial `LinearRegression` model.
   - Extracts the model intercept and coefficients.
   - Visualizes coefficient values to support initial feature interpretation.

4. **Feature Normalization**
   - Starts preparing the feature matrix with `StandardScaler` for further model experiments.

### Neural Networks Template
The fourth notebook contains an `MLPRegressor` workflow with standardized features, early stopping, training-loss inspection, and validation-score plots. It is a template rather than a runnable project workflow at present: its input filenames (`train_data.csv` and `test_data.csv`) are not generated or included in this repository.

### SVM Experiments
The fifth notebook loads the generated essay-level feature table and score splits, then scales the input features. It explores SVC as a binary classifier using a score threshold of 3, and several SVR configurations for score regression, including a linear-kernel SVR and `LinearSVR`.

### Tree-Based Models
The sixth notebook compares decision trees at several maximum depths, inspects tree structure and feature importance, and evaluates `HistGradientBoostingRegressor`. With the current settings and the full 1,976/495 train/validation split, the validation RMSE is approximately `0.7190`. Permutation importance ranks `words_per_essay`, `chars_per_essay`, and `chars_per_minute` highest in this run. Results depend on the selected split and model parameters.

### Natural Language Processing
The seventh notebook is a separate text-sentiment experiment using `train.csv` and `test.csv`. It explores bag-of-words and TF-IDF features with logistic regression, Word2Vec embeddings, and a Hugging Face sentiment-analysis pipeline using the default DistilBERT model fine-tuned on SST-2. The transformer predicts only `POSITIVE` or `NEGATIVE`; because the labeled test data also contains neutral examples, its current accuracy of approximately `0.5028` should not be interpreted as a three-class score.

### Building a Recommender System
The eighth notebook is a separate MovieLens (`ml-latest-small`) experiment. It builds movie features (release year and one-hot genres) and user features (number of watched movies, rating statistics, activity time range, and tag-length statistics), splits users 80/20, and generates negative samples from all user-movie combinations. A `LogisticRegression` model predicts whether a user watched a movie, and recommendations are evaluated with mean average precision per user: approximately `0.0467` versus `0.0238` for a random baseline. Downsampling negatives and adding `StandardScaler` raises it slightly to approximately `0.0482`.

### Plan and Generation of New Features
The ninth notebook rebuilds the essay-level feature table from `train_logs.csv` (see the Feature Preparation Notebook) and extends it with advanced features: `chars_per_words`, `perc_time_0.1` to `perc_time_0.9` (the fraction of writing time elapsed when the essay first reaches 10%-90% of its final word count), and burst features (`words_no_pause`, `writing_time_no_pause`, `group`, `bursts_per_words`, and per-word/per-time ratios). The result is saved to `grouped_train_logs_advanced.csv`.

### Training XGBoost and Prediction Confidence
The tenth notebook trains a native XGBoost model (`max_depth=2`, 20 rounds) on the advanced features with a 90/10 train/validation split of the training scores and plots feature importance. The test RMSE is approximately `0.6991`. It then analyses prediction confidence: absolute residuals are correlated with the features, and RMSE is compared on essays with `writing_time_no_pause` below several thresholds.

### Model Ensembling
The eleventh notebook scales the advanced features and compares `LinearRegression`, `MLPRegressor`, and `RandomForestRegressor(max_depth=4)`, with test RMSE of approximately `0.7171`, `0.7198`, and `0.6992`. Their `VotingRegressor` ensemble reaches approximately `0.6847`.

### Hyperparameter Optimization
The twelfth notebook uses Hyperopt (`fmin` with `tpe.suggest`, 50 evaluations) to search `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf` of a `RandomForestRegressor`, minimizing 5-fold cross-validated MSE. The best found settings are `n_estimators=153`, `max_depth=16`, `min_samples_split=16`, and `min_samples_leaf=13`, giving a test RMSE of approximately `0.6830` (train RMSE approximately `0.5442`). The notebook also tabulates the trials and plots the `max_depth` distribution. It requires a Hyperopt version that accepts a NumPy `Generator` as `rstate` (the notebook uses `np.random.default_rng(42)`).

### Explainability with SHAP
The thirteenth notebook repeats the data preparation, XGBoost model (test RMSE approximately `0.6943`), and prediction-confidence analysis from notebook 10. It then uses `shap.Explainer` on the trained booster to compute SHAP values and shows a summary (beeswarm) plot for the training and test sets, a bar plot of mean absolute SHAP values, and a force plot for the first test essay (predicted score approximately `3.33` versus actual `3.5`).

### Preparing and Publishing a Model on Kaggle
The fourteenth notebook packages the final workflow for submission to the competition. It defines `generate_features`, which builds the essay-level features (basic, pause, `perc_time_*`, and burst features) from raw event logs, and loads the data from `/kaggle/input/linking-writing-processes-to-writing-quality` when that folder exists, otherwise from the local project folder. It scales the training features with `StandardScaler`, fits a `VotingRegressor` of `LinearRegression`, `MLPRegressor`, and `RandomForestRegressor(max_depth=4)` on all 2,471 training essays, applies the same feature generation and scaling to `test_logs.csv`, and saves the predictions as `submission.csv` (columns `id` and `score`). Run the cells in order (or use "Restart kernel and run all"), because the model cells depend on variables defined earlier.

### Purpose
This project currently lays the groundwork for feature engineering and model building by:
- Validating data integrity (no unexpected duplicates, consistent schema between train/test).
- Identifying data quality issues (missing values, outliers).
- Understanding the distributions of key behavioral features (timing, activity type, word count).
- Understanding the distribution of the target variable (essay score) to inform modeling choices.
- Producing an initial aggregated feature table at essay level for future model experiments.
- Verifying worksheet assumptions about `word_count` and event timing behavior directly from `train_logs.csv`.
- Establishing a mean-value baseline and comparing linear regression, SVM, decision-tree, histogram-based gradient-boosting, XGBoost, ensemble, and tuned random-forest approaches. The neural-network notebook remains a template that needs its input data wiring completed.
- Including separate NLP and recommender-system notebooks; they are not part of the essay-quality competition workflow.
- Explaining model predictions with SHAP (notebook 13).
- Packaging feature generation, ensemble training, and prediction into a single notebook that produces a Kaggle `submission.csv` (notebook 14).

---

## Polski

### Opis ogólny
Ten projekt zawiera wstępną analizę eksploracyjną danych oraz pierwsze przygotowanie cech dla konkursu Kaggle **["Linking Writing Processes to Writing Quality"](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality/overview)**. Celem konkursu jest przewidywanie oceny jakości napisanego eseju na podstawie zachowań zarejestrowanych podczas pisania i edycji tekstu (naciśnięcia klawiszy, akcje myszy, czasy trwania, pauzy, edycje tekstu itd.).

Przepływ pracy dotyczący jakości esejów jest zorganizowany w dwunastu notatnikach, od eksploracji danych i przygotowania cech po eksperymenty modelowe, łączenie modeli, strojenie hiperparametrów, wyjaśnialność i przygotowanie zgłoszenia do Kaggle:
- `1) Preliminary-Data-Analysis.ipynb` - eksploracyjna analiza danych (EDA), sprawdzenie jakości danych, analiza rozkładów cech i analiza zmiennej docelowej.
- `2) Future Preparation for the Model.ipynb` - pierwsze inżynierowanie cech z logów zdarzeń, generowanie cech na poziomie eseju, profilowanie przygotowanych danych oraz podział ocen na zbiór treningowy i testowy w proporcji 80/20.
- `3) Model Selection Based on Data. Benchmark. Linear Regression.ipynb` - przygotowanie danych modelowych, benchmark stałej wartości, ocena regresji liniowej, analiza współczynników oraz normalizacja cech.
- `4) Neural Networks Template.ipynb` - szablon `MLPRegressor` ze skalowaniem cech, wczesnym zatrzymaniem i podglądem krzywej straty/wyników walidacyjnych. Obecnie oczekuje plików `train_data.csv` i `test_data.csv`, których repozytorium nie zawiera, więc nie jest jeszcze podłączony do lokalnych danych modelowych.
- `5)SVM.ipynb` - eksperymenty klasyfikacji SVC (ocena powyżej 3 lub nie) oraz regresji SVR/LinearSVR na przeskalowanych cechach esejów.
- `6)Tree-Based-Models.ipynb` - eksperymenty z różnymi głębokościami `DecisionTreeRegressor`, analiza ważności cech oraz ocena `HistGradientBoostingRegressor` i permutation importance.
- `9)Plan and Generation of New Features.ipynb` - odtwarza cechy na poziomie eseju i dodaje zaawansowane: progi pauz, wskaźniki względem liczby słów, cechy `perc_time_*` (czas w kamieniach milowych liczby słów) oraz cechy serii pisania (burst). Wynik jest zapisywany do `grouped_train_logs_advanced.csv`.
- `10)Training XGBoost on the New Features and Prediction Confidence.ipynb` - trenuje natywny model XGBoost na zaawansowanych cechach, rysuje ważność cech i analizuje pewność predykcji, porównując błędy dla różnych progów `writing_time_no_pause`.
- `11) Model Ensembling.ipynb` - porównuje `LinearRegression`, `MLPRegressor` i `RandomForestRegressor`, a następnie łączy je w `VotingRegressor`.
- `12)Hyperparameter Optimization.ipynb` - stroi `RandomForestRegressor` za pomocą Hyperopt (TPE, 50 ewaluacji, 5-krotna walidacja krzyżowa MSE) i analizuje przebieg poszukiwań.
- `13) Explainability with SHAP.ipynb` - powtarza model XGBoost i analizę błędów z notatnika 10, a następnie wyjaśnia model za pomocą SHAP: wykresy podsumowujące dla zbioru treningowego i testowego, wykres słupkowy średnich bezwzględnych wartości SHAP oraz wykres force dla pojedynczego eseju testowego.
- `14) Preparing and Publishing a Model on Kaggle.ipynb` - samodzielny przepływ do zgłoszenia w konkursie: generuje cechy na poziomie eseju z surowych logów, trenuje przeskalowany zespół `VotingRegressor` (`LinearRegression`, `MLPRegressor`, `RandomForestRegressor`) na wszystkich esejach treningowych, przewiduje oceny dla logów testowych i zapisuje `submission.csv`. Ścieżki do danych działają zarówno na Kaggle (`/kaggle/input/...`), jak i lokalnie.

Repozytorium zawiera również dwa osobne notatniki, które nie są częścią przepływu pracy dotyczącego jakości esejów:
- `7) Natural Language Processing.ipynb` - porównuje reprezentacje tekstu bag-of-words, TF-IDF i Word2Vec, a następnie testuje transformerowy pipeline sentymentu z biblioteki Hugging Face.
- `8)Building a Recommender System.ipynb` - buduje system rekomendacji filmów na zbiorze MovieLens `ml-latest-small`: cechy filmów i użytkowników, próbkowanie negatywne, model regresji logistycznej przewidujący obejrzenie filmu oraz ocena średnią precyzją (MAP) względem losowego punktu odniesienia.

### Wykorzystane zbiory danych
- `train_logs.csv` - szczegółowe logi zdarzeń dla 2 471 esejów treningowych; 8 405 898 wierszy i 11 kolumn.
- `test_logs.csv` - logi zdarzeń dla zbioru testowego konkursu; lokalny przykładowy plik ma 6 wierszy i 11 kolumn.
- `train_scores.csv` - docelowe oceny jakości esejów; 2 471 wierszy i 2 kolumny (`id`, `score`).
- `sample_submission.csv` - przykładowy format zgłoszenia wyników w konkursie.
- `grouped_train_logs.csv` - wygenerowana tabela cech na poziomie eseju utworzona z `train_logs.csv`; 2 471 wierszy i 22 kolumny (wraz z `id`).
- `grouped_train_logs_advanced.csv` - wygenerowana tabela cech na poziomie eseju z zaawansowanymi cechami z notatnika 9; 2 471 wierszy i 39 kolumn (wraz z `id` i zapisaną kolumną indeksu). Jest wejściem notatników 10-13.
- `X_train_scores.csv` - wygenerowany treningowy podział tabeli ocen; 1 976 wierszy oraz zapisana kolumna indeksu.
- `X_test_scores.csv` - wygenerowany testowy podział tabeli ocen; 495 wierszy oraz zapisana kolumna indeksu.

`X_test_scores.csv` jest wewnętrznym zbiorem walidacyjnym używanym do oceny modeli; nie jest tym samym co konkursowy plik `test_logs.csv`. Notatnik modeli drzewiastych korzysta z `HistGradientBoostingRegressor` z scikit-learn; notatniki 10 i 13 używają natywnej biblioteki XGBoost, która wymaga działającego systemowego runtime OpenMP (`libomp`).

Notatnik NLP korzysta z oznaczonych zbiorów tekstowych `train.csv` (27 481 wierszy) i `test.csv` (4 815 wierszy), które są niezależne od wymienionych wyżej danych konkursu esejowego. Sekcja transformerowa używa domyślnego, binarnego pipeline'u DistilBERT SST-2; w aktualnym zapisanym przebiegu uzyskana dokładność wynosi około `0.5028`. Pipeline przewiduje wyłącznie sentyment pozytywny lub negatywny, podczas gdy zbiór zawiera również etykietę neutralną, więc wynik nie jest oceną klasyfikacji trójklasowej. Repozytorium zawiera też pliki `training.1600000.processed.noemoticon.csv` i `testdata.manual.2009.06.14.csv`, których żaden notatnik obecnie nie używa.

Notatnik systemu rekomendacji korzysta z folderu MovieLens `ml-latest-small/` (`ratings.csv` ze 100 836 ocenami, `movies.csv`, `tags.csv` i `links.csv`).

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

### Notatnik wyboru modelu i regresji liniowej
Trzeci notatnik przenosi projekt od przygotowania cech do oceny modeli.

Aktualnie zaimplementowane kroki:

1. **Przygotowanie danych modelowych**
   - Wczytuje zagregowane cechy na poziomie eseju oraz treningowy i testowy podział ocen.
   - Łączy tabelę cech z ocenami za pomocą identyfikatora eseju `id`.
   - Usuwa identyfikatory, zapisane kolumny indeksu oraz wybrane cechy wyłączone z pierwszych eksperymentów.
   - Oddziela macierz cech od zmiennej docelowej `score`.

2. **Benchmark stałej wartości**
   - Tworzy model bazowy `DummyRegressor`, który przewiduje średnią wartość.
   - Ocenia model bazowy za pomocą pierwiastka z błędu średniokwadratowego (RMSE).
   - Oblicza wariancję i obciążenie predykcji, aby lepiej zrozumieć zachowanie modelu bazowego.

3. **Regresja liniowa**
   - Dodaje wielokrotnie używaną funkcję `train_and_evaluate` do trenowania modeli i porównywania RMSE dla zbioru treningowego i testowego.
   - Trenuje pierwszy model `LinearRegression`.
   - Wyodrębnia wyraz wolny oraz współczynniki modelu.
   - Wizualizuje wartości współczynników, aby rozpocząć interpretację wpływu cech.

4. **Normalizacja cech**
   - Rozpoczyna przygotowanie macierzy cech za pomocą `StandardScaler`, aby wykorzystać znormalizowane cechy w kolejnych eksperymentach.

### Szablon sieci neuronowej
Czwarty notatnik zawiera przepływ pracy `MLPRegressor` ze standaryzacją cech, wczesnym zatrzymaniem, analizą straty treningowej i wykresami wyników walidacyjnych. Na razie jest to szablon, a nie działający przepływ projektu: używa nazw wejściowych `train_data.csv` i `test_data.csv`, których repozytorium nie generuje ani nie zawiera.

### Eksperymenty SVM
Piąty notatnik wczytuje zagregowaną tabelę cech esejów i podziały ocen, a następnie skaluje cechy wejściowe. Sprawdza SVC jako klasyfikator binarny z progiem oceny 3 oraz kilka konfiguracji SVR do regresji oceny, w tym SVR z jądrem liniowym i `LinearSVR`.

### Modele drzewiaste
Szósty notatnik porównuje drzewa decyzyjne o różnych maksymalnych głębokościach, analizuje strukturę drzewa i ważność cech, a następnie ocenia `HistGradientBoostingRegressor`. Przy aktualnych ustawieniach i pełnym podziale 1 976/495 RMSE walidacyjne wynosi około `0.7190`. Permutation importance wskazuje w tym przebiegu najwyżej `words_per_essay`, `chars_per_essay` i `chars_per_minute`. Wyniki zależą od podziału danych i parametrów modelu.

### Przetwarzanie języka naturalnego
Siódmy notatnik zawiera osobny eksperyment analizy sentymentu tekstu na podstawie `train.csv` i `test.csv`. Sprawdza cechy bag-of-words i TF-IDF z regresją logistyczną, embeddingi Word2Vec oraz pipeline analizy sentymentu Hugging Face z domyślnym modelem DistilBERT dostrojonym na SST-2. Transformer przewiduje wyłącznie etykiety `POSITIVE` i `NEGATIVE`; ponieważ oznaczony zbiór testowy zawiera również przykłady neutralne, aktualna dokładność około `0.5028` nie powinna być interpretowana jako wynik klasyfikacji trójklasowej.

### System rekomendacji
Ósmy notatnik to osobny eksperyment na zbiorze MovieLens (`ml-latest-small`). Buduje cechy filmów (rok premiery i gatunki zakodowane one-hot) oraz użytkowników (liczba obejrzanych filmów, statystyki ocen, zakres czasu aktywności i statystyki długości tagów), dzieli użytkowników w proporcji 80/20 i generuje próbki negatywne ze wszystkich kombinacji użytkownik-film. Model `LogisticRegression` przewiduje, czy użytkownik obejrzał film, a rekomendacje są oceniane średnią precyzją (MAP) na użytkownika: około `0.0467` wobec `0.0238` dla losowego punktu odniesienia. Redukcja próbek negatywnych i dodanie `StandardScaler` podnosi wynik nieznacznie, do około `0.0482`.

### Plan i generowanie nowych cech
Dziewiąty notatnik odtwarza tabelę cech na poziomie eseju z `train_logs.csv` (zob. notatnik przygotowania cech) i rozszerza ją o cechy zaawansowane: `chars_per_words`, `perc_time_0.1` do `perc_time_0.9` (ułamek czasu pisania, który upłynął, gdy esej po raz pierwszy osiąga 10%-90% końcowej liczby słów) oraz cechy serii pisania (`words_no_pause`, `writing_time_no_pause`, `group`, `bursts_per_words` i wskaźniki względem słów/czasu). Wynik jest zapisywany do `grouped_train_logs_advanced.csv`.

### Trenowanie XGBoost i pewność predykcji
Dziesiąty notatnik trenuje natywny model XGBoost (`max_depth=2`, 20 rund) na zaawansowanych cechach, z podziałem treningowych ocen 90/10 na zbiór treningowy i walidacyjny, i rysuje ważność cech. RMSE na zbiorze testowym wynosi około `0.6991`. Następnie analizuje pewność predykcji: koreluje bezwzględne reszty z cechami i porównuje RMSE dla esejów z `writing_time_no_pause` poniżej kilku progów.

### Łączenie modeli
Jedenasty notatnik skaluje zaawansowane cechy i porównuje `LinearRegression`, `MLPRegressor` oraz `RandomForestRegressor(max_depth=4)`, z RMSE na zbiorze testowym odpowiednio około `0.7171`, `0.7198` i `0.6992`. Ich zespół `VotingRegressor` osiąga około `0.6847`.

### Optymalizacja hiperparametrów
Dwunasty notatnik używa Hyperopt (`fmin` z `tpe.suggest`, 50 ewaluacji) do przeszukania `n_estimators`, `max_depth`, `min_samples_split` i `min_samples_leaf` modelu `RandomForestRegressor`, minimalizując MSE z 5-krotnej walidacji krzyżowej. Najlepsze znalezione ustawienia to `n_estimators=153`, `max_depth=16`, `min_samples_split=16` i `min_samples_leaf=13`, co daje RMSE na zbiorze testowym około `0.6830` (RMSE treningowe około `0.5442`). Notatnik zestawia też próby w tabeli i rysuje rozkład `max_depth`. Wymaga wersji Hyperopt akceptującej generator NumPy `Generator` jako `rstate` (notatnik używa `np.random.default_rng(42)`).

### Wyjaśnialność z SHAP
Trzynasty notatnik powtarza przygotowanie danych, model XGBoost (RMSE testowe około `0.6943`) i analizę pewności predykcji z notatnika 10. Następnie używa `shap.Explainer` na wytrenowanym modelu do obliczenia wartości SHAP i pokazuje wykres podsumowujący (beeswarm) dla zbioru treningowego i testowego, wykres słupkowy średnich bezwzględnych wartości SHAP oraz wykres force dla pierwszego eseju testowego (przewidywana ocena około `3.33` wobec rzeczywistej `3.5`).

### Przygotowanie i publikowanie modelu w Kaggle
Czternasty notatnik pakuje końcowy przepływ pracy do zgłoszenia w konkursie. Definiuje `generate_features`, która buduje cechy na poziomie eseju (podstawowe, pauzy, `perc_time_*` i cechy serii pisania) z surowych logów zdarzeń, i wczytuje dane z `/kaggle/input/linking-writing-processes-to-writing-quality`, jeśli ten folder istnieje, a w przeciwnym razie z lokalnego folderu projektu. Skaluje cechy treningowe za pomocą `StandardScaler`, trenuje `VotingRegressor` złożony z `LinearRegression`, `MLPRegressor` i `RandomForestRegressor(max_depth=4)` na wszystkich 2 471 esejach treningowych, stosuje to samo generowanie cech i skalowanie do `test_logs.csv` i zapisuje predykcje jako `submission.csv` (kolumny `id` i `score`). Komórki należy uruchamiać po kolei (lub użyć „Restart kernel and run all”), ponieważ komórki modelu zależą od wcześniej zdefiniowanych zmiennych.

### Cel
Ten projekt stanowi obecnie podstawę do dalszego inżynierowania cech i budowy modelu poprzez:
- Weryfikację integralności danych (brak nieoczekiwanych duplikatów, spójny schemat między zbiorem treningowym i testowym).
- Identyfikację problemów z jakością danych (braki danych, wartości odstające).
- Zrozumienie rozkładów kluczowych cech behawioralnych (czasy, typ aktywności, liczba słów).
- Zrozumienie rozkładu zmiennej docelowej (ocena eseju), co pomaga w podejmowaniu decyzji modelowych.
- Wygenerowanie pierwszej zagregowanej tabeli cech na poziomie eseju do przyszłych eksperymentów modelowych.
- Zweryfikowanie założeń z arkusza dotyczących `word_count` oraz zachowania czasów zdarzeń bezpośrednio na podstawie `train_logs.csv`.
- Utworzenie benchmarku średniej oraz porównanie regresji liniowej, SVM, drzew decyzyjnych, histogramowego gradient boostingu, XGBoost, zespołu modeli i nastrojonego lasu losowego. Notatnik sieci neuronowej pozostaje szablonem wymagającym podłączenia danych wejściowych.
- Uwzględnienie osobnych notatników NLP i systemu rekomendacji; nie są one częścią przepływu pracy konkursu dotyczącego jakości esejów.
- Wyjaśnianie predykcji modelu z użyciem SHAP (notatnik 13).
- Spakowanie generowania cech, trenowania zespołu modeli i predykcji w jednym notatniku tworzącym plik `submission.csv` dla Kaggle (notatnik 14).
