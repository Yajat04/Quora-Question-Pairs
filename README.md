# 🔍 Quora Question Pairs Duplicate Detector

> A machine learning web application built to identify whether two Quora questions share identical intent.

---

## 🛠️ Tech Stack

| Category | Tools & Libraries |
| :--- | :--- |
| **Language** | `Python` |
| **Interface** | `Streamlit` |
| **NLP & Text Processing** | `NLTK`, `BeautifulSoup4`, `FuzzyWuzzy`, `Python-Levenshtein` |
| **ML & Vectorization** | `Scikit-Learn`, `Pandas`, `NumPy` |

---

## 🔄 Preprocessing Pipeline

Every input text string goes through a multi-step cleanup process:

1. **HTML & Entity Cleaning**
   * Removes residual HTML markup using `BeautifulSoup`.
2. **Normalization**
   * Converts text to lowercase and expands contractions (e.g., `can't` $\rightarrow$ `cannot`).
3. **Symbol Mapping**
   * Replaces special characters and currencies (e.g., `$` $\rightarrow$ `dollar`, `k` $\rightarrow$ `000`).
4. **Stemming**
   * Applies NLTK's `PorterStemmer` to standardize token roots.

---

## 🔬 Feature Extraction Deep-Dive

### 1. Basic & Lexical Features
* **Length Statistics:** `q1_len`, `q2_len`, `q1_num_words`, `q2_num_words`
* **Word Ratios:** Shared unique word counts (`word_common`) and word share ratio (`word_common / word_total`).

### 2. Advanced Token-Level Features (NLTK)
Calculates token overlap split by stop-words and core content words:
* `cwc_min` / `cwc_max` — Ratio of common non-stop words (content words) relative to the minimum and maximum content word counts between the two questions.
* `csc_min` / `csc_max` — Ratio of common stop words relative to the minimum and maximum stop word counts.
* `ctc_min` / `ctc_max` — Ratio of total common tokens relative to the minimum and maximum total token counts.
* **Position & Substring Metrics:** 
  * `first_word_eq` (1 if first words match)
  * `last_word_eq` (1 if last words match)
  * `absolute word count difference`
  * `longest_substr_ratio`

### 3. Fuzzy String Matching Features
Leverages Levenshtein distance metrics via `FuzzyWuzzy`:
* `fuzz_ratio` — Direct character-level similarity.
* `fuzz_partial_ratio` — Substring match scoring.
* `token_sort_ratio` — Alphabetically sorts tokens before matching (handles inverted word order).
* `token_set_ratio` — Separates shared tokens from unique remainders to improve scoring on questions of unequal length.

### 4. TF-IDF & Vector Space Extraction
* **TF-IDF Vectorizer:** Fits a corpus-wide matrix to assign inverse document frequency weights to terms, ensuring rare, high-information words carry more weight than generic terms.

---

## ⚖️ Model Selection: Random Forest vs. XGBoost

> **Decision Rationale:** In duplicate detection, false positives (marking two distinct questions as duplicates) severely damage user experience.

- [x] **Random Forest (Chosen)**
  * **Priority:** High Precision.
  * **Advantage:** Handles sparse, high-dimensional TF-IDF vectors well while minimizing false positive duplicate classifications.
- [ ] **XGBoost**
  * **Focus:** Optimizes for overall recall and log-loss.
  * **Drawback:** Yielded slightly higher false positive rates in this pipeline configuration.

---
