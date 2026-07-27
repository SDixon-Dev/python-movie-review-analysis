# Movie Review Analysis (Python)

A Python data analysis project that automatically classifies movie reviews by sentiment and extracts common language patterns from positive and negative audience feedback.

The project analyses a dataset of **10,000 Amazon movie reviews** using **TextBlob** for sentiment analysis and **NLTK** for tokenisation, part-of-speech tagging and collocation extraction.

The analysis was developed through several iterations, progressing from basic frequency counting to **part-of-speech-filtered Pointwise Mutual Information (PMI)** scoring.

---

## Overview

The project was developed as a tool for movie production companies' Research and Development teams analysing the language used in existing movie reviews.
The resulting insights can be interpreted by production companies’ marketing teams to determine how to inform trailers, posters and promotional campaigns for a film

The project consists of two main stages:

1. Classifying each review as positive or negative.
2. Extracting and ranking collocations from each sentiment category.

---

## Features

- Process all reviews.
- Split reviews into individual sentences.
- Calculate sentence-level polarity scores using TextBlob.
- Calculate an average sentiment score for each review.
- Classify reviews as positive or negative.
- Save classified reviews to a tab-separated output file.
- Tokenise review text using NLTK.
- Convert text to lowercase for consistent counting.
- Remove punctuation-only tokens.
- Generate adjacent word pairs known as bigrams.
- Count collocations separately for positive and negative reviews.
- Apply part-of-speech filtering.
- Retain adjective–noun and noun–noun patterns.
- Calculate Pointwise Mutual Information scores.
- Apply a minimum-frequency threshold to remove rare collocations.
- Display the 40 highest-ranked positive and negative collocations.
- Track malformed or unsupported rows.

---

## Project Workflow

The project is divided into two connected Python programs.

### Sentiment Analysis

The `sentiments.py` program reads the original movie review dataset.

For each review, it:

1. Divides the review into individual sentences.
2. Assigns a positive or negative sentiment label.
3. Writes the result to a new tab-separated file.

---

### Collocation Analysis

The `collocations.py` program reads the sentiment-labelled reviews and extracts adjacent word pairs.

Collocations are analysed separately for:

- Positive reviews.
- Negative reviews.

This allows the language used by satisfied and dissatisfied audiences to be compared.

---

## Sentiment Classification

TextBlob assigns polarity scores between:

-1.0 = strongly negative

1.0 = strongly positive

The program calculates the polarity of each sentence and then averages the scores to produce an assinged review sentiment.

---

## Development Iterations

The collocation analysis was developed through three main iterations.

### Iteration 1: Unfiltered Frequency Analysis

The first version generated and counted every adjacent word pair (bigram) in the dataset.

### Iteration 2: Part-of-Speech Filtering

The second version filtered the extracted bigrams, retaining only **adjective–noun** and **noun–noun** combinations to produce more meaningful collocations.

### Iteration 3: PMI Ranking

The final version ranked collocations using **Pointwise Mutual Information (PMI)** to identify word pairs that occur together more strongly than would be expected by chance.

---

## Marketing Applications

The analysis can help a movie production company understand which aspects of films audiences repeatedly praise or criticise.

Positive collocations may highlight areas such as:

```text
great acting
special effects
strong performance
excellent cast
```

Negative collocations may highlight areas such as:

```text
bad acting
poor script
slow pace
worst movie
```

These findings could help marketing teams decide which elements to emphasise in trailers, posters and promotional campaigns.

---

## Technologies

- Python
- TextBlob
- NLTK
- Visual Studio Code

---

## Files

| File | Description |
|------|-------------|
| `sentiments.py` | Classifies reviews as positive or negative |
| `collocations.py` | Extracts and ranks positive and negative collocations |
| `Movie_reviews.csv` | Original dataset containing 10,000 movie reviews |
| `reviews with sentiment.tsv` | Generated dataset containing sentiment labels |

---

## Running the Project

### 1. Run the sentiment analysis

```bash
python sentiments.py
```

This creates:

```text
reviews with sentiment.tsv
```

---

### 2. Run the collocation analysis

```bash
python collocations.py
```

The program displays the top 40 positive and negative PMI-ranked collocations.

---

## Example Output

```text
Top 40 positive POS-filtered PMI collocations:

special effects - PMI: 5.721, Frequency: 114
main character - PMI: 4.826, Frequency: 93
horror movie - PMI: 4.205, Frequency: 87

Top 40 negative POS-filtered PMI collocations:

worst movie - PMI: 5.312, Frequency: 285
bad acting - PMI: 4.917, Frequency: 76
poor script - PMI: 4.581, Frequency: 31

Collocation extraction complete.
Skipped rows: 0
```

The example demonstrates the output format. Exact results will vary depending on the dataset and program configuration.

---

## Author

Sean Dixon
