import math
from collections import Counter

import nltk
from nltk import bigrams
from nltk import pos_tag
from nltk import word_tokenize


# Download the tokenizer resources required by NLTK
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")


# Control whether POS filtering is applied
filter_pos = True

# Ignore bigrams that occur fewer than this number of times
minimum_frequency = 5

# Store the frequency of positive and negative bigrams
positive_bigrams = Counter()
negative_bigrams = Counter()

# Stores individual word frequencies for PMI calculations
positive_words = Counter()
negative_words = Counter()

# Stores the total number of words in each sentiment category
positive_word_total = 0
negative_word_total = 0

# Count rows that cannot be processed
skipped_count = 0

# POS tag groups accepted by the filter
noun_tags = {"NN", "NNS", "NNP", "NNPS"}
adjective_tags = {"JJ", "JJR", "JJS"}


# PMI Function
def calculate_pmi(bigram_counts, word_counts, total_word_count, minimum_bigram_frequency):

    """
    Calculate PMI scores for bigrams that meet the minimum frequency threshold.
    
    Returns a list containing each bigram, its PMI score, and its frequency
    """

    pmi_results = []

    for bigram, bigram_frequency in bigram_counts.items():

        # Skip bigrams that do not meet the minimum frequency threshold
        if bigram_frequency < minimum_bigram_frequency:
            continue

        first_word, second_word = bigram

        # Calculate the probability of each word and the bigram
        first_word_frequency = word_counts[first_word]
        second_word_frequency = word_counts[second_word]

        # Avoid division by zero if a word has no occurrences
        if first_word_frequency == 0 or second_word_frequency == 0:
            continue
        
        # Calculate PMI using the formula: PMI = log2(P(x,y) / (P(x) * P(y)))
        pmi_score = math.log2((bigram_frequency * total_word_count) / (first_word_frequency * second_word_frequency))
        pmi_results.append((bigram, pmi_score, bigram_frequency))

    # Sort the results by PMI score in descending order
    pmi_results.sort(key=lambda result: result[1], reverse=True)

    return pmi_results


with open("reviews with sentiment.tsv","r",encoding="utf-8") as reviews:

    for line_number, line in enumerate(reviews, start=1):

        try:
            # Separate the four tab-separated fields
            review_id, movie_id, sentiment_label, review_text = (
                line.rstrip("\n").split("\t", maxsplit=3)
            )

        except ValueError:
            print(f"Skipping malformed row {line_number}")
            skipped_count += 1
            continue

        # Skip rows with an unknown sentiment label
        if sentiment_label not in {"positive", "negative"}:
            print(
                f"Skipping row {line_number}: "
                f"unknown sentiment '{sentiment_label}'"
            )
            skipped_count += 1
            continue

        # Convert the review text to lowercase and tokenise it
        words = word_tokenize(review_text.lower())

        # Keep only tokens that contain letters or numbers
        words = [
            word
            for word in words
            if any(character.isalnum() for character in word)
        ]

        # Count individual words for appropriate sentiment category
        if sentiment_label == "positive":
            positive_words.update(words)
            positive_word_total += len(words)

        else:
            negative_words.update(words)
            negative_word_total += len(words)

        # Assign a POS tag to each token
        tagged_words = pos_tag(words)

        # Create adjacent tagged word pairs
        tagged_bigrams = bigrams(tagged_words)

        for first_item, second_item in tagged_bigrams:

            first_word, first_tag = first_item
            second_word, second_tag = second_item

            # Check whether bigram is noun-noun
            noun_noun = (
                first_tag in noun_tags
                and second_tag in noun_tags
            )

            # Check whether bigram is adjective-noun
            adjective_noun = (
                first_tag in adjective_tags
                and second_tag in noun_tags
            )

            # Remove bigrams that do not match the accepted patterns
            if filter_pos and not (noun_noun or adjective_noun):
                continue

            filtered_bigram = (first_word, second_word)

            # Add the bigram to the appropriate counter
            if sentiment_label == "positive":
                positive_bigrams.update([filtered_bigram])

            else:
                negative_bigrams.update([filtered_bigram])

        # Display a progress message every 100 reviews
        if line_number % 100 == 0:
            print(f"Processed {line_number} rows")

# Calculate positive & negative PMI scores

positive_pmi_results = calculate_pmi(
    positive_bigrams, positive_words, positive_word_total, minimum_frequency)

negative_pmi_results = calculate_pmi(
    negative_bigrams, negative_words, negative_word_total, minimum_frequency)

print("\nTop 40 positive POS-filtered PMI collocations:\n")

for bigram, pmi_score, frequency in positive_pmi_results[:40]:
    first_word, second_word = bigram

    print(
        f"{first_word} {second_word} - "
        f"PMI: {pmi_score:.3f}, "f"Frequency: {frequency}")


print("\nTop 40 negative POS-filtered PMI collocations:\n")

for bigram, pmi_score, frequency in negative_pmi_results[:40]:
    first_word, second_word = bigram
    print(
        f"{first_word} {second_word} - "
        f"PMI: {pmi_score:.3f}, "f"Frequency: {frequency}")


print("\nCollocation extraction complete.")
print(f"POS filtering enabled: {filter_pos}")
print(f"Minimum bigram frequency: {minimum_frequency}")
print(f"Skipped rows: {skipped_count}")