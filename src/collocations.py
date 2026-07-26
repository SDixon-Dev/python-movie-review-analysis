from collections import Counter

import nltk
from nltk import bigrams
from nltk import word_tokenize

# Download the tokenizer resources required by nltk
nltk.download("punkt")
nltk.download("punkt_tab")

#Stores the frequency of positive and negative bigrams in the reviews
positive_bigrams = Counter()
negative_bigrams = Counter()

#Count rows that cannot be processed
skipped_count = 0

with open("reviews with sentiment.tsv", "r", encoding="utf-8") as reviews:

    for line_number, line in enumerate(reviews, start=1):

        try:
            #Separate the four tab-separated fields
            review_id, movie_id, sentiment_label, review_text = (line.rstrip("\n").split("\t", maxsplit=3))

        except ValueError:
            print(f"Skipping malformed row {line_number}")
            skipped_count += 1
            continue

        #Convert the review text to lowercase and tokenise it
        words = word_tokenize(review_text.lower())

        #Keep only tokens that contain letters or numbers
        words = [word for word in words if any(character.isalnum() for character in word)]

        #Create adjacent word pairs from the review
        review_bigrams = bigrams(words)

        #Add bigrams to the appropriate sentiment counter
        if sentiment_label == "positive":
            positive_bigrams.update(review_bigrams)

        elif sentiment_label == "negative":
            negative_bigrams.update(review_bigrams)

        else:
            print(
                 f"Skipping row {line_number}: "
                 f"unknown sentiment '{sentiment_label}'"
            )
            skipped_count += 1
            continue

        #Display progress message every 100 reviews
        if line_number % 100 == 0:
            print(f"Processed {line_number} rows")

print("\nTop 40 positive collocations:\n")

for bigram, frequency in positive_bigrams.most_common(40):
        first_word, second_word = bigram
        print(f"{first_word} {second_word} - {frequency}")

print("\nTop 40 negative collocations:\n")

for bigram, frequency in negative_bigrams.most_common(40):
        first_word, second_word = bigram
        print(f"{first_word} {second_word} - {frequency}")

print("\nCollocation extraction complete.")
print(f"Skipped rows: {skipped_count}")