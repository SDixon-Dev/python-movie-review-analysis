from collections import Counter

import nltk
from nltk import bigrams
from nltk import pos_tag
from nltk import word_tokenize

# Download the tokenizer resources required by nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")

#Control whether POS filtering is applied
filter_pos = True

#Stores the frequency of positive and negative bigrams in the reviews
positive_bigrams = Counter()
negative_bigrams = Counter()

#Count rows that cannot be processed
skipped_count = 0

#POS tag groups accepted by the filter
noun_tags = {"NN", "NNS", "NNP", "NNPS"}
adjective_tags = {"JJ", "JJR", "JJS"}

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

        #Assign a POS tag to each token
        tagged_words = pos_tag(words)

        #Create adjacent tagged words pairs
        tagged_bigrams = bigrams(tagged_words)

        for first_item, second_item in tagged_bigrams:

            first_word, first_tag = first_item
            second_word, second_tag = second_item

            #If POS filtering is enabled, skip bigrams that do not contain at least one noun or adjective
            noun_noun = (
                 first_tag in noun_tags
                and second_tag in noun_tags
                )

            adjective_noun = (
                first_tag in adjective_tags
                and second_tag in noun_tags
                )

            #Remove bigrams that do not match the accepted POS tag patterns
            if filter_pos and not (noun_noun or adjective_noun):
                continue

            filtered_bigram = (first_word, second_word)            

            #Add bigrams to the appropriate sentiment counter
            if sentiment_label == "positive":
                positive_bigrams.update([filtered_bigram])

            elif sentiment_label == "negative":
                negative_bigrams.update([filtered_bigram])

            else:
                print(
                    f"Skipping row {line_number}: "
                    f"unknown sentiment '{sentiment_label}'"
                )
                skipped_count += 1
                break

        #Display progress message every 100 reviews
        if line_number % 100 == 0:
            print(f"Processed {line_number} rows")

print("\nTop 40 positive POS-filtered collocations:\n")

for bigram, frequency in positive_bigrams.most_common(40):
        first_word, second_word = bigram
        print(f"{first_word} {second_word} - {frequency}")

print("\nTop 40 negative POS-filtered collocations:\n")

for bigram, frequency in negative_bigrams.most_common(40):
        first_word, second_word = bigram
        print(f"{first_word} {second_word} - {frequency}")

print("\nCollocation extraction complete.")
print(f"POS filtering enabled: {filter_pos}")
print(f"Skipped rows: {skipped_count}")