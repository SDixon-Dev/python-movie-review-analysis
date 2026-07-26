
from textblob import TextBlob


def calculate_review_sentiment(review_text):

    """
    Calculates the overall sentiment of a movie review.

    Each sentence is assigned a polarity score between -1 and 1.
    Sentence scores are averaged to get an overall score for the review.

    Parameters:
    review_text (str): The text of the movie review.

    Returns:
    A tuple containing the overall sentiment score (float) and the sentiment label (str).   
    """
    
    #Create a TextBlob object from the review text
    blob = TextBlob(review_text)

    #Stores the polarity score for each sentence
    sentence_scores = []

    for sentence in blob.sentences:
        sentence_scores.append(sentence.sentiment.polarity)

    #Avoids division by zero if there are no sentences in the review
    if not sentence_scores:
        return 0.0, "positive"

    #Calculate the mean sentiment score of all sentences in the review
    overall_score = sum(sentence_scores) / len(sentence_scores)

    #Convert the numerical score to a sentiment label
    if overall_score >= 0:
        sentiment_label = "positive"
    else:
        sentiment_label = "negative"

    return overall_score, sentiment_label

positive_count = 0
negative_count = 0
skipped_count = 0

with open(
    "Movie_reviews.csv", "r", encoding="utf-8"
) as reviews, open(
    "reviews with sentiment.tsv", "w", encoding="utf-8"
) as output_file:

    for line_number, line in enumerate(reviews, start=1):

        try:
            #Separates review ID, movie ID, and review text
            review_id, movie_id, review_text = (line.rstrip("\n").split("\t", maxsplit=2))

        except ValueError:
            print(f"Skipping malformed row {line_number}")
            skipped_count += 1
            continue

        #Skip rows that contain no review text
        if not review_text.strip():
            print(f"Skipping review {review_id}:  no review text")
            skipped_count += 1
            continue

        #Calculate the score and classification
        overall_score, sentiment_label = calculate_review_sentiment(review_text)

        #Update the sentiment totals
        if sentiment_label == "positive":
            positive_count += 1
        else:
            negative_count += 1

        #Write the review and classification to the output file
        output_file.write(f"{review_id}\t{movie_id}\t{sentiment_label}\t{review_text}\n")

        #Display progress updates
        if line_number % 100 == 0:
            print(f"Processed {line_number} rows")

print("\nSentiment analysis complete.")
print(f"Positive reviews: {positive_count}")
print(f"Negative reviews: {negative_count}")
print(f"Skipped reviews: {skipped_count}")