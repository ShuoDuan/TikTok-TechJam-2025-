import pandas as pd
import re
import numpy as np

#Count total words in review
def count_words(text):
    if text is None or pd.isna(text):
        return 0
    return len(str(text).split())


def sentiment_words_count(text):
    """Count positive and negative sentiment words"""
    if text is None or pd.isna(text):
        return 0

    text = str(text).lower()

    # Positive sentiment words
    positive_words = ['good', 'great', 'excellent', 'amazing', 'delicious', 'perfect',
                      'wonderful', 'fantastic', 'outstanding', 'love', 'best', 'nice',
                      'awesome', 'brilliant', 'superb', 'magnificent']

    # Negative sentiment words
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disgusting', 'worst',
                      'hate', 'disappointing', 'poor', 'unacceptable', 'rude', 'nasty',
                      'terrible', 'pathetic', 'useless']

    sentiment_count = 0
    for word in positive_words + negative_words:
        import re
        pattern = r'\b' + re.escape(word) + r'\b'
        sentiment_count += len(re.findall(pattern, text))

    return sentiment_count


def count_website_words(text):
    """Count website-related keywords (.com, https, etc.)"""
    if text is None or pd.isna(text):
        return 0

    text = str(text).lower()
    website_keywords = ['.com', '.net', '.org', 'http', 'https', 'www.', 'website', 'link']

    count = 0
    for keyword in website_keywords:
        count += text.count(keyword)

    return count


def count_superlatives(text):
    """Count superlative words (both positive and negative)"""
    if text is None or pd.isna(text):
        return 0

    text = str(text).lower()

    # Food-specific superlatives
    superlatives = [
        # Positive superlatives
        'divine', 'heavenly', 'perfectly cooked', 'to die for', 'best', 'most delicious',
        'incredible', 'unbelievable', 'extraordinary', 'phenomenal',

        # Negative superlatives
        'inedible', 'unbearable', 'unacceptable', 'worst', 'terrible', 'disgusting'
    ]

    count = 0
    for superlative in superlatives:
        count += text.count(superlative)

    return count


def count_generic_words(text):
    """Count generic/vague words that may indicate spam"""
    if text is None or pd.isna(text):
        return 0

    text = str(text).lower()
    generic_words = ['good', 'nice', 'ok', 'fine', 'alright', 'decent', 'average']

    count = 0
    for word in generic_words:
        count += text.count(' ' + word + ' ')  # Word boundaries
        # Also check at start/end of text
        if text.startswith(word + ' ') or text.endswith(' ' + word):
            count += 1

    return count


def business_experience_score(text):
    """Count business experience related words"""
    if text is None or pd.isna(text):
        return 0

    text = str(text).lower()
    business_words = ['service', 'food', 'menu', 'price', 'staff', 'quality',
                      'experience', 'restaurant', 'establishment', 'place',
                      'meal', 'dish', 'cuisine', 'dining']

    count = 0
    for word in business_words:
        count += text.count(word)

    return count


def promotional_language_count(text):
    """Count promotional language indicators"""
    if text is None or pd.isna(text):
        return 0

    text = str(text).lower()
    promo_phrases = ["don't miss", "must try", "highly recommend", "special offer",
                     "limited time", "exclusive", "deal", "discount", "promotion",
                     "hurry", "act now", "call now"]

    count = 0
    for phrase in promo_phrases:
        count += text.count(phrase)

    return count


def story_telling_words_count(text):
    """Count story-telling words that may indicate irrelevant content"""
    if text is None or pd.isna(text):
        return 0

    text = str(text).lower()
    story_words = ['meanwhile', 'during', 'then', 'after', 'before', 'suddenly',
                   'finally', 'later', 'yesterday', 'today', 'when i', 'so i',
                   'first', 'next', 'afterwards', 'eventually']

    count = 0
    for word in story_words:
        count += text.count(word)

    return count


def rate_authenticity_score_vectorized(df):
    """Calculate authenticity score for all rows efficiently"""
    authenticity_scores = []

    # Pre-calculate restaurant averages using 'name' column
    restaurant_avg_ratings = df.groupby('name')['rating'].mean().to_dict()
    restaurant_counts = df.groupby('name').size().to_dict()

    for idx, row in df.iterrows():
        restaurant = row['name']
        current_rating = row['rating']

        if restaurant_counts[restaurant] <= 1:
            authenticity_scores.append(0)
            continue

        total_sum = restaurant_avg_ratings[restaurant] * restaurant_counts[restaurant]
        avg_without_current = (total_sum - current_rating) / (restaurant_counts[restaurant] - 1)

        deviation = abs(current_rating - avg_without_current)

        if deviation >= 3:
            authenticity_scores.append(-2)
        elif deviation >= 2:
            authenticity_scores.append(-1)
        else:
            authenticity_scores.append(1)

    return authenticity_scores


def sensory_words_count(text):
    """Count sensory words (taste, texture, appearance)"""
    if text is None or pd.isna(text):
        return 0

    text = str(text).lower()

    sensory_words = [
        # Taste words
        'flavorful', 'tasty', 'delicious', 'savory', 'sweet', 'bitter',
        'spicy', 'bland', 'flavorless', 'aromatic',

        # Texture words
        'crispy', 'crunchy', 'tender', 'chewy', 'creamy', 'smooth',
        'tough', 'soggy', 'dry', 'juicy',

        # Appearance words
        'beautiful', 'appealing', 'colorful', 'presentation',
        'unappetizing', 'messy', 'burnt', 'overcooked'
    ]

    count = 0
    for word in sensory_words:
        count += text.count(word)

    return count


def service_words_count(text):
    """Count service-related words"""
    if text is None or pd.isna(text):
        return 0

    text = str(text).lower()

    service_words = [
        'service', 'waitstaff', 'server', 'bartender', 'host',
        'friendly', 'attentive', 'professional', 'rude', 'ignored',
        'wait time', 'reservation', 'seated', 'prompt', 'slow'
    ]

    count = 0
    for word in service_words:
        count += text.count(word)

    return count


def atmosphere_words_count(text):
    """Count atmosphere/ambiance words"""
    if text is None or pd.isna(text):
        return 0

    text = str(text).lower()

    atmosphere_words = [
        'atmosphere', 'ambiance', 'decor', 'lighting', 'music',
        'romantic', 'cozy', 'lively', 'noisy', 'quiet',
        'crowded', 'empty', 'clean', 'dirty', 'spacious'
    ]

    count = 0
    for word in atmosphere_words:
        count += text.count(word)

    return count


def process_all_features(filename=r'C:\Users\zhang\Desktop\reviews_cleaned.csv'):
    try:
        # Load data
        df = pd.read_csv(filename)
        print(f"Loaded data with columns: {df.columns.tolist()}")
        print(f"Data shape: {df.shape}")

        # Drop label column if it exists
        if 'label' in df.columns:
            df = df.drop('label', axis=1)
            print("Dropped 'label' column")

        # Fix rating data type
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df['rating'] = df['rating'].fillna(df['rating'].median())

        # Add numerical features as new columns
        df['word_count'] = df['text'].apply(count_words)
        df['sentiment_words'] = df['text'].apply(sentiment_words_count)
        df['website_words'] = df['text'].apply(count_website_words)
        df['superlatives'] = df['text'].apply(count_superlatives)
        df['generic_words'] = df['text'].apply(count_generic_words)
        df['business_experience'] = df['text'].apply(business_experience_score)
        df['promotional_language'] = df['text'].apply(promotional_language_count)
        df['story_telling_words'] = df['text'].apply(story_telling_words_count)
        df['sensory_words'] = df['text'].apply(sensory_words_count)
        df['service_words'] = df['text'].apply(service_words_count)
        df['atmosphere_words'] = df['text'].apply(atmosphere_words_count)

        # Calculate authenticity score using vectorized approach
        df['authenticity_score'] = rate_authenticity_score_vectorized(df)

        return df

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


if __name__ == "__main__":
    # Process all features
    df = process_all_features()

    if df is not None:
        # Save with both raw data and features to Desktop
        output_path = r'C:\Users\zhang\Desktop\enhanced_reviews_with_features_testing.csv'
        df.to_csv(output_path, index=False)
        print(f"Enhanced dataset saved successfully to: {output_path}")
        print(f"Final dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    else:
        print("Failed to process features.")