# 🦜 The Travel Parrot

**Machine Learning-Powered Restaurant Review Classification System**

The Travel Parrot is an intelligent review classification system that automatically identifies and filters low-quality restaurant reviews, helping travelers find authentic dining recommendations by removing spam, advertisements, irrelevant content, and non-constructive rants.

## Overview

Restaurant review platforms are increasingly polluted with fake reviews, spam, and irrelevant content that make it difficult for users to find genuine dining experiences. The Travel Parrot addresses this problem by using advanced natural language processing and machine learning to automatically classify reviews into quality categories.

### Classification Categories

- **🔍 Good Review**: Authentic, helpful reviews providing genuine customer insights
- **🚫 Spam**: Fake, gibberish, automated, or repetitively posted content  
- **📢 Advertisement**: Reviews promoting other businesses or containing promotional links
- **😤 Rant**: Overly emotional content without constructive critique
- **❓ Irrelevant Content**: Reviews unrelated to the business experience

## Architecture

```
Stage 1: Model Training Pipeline
Raw Review Data → OpenAI GPT-4 Labeling → Data Cleaning → Feature Engineering → Random Forest Training

Stage 2: Production Classification Pipeline  
New Reviews → Data Cleaning → Feature Engineering → Trained Model → Review Classification
```

## Features

### 🧠 Intelligent Feature Engineering

The system uses 11 sophisticated features designed specifically for restaurant review analysis:

**Core Linguistic Features:**
- Word count analysis for spam detection
- Sentiment word patterns for authenticity assessment
- Generic language detection for quality filtering

**Policy-Specific Detection:**
- Website/URL pattern matching for advertisements
- Promotional language identification
- Superlative language analysis for exaggerated content

**Domain-Specific Restaurant Analysis:**
- Business experience vocabulary density
- Sensory word validation (taste, texture, appearance)
- Service and atmosphere description analysis


### 🤖 Advanced ML Pipeline

- **Random Forest Classifier** optimized for mixed feature types
- **Comprehensive validation** with cross-validation and learning curves
- **Hyperparameter tuning** for optimal performance
- **Separate training/testing pipelines** for production scalability

## Setup Instructions

### Prerequisites
```bash
# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Repository Setup
```bash
# Clone the repository
git clone https://github.com/ShuoDuan/travel-parrot.git
cd travel-parrot

# Ensure you have the pre-trained model file
# Download random_forest_review_classifier.pkl to the project directory
```

### Using the Pre-trained Model

**Step 1: Prepare Your Review Data**

Create a CSV file with your restaurant reviews containing these columns:
- `name` (restaurant name)
- `rating` (numerical rating, e.g., 1-5)
- `text` (review content)

Example `my_reviews.csv`:
```csv
name,rating,text
Joe's Pizza,5,Amazing pizza with great service and friendly staff!
Cafe Luna,2,Terrible food and rude waitstaff. Would not recommend.
Best Burgers,1,Visit our new location at www.competitor.com for better deals!
```

**Step 2: Run Classification**

```python
# Import the functions we need
from feature_engineering_for_testing import process_all_features
from model_training import load_model, predict_new_data

# Step 2a: Extract features from your raw reviews
# This function reads your CSV and automatically calculates all 12 features
# (word count, sentiment words, website words, etc.) needed for classification
df_with_features = process_all_features("my_reviews.csv")

# Step 2b: Load the pre-trained model
# This .pkl file contains our trained Random Forest model + preprocessing components
# You need to download this file from our GitHub releases or model repository
model_dict = load_model('random_forest_review_classifier.pkl')

# Step 2c: Select the feature columns the model expects
# These are the same 12 features the model was trained on
feature_columns = [
    'rating', 'word_count', 'sentiment_words', 'website_words',
    'superlatives', 'generic_words', 'business_experience',
    'promotional_language', 'story_telling_words', 'sensory_words',
    'service_words', 'atmosphere_words', 'authenticity_score'
]
X_features = df_with_features[feature_columns]

# Step 2d: Get predictions from the model
# predictions = array of labels like ['Good Review', 'Spam', 'Advertisement']
# probabilities = confidence scores for each prediction
predictions, probabilities = predict_new_data(model_dict, X_features)

# Step 2e: Add results back to your dataframe
df_with_features['predicted_label'] = predictions
df_with_features['confidence_score'] = probabilities.max(axis=1)  # Highest probability

# Step 2f: Save the results
df_with_features.to_csv('labeled_reviews.csv', index=False)
print("Reviews classified and saved to 'labeled_reviews.csv'")
```

**About the Model File (`random_forest_review_classifier.pkl`):**

This file contains:
- The trained Random Forest classifier
- The scaler used to normalize features
- The label encoder that maps predictions to text labels

**Getting the Model File:**
- **Option 1**: Download from our GitHub repository's releases section
- **Option 2**: Contact us for the model file (it may be too large for direct GitHub storage)
- **Option 3**: Train your own using our training scripts (see Model Training Process section)

**Model File Size:** ~50-100MB (typical for Random Forest with our feature set)

**Step 3: Review Output**

Your output CSV will contain:
- Original review data (name, rating, text)
- All computed features
- `predicted_label`: Classification result (Good Review, Spam, Advertisement, Rant, Irrelevant Content)
- `confidence_score`: Model confidence (0-1, higher = more confident)

```python
# View classification summary
import pandas as pd
results = pd.read_csv('labeled_reviews.csv')
print("Classification Distribution:")
print(results['predicted_label'].value_counts())
```

## Model Training Process

If you want to understand how we built the model or train your own:

### Data Labeling
We used OpenAI GPT-4 to automatically label 10,000+ restaurant reviews:
```python
# Run automated labeling (requires OpenAI API key)
python trytry.py
```

### Feature Engineering
Our system extracts 12 sophisticated features from review text:
```python
# Generate features for training data
python feature_engineering_for_training_final.py
```

### Model Training and Evaluation
```python
# Train Random Forest with comprehensive validation
python model_training.py
```

The training process includes:
- 80/20 train-test split with stratification
- Cross-validation for model selection
- Hyperparameter optimization via grid search
- Learning curves and validation curves
- Feature importance analysis
- Comprehensive evaluation metrics

## File Structure

```
travel-parrot/
├── feature_engineering_for_training.py      # Feature extraction for training data
├── feature_engineering_for_testing.py       # Feature extraction for new data  
├── feature_engineering_for_training_final.py # Optimized training features
├── model_training.py                        # ML model training and evaluation
├── trytry.py                               # OpenAI GPT-4 labeling system
├── jsson_to_csv.py                         # Data format conversion utilities
├── review_labelled_cleaned.csv             # Sample labeled dataset
├── README.md                               # Project documentation
└── requirements.txt                        # Python dependencies
```

## Data Labeling with OpenAI

The system includes an automated labeling pipeline using OpenAI's GPT-4:

```python
# Configure your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Run automated labeling
python trytry.py
```

**Features:**
- Robust error handling and retry mechanisms
- Rate limiting and quota management
- Progress tracking for large datasets
- Comprehensive logging and fallback strategies

## Model Performance

The Random Forest classifier achieves:
- **High accuracy** on multi-class review classification
- **Robust performance** across different restaurant types
- **Interpretable results** with feature importance analysis
- **Production-ready** with comprehensive validation

### Evaluation Metrics

The system provides detailed performance analysis including:
- Classification reports with precision, recall, and F1-scores
- Confusion matrices for class-wise performance
- Feature importance rankings
- Learning curves for bias/variance analysis
- Cross-validation scores for generalization assessment

## Advanced Features

### Hyperparameter Optimization

```python
# Automated hyperparameter tuning
best_model = tune_hyperparameters(X_train, y_train)
```

### Comprehensive Validation

```python
# Full model validation suite  
comprehensive_validation(model, X, y)
```

### Model Persistence

```python
# Save trained model for production use
save_model(model, scaler, label_encoder, 'classifier.pkl')

# Load model for inference
model_dict = load_model('classifier.pkl')
```

## API Integration

The system is designed for easy integration with existing review platforms:

- **RESTful API endpoints** for real-time classification
- **Batch processing** for large-scale historical data
- **Confidence scoring** for nuanced decision-making
- **Scalable architecture** for high-throughput scenarios

## Reproducing Results

To reproduce our hackathon results from scratch:

### 1. Data Collection and Preprocessing
```bash
# Process raw review data from Google Local Reviews dataset
python jsson_to_csv.py  # Convert JSONL to CSV format
```

### 2. Data Labeling (Optional - we provide pre-labeled data)
```bash
# Generate labels using OpenAI GPT-4 (requires API key)
python trytry.py
```

### 3. Feature Engineering and Model Training
```bash
# Train the model with comprehensive validation
python model_training.py
```

### 4. Model Evaluation
The training script provides:
- Cross-validation scores
- Confusion matrices
- Feature importance analysis
- Learning curves
- Hyperparameter optimization results

### Expected Results
- Multi-class classification accuracy: 85-90%
- F1-scores across all categories: 0.80+
- Robust performance on restaurant review classification
- Production-ready model saved as `random_forest_review_classifier.pkl`

## Team Contributions

This project was developed during a hackathon focused on improving online review quality:

**Core Development:**
- **Data Pipeline**: Implemented robust data collection, cleaning, and preprocessing systems
- **Feature Engineering**: Designed 12 sophisticated features for restaurant review analysis
- **Machine Learning**: Built and optimized Random Forest classifier with comprehensive validation
- **API Integration**: Created reliable OpenAI GPT-4 labeling system with error handling
- **Production Pipeline**: Developed separate training/inference systems for scalability

**Technical Implementation:**
- **Policy Enforcement**: Translated hackathon policy requirements into actionable ML features
- **Quality Assurance**: Implemented comprehensive testing and validation frameworks  
- **Documentation**: Created detailed code documentation and user guides
- **Performance Optimization**: Optimized for both accuracy and computational efficiency

## Contributing

We welcome contributions to improve The Travel Parrot! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Future Enhancements

- **🤖 Transformer Integration**: Experiment with BERT and RoBERTa models
- **🌐 Multi-language Support**: Expand to support international reviews
- **📊 Real-time Dashboard**: Administrative interface for review management
- **🔍 Explainable AI**: Model interpretability for classification decisions
- **🛡️ Adversarial Robustness**: Defense against manipulation attempts

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built during a hackathon focused on improving online review quality
- Uses Google Local Reviews dataset for training and validation
- Leverages OpenAI's GPT-4 for intelligent data labeling
- Inspired by the need for trustworthy travel recommendations

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue or reach out to the development team.

---

**The Travel Parrot** - *Your reliable companion for authentic dining discoveries* 🦜✨
