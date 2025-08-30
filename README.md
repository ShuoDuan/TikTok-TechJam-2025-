# 🦜 The Travel Parrot  

The Travel Parrot is a machine learning system that filters out **low-quality, fake, or irrelevant reviews** from restaurant review platforms. Using advanced NLP, custom feature engineering, and a robust classification model, it identifies and removes spam, ads, and unhelpful rants — leaving only **authentic, constructive reviews** that enhance trust in online review ecosystems.  

---

## 🌟 Project Overview  
- **Problem**: Online restaurant reviews are increasingly polluted with spam, irrelevant content, and fake entries, reducing trust and usability.  
- **Solution**: A two-stage ML pipeline combining **LLM-based labeling (GPT-4o)** and a **Random Forest classifier** with 12 custom-designed linguistic and domain features.  
- **Outcome**: Accurate classification of reviews into 5 categories: *Irrelevant Content, Advertisement, Rant, Spam, Good Reviews*.  
- **Impact**: Platforms can deliver more reliable recommendations, improving user trust and customer decision-making.  

---

## ⚙️ Setup Instructions  

### 1. Clone Repository  
```bash
git clone https://github.com/<your-repo>/travel-parrot.git
cd travel-parrot
```

### 2. Install Dependencies  
We recommend using Python 3.9+. Install requirements via:  
```bash
pip install -r requirements.txt
```

### 3. Environment Setup  
Create a `.env` file in the project root and add your OpenAI API key:  
```
OPENAI_API_KEY=your_api_key_here
```

### 4. Dataset  
- Primary source: [Google Local Reviews (Kaggle)](https://www.kaggle.com/datasets) and Google Local Data (2021).  
- Place raw datasets in the `data/raw/` directory.  
- Preprocessed/cleaned datasets will be saved to `data/processed/`.  

---

## 🔄 How to Reproduce Results  

1. **Data Cleaning & Preprocessing**  
```bash
python scripts/preprocess.py
```

2. **Label Generation with GPT-4o**  
```bash
python scripts/label_reviews.py
```
(Automatically applies GPT-4o labeling with retry, quota management, and error logging.)  

3. **Feature Engineering**  
```bash
python scripts/feature_engineering.py
```

4. **Model Training**  
```bash
python scripts/train_model.py
```
Trains the Random Forest classifier on labeled data.  

5. **Run Inference on New Reviews**  
```bash
python scripts/predict.py --input data/new_reviews.csv --output results/classified_reviews.csv
```

6. **Evaluate Results**  
```bash
python scripts/evaluate.py
```
Outputs precision, recall, and F1-score as per hackathon requirements.  

---

## 👩‍💻 Team Member Contributions  

- **Data Engineering & Preprocessing**: Standardized datasets, handled missing values, implemented cleaning pipeline.  
- **Labeling & API Integration**: Designed GPT-4o prompt strategy, built error-handling and quota management system.  
- **Feature Engineering**: Developed 12 custom features spanning linguistic, policy-specific, authenticity, and restaurant-domain indicators.  
- **Machine Learning**: Random Forest model training, validation, and pipeline scalability design.  
- **Documentation & Deployment**: Wrote README, structured repo, ensured reproducibility for hackathon evaluation.  

*(If you want, we can replace this generic list with your actual teammates’ names and roles.)*  

---

## 🔮 What’s Next  
- Real-time REST API for integration with platforms  
- Advanced transformer models (BERT, RoBERTa) for accuracy gains  
- Expansion to multi-language and multi-domain reviews  
- Administrative dashboards and user appeal systems  

---

## 📌 Conclusion  
**The Travel Parrot** enables cleaner, more trustworthy review platforms by filtering out irrelevant, fake, and spammy content. Through reproducible pipelines and robust ML engineering, it demonstrates how intelligent automation can significantly improve digital ecosystems for both businesses and consumers.  
