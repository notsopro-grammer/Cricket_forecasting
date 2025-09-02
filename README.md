InningsInsight : Cricket Score Forecasting
This project predicts the final T20I innings score of India’s national team in real time using a hybrid BiLSTM–Transformer model. It leverages enriched ball-by-ball data, advanced feature engineering, and a custom-designed loss function to produce highly accurate, interpretable forecasts-significantly outperforming ESPN’s over-wise predictions.

📌Objective To build a machine learning pipeline that forecasts the final score of India’s innings in T20I matches using ball-level context, temporal features, and domain knowledge.

🧠 Approach 📊 Data Collection & Enrichment:

Retrieved all T20 Internationals played by India from 2006 to 2024 via CricSheet.
Used BeautifulSoup and Selenium to augment with: Venue country, toss outcome, match time (day/night), and player identifiers.
Mapped stadiums to their countries to infer home/away status.
🧹 Preprocessing & Feature Engineering

Standardized ball notations, dates, and resolved duplicates/missing values.
Engineered 30+ features such as: Cumulative run rate, wickets remaining, 5/20-ball performance summaries.
Partnership runs, bowler fatigue, and venue scoring impact.
Recency score (days since earliest match), used to compute weight: w=1+α×recency_norm
Converted matches into overlapping 30-ball windows with stride 6, each paired with final innings score.
🏗 Model Architecture

Designed a hybrid BiLSTM–Transformer network: Linear projection → 2-layer BiLSTM → [CLS] token insertion → Transformer encoder → regression head.
Included Positional Encoding and LayerNorm.
Implemented a custom loss combining: MAPE + Huber Loss + under/overprediction penalties based on run rate and wickets left.
🎯 Training Strategy

Optimized using AdamW, Cosine Annealing, gradient clipping, and early stopping.
Training samples weighted by recency to prioritize modern matches.
📈 Evaluation

Tested on a held-out India T20I match.
Conducted phase-wise evaluation: Powerplay, Middle Overs, and Death Overs.
Visualized predictions and wicket events with matplotlib.
