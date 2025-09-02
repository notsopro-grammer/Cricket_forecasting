# InningsInsight : Cricket Score Forecasting
This project predicts the final T20I innings score of India’s national team in real time using a hybrid BiLSTM–Transformer model. It leverages enriched ball-by-ball data, advanced feature engineering, and a custom-designed loss function to produce highly accurate, interpretable forecasts-significantly outperforming ESPN’s over-wise predictions.

📌Objective
To build a machine learning pipeline that forecasts the final score of India’s innings in T20I matches using ball-level context, temporal features, and domain knowledge.

🧠 Approach
📊 Data Collection & Enrichment:
 1) Retrieved all T20 Internationals played by India from 2006 to 2024 via CricSheet.
 2) Used BeautifulSoup and Selenium to augment with:
      Venue country, toss outcome, match time (day/night), and player identifiers.
 3) Mapped stadiums to their countries to infer home/away status.

🧹 Preprocessing & Feature Engineering
  1) Standardized ball notations, dates, and resolved duplicates/missing values.
  2) Engineered 30+ features such as:
      Cumulative run rate, wickets remaining, 5/20-ball performance summaries.
  3) Partnership runs, bowler fatigue, and venue scoring impact.
  4) Recency score (days since earliest match), used to compute weight:
        w=1+α×recency_norm
  5) Converted matches into overlapping 30-ball windows with stride 6, each paired with final innings score.

🏗 Model Architecture
  1) Designed a hybrid BiLSTM–Transformer network:
      Linear projection → 2-layer BiLSTM → [CLS] token insertion → Transformer encoder → regression head.
  2) Included Positional Encoding and LayerNorm.
  3) Implemented a custom loss combining:
      MAPE + Huber Loss + under/overprediction penalties based on run rate and wickets left.

🎯 Training Strategy
  1) Optimized using AdamW, Cosine Annealing, gradient clipping, and early stopping.
  2) Training samples weighted by recency to prioritize modern matches.


📈 Evaluation
  1) Tested on a held-out India T20I match.
  2) Conducted phase-wise evaluation:
      Powerplay, Middle Overs, and Death Overs.
  3) Visualized predictions and wicket events with matplotlib.
