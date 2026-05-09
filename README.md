# Engagement-Aware Movie Recommendation System

**CSCI 567 — Machine Learning | University of Southern California**  
Rhea Pandita · Sanjay Balasubramaniam · Arnav Kamra · Illy Hoang 

## Description

Sequential recommendation system built on a real-world Netflix UK clickstream dataset (649,250 interactions, 161,618 users, 8,411 movies). Models predict a user's next movie from their watch history, extended with item content features (genres, popularity, metadata) and user behavioural features (session patterns, genre preferences).

Evaluation uses leave-one-out full ranking over all 8,411 items — the same protocol across all models for fair comparison.

## Models

| Model | Type |
|---|---|
| BERT4Rec v2 | Bidirectional transformer |
| SASRec v3 | Causal transformer |
| LightGCN | Graph collaborative filtering |
| GRU-LightGCN | Sequential + graph hybrid |

## Results 
| Model | HR@10 | NDCG@10 | MRR |
|---|---|---|---|
| **BERT4Rec v2** | **0.3085** | **0.2081** | **0.1855** |
| SASRec v3 | 0.2849 | 0.1853 | 0.1634 |
| GRU-LightGCN | 0.1836 | 0.1082 | 0.0947 |
| LightGCN | 0.1636 | 0.0982 | 0.0847 |

---

## Installation

```bash
git clone https://github.com/RP-1106/ML-Project-Movie-Recommendation
cd ML-Project-Movie-Recommendation
pip install torch numpy pandas polars scipy scikit-learn matplotlib requests python-dotenv
```

To run feature engineering:
```bash
export TMDB_API_KEY=your_key  
python feature_engineering.py
```

---

## Dataset

[Netflix Audience Behaviour — UK Movies](https://www.kaggle.com/) on Kaggle. Place `netflix_uk_data_filtered.csv` in the project root before running `feature_engineering.py`.

---

## References

- Sun et al. (2019) BERT4Rec — [arXiv:1904.06690](https://arxiv.org/abs/1904.06690)
- Kang & McAuley (2018) SASRec — [arXiv:1808.09781](https://arxiv.org/abs/1808.09781)
- He et al. (2020) LightGCN — [arXiv:2002.02126](https://arxiv.org/abs/2002.02126)
- Rafailov et al. (2023) DPO — [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
