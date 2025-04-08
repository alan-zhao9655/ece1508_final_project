# Movie Recommendation System Project

## Overview
This project explores different approaches to movie recommendation systems, comparing traditional collaborative and content-based filtering methods with modern large language model (LLM) approaches. We used the MovieLens dataset to train and evaluate various models, demonstrating the strengths and limitations of each approach for personalized movie recommendations.

## Dataset
- **MovieLens small dataset** (100k ratings)
- Contains user ratings (userId, movieId, rating, timestamp)
- Movie metadata (title, genres)
- User-generated tags providing additional context

## Approaches Implemented

### 1. Traditional Recommendation Models
Three baseline models were implemented and evaluated:

| Model | Description | Precision | Recall | F1 Score |
|-------|-------------|-----------|--------|----------|
| **Popularity Baseline** | Recommends movies based solely on their average ratings across all users | 0.0000 | 0.0000 | 0.0000 |
| **Matrix Factorization** | Decomposes the user-item rating matrix into latent factors | 0.1738 | 0.0676 | 0.0973 |
| **Factorization Machine** | Extends matrix factorization by incorporating movie features | 0.1033 | 0.0311 | 0.0478 |

### 2. LLM-Based Approaches

#### a) Fine-tuned GPT-2 Model
- Formatted user histories and preferences as natural language
- Fine-tuned a GPT-2 model to generate personalized recommendations
- Training sequences included user history followed by recommendation examples

**Example output:**
```
User ID: 425
History: User 425 has watched Trainspotting (1996), rated 5.0/5...
Recommendations: "Prefontaine (1997) - This drama film aligns with the user's preferred genres"
```

#### b) OpenAI API (GPT-4o)
- Leveraged OpenAI's API for high-quality recommendations
- Formatted user histories as detailed natural language prompts

**Example output:**
```
User ID: 425
History: User 425 has watched Trainspotting (1996), rated 5.0/5...
Recommendations: "Requiem for a Dream (2000) - This film explores the harrowing effects 
of drug addiction, similar to 'Trainspotting.' Its intense narrative and dark themes would 
resonate with the user's appreciation for movies that delve into substance abuse and 
psychological depth."
```

## Key Findings
- Traditional models provide measurable performance but lack detailed reasoning
- Matrix Factorization surprisingly outperformed the more complex Factorization Machine
- Fine-tuned GPT-2 shows potential but suffers from limitations in model size
- OpenAI's GPT-4o demonstrates superior understanding of user preferences with human-like reasoning

## Future Improvements
1. **Retrieval-Augmented Generation (RAG):** Combine LLMs with a knowledge base of movie information to reduce hallucinations
2. **Prompt Engineering:** Develop more effective prompts with few-shot learning and structured formats
3. **Advanced Fine-tuning:** Experiment with larger models and domain-specific fine-tuning techniques
4. **Hybrid Systems:** Combine the precision of traditional algorithms with the reasoning capabilities of LLMs

## Project Structure
```
project/
├── data/
│   └── ml-latest-small/         # MovieLens dataset files
├── models/
│   ├── baseline_models.py       # Traditional recommendation models
│   ├── llm_finetuning.py        # GPT-2 fine-tuning implementation
│   └── openai_recommender.py    # OpenAI API recommendation system
├── notebooks/
│   ├── data_exploration.ipynb   # Dataset analysis
│   └── model_comparison.ipynb   # Results comparison
├── results/
│   ├── baseline_metrics.json    # Performance metrics for traditional models
│   ├── openai_recommendations.json  # OpenAI-generated recommendations
│   └── gpt2_recommendations.json    # Fine-tuned GPT-2 recommendations
├── requirements.txt
└── README.md
```

## Requirements
- Python 3.8+
- PyTorch
- Pandas, NumPy, Scikit-learn
- Transformers (Hugging Face)
- OpenAI API key (for OpenAI-based recommendations)
- CUDA-capable GPU for model training

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Download MovieLens dataset (ml-latest-small)
3. Run traditional models: `python models/baseline_models.py`
4. Fine-tune GPT-2: `python models/llm_finetuning.py`
5. Generate OpenAI recommendations: `python models/openai_recommender.py --use_openai --num_openai_users 5`
6. Compare results in the provided notebooks

## Conclusion
While traditional recommendation methods provide a quantifiable baseline, LLM-based approaches demonstrate superior ability to generate personalized, contextual, and explainable recommendations. The OpenAI API offers the highest quality recommendations currently, but fine-tuned models show promise for cost-effective deployments.


## License
This project is licensed under the MIT License - see the LICENSE file for details.