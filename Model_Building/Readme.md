## Table of Contents

1. [Introduction](#introduction)
2. [Imports and Setup](#imports-and-setup)
3. [Data Loading](#data-loading)
4. [Model Training](#model-training)
5. [Model Evaluation](#model-evaluation)
6. [Testing](#testing)

## Introduction

This notebook is designed to train a machine learning model for job recommendation based on input skills. The model uses techniques like TF-IDF for text processing and FAISS for approximate nearest neighbor search.

## Imports and Setup

### Code

```python
import pandas as pd
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
```

### Description

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical computations.
- **faiss**: For efficient similarity search and clustering of dense vectors.
- **sklearn**: For machine learning algorithms and preprocessing.

## Data Loading

### Code

```python
df = pd.read_csv('job_skills.csv')
df.head()
```

### Description

- **df**: DataFrame containing job skills data loaded from a CSV file.

## Model Training

### Code

```python
def train_faiss_index(df):
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=20000)
    tfidf_matrix = tfidf.fit_transform(df['skills'])
    
    # Dimensionality Reduction
    svd = TruncatedSVD(n_components=100)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    lsa_matrix = lsa.fit_transform(tfidf_matrix)
    
    # Building FAISS index
    faiss_index = faiss.IndexFlatL2(lsa_matrix.shape[1])
    faiss_index.add(lsa_matrix)
    
    return faiss_index, tfidf, svd
```

### Description

- **train_faiss_index**: Function to train the FAISS index using TF-IDF and SVD for dimensionality reduction.
- **TF-IDF Vectorization**: Converts text to a matrix of TF-IDF features.
- **Dimensionality Reduction**: Uses TruncatedSVD and Normalizer to reduce the dimensions of the TF-IDF matrix.
- **FAISS Index**: Builds a FAISS index for fast similarity search.

## Model Evaluation

### Code

```python
def mean_average_precision(df, n_recommendations=10):
    # Custom evaluation function for calculating MAP
    scores = []
    for _, row in df.iterrows():
        input_skills = row['skills']
        actual_jobs = row['job_id']
        
        recommended_jobs = get_recommendations(faiss_index, tfidf, svd, input_skills, n_recommendations)
        score = average_precision_at_k(actual_jobs, recommended_jobs, n_recommendations)
        scores.append(score)
    
    return np.mean(scores)
```

### Description

- **mean_average_precision**: Function to calculate the Mean Average Precision (MAP) for evaluating the recommendation model.
- **get_recommendations**: Retrieves job recommendations based on input skills.
- **average_precision_at_k**: Calculates precision at k for evaluating the quality of recommendations.

## Testing

### Code

```python
test_data = df.sample(n=100, random_state=2)
map_score = mean_average_precision(test_data, n_recommendations=10)
print(f"Mean Average Precision: {map_score:.2f}")
```

### Description

- **test_data**: Random sample of 100 rows from the DataFrame for testing.
- **mean_average_precision**: Calculates the MAP score for the test data.
- **print**: Displays the MAP score.



