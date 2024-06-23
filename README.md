# Machine_learning_repo_bangkit
repository for bangkit capstone project  
Our project aims to address a pressing issue in the Indonesian job market, where rapid technological advancements and changing industry demands have created a significant skills gap. Recent reports highlight that many job seekers in Indonesia struggle to find positions that match their qualifications, and traditional job posting platforms often fail to bridge this gap effectively. JobLens will address this challenge by developing a state-of-the-art skills-based recommendation system. The project begins with requirements engineering and UI/UX design, followed by development of the machine learning model, app API, and the mobile app itself. Continuous testing and feedback will be integrated throughout the development phase to refine our approach. By significantly reducing the time and effort required by job seekers to find suitable positions, this initiative is designed to increase employment rates and create an inclusive job market free from bias related to gender, race, or socioeconomic status.   
Here’s how we organized our work:

* Machine Learning: determine a model that uses machine learning to match job seekers with the right job openings based on their skills, and that looks at past data to make the job matching more accurate over time. 
* Mobile Development: Launching the app with powerful cloud technology to make sure it performs well and can handle data in real time, making the app has features for both users and administrators, all working with real-time data syncing through Firebase, which means it can still work when there's no internet.
* Cloud Computing: Creating a secure and capable database to manage a lot of user data and job information, developing the backend services, API endpoints, and detailed documentation, all running on Google Cloud Platform to ensure everything works smoothly and is easy to access.

The final result is a working mobile app that not only finds the right jobs for users based on their skills but also improves their chances of getting hired. 

# How machine learning team work
## data exploratory
   for this project we search for data in kaggle that contains job skills and the job link, you can check the dataset in this link  
     
   [1.3M LinkedIn Jobs and Skills 2024 Dataset on Kaggle](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024) 
     
   the dataset have about 500000 unique job but for this project we only use about 50000 because of computing limitation  
   **Loading the Dataset**: Load the dataset into preferred data analysis environment (Python).  
   **Understanding the Structure**: Explore the columns and their descriptions, Check for any missing or null values.  
   **Summary Statistics**: Count unique values and frequency distributions for categorical columns.  
   we also do data preprocessing in this process by merge the dataset and taking sample, we take only the skill and job_title for the training process for efficiency. we do data exploratory and preprocessing in kaggle.  
## train the model  
1. **Imports**: Libraries like `pandas`, `numpy`, `faiss`, and `sklearn` are imported.
2. **Data Loading**: Loads job skills data into a DataFrame.
    ```python
    df = pd.read_csv('job_skills.csv')
    ```
3. **Model Training**:
    ```python
    def train_faiss_index(df):
        tfidf = TfidfVectorizer(max_features=20000)
        tfidf_matrix = tfidf.fit_transform(df['skills'])
        svd = TruncatedSVD(n_components=100)
        lsa = make_pipeline(svd, Normalizer(copy=False))
        lsa_matrix = lsa.fit_transform(tfidf_matrix)
        faiss_index = faiss.IndexFlatL2(lsa_matrix.shape[1])
        faiss_index.add(lsa_matrix)
        return faiss_index, tfidf, svd
    ```
4. **Model Evaluation**:
    ```python
    def mean_average_precision(df, n_recommendations=10):
        scores = []
        for _, row in df.iterrows():
            input_skills = row['skills']
            actual_jobs = row['job_id']
            recommended_jobs = get_recommendations(faiss_index, tfidf, svd, input_skills, n_recommendations)
            score = average_precision_at_k(actual_jobs, recommended_jobs, n_recommendations)
            scores.append(score)
        return np.mean(scores)
    ```
5. **Testing**:
    ```python
    test_data = df.sample(n=100, random_state=2)
    map_score = mean_average_precision(test_data, n_recommendations=10)
    print(f"Mean Average Precision: {map_score:.2f}")
    ```
## Load the Model
1. **Load the Models and Data**:
    The script loads the pre-trained models and data files. Ensure the paths to these files are correctly specified:

    ```python
    tfidf_path = 'path/to/tfidf_model.pkl'
    svd_path = 'path/to/svd_model.pkl'
    faiss_path = 'path/to/faiss_index.idx'
    df_path = 'path/to/jobs3.csv'
    
    tfidf = joblib.load(tfidf_path)
    svd = joblib.load(svd_path)
    faiss_index = faiss.read_index(faiss_path)
    df = pd.read_csv(df_path)
    ```

2. **Get Recommendations**:
    The `get_recommendations` function takes the following parameters:
    - `index`: FAISS index
    - `tfidf`: TF-IDF model
    - `svd`: SVD model
    - `skills_input`: A string of input skills
    - `n_recommendations`: Number of recommendations to return (default is 10)

    Example usage:

    ```python
    skills_input = "machine learning, data analysis, python"
    recommendations = get_recommendations(faiss_index, tfidf, svd, skills_input)
    print(recommendations)
    ```

### Function Definition

#### `get_recommendations`

```python
def get_recommendations(index, tfidf, svd, skills_input, n_recommendations=10):
    """
    Get job recommendations based on input skills.

    Parameters:
        index (faiss.Index): Pre-trained FAISS index.
        tfidf (TfidfVectorizer): Pre-trained TF-IDF model.
        svd (TruncatedSVD): Pre-trained SVD model.
        skills_input (str): Input skills as a string.
        n_recommendations (int): Number of recommendations to return.

    Returns:
        pd.Series: Recommended job titles.
    """
    skills_tfidf = tfidf.transform([skills_input])
    skills_reduced = svd.transform(skills_tfidf)

    distances, indices = index.search(skills_reduced, n_recommendations)
    
    return df['job_title'].iloc[indices[0]]
```

### Example

```python
# Load models and data
tfidf = joblib.load('path/to/tfidf_model.pkl')
svd = joblib.load('path/to/svd_model.pkl')
faiss_index = faiss.read_index('path/to/faiss_index.idx')
df = pd.read_csv('path/to/jobs3.csv')

# Input skills
skills_input = "machine learning, data analysis, python"

# Get recommendations
recommendations = get_recommendations(faiss_index, tfidf, svd, skills_input)
print(recommendations)
```

This should provide a list of job titles that best match the input skills based on the pre-trained models.

   

   

