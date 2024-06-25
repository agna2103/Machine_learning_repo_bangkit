# Exploratory Data and Data Preprocessing
## Exploratory data 
Exploratory Data

The first step in our machine learning workflow was to perform an exploratory data analysis (EDA) to understand the structure and characteristics of our dataset. We utilized the "1.3M LinkedIn Jobs and Skills 2024 Dataset" from Kaggle, which contains approximately 500,000 unique job listings. For our project, due to computing limitations, we focused on a subset of 50,000 job listings.

Loading the Dataset:

The dataset was loaded into our preferred data analysis environment, Python, using the Pandas library.
python
Salin kode
import pandas as pd

### Load the dataset
dataset_url = 'path_to_your_kaggle_dataset'
df = pd.read_csv(dataset_url)
Understanding the Structure:

We explored the columns and their descriptions to get a sense of the data.
python
Salin kode
### Display the first few rows of the dataset
print(df.head())

### Display the dataset's columns
print(df.columns)
We checked for any missing or null values in the dataset.
python
Salin kode
### Check for missing values
print(df.isnull().sum())
Summary Statistics:

We counted unique values and calculated frequency distributions for the categorical columns, such as job titles and skills.
python
Salin kode
### Count unique values
unique_jobs = df['job_title'].nunique()
unique_skills = df['skills'].nunique()
print(f'Unique job titles: {unique_jobs}')
print(f'Unique skills: {unique_skills}')

### Frequency distribution for job titles
job_title_distribution = df['job_title'].value_counts()
print(job_title_distribution.head())
## Data Preprocessing
After the exploratory data analysis, we proceeded with data preprocessing to prepare the data for model training. This step involved several tasks to clean and transform the data into a suitable format.

Steps in Data Preprocessing:
Data Merging and Sampling:

Given the size of the dataset, we merged the relevant columns and took a sample of 50,000 records to ensure efficiency during the training process.
python
Salin kode
### Take a sample of 50,000 records
df_sample = df.sample(n=50000, random_state=42)

### Select relevant columns
df_sample = df_sample[['job_title', 'skills']]
Handling Missing Values:

We handled missing values by either filling them with appropriate values or dropping the rows/columns with significant amounts of missing data.
python
Salin kode
### Drop rows with missing values
df_sample = df_sample.dropna()
Data Transformation:

We transformed the text data into a format suitable for machine learning algorithms. This included tokenizing the job titles and skills, and converting them into numerical representations using techniques like one-hot encoding or TF-IDF.
python
Salin kode
from sklearn.feature_extraction.text import TfidfVectorizer

### TF-IDF vectorization of job titles
tfidf_vectorizer = TfidfVectorizer()
job_title_tfidf = tfidf_vectorizer.fit_transform(df_sample['job_title'])

### TF-IDF vectorization of skills
skills_tfidf = tfidf_vectorizer.fit_transform(df_sample['skills'])
Feature Engineering:

We engineered additional features if necessary, to improve the model's performance.
python
Salin kode
### Example: Creating a feature that counts the number of skills per job listing
df_sample['num_skills'] = df_sample['skills'].apply(lambda x: len(x.split(',')))
Splitting the Dataset:

Finally, we split the dataset into training and testing sets to evaluate the model's performance.
python
Salin kode
from sklearn.model_selection import train_test_split

### Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(job_title_tfidf, skills_tfidf, test_size=0.2, random_state=42)
By following these steps, we ensured that our dataset was clean, well-understood, and ready for the subsequent phases of model training and evaluation. This rigorous approach to data exploration and preprocessing is crucial for building a robust and effective skills-based recommendation system.
