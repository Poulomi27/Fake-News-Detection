# Fake-News-Detection

Problem Definition:

What is the problem all about?
The problem is to develop a machine learning-based system for the detection of fake news
within textual content. Given a dataset of news articles labelled as either real or fake, the
objective is to design a model capable of accurately classifying news articles based on their
authenticity. The system should analyse linguistic patterns, contextual cues, and features
indicative of misinformation to discern between genuine and false information. The goal is to
provide a tool that aids in identifying and mitigating the spread of fake news, contributing to
the enhancement of information reliability and fostering a more informed public discourse.

Why is this an important problem to solve?
Fake news basically means information which is false. False information misleads us which is often done to advance or enforce specific beliefs and is frequently accomplished through political agendas. So it becomes crucial to differentiate between real and fake information to recognize fake news.
 
Business/real world impact to solve this problem-
By analysing the content of news articles and identifying patterns and characteristics of fake news, ML algorithms can learn to distinguish between real and fake news with high accuracy. This can help businesses and individuals make informed decisions based on accurate information, which can lead to better outcomes and improved performance.For example, it can help businesses avoid making decisions based on false information, which can lead to financial losses, reputational damage, and other negative consequences. It can also help individuals avoid falling prey to scams, misinformation, and propaganda, which can affect their personal lives and well-being. Furthermore, it can help prevent the spread of hate speech, fake news, and other harmful content on social media and other online platforms, which can have a negative impact on society as a whole.

Dataset:

Source of the dataset.
Fake news Detection Dataset: https://drive.google.com/drive/folders/1ByadNwMrPyds53cA6SDCHLelTAvIdoF_

Explanation of each feature and datapoint available.
Each datapoint in the dataset represents a single article and contains information about the title, subject, text, and date of publication.

Data Size and any challenges you foresee to process it?
The first dataset has 21417 rows and 5 columns, while the second dataset has 23481 rows and 5 columns.
Challenges may include:
Data cleaning: The text data may contain special characters, punctuation marks, or other unwanted elements that need to be removed before processing.


Tools (Pandas, SQL, Spark etc) that you will use to process this data.
We have used Python and several libraries such as Pandas, Matplotlib, Seaborn, NLTK, and Scikit-learn to process the data. 
Pandas is a popular data manipulation library in Python that provides easy-to-use data structures and data analysis tools. 
Matplotlib and Seaborn are data visualisation libraries that can be used to create various types of plots and charts. 
NLTK is a natural language processing library that can be used to preprocess text data. Scikit-learn is a machine learning library that provides various algorithms for classification, regression, clustering, and other tasks.
