# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.porter import PorterStemmer
import string
from collections import Counter  # Add this import statement


# Downloading NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin1')
df.head()
df.info()
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

# Label encoding
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Check for missing values and duplicates
df.isnull().sum()
df.duplicated().sum()

# Remove duplicates
df = df.drop_duplicates(keep='first')
df.shape

# EDA - Distribution of target labels
values = df['target'].value_counts()
total = values.sum()
percentage_0 = (values[0] / total) * 100
percentage_1 = (values[1] / total) * 100
print('Percentage of ham:', percentage_0)
print('Percentage of spam:', percentage_1)

# Plot pie chart
colors = ['#FF5733', '#33FF57']
explode = (0, 0.1)
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor('white')
wedges, texts, autotexts = ax.pie(
    values, labels=['ham', 'spam'],
    autopct='%0.2f%%',
    startangle=90,
    colors=colors,
    wedgeprops={'linewidth': 2, 'edgecolor': 'white'},
    explode=explode, shadow=True
)
for text, autotext in zip(texts, autotexts):
    text.set(size=14, weight='bold')
    autotext.set(size=14, weight='bold')
ax.set_title('Email Classification', fontsize=16, fontweight='bold')
ax.axis('equal')
plt.show()

# Adding text features
df['num_characters'] = df['text'].apply(len)
df['num_words'] = df['text'].apply(lambda x: len(word_tokenize(x)))
df['num_sentence'] = df['text'].apply(lambda x: len(sent_tokenize(x)))

# Describe text features
df[['num_characters', 'num_words', 'num_sentence']].describe()
df[df['target'] == 0][['num_characters', 'num_words', 'num_sentence']].describe()
df[df['target'] == 1][['num_characters', 'num_words', 'num_sentence']].describe()

# Plot histograms
plt.figure(figsize=(10, 6))
sns.histplot(df[df['target'] == 0]['num_characters'], color='blue', label='Ham', kde=True)
sns.histplot(df[df['target'] == 1]['num_characters'], color='red', label='Spam', kde=True)
plt.xlabel('Number of Characters', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Number of Characters by Target', fontsize=16, fontweight='bold')
plt.legend()
sns.set(style='whitegrid')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df[df['target'] == 0]['num_words'], color='blue', label='Ham', kde=True)
sns.histplot(df[df['target'] == 1]['num_words'], color='red', label='Spam', kde=True)
plt.xlabel('Number of Words', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Distribution of Number of Words by Target', fontsize=16, fontweight='bold')
plt.legend()
sns.set(style='whitegrid')
plt.show()

# Pairplot
sns.set(style='ticks', color_codes=True)
g = sns.pairplot(df, hue='target', diag_kind='kde', markers=["o", "s"])
g.fig.suptitle("Pairplot of Data by Target", fontsize=16, fontweight='bold')
plt.subplots_adjust(top=0.95)
g._legend.set_title('Target')
for t, l in zip(g._legend.texts, ["Ham", "Spam"]):
    t.set_text(l)
plt.show()

# Correlation heatmap
correlation_matrix = df[['target', 'num_characters', 'num_words', 'num_sentence']].corr()
plt.figure(figsize=(10, 6))
sns.set(font_scale=1.2)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title("Correlation Heatmap", fontsize=16, fontweight='bold')
plt.xticks(rotation=45)
plt.show()

# Text preprocessing function
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

# Apply text transformation
df['transformed_text'] = df['text'].apply(transform_text)

# Word clouds
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15, 6))
plt.imshow(spam_wc)
plt.show()

ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(15, 6))
plt.imshow(ham_wc)
plt.show()

# Most common words bar plots
spam_words = [word for sentence in df[df['target'] == 1]['transformed_text'] for word in sentence.split()]
ham_words = [word for sentence in df[df['target'] == 0]['transformed_text'] for word in sentence.split()]

spam_word_counts = pd.DataFrame(Counter(spam_words).most_common(30))
ham_word_counts = pd.DataFrame(Counter(ham_words).most_common(30))

sns.barplot(data=spam_word_counts, x=0, y=1, palette='bright')
plt.xticks(rotation=90)
plt.show()

sns.barplot(data=ham_word_counts, x=0, y=1, palette='cool')
plt.xticks(rotation=90)
plt.show()

# Feature extraction and data split
from sklearn.feature_extraction.text import TfidfVectorizer
tfid = TfidfVectorizer(max_features=3000)
X = tfid.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)

# Model training and evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score

# Instantiate classifiers
svc = SVC(kernel="sigmoid", gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb = XGBClassifier(n_estimators=50, random_state=2)

clfs = {
    'SVC': svc,
    'KNN': knc,
    'NB': mnb,
    'DT': dtc,
    'LR': lrc,
    'RF': rfc,
    'Adaboost': abc,
    'Bgc': bc,
    'ETC': etc,
    'GBDT': gbdt,
    'xgb': xgb
}

# Function to train and evaluate classifiers
def train_classifier(clfs, X_train, y_train, X_test, y_test):
    clfs.fit(X_train, y_train)
    y_pred = clfs.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return accuracy, precision

# Evaluate all classifiers
accuracy_scores = []
precision_scores = []
for name, clfs in clfs.items():
    current_accuracy, current_precision = train_classifier(clfs, X_train, y_train, X_test, y_test)
    print()
    print("For: ", name)
    print("Accuracy: ", current_accuracy)
    print("Precision: ", current_precision)
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)
