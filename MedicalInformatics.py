
# DRUG INFORMATION PROJECT:
import pandas as pd

df = pd.read_csv("MID.csv")
# PHASE 1:
# #  basic info
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())


import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# output: Shape: (192807, 15)
# Columns: ['Name', 'Link', 'Contains', 'ProductIntroduction', 'ProductUses', 'ProductBenefits', 'SideEffect', 'HowToUse', 'HowWorks', 'QuickTips', 'SafetyAdvice', 'Chemical_Class', 'Habit_Forming', 'Therapeutic_Class', 'Action_Class']
#                       Name  ...                                       Action_Class
# 0       Andol 0.5mg Tablet  ...                              Typical Antipsychotic
# 1  Avastin 100mg Injection  ...    Vascular endothelial growth factor (VEGF) in...
# 2    Actorise 40 Injection  ...              Erythropoiesis-stimulating agent (ESA
# 3    Actorise 25 Injection  ...              Erythropoiesis-stimulating agent (ESA
# 4    Actorise 60 Injection  ...              Erythropoiesis-stimulating agent (ESA


# Missing Values
print('\nMissing values (%):')
print(df.isnull().mean().round(4) * 100)

# output: Missing values (%):
# name                    0.00
# link                    0.00
# contains                0.00
# productintroduction     6.22
# productuses             0.00
# productbenefits         0.00
# sideeffect              0.00
# howtouse                0.05
# howworks                0.12
# quicktips               0.00
# safetyadvice            0.00
# chemical_class         47.37
# habit_forming           0.00
# therapeutic_class       0.00
# action_class           55.56


# SUMMARY STAT
print(df.describe(include='all').T)

# output: count  ...    freq
# name                 192807  ...      12
# link                 192807  ...       5
# contains             192807  ...    2205
# productintroduction  180821  ...     506
# productuses          192807  ...   19283
# productbenefits      192807  ...    2994
# sideeffect           192807  ...    3558
# howtouse             192712  ...   17018
# howworks             192571  ...     676
# quicktips            192807  ...    1463
# safetyadvice         192807  ...     229
# chemical_class       101473  ...    4805
# habit_forming        192807  ...  188428
# therapeutic_class    192807  ...   20291
# action_class          85690  ...    5888

# CHECKING UNIQUE VALUES:

print("Unique drug names:", df['name'].nunique())
print("Unique therapeutic classes:", df['therapeutic_class'].nunique())
print("Top therapeutic classes:")
print(df['therapeutic_class'].value_counts().head(10))

# OUTPUT:
# Unique drug names: 147872
# Unique therapeutic classes: 44
# Top therapeutic classes:
# therapeutic_class
# ANTI INFECTIVES       20291
# PAIN ANALGESIC        18861
# RESPIRATOR            16392
# GASTRO INTESTINA      15716
# ANTI INFECTIVE        13333
# NEURO CNS             11995
# GASTRO INTESTINAL     11743
# CARDIA                 8486
# ANTI DIABETI           8466
# NEURO CN               8125


text_columns = [
    'productintroduction', 'productuses', 'productbenefits', 'sideeffect',
    'howtouse', 'howworks', 'quicktips', 'safetyadvice'
]
def clean_text(text):
    if pd.isnull(text):
        return ""
    return (
        str(text)
        .replace('\n', ' ')
        .replace('\r', ' ')
        .replace('\"', '')
        .replace('  ', ' ')
        .strip()
    )

# Apply to all text columns
for col in text_columns:
    df[col] = df[col].apply(clean_text)


print("Before Cleaning:\n", df.loc[86504, 'productintroduction'])
print("\nAfter Cleaning:\n", clean_text(df.loc[86504, 'productintroduction']))

df.to_csv("MID_cleaned.csv", index=False)




# PHASE 2: VISUALISATION
import matplotlib.pyplot as plt

# df['therapeutic_class'].value_counts().head(10).plot(kind='bar', figsize=(10,5), color='skyblue')
# plt.title('Top 10 Therapeutic Classes')
# plt.ylabel('Number of Drugs')
# plt.xlabel('Therapeutic Class')
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=12)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

from wordcloud import WordCloud
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'<.*?>', ' ', text)          # Remove HTML tags like <ul>, <li>, <span>
    text = re.sub(r'\\[a-z]+', ' ', text)       # Remove \n, \ul, \li, etc.
    text = re.sub(r'\b(span|ul|li|br|nbsp|amp|n)\b', ' ', text, flags=re.IGNORECASE)  # Remove leftover keywords
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)    # Remove numbers and special characters
    text = re.sub(r'\s+', ' ', text).strip()    # Remove extra whitespace
    return text.lower()
#
#
df['clean_productuses'] = df['productuses'].apply(clean_text)

import matplotlib.pyplot as plt

text = " ".join(df['clean_productuses'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Most Common Drug Uses", fontsize=18)
plt.show()


# piechart
df['habit_forming'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.title('Habit Forming Drugs Distribution')
plt.ylabel('')
plt.show()

import seaborn as sns

plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()



# PHASE 3: DRUG CLASSIFICATION ML MODEL:
# combine and clean:
print(df.columns.tolist())

df = df.dropna(subset=['therapeutic_class'])
df['text'] = (df['productuses'] + ' ' + df['productintroduction']).fillna('')
df['text'] = df['text'].apply(clean_text)
#
from sklearn.feature_extraction.text import TfidfVectorizer
X = TfidfVectorizer(max_features=5000).fit_transform(df['text'])
y = df['therapeutic_class']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)

from sklearn.metrics import classification_report, accuracy_score

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

#  output:
# precision    recall  f1-score   support
#
#                   ANTI DIABETI       0.88      0.95      0.91      1676
#                 ANTI DIABETIC        0.96      0.70      0.81       804
#                 ANTI INFECTIVE       0.81      0.80      0.81      2631
#               ANTI INFECTIVES        0.59      0.98      0.73      4029
#                  ANTI MALARIAL       1.00      0.69      0.82        52
#                ANTI MALARIALS        0.93      0.93      0.93       129
#                ANTI NEOPLASTIC       1.00      0.23      0.37        79
#              ANTI NEOPLASTICS        0.83      0.85      0.84       486
#                   BLOOD RELATE       1.00      0.48      0.64       223
#                 BLOOD RELATED        0.79      0.87      0.83       346
#                         CARDIA       0.96      0.93      0.94      1644
#                       CARDIAC        0.97      0.87      0.92      1464
#                           DERM       0.72      0.91      0.80      1185
#                         DERMA        0.88      0.68      0.77       786
#               GASTRO INTESTINA       0.93      0.74      0.83      3143
#             GASTRO INTESTINAL        0.98      0.81      0.89      2397
#                  GYNAECOLOGICA       0.90      0.55      0.69       164
#                GYNAECOLOGICAL        0.88      0.93      0.91       703
#                        HORMONE       0.98      0.96      0.97       436
#                      HORMONES        0.89      0.77      0.82       318
#                       NEURO CN       0.90      0.84      0.87      1659
#                     NEURO CNS        0.96      0.88      0.92      2352
#                         OPHTHA       0.64      0.72      0.68       350
#                       OPHTHAL        0.78      0.86      0.82       655
#             OPHTHAL OTOLOGICAL       0.93      0.09      0.16       152
#           OPHTHAL OTOLOGICALS        0.97      0.24      0.39       116
#                          OTHER       0.00      0.00      0.00         2
#                        OTHERS        1.00      0.63      0.78        41
#                     OTOLOGICAL       0.95      0.29      0.44        63
#                   OTOLOGICALS        0.00      0.00      0.00        28
#                 PAIN ANALGESIC       0.87      0.91      0.89      3818
#               PAIN ANALGESICS        0.89      0.69      0.78      1192
#                     RESPIRATOR       0.94      0.84      0.89      3273
#                   RESPIRATORY        0.80      0.91      0.85       727
#     SEX STIMULANTS REJUVENATOR       1.00      0.28      0.44        67
#   SEX STIMULANTS REJUVENATORS        0.79      0.97      0.87       116
#                 STOMATOLOGICAL       1.00      0.21      0.35        42
#               STOMATOLOGICALS        0.00      0.00      0.00         4
#                         UROLOG       1.00      0.48      0.65       168
#                       UROLOGY        0.77      0.75      0.76       251
#                        VACCINE       0.81      0.62      0.70        42
#                      VACCINES        1.00      0.56      0.71        18
#     VITAMINS MINERALS NUTRIENT       0.76      0.58      0.66       572
#   VITAMINS MINERALS NUTRIENTS        0.99      0.48      0.65       159
#
#                       accuracy                           0.84     38562
#                      macro avg       0.83      0.65      0.69     38562
#                   weighted avg       0.86      0.84      0.84     38562


# PHASE 4: TF-IDF + Logistic Regression with class name merging and filtering
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------
# 1. Load and clean
# -------------------
df = pd.read_csv("MID.csv")  # Replace with your file path
df.columns = df.columns.str.lower()
df = df.dropna(subset=['therapeutic_class'])

# Optional: Reset index after dropna
df = df.reset_index(drop=True)

# Combine columns into a single text feature
df['text'] = (df['productuses'].fillna('') + ' ' + df['productintroduction'].fillna(''))

# -----------------------------
# 2. Merge similar class names
# -----------------------------
merge_map = {
    'ANTI INFECTIVES': 'ANTI INFECTIVE',
    'CARDIA': 'CARDIAC',
    'NEURO CN': 'NEURO CNS',
    'GASTRO INTESTINA': 'GASTRO INTESTINAL',
    'HORMONE': 'HORMONES',
    'OPHTHAL': 'OPHTHAL OTOLOGICALS',
    'DERM': 'DERMA',
    'PAIN ANALGESIC': 'PAIN ANALGESICS',
    'RESPIRATOR': 'RESPIRATORY',
    'VACCINE': 'VACCINES',
    'UROLOG': 'UROLOGY',
    'STOMATOLOGICAL': 'STOMATOLOGICALS',
    'SEX STIMULANTS REJUVENATOR': 'SEX STIMULANTS REJUVENATORS',
    'ANTI MALARIAL': 'ANTI MALARIALS',
    'ANTI DIABETI': 'ANTI DIABETIC',
    'BLOOD RELATE': 'BLOOD RELATED',
    'OPHTHAL OTOLOGICAL': 'OPHTHAL OTOLOGICALS',
    'OTOLOGICAL': 'OTOLOGICALS',
    'OTHER': 'OTHERS',
}
df['therapeutic_class'] = df['therapeutic_class'].replace(merge_map)

# -------------------------------
# 3. Filter classes with >=50 samples
# -------------------------------
class_counts = df['therapeutic_class'].value_counts()
major_classes = class_counts[class_counts >= 50].index
df = df[df['therapeutic_class'].isin(major_classes)]

# -------------------
# 4. TF-IDF Vectorizer
# -------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['therapeutic_class']

# --------------------------
# 5. Train/test split & model
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------------------
# 6. Evaluation report
# ---------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------------------
# 7. Confusion Matrix Plot
# ---------------------
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
sns.heatmap(cm, annot=False, fmt='d', cmap='viridis',
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Therapeutic Class Prediction)")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# output: precision    recall  f1-score   support
#
#                   ANTI DIABETI       0.97      0.99      0.98      1693
#                 ANTI DIABETIC        0.99      0.95      0.97       810
#                 ANTI INFECTIVE       0.97      0.94      0.95      2667
#               ANTI INFECTIVES        0.96      0.98      0.97      4058
#                  ANTI MALARIAL       1.00      0.96      0.98        47
#                ANTI MALARIALS        0.97      0.95      0.96       122
#                ANTI NEOPLASTIC       0.94      0.74      0.83        68
#              ANTI NEOPLASTICS        0.95      0.98      0.96       507
#                   BLOOD RELATE       0.99      0.91      0.95       197
#                 BLOOD RELATED        0.96      0.99      0.97       349
#                         CARDIA       0.99      0.97      0.98      1697
#                       CARDIAC        0.97      0.98      0.97      1438
#                           DERM       0.97      0.96      0.97      1165
#                         DERMA        0.94      0.97      0.95       774
#               GASTRO INTESTINA       0.98      0.99      0.99      3143
#             GASTRO INTESTINAL        0.98      0.99      0.98      2349
#                  GYNAECOLOGICA       0.92      0.86      0.89       173
#                GYNAECOLOGICAL        1.00      0.98      0.99       705
#                        HORMONE       0.97      1.00      0.98       444
#                      HORMONES        1.00      0.93      0.96       324
#                       NEURO CN       0.99      0.97      0.98      1625
#                     NEURO CNS        0.98      0.99      0.99      2399
#                         OPHTHA       0.90      0.93      0.92       346
#                       OPHTHAL        0.97      0.97      0.97       636
#             OPHTHAL OTOLOGICAL       0.95      0.87      0.91       144
#           OPHTHAL OTOLOGICALS        1.00      0.92      0.96       123
#                        OTHERS        0.95      0.89      0.92        44
#                     OTOLOGICAL       0.97      0.97      0.97        64
#                   OTOLOGICALS        1.00      0.93      0.96        28
#                 PAIN ANALGESIC       0.98      0.99      0.99      3772
#               PAIN ANALGESICS        0.98      0.95      0.97      1178
#                     RESPIRATOR       0.99      1.00      0.99      3278
#                   RESPIRATORY        0.97      0.95      0.96       745
#     SEX STIMULANTS REJUVENATOR       1.00      0.95      0.98        62
#   SEX STIMULANTS REJUVENATORS        0.99      0.99      0.99       114
#                 STOMATOLOGICAL       0.97      0.89      0.93        38
#                         UROLOG       0.95      0.96      0.96       164
#                       UROLOGY        0.98      0.97      0.98       251
#                        VACCINE       0.95      0.89      0.92        44
#                      VACCINES        0.77      0.83      0.80        12
#     VITAMINS MINERALS NUTRIENT       0.94      0.97      0.95       580
#   VITAMINS MINERALS NUTRIENTS        0.96      0.84      0.90       179
#
#                       accuracy                           0.97     38556
#                      macro avg       0.97      0.94      0.95     38556
#                   weighted avg       0.97      0.97      0.97     38556