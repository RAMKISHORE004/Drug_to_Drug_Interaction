#Drug to Drug Interaction code
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
# Load the dataset
file_path = "C:/Users/ahana/OneDrive/Desktop/capstone/Drug_to_Drug_Interaction/DDICorpus2013.xlsx"
df = pd.read_excel(file_path)
print(df.head(10))
print(df.shape)
 #Check for duplicate rows
print(f"Number of duplicate rows: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(df.shape)



# Step 1: Handle Missing Values (Remove rows with missing drug names or interaction sentences)
df_preprocessed = df.dropna(subset=["Drug_1_Name", "Drug_2_Name", "Sentence_Text"])

# Step 2: Normalize Drug Names (Lowercase, strip spaces)
df_preprocessed["Drug_1_Name"] = df_preprocessed["Drug_1_Name"].str.lower().str.strip()
df_preprocessed["Drug_2_Name"] = df_preprocessed["Drug_2_Name"].str.lower().str.strip()

# Step 3: Expand Drug Synonyms (Mapping common synonyms)
synonym_map = {
    "aspirin": "acetylsalicylic acid",
    "paracetamol": "acetaminophen",
    "tylenol": "acetaminophen",
    "ibuprofen": "advil",
    "omeprazole": "prilosec",
}

def replace_synonyms(drug_name):
    return synonym_map.get(drug_name, drug_name)

df_preprocessed["Drug_1_Name"] = df_preprocessed["Drug_1_Name"].apply(replace_synonyms)
df_preprocessed["Drug_2_Name"] = df_preprocessed["Drug_2_Name"].apply(replace_synonyms)

# Step 4: Text Cleaning (Removing special characters, multiple spaces)
def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Normalize spaces
    return text

df_preprocessed["Sentence_Text"] = df_preprocessed["Sentence_Text"].apply(clean_text)

import matplotlib.pyplot as plt
import seaborn as sns
# Investigate the frequency of drug interactions
interaction_counts = df.groupby(['Drug_1_Name', 'Drug_2_Name']).size().reset_index(name='Interaction_Count')
interaction_counts = interaction_counts.sort_values('Interaction_Count', ascending=False)

# Sort the interaction counts in descending order
top_interactions = interaction_counts.sort_values('Interaction_Count', ascending=False)

# Create a bar plot of the top drug interactions
plt.figure(figsize=(12, 8))
sns.barplot(x='Interaction_Count', y='Drug_1_Name', hue='Drug_2_Name', data=top_interactions.head(20))
plt.title('Top 20 Drug Interactions')
plt.xlabel('Interaction Count')
plt.ylabel('Drug 1 Name')
plt.legend(title='Drug 2 Name', loc='upper left')
plt.show()

# Create a histogram of the interaction counts
plt.figure(figsize=(12, 6))
sns.histplot(interaction_counts['Interaction_Count'], bins=30, kde=True)
plt.title('Distribution of Drug Interaction Counts')
plt.xlabel('Interaction Count')
plt.ylabel('Frequency')
plt.show()

# Group the data by drug categories and calculate the total interaction counts
interaction_by_category = df.groupby(['Drug_1_Drugbankid', 'Drug_2_Drugbankid'])['Interaction_Count'].sum().reset_index()
interaction_by_category = interaction_by_category.sort_values('Interaction_Count', ascending=False)

# Create a heatmap of the interaction counts by drug categories
plt.figure(figsize=(12, 10))
pivot_table = interaction_by_category.pivot(index='Drug_1_Drugbankid', columns='Drug_2_Drugbankid', values='Interaction_Count')
sns.heatmap(pivot_table, annot=True, cmap='YlOrRd')
plt.title('Drug Interaction Counts by Categories')
plt.xlabel('Drug 2 Category')
plt.ylabel('Drug 1 Category')
plt.show()


import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Assuming we have our dataframe 'df' with interaction counts
# First, let's create a simplified target variable based on interaction frequency
def categorize_interaction(count):
    if count == 1:
        return 'rare'
    elif count <= 3:
        return 'moderate'
    else:
        return 'frequent'

# Add interaction category column
df['interaction_category'] = df['Interaction_Count'].apply(categorize_interaction)

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df['interaction_category']),
    y=df['interaction_category']
)

# Create a dictionary of class weights
weight_dict = dict(zip(np.unique(df['interaction_category']), class_weights))

print("Class weights:", weight_dict)

# Now let's see the distribution before and after applying weights
print("\nBefore weighting:")
print(df['interaction_category'].value_counts())

print("\nAfter weighting (normalized):")
weighted_counts = df['interaction_category'].value_counts() * pd.Series(weight_dict)
print(weighted_counts / weighted_counts.sum())

# Visualize the effect of weighting
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
df['interaction_category'].value_counts().plot(kind='bar', alpha=0.5, label='Original')
(df['interaction_category'].value_counts() * pd.Series(weight_dict)).plot(kind='bar', alpha=0.5, label='Weighted')
plt.title('Distribution Before and After Class Weighting')
plt.xlabel('Interaction Category')
plt.ylabel('Count')
plt.legend()
plt.show()



