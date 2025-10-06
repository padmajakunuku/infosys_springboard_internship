import pandas as pd
import numpy as np

# Load the dataset
file_path = "education_jobs_dataset_500.csv"   # put your file path here
df = pd.read_csv(file_path)

# Peek at first rows
print("First 5 rows of the dataset:")
print(df.head())

# Check shape
print("\nDataset shape:", df.shape)
# Replace 'None', 'nan', and blanks with real NaN
df = df.replace(r'^\s*$', np.nan, regex=True)
df = df.replace({'None': np.nan, 'nan': np.nan})

# Strip whitespace
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].str.strip()

# Check missing counts
print("\nMissing values per column:")
print(df.isnull().sum())
def standardize_education(e):
    if pd.isna(e): return np.nan
    e = str(e).strip().replace("  ", " ")
    e = e.title().replace("Btech", "BTech").replace("Bsc", "BSc")
    return e

df['Education_std'] = df['Education'].apply(standardize_education)

# Split Degree and Branch
df[['Degree','Branch']] = df['Education_std'].apply(
    lambda e: pd.Series([np.nan, np.nan]) if pd.isna(e) 
    else pd.Series([e.split()[0], " ".join(e.split()[1:]) if len(e.split())>1 else np.nan])
)

print("\nStandardized Education sample:")
print(df[['Education','Education_std','Degree','Branch']].head())
mode_skill = df.groupby('Education_std')['Skill'] \
               .agg(lambda x: x.dropna().mode().iloc[0] if len(x.dropna().mode())>0 else np.nan)

mode_job = df.groupby('Education_std')['Job Role'] \
             .agg(lambda x: x.dropna().mode().iloc[0] if len(x.dropna().mode())>0 else np.nan)

modes_df = pd.DataFrame({'mode_skill': mode_skill, 'mode_job': mode_job})
print("\nMost common Skill/Job Role per Education:")
print(modes_df)
print("\nMissing Skill count by Education:")
print(df.groupby('Education_std')['Skill'].apply(lambda s: s.isna().sum()))

print("\nMissing Job Role count by Education:")
print(df.groupby('Education_std')['Job Role'].apply(lambda s: s.isna().sum()))
print("\nExample rows with missing Skill:")
print(df[df['Skill'].isnull()].head())

print("\nExample rows with missing Job Role:")
print(df[df['Job Role'].isnull()].head())
df_imp = df.copy()
rng = np.random.default_rng(42)   # reproducible

for idx, row in df_imp.iterrows():
    edu = row['Education_std']
    # Fill Skill with mode 70% chance
    if pd.isna(row['Skill']) and edu in mode_skill.index and pd.notna(mode_skill.loc[edu]):
        if rng.random() < 0.7:
            df_imp.at[idx, 'Skill'] = mode_skill.loc[edu]
    # Fill Job Role with mode 80% chance
    if pd.isna(row['Job Role']) and edu in mode_job.index and pd.notna(mode_job.loc[edu]):
        if rng.random() < 0.8:
            df_imp.at[idx, 'Job Role'] = mode_job.loc[edu]

print("\nMissing counts AFTER imputation:")
print(df_imp.isnull().sum())
comparison = df[['Name','Education_std','Skill','Job Role']] \
    .merge(df_imp[['Skill','Job Role']], left_index=True, right_index=True, 
           suffixes=('_before','_after'))

print("\nRows where something changed (first 10):")
print(comparison[comparison['Skill_before']!=comparison['Skill_after']].head(10))
print(comparison[comparison['Job Role_before']!=comparison['Job Role_after']].head(10))
df_imp.to_csv("education_jobs_dataset_500_imputed.csv", index=False)
print("Cleaned dataset saved as education_jobs_dataset_500_imputed.csv")
