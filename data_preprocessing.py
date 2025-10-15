import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def preprocess_dataset(input_csv: str = 'education_jobs_dataset_500_imputed.csv',
                       output_csv: str = 'preprocessed_education_jobs_dataset.csv',
                       keep_top_n: int | None = 10,
                       min_count: int = 10,
                       target_encode_cols: list | None = None,
                       te_n_splits: int = 5):
    """Simple preprocessing pipeline that finishes after feature scaling.

    Steps:
    - Load CSV
    - Impute missing values (mode for categorical, mean for numeric)
    - Separate target 'Job Role' from features
    - Reduce categorical cardinality (keep_top_n or group rares by min_count)
    - One-hot encode remaining categoricals
    - Scale numeric features (StandardScaler then MinMaxScaler) if any
    - Save processed DataFrame (features + target) to CSV

    Returns: X, y, processed_df
    """
    df = pd.read_csv(input_csv)
    print('Loaded', input_csv)

    # Impute missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    # Separate target
    if 'Job Role' not in df.columns:
        raise ValueError("Expected 'Job Role' column in input CSV")
    y = df['Job Role']
    X = df.drop('Job Role', axis=1)

    # Categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Reduce cardinality
    for col in categorical_cols:
        if keep_top_n is not None:
            top_levels = X[col].value_counts().nlargest(keep_top_n).index
            X[col] = X[col].where(X[col].isin(top_levels), other='Other')
        else:
            counts = X[col].value_counts()
            rare_levels = counts[counts < min_count].index
            if len(rare_levels) > 0:
                X[col] = X[col].replace(rare_levels, 'Other')

    # Target encoding (Option A): per-class KFold-safe probability encoding for selected columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if target_encode_cols is not None:
        te_cols = [c for c in target_encode_cols if c in categorical_cols]
        if len(te_cols) > 0:
            from sklearn.model_selection import KFold
            import numpy as _np
            classes = list(y.unique())
            # global prior probabilities per class (fallback)
            global_prior = y.value_counts(normalize=True).to_dict()
            kf = KFold(n_splits=te_n_splits, shuffle=True, random_state=42)
            for col in te_cols:
                print(f'Applying per-class KFold target encoding to {col} with {te_n_splits} folds')
                encoded = _np.zeros((len(X), len(classes)), dtype=float)
                for train_idx, val_idx in kf.split(X):
                    train_idx = list(train_idx)
                    val_idx = list(val_idx)
                    X_train_fold = X.iloc[train_idx]
                    y_train_fold = y.iloc[train_idx]
                    ct = pd.crosstab(X_train_fold[col], y_train_fold)
                    probs = ct.div(ct.sum(axis=1), axis=0)
                    for cls_i, cls in enumerate(classes):
                        if cls in probs.columns:
                            mapped = X[col].iloc[val_idx].map(probs[cls])
                            mapped = mapped.fillna(global_prior.get(cls, 0.0))
                        else:
                            mapped = pd.Series(global_prior.get(cls, 0.0), index=val_idx)
                        encoded[val_idx, cls_i] = mapped.values
                # attach per-class encoded columns to X
                for cls_i, cls in enumerate(classes):
                    safe_cls = str(cls).replace(' ', '_').replace('/', '_')
                    new_col = f"{col}_te_{safe_cls}"
                    X[new_col] = encoded[:, cls_i]
                # remove original column from further one-hot encoding
                categorical_cols.remove(col)

    # One-hot encode remaining categorical columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        # convert boolean dummies to ints
        X = X.apply(lambda s: s.astype('int8') if s.dtype == 'bool' else s)

    # Scale numeric features if present
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(numerical_cols) > 0:
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        minmax = MinMaxScaler()
        X[numerical_cols] = minmax.fit_transform(X[numerical_cols])
    else:
        print('No numeric columns to scale after encoding.')

    processed_df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    processed_df.to_csv(output_csv, index=False)
    print('Saved processed file to', output_csv)

    return X, y, processed_df


if __name__ == '__main__':
    # Example: apply per-class target encoding to 'Skill' and 'Certification'
    preprocess_dataset(target_encode_cols=['Skill', 'Certification'], te_n_splits=5)


def preprocess_dataset(input_csv: str = 'education_jobs_dataset_500_imputed.csv',
                       output_csv: str = 'preprocessed_education_jobs_dataset.csv',
                       min_count: int = 10,
                       keep_top_n: int | None = 10,
                       remove_high_vif: bool = False,
                       vif_threshold: float = 10.0,
                       target_encode_cols: list | None = None,
                       te_n_splits: int = 5):
    """Load raw CSV, handle missing values, group rare categories, one-hot encode features,
    scale numeric features if present, save processed DataFrame to CSV.

    Returns:
        X (pd.DataFrame): preprocessed feature matrix
        y (pd.Series): target column 'Job Role'
        processed_df (pd.DataFrame): concatenated features + target saved to output_csv
    """
    df = pd.read_csv(input_csv)
    print('Loaded', input_csv)
    print(df.head())

    # Step 2: Handling Missing Values
    for col in df.columns:
        if df[col].dtype == "object":  # categorical → use mode
            df[col] = df[col].fillna(df[col].mode()[0])
        else:  # numeric → use mean
            df[col] = df[col].fillna(df[col].mean())
    print('Missing values after imputation:\n', df.isnull().sum())

    # Separate target from features before encoding/scaling
    y = df['Job Role']
    X = df.drop('Job Role', axis=1)

    # Encode categorical feature columns only (leave the target untouched)
    categorical_cols = X.select_dtypes(include=['object']).columns

    # Reduce cardinality: either keep top-N categories per column or group rares by min_count
    for col in categorical_cols:
        if keep_top_n is not None:
            top_levels = X[col].value_counts().nlargest(keep_top_n).index
            X[col] = X[col].where(X[col].isin(top_levels), other='Other')
        else:
            counts = X[col].value_counts()
            rare_levels = counts[counts < min_count].index
            if len(rare_levels) > 0:
                X[col] = X[col].replace(rare_levels, 'Other')

    # Recompute categorical columns after grouping
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Target encoding for selected columns (KFold-safe out-of-fold mean encoding)
    if target_encode_cols is not None:
        # only consider columns that exist and are categorical
        te_cols = [c for c in target_encode_cols if c in categorical_cols]
        if len(te_cols) > 0:
            from sklearn.model_selection import KFold
            global_mean = y.mean()
            kf = KFold(n_splits=te_n_splits, shuffle=True, random_state=42)
            for col in te_cols:
                print(f'Applying KFold target encoding to {col} with {te_n_splits} folds')
                encoded = pd.Series(index=X.index, dtype=float)
                for train_idx, val_idx in kf.split(X):
                    train_idx = list(train_idx)
                    val_idx = list(val_idx)
                    mapping = y.iloc[train_idx].groupby(X[col].iloc[train_idx]).mean()
                    # map validation fold
                    encoded.iloc[val_idx] = X[col].iloc[val_idx].map(mapping).fillna(global_mean)
                # replace column with encoded values
                X[col] = encoded
                # remove from categorical columns list so it's not one-hot encoded
                categorical_cols.remove(col)

    # One-hot encode remaining categorical columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        # Ensure dummy columns are 0/1 integers (not booleans)
        X = X.apply(lambda s: s.astype('int8') if s.dtype == 'bool' else s)
    print('Features after encoding (sample):')
    print(X.head())

    # Normalize / scale numerical features if any exist
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(numerical_cols) > 0:
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        print('After StandardScaler (sample):')
        print(X.head())
        min_max_scaler = MinMaxScaler()
        X[numerical_cols] = min_max_scaler.fit_transform(X[numerical_cols])
        print('After MinMaxScaler (sample):')
        print(X.head())
    else:
        print('No numerical columns to scale.')

    # Save processed data (features + target)
    # Optional: remove features with high multicollinearity using VIF
    if remove_high_vif:
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            import numpy as _np

            print(f'Removing features with VIF > {vif_threshold} (iterative)...')
            # Work on a copy
            X_vif = X.copy()
            # Ensure numeric dtype for VIF calculation
            X_vif = X_vif.astype(float)
            while True:
                vif_data = []
                cols = X_vif.columns
                for i in range(len(cols)):
                    try:
                        vif = variance_inflation_factor(X_vif.values, i)
                    except Exception:
                        vif = float('inf')
                    vif_data.append((cols[i], vif))
                vif_df = pd.DataFrame(vif_data, columns=['feature', 'vif']).sort_values('vif', ascending=False)
                max_vif = vif_df['vif'].iloc[0]
                max_feat = vif_df['feature'].iloc[0]
                print('Max VIF:', max_vif, 'feature:', max_feat)
                if max_vif <= vif_threshold or len(X_vif.columns) <= 1:
                    break
                # Drop the feature with highest VIF
                X_vif = X_vif.drop(columns=[max_feat])

            dropped = set(X.columns) - set(X_vif.columns)
            if dropped:
                print('Dropped features due to high VIF:', dropped)
                X = X_vif
            else:
                print('No features dropped by VIF')
        except Exception as e:
            print('VIF reduction failed:', e)

    processed_df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    processed_df.to_csv(output_csv, index=False)
    print('Saved processed dataset to', output_csv)

    return X, y, processed_df


# Run preprocessing and get X, y
X, y, processed_df = preprocess_dataset()

#Step 7: Feature Selection / Splitting
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# X and y are already defined (X = preprocessed features, y = original target)
#Step 7: Train-Test Split (stratify when possible)
if y.value_counts().min() >= 2:
    stratify_arg = y
else:
    stratify_arg = None
    print('Warning: some classes have fewer than 2 samples — proceeding without stratify.')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_arg
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#Step 8: Model selection based on target type
import numpy as np
import pandas as _pd
if _pd.api.types.is_numeric_dtype(y):
    # If target is numeric, proceed with OLS to check linear regression suitability
    import statsmodels.api as sm
    X_train_sm = sm.add_constant(X_train)  # adding a constant
    model = sm.OLS(y_train, X_train_sm).fit()
    print(model.summary())
else:
    # Target is categorical -> run classifiers (LogisticRegression baseline + RandomForest)
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    
    # Logistic Regression baseline
    lr = LogisticRegression(max_iter=2000, solver='lbfgs')
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print('LogisticRegression accuracy:', accuracy_score(y_test, y_pred_lr))
    print('\nLogistic Regression classification report:\n', classification_report(y_test, y_pred_lr))

    # RandomForest with class weights for imbalance
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print('RandomForest accuracy:', accuracy_score(y_test, y_pred_rf))
    print('\nRandomForest classification report:\n', classification_report(y_test, y_pred_rf))
    print('\nRandomForest confusion matrix:\n', confusion_matrix(y_test, y_pred_rf))

    # Cross-validated accuracy for RandomForest (use StratifiedKFold if possible)
    if stratify_arg is not None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    else:
        cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
    print('\nRandomForest CV accuracies (5-fold):', cv_scores)
    print('RandomForest CV mean accuracy: {:.3f} +/- {:.3f}'.format(cv_scores.mean(), cv_scores.std()))


