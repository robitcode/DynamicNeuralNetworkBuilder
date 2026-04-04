import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def process_csv_to_tensors(df, target_column):
    """
    Ingests a CSV, handles missing values, encodes text, scales features,
    and returns PyTorch tensors ready for training.
    """
    try:
        # 2. Isolate Features (X) and Target (y)
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")
            
        y = df.pop(target_column)
        X = df
        
        # 3. Handle Missing Data
        # For a rapid prototype, dropping missing rows is safest. 
        # In a V2, you could use SimpleImputer here.
        X = X.dropna()
        y = y[X.index] # Ensure y drops the same rows as X
        
        # 4. Encode Categorical Text in X
        # pd.get_dummies automatically finds text columns and turns them into 0s and 1s.
        X = pd.get_dummies(X, drop_first=True)
        
        # 5. SMART TARGET ENCODING (Regression vs Classification)
        # If it's a float, or if it has a ton of unique numbers, it's Regression
        if y.dtype in ['float64', 'float32'] or (y.dtype in ['int64', 'int32'] and y.nunique() > 20):
            task_type = 'regression'
            y_processed = y.values
            num_classes = 1 # Output layer just needs 1 node to spit out a number
            y_tensor = torch.tensor(y_processed, dtype=torch.float32).view(-1, 1)
        else:
            task_type = 'classification'
            le = LabelEncoder()
            y_processed = le.fit_transform(y)
            num_classes = len(le.classes_)
            y_tensor = torch.tensor(y_processed, dtype=torch.long)
            
        # 6. Standard Scale the Features (X)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        # 7. The 80/20 Train/Test Split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
        
        input_dim = X.shape[1]
        
        # CHANGED: We now return task_type as well!
        return X_train, X_test, y_train, y_test, input_dim, num_classes, task_type
        
    except Exception as e:
        return None, None, None, None, None, None, str(e)