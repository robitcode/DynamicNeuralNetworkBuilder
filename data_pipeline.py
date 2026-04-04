import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder

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
        
        # 5. Encode the Target (y)
        # If the user is trying to predict "Yes/No" or "Malignant/Benign", we make it 1/0
        if y.dtype == 'object' or y.name == 'category':
            le = LabelEncoder()
            y_processed = le.fit_transform(y)
        else:
            y_processed = y.values
            
        # 6. Standard Scale the Features
        # Neural networks need all inputs to be on a similar scale (mean=0, variance=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 7. Convert to PyTorch Tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        # y must be shaped as a column vector for Binary Cross Entropy Loss
        y_tensor = torch.tensor(y_processed, dtype=torch.float32).view(-1, 1)
        
        # We also return X.shape[1] so the neural network knows how many input nodes it needs
        input_dim = X.shape[1]
        
        return X_tensor, y_tensor, input_dim
        
    except Exception as e:
        return None, None, str(e)