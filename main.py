import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from data_pipeline import process_csv_to_tensors

st.title("🔥 Dynamic PyTorch Network Builder")

# ==========================================
# NEW: DATA UPLOAD & INGESTION SECTION
# ==========================================
st.sidebar.header("📁 1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your clean CSV", type=["csv"])

if uploaded_file is not None:
    # Preview the data to pick the target column
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)

    st.write("### Data Preview")
    st.dataframe(df.head(5))
    
    target_col = st.sidebar.selectbox("Select Target Column to Predict", df.columns)
    
    # Process the data when the user clicks the button
    if st.sidebar.button("Process Data"):
        with st.spinner('Converting data to PyTorch Tensors...'):
            X_tensor, y_tensor, input_dim_or_error = process_csv_to_tensors(df.copy(), target_col)
            uploaded_file.seek(0)
            if X_tensor is not None:
                st.sidebar.success(f"Success! Detected {input_dim_or_error} input features.")
                # Save the real data to session state
                st.session_state['X_tensor'] = X_tensor
                st.session_state['y_tensor'] = y_tensor
                st.session_state['input_dim'] = input_dim_or_error
            else:
                st.sidebar.error(f"Error processing data: {input_dim_or_error}")

st.sidebar.markdown("---")

# ==========================================
# MODEL ARCHITECTURE CONFIGURATION
# ==========================================
st.sidebar.header("⚙️ 2. Model Architecture")
num_layers = st.sidebar.slider("Number of Hidden Layers", 1, 5, 2)
neurons = st.sidebar.slider("Neurons per Layer", 8, 128, 32)
activation_choice = st.sidebar.selectbox("Activation Function", ["ReLU", "Sigmoid", "Tanh"])
epochs = st.sidebar.slider("Epochs", 10, 500, 100)
learning_rate = st.sidebar.number_input("Learning Rate", value=0.01, format="%.4f")

activation_dict = {
    "ReLU": nn.ReLU(),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh()
}

# ==========================================
# TRAINING LOOP
# ==========================================
if st.button("Build and Train Model"):
    # 🚨 SECURITY CHECK: Ensure data was uploaded first
    if 'X_tensor' not in st.session_state:
        st.error("⚠️ Please upload and process a CSV file first!")
    else:
        # Load the real data from session state
        X_tensor = st.session_state['X_tensor']
        y_tensor = st.session_state['y_tensor']
        input_dim = st.session_state['input_dim']
        
        # 1. DYNAMICALLY BUILD THE PYTORCH MODEL
        layers = []
        
        # Input Layer (Uses the real input_dim from your CSV)
        layers.append(nn.Linear(input_dim, neurons))
        layers.append(activation_dict[activation_choice])
        
        # Hidden Layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(activation_dict[activation_choice])
            
        # Output Layer
        layers.append(nn.Linear(neurons, 1))
        layers.append(nn.Sigmoid())
        
        model = nn.Sequential(*layers)
        
        st.write("### Model Architecture Built!")
        st.code(model, language='python') 
        
        # 2. THE REAL PYTORCH TRAINING LOOP
        criterion = nn.BCELoss() 
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        loss_chart = st.empty() 
        
        loss_history = []
        st.write("### Training Progress")
        
        # We uncommented the real math!
        for epoch in range(epochs):
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Record the actual loss
            loss_history.append(loss.item())
            
            # Update the UI every 10 epochs or on the last epoch
            if epoch % 10 == 0 or epoch == epochs - 1:
                progress_bar.progress(int((epoch + 1) / epochs * 100))
                status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
                loss_chart.line_chart(loss_history)
                
        st.success("Training Complete!")