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
            X_train, X_test, y_train, y_test, input_dim, num_classes, task_type = process_csv_to_tensors(df.copy(), target_col)
        
            if X_train is not None:
                st.sidebar.success(f"Task detected as: {task_type}")
                # Save ALL the splits to session state
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['input_dim'] = input_dim
                st.session_state['num_classes'] = num_classes
                st.session_state['task_type'] = task_type
                
            else:
                st.sidebar.error(f"Error processing data: {num_classes_or_error}")

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
        X_tensor = st.session_state['X_train']
        y_tensor = st.session_state['y_train']
        input_dim = st.session_state['input_dim']
        num_classes = st.session_state['num_classes']

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
        layers.append(nn.Linear(neurons, num_classes))
        
        model = nn.Sequential(*layers)
        
        st.write("### Model Architecture Built!")
        st.code(model, language='python') 
        
        # 2. THE REAL PYTORCH TRAINING LOOP
        task = st.session_state['task_type']
        if task == "regression":
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
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

        # Model Evaluation
        st.write("### 📊 Model Evaluation")
        model.eval() 
        
        with torch.no_grad():
            test_outputs = model(st.session_state['X_test'])
            task_type = st.session_state['task_type']
            if task_type == 'classification':
                _, predicted = torch.max(test_outputs.data, 1)
                total = st.session_state['y_test'].size(0)
                correct = (predicted == st.session_state['y_test']).sum().item()
                accuracy = 100 * correct / total
                st.metric(label="Test Accuracy", value=f"{accuracy:.2f}%")
                if accuracy > 80: st.balloons()
                
            elif task_type == 'regression':
                # Calculate Mean Absolute Error (How far off we are on average)
                mae = torch.mean(torch.abs(test_outputs - st.session_state['y_test'])).item()
                st.metric(label="Average Prediction Error (MAE)", value=f"{mae:.2f}")
                st.caption("Lower is better! This is how far off your predictions are from the actual sales numbers on average.")