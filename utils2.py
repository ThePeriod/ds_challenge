import polars as pl
import torch
import torch.nn as nn
import logging
import os
import numpy as np
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def configure_logging():
    # Configuración de logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger()

def create_checkpoint_dir(checkpoint_dir="checkpoints"):
    # Directorio para guardar los checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

def save_checkpoint(epoch, model, optimizer, loss, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")

def load_data(filepath):
    logger.info("Loading data")
    df = pl.read_csv(filepath)
    return df

def handle_missing_values(df):
    logger.info("Handling missing values")
    df = df.with_columns([
        pl.when(pl.col(col).is_null() | (pl.col(col) == "?")).then(None).otherwise(pl.col(col)).alias(col)
        for col in df.columns if df[col].dtype == pl.Utf8
    ])
    return df

def convert_categorical_to_numeric(df, categorical_cols):
    logger.info("Converting categorical columns to numerical")
    mappings = {}
    for col in categorical_cols:
        if col == 'readmitted':
            continue  # Skip converting 'readmitted' here
        unique_vals = df[col].unique()
        mappings[col] = {val: idx for idx, val in enumerate(unique_vals)}

    for col, mapping in mappings.items():
        df = df.with_columns(
            pl.col(col).replace(mapping).cast(pl.Int64).alias(col)
        )
    return df


def preprocess_data(filepath, categorical_cols):
    df = load_data(filepath)
    df = handle_missing_values(df)
    df = convert_categorical_to_numeric(df, categorical_cols)
    # Ajuste de la etiqueta para clasificación multiclase
    df = df.with_columns(
        pl.when(pl.col('readmitted') == 'NO').then(0)
        .when(pl.col('readmitted') == '<30').then(1)
        .otherwise(2).alias('readmitted')
    )
    return df

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Sin softmax aquí
        return x


class CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        
        # Calcular el tamaño de la salida de las capas convolucionales
        conv1_output_dim = (input_dim - 3 + 2 * 1) // 1 + 1
        pool1_output_dim = (conv1_output_dim + 1) // 2
        conv2_output_dim = (pool1_output_dim - 3 + 2 * 1) // 1 + 1
        pool2_output_dim = (conv2_output_dim + 1) // 2
        
        self.fc1 = nn.Linear(256, 128)  # Ajuste aquí
        self.fc2 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        #x = self.softmax(x)
        return x

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, device):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(self.device)
        
        out, _ = self.rnn(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        #out = self.softmax(out)
        return out

#class SVM(torch.nn.Module):
#    def __init__(self, input_dim):
#        super(SVM, self).__init__()
#        self.linear = torch.nn.Linear(input_dim, 3)

#    def forward(self, x):
#        return self.linear(x)

class SVM(nn.Module):
    def __init__(self, input_dim):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim, 3)  # Para 3 clases

    def forward(self, x):
        return self.linear(x)


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

def train_model(model, train_loader, num_epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        logger.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    
    return model

def evaluate_model(model, X_test_tensor, y_test_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, y_pred_test = torch.max(outputs, 1)
    accuracy = accuracy_score(y_test_tensor, y_pred_test)
    report = classification_report(y_test_tensor, y_pred_test)
    return accuracy, report


# Funciones para el EDA

def summarize_data(df):
    return df.describe()

def get_missing_values(df):
    return df.select([(pl.col(c).is_null().sum().alias(c)) for c in df.columns])

def visualize_distribution(df, column):
    import matplotlib.pyplot as plt
    df.select(column).to_pandas().plot(kind='hist', bins=30, title=f'Distribution of {column}')
    plt.show()

def visualize_categorical_distribution(df, column):
    import matplotlib.pyplot as plt
    df.select(column).to_pandas().value_counts().plot(kind='bar', title=f'Distribution of {column}')
    plt.show()

def correlation_matrix(df):
    # Filtrar solo las columnas numéricas
    numeric_columns = [col for col in df.columns if df[col].dtype in [pl.Int64, pl.Float64]]
    numeric_df = df.select(numeric_columns)
    # Remover filas con valores nulos
    numeric_df = numeric_df.drop_nulls()
    # Convertir a matriz de numpy y calcular la correlación
    correlation_matrix = np.corrcoef(numeric_df.to_numpy().T)
    # Crear un DataFrame de Polars a partir de la matriz de correlación
    corr_df = pl.DataFrame(correlation_matrix)
    corr_df.columns = numeric_columns
    return corr_df
