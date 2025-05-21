from imports import *

class VISL(nn.Module):
    def __init__(self):
        super(VISL, self).__init__()

        # multilayer perceptron layer with linear layer follwed buy relu activation function and then bach normalization
        # 39 -> 64 -> 128 -> 64 -> 32 -> 1 (final prediction)
        self.mlp = nn.Sequential(
            nn.Linear(39, 64), 
            nn.ReLU(), 
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            # final prediction - temperature / rain etc..
            nn.Linear(32, 1)  
        )

    def forward(self, x):
        return self.mlp(x)

# weather dataset object
class WeatherDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# function that takes features and labels columns, split the dataset (0.8, 0.1, 0.1)) & return weather dataset object for training, validating and testing phases
def prepare_data(df, feature_cols, label_cols):
    X = df[feature_cols].values
    y = df[label_cols].values

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)

    return (
        WeatherDataset(X_train, y_train),
        WeatherDataset(X_val, y_val),
        WeatherDataset(X_test, y_test)
    )

def train_model(model, train_loader, val_loader, epochs, lr=5e-3, weight_decay=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.L1Loss() # MAE loss function (mean absolut error)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * features.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * features.size(0)

        val_loss /= len(val_loader.dataset)

        #if (epoch + 1) % 10 == 0:
            #print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return model

def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)

    predictions, actuals = [], []

    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            outputs = model(features).cpu().numpy()
            predictions.append(outputs)
            actuals.append(targets.numpy())

    predictions = np.vstack(predictions)
    actuals = np.vstack(actuals)

    # 3 metrices: MAE, R^2 & EVS
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    evs = explained_variance_score(actuals, predictions)
    
    #print(f"\nTest MAE: {mae:.4f}, R²: {r2:.4f}, EVS: {evs:.4f}")
    return (round(mae, 4), round(r2, 4), round(evs, 4))