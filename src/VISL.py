from imports import *

class VISL(nn.Module):
    def __init__(self):
        super(VISL, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(34, 64), 
            nn.LeakyReLU(0.1), 
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(32),

            # final prediction - temperature
            nn.Linear(32, 1)  
        )

    def forward(self, x):
        return self.mlp(x)


class WeatherDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

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

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

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

    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    evs = explained_variance_score(actuals, predictions)
    
    print(f"\nTest MAE: {mae:.4f}, R²: {r2:.4f}, EVS: {evs:.4f}")
    return predictions, actuals