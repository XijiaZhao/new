import torch
import torch.nn as nn
from transformer import TimeSeriesTransformer, SSLTransformer, ContrastiveLosses, ClassificationModel # Ensure this is correctly imported
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
torch.manual_seed(0)

class Transformer_Train:
    def __init__(self, num_features, num_classes, d_model=256, nhead=4, num_encoder_layers=3, dim_feedforward=512, dropout=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.create_model(num_features, num_classes, d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.model.to(self.device)

    def create_model(self, num_features, num_classes, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        """Create the Transformer model."""
        model = TimeSeriesTransformer(
            num_features=num_features, 
            num_classes=num_classes, 
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout
        )
        return model

    def train(self, train_loader, test_loader, criterion, optimizer, epochs):
        """Train the model."""
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            num_batch = 0

            for batch_data, batch_labels in train_loader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)

                optimizer.zero_grad()
                output = self.model(batch_data)
                loss = criterion(output, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batch += 1

            if num_batch > 0:
                avg_loss = total_loss / num_batch
            else:
                avg_loss = 0
                print(f"No batches processed in Epoch {epoch+1}. Check the data loader and dataset.")

            print(f'Epoch {epoch+1}, Loss: {avg_loss}')

            self.evaluate(test_loader)

    def evaluate(self, test_loader):
        """Evaluate the model."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)

                output = self.model(batch_data)
                _, predicted = torch.max(output.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy}%')


class SSL_Transformer_Train:
    def __init__(self, num_features, d_model=256, nhead=4, num_encoder_layers=3, dim_feedforward=512, proj_dimension=256, dropout=0.1, margin=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.create_model(num_features, d_model, nhead, num_encoder_layers, dim_feedforward, proj_dimension, dropout)
        self.model.to(self.device)
        self.loss_function = ContrastiveLosses(margin).standard_contrastive_loss

    def create_model(self, num_features, d_model, nhead, num_encoder_layers, dim_feedforward, proj_dimension, dropout):
        """Create the SSL Transformer model."""
        model = SSLTransformer(
            num_features=num_features, 
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            dim_feedforward=dim_feedforward, 
            proj_dimension = proj_dimension,
            dropout=dropout
        )
        return model

    def train(self, train_loader, test_loader=None, lr=0.001, num_epochs=10):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0

            for (sample1, sample2), labels in train_loader:
                # the sample1 and sample2 are the two tsdata, get by the pairs of index, label is whether they are nega or posi
                sample1, sample2, labels = sample1.to(self.device), sample2.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                output1, output2 = self.model(sample1), self.model(sample2)
                # print("Output1 size:", output1.size())
                # print("Output2 size:", output2.size())
                # print("Label size:", labels.size())
                loss = self.loss_function(output1, output2, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

            if test_loader is not None:
                self.validate(test_loader)

    def validate(self, test_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for (sample1, sample2), labels in test_loader:
                sample1, sample2, labels = sample1.to(self.device), sample2.to(self.device), labels.to(self.device)

                output1, output2 = self.model(sample1), self.model(sample2)
                loss = self.loss_function(output1, output2, labels)
                total_loss += loss.item()

        avg_loss = total_loss / len(test_loader)
        print(f"Validation Loss: {avg_loss:.4f}")
    
    def get_model(self):
        return self.model


class Classification_Train:
    def __init__(self, ssl_model, num_classes, feature_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.create_model(ssl_model, num_classes, feature_size=256)
        self.model.to(self.device)

    def create_model(self, ssl_model, num_classes, feature_size):
        """Create the Transformer model."""
        model = ClassificationModel(
            ssl_model=ssl_model, 
            num_classes=num_classes, 
            feature_size=feature_size
        )
        return model

    def train(self, train_loader, test_loader, criterion, optimizer, epochs):
        """Train the model."""
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            num_batch = 0

            for batch_data, batch_labels in train_loader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)

                optimizer.zero_grad()
                output = self.model(batch_data)
                loss = criterion(output, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batch += 1

            if num_batch > 0:
                avg_loss = total_loss / num_batch
            else:
                avg_loss = 0
                print(f"No batches processed in Epoch {epoch+1}. Check the data loader and dataset.")

            print(f'Epoch {epoch+1}, Loss: {avg_loss}')

            self.evaluate(test_loader)

    def evaluate(self, test_loader):
        """Evaluate the model."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)

                output = self.model(batch_data)
                _, predicted = torch.max(output.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy}%')

    def get_model(self):
        return self.model
    
    def print_model_trainable_parameters(self):
        for name, param in self.model.named_parameters():
            print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")
        print("\nModel Structure:\n")
        print(self.model)
    
    def get_predictions_and_true_labels(self, test_loader):
        """Collect predictions and true labels for the test dataset."""
        self.model.eval()
        all_predictions = []
        all_true_labels = []
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)
                output = self.model(batch_data)
                _, predicted = torch.max(output.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_true_labels.extend(batch_labels.cpu().numpy())
        return np.array(all_predictions), np.array(all_true_labels)

    def plot_confusion_matrix(self, test_loader, num_classes):
        """Plot the confusion matrix given the true and predicted labels."""
        # Get predictions and true labels
        predictions, true_labels = self.get_predictions_and_true_labels(test_loader)
        
        # Compute the confusion matrix
        cm = confusion_matrix(true_labels, predictions, labels=np.arange(num_classes))
        
        # Plot the confusion matrix
        plt.figure(figsize=(4, 4))
        font = {'family': 'Times New Roman',
        'color':  'black',
        'weight': 'normal',
        'size': 14
        }
        sns.heatmap(cm, annot=True, annot_kws= font, fmt="d", cmap='Blues', xticklabels=np.arange(num_classes), yticklabels=np.arange(num_classes),cbar=False)
        # plt.title('Expulsion Classification Confusion Matrix', fontdict=font, fontsize=16)
        plt.xlabel('Predicted Labels', fontdict=font, fontsize=16)
        plt.ylabel('True Labels', fontdict=font, fontsize=16)
        plt.show()