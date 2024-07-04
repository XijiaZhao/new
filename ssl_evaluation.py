import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

class ModelEvaluator:
    def __init__(self, model, dataloader, label_mapping):
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.label_mapping = label_mapping
        # Creating a reverse mapping from integer labels to original string labels
        self.reverse_label_mapping = {v: k for k, v in label_mapping.items()}

    def get_embeddings_and_labels(self):
        self.model.eval()
        embeddings = []
        labels_list = []

        with torch.no_grad():
            for data, labels in self.dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                embeddings.append(self.model(data).cpu().numpy())
                labels_list.append(labels.cpu().numpy())

        return np.concatenate(embeddings, axis=0), np.concatenate(labels_list)

    def visualize_with_tsne(self, title, perplexity=30, n_iter=1000):
        embeddings, labels = self.get_embeddings_and_labels()

        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)
        tsne_results = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            idxs = labels == label
            # Use reverse mapping to get the original string label
            original_label = self.reverse_label_mapping[label]
            plt.scatter(tsne_results[idxs, 0], tsne_results[idxs, 1], s = 70, color=colors[i], label=original_label, alpha=0.5)

        
        font = {'family': 'Times New Roman',
        'color':  'black',
        'weight': 'normal',
        'size': 18
        }
        plt.title(title, family = "Times New Roman", fontsize=20)
        plt.legend(loc='center right', bbox_to_anchor=(1, 0.5), fontsize=12)
        plt.xlabel('t-SNE component 1', fontdict = font)
        plt.ylabel('t-SNE component 2', fontdict = font)
        font_properties = font_manager.FontProperties(family='Times New Roman', size=13)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop=font_properties)
        plt.subplots_adjust(right=0.8)
        plt.show()


# Example
# evaluator = ModelEvaluator(your_ssl_model, your_dataloader)
# evaluator.visualize_with_tsne()

