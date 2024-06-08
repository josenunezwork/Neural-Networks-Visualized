import sys
import os
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QCheckBox
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import math

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Check for Metal device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 6)
        self.fc2 = nn.Linear(6, 3)
        self.fc3 = nn.Linear(3, 10)
        self.fc4 = nn.Linear(10, 8)
        self.fc5 = nn.Linear(8, 3)
        self.fc6 = nn.Linear(3, 8)
        self.fc7 = nn.Linear(8, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)  # Create an instance of LeakyReLU

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))  # Use the LeakyReLU instance
        x = self.leaky_relu(self.fc2(x))  # Use the LeakyReLU instance
        x = self.leaky_relu(self.fc3(x))  # Use the LeakyReLU instance
        x = self.leaky_relu(self.fc4(x))  # Use the LeakyReLU instance
        x = self.leaky_relu(self.fc5(x))  # Use the LeakyReLU instance
        x = self.leaky_relu(self.fc6(x))  # Use the LeakyReLU instance
        x = self.fc7(x)
        return x

class TrainingThread(QThread):
    update_plot = pyqtSignal(list, list)
    update_network = pyqtSignal(object, int, int, dict)

    def __init__(self, train_loader):
        super().__init__()
        self.train_loader = train_loader
        self.model = SimpleNN().to(device)
        self.criterion = nn.MSELoss().to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.09, weight_decay=0.001)  # Use SGD with momentum and weight decay
        self.epochs = 500
        self.losses = []

    def run(self):
        try:
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                for step, (inputs, targets) in enumerate(self.train_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()

                    if step % 5 == 0:  # Update network and plot every 10 steps
                        # Capture gradients
                        gradients = {}
                        for name, param in self.model.named_parameters():
                            if 'weight' in name:
                                gradients[name] = param.grad.cpu().detach().numpy()
                        self.update_network.emit(self.model, epoch, step, gradients)

                # Update losses and plot at the end of each epoch
                self.losses.append(epoch_loss / len(self.train_loader))
                self.update_plot.emit(list(range(len(self.losses))), self.losses)
        except Exception as e:
            print(f"Error during training: {e}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Real-Time NN Training Visualization with PyTorch")
        self.setGeometry(100, 100, 1200, 800)

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        layout = QVBoxLayout(self.main_widget)
        self.canvas = FigureCanvas(plt.figure(figsize=(10, 6)))
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        title = QLabel("Neural Network Training Progress", self)
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("TitleLabel")

        self.display_edge_labels = False
        self.edge_labels_checkbox = QCheckBox("Display Edge Labels", self)
        self.edge_labels_checkbox.stateChanged.connect(self.toggle_edge_labels)

        layout.addWidget(title)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.edge_labels_checkbox)

        self.network_canvas = FigureCanvas(plt.figure(figsize=(15, 8)))
        layout.addWidget(self.network_canvas)

        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.set_title('Training Loss')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.grid(True)

        self.apply_styles()
        self.show()

        train_loader = self.generate_data()
        self.thread = TrainingThread(train_loader)
        self.thread.update_plot.connect(self.update_plot)
        self.thread.update_network.connect(self.update_network)

        # Generate initial network layout
        self.G, self.pos = self.create_network_layout()

        self.thread.start()

    def generate_data(self):
        x = np.random.rand(500, 4).astype(np.float32)
        y = (np.sum(x, axis=1) + np.random.rand(500)).astype(np.float32).reshape(-1, 1)
        dataset = TensorDataset(torch.tensor(x), torch.tensor(y))
        train_loader = DataLoader(dataset, batch_size=20, shuffle=True)  # Adjust batch size here
        return train_loader

    def apply_styles(self):
        style = """
            QMainWindow {
                background-color: #f0f0f0;
            }
            QLabel#TitleLabel {
                font-size: 24pt;
                font-weight: bold;
                color: #333;
                padding: 20px;
                border: 2px solid #555;
                border-radius: 10px;
                background-color: #e0e0e0;
            }
            QToolBar {
                background: #d0d0d0;
            }
        """
        QApplication.instance().setStyleSheet(style)

    def update_plot(self, epochs, losses):
        self.ax.clear()

        # Show only the last 20 epochs
        if len(epochs) > 80:
            epochs = epochs[-40:]
            losses = losses[-40:]
        elif len(epochs) > 50:
            epochs = epochs[-30:]
            losses = losses[-30:]
        elif len(epochs) > 30:
            epochs = epochs[-20:]
            losses = losses[-20:]
        elif len(epochs) > 10:
            epochs = epochs[-10:]
            losses = losses[-10:]

        self.ax.plot(epochs, losses, label='Loss', color='blue', linewidth=2)
        self.ax.legend()
        self.ax.set_title('Training Loss (Last 20 Epochs)')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.grid(True)

        # Dynamically adjust y-axis limits based on the current loss values
        min_loss = min(losses)
        max_loss = max(losses)
        if min_loss == max_loss:
            # If min_loss and max_loss are equal, set a small range around the value
            self.ax.set_ylim(min_loss - 0.1, max_loss + 0.1)
        else:
            range_loss = max_loss - min_loss
            margin = range_loss * 0.1  # Add a 10% margin on both sides
            self.ax.set_ylim(min_loss - margin, max_loss + margin)

        self.canvas.draw()

    def create_network_layout(self):
        model = SimpleNN()
        G = nx.DiGraph()

        for i, layer in enumerate(model.children()):
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                out_features = layer.out_features
                for j in range(in_features):
                    G.add_node(f"layer_{i}_node_{j}", layer=i, weight=0)
                for j in range(out_features):
                    G.add_node(f"layer_{i+1}_node_{j}", layer=i+1, weight=0)

        for i, (name, layer) in enumerate(model.named_children()):
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                out_features = layer.out_features
                for j in range(in_features):
                    for k in range(out_features):
                        weight = layer.weight[k, j].item()
                        G.add_edge(f"layer_{i}_node_{j}", f"layer_{i+1}_node_{k}", weight=weight, color='red')

        pos = self.fixed_position_layout(G)
        return G, pos

    def fixed_position_layout(self, G):
        pos = {}
        layer_nodes = {}
        node_spacing = 1  # Decrease spacing between nodes within each layer
        layer_spacing = 8.0  # Keep spacing between layers to avoid overlap

        # Group nodes by layer
        for node, data in G.nodes(data=True):
            layer = data['layer']
            if layer not in layer_nodes:
                layer_nodes[layer] = []
            layer_nodes[layer].append(node)

        # Calculate positions in a grid layout for each layer
        for layer, nodes in layer_nodes.items():
            num_nodes = len(nodes)
            grid_size = math.ceil(math.sqrt(num_nodes))  # Calculate the grid size (rows and columns)
            half_grid = grid_size / 2

            for i, node in enumerate(nodes):
                row = i // grid_size
                col = i % grid_size

                # Position nodes in a grid layout
                x = col * node_spacing
                y = row * node_spacing

                # Apply rotation transformation
                angle = math.radians(40)  # Adjust rotation angle
                x_rotated = x * math.cos(angle) - y * math.sin(angle)
                y_rotated = x * math.sin(angle) + y * math.cos(angle)

                # Offset by layer spacing
                pos[node] = (layer * layer_spacing, y_rotated - half_grid * node_spacing)

        return pos

    def update_network(self, model, current_epoch, current_step, gradients):
        self.network_canvas.figure.clf()
        ax = self.network_canvas.figure.add_subplot(111)

        edges = list(self.G.edges(data=True))
        node_weights = {node: 0 for node in self.G.nodes()}

        for i, (name, layer) in enumerate(model.named_children()):
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                out_features = layer.out_features
                max_grad = 1e-5  # Set a default value for max_grad
                if gradients:  # Check if gradients is not empty
                    max_grad = max(gradients.get(f"{name}.weight", torch.tensor(1e-5)).max(), 1e-5)  # Set minimum gradient threshold
                for j in range(in_features):
                    for k in range(out_features):
                        weight = layer.weight[k, j].item()
                        grad = 1e-5  # Set a default value for grad
                        if gradients:  # Check if gradients is not empty
                            grad = max(gradients.get(f"{name}.weight", torch.zeros_like(layer.weight))[k, j], 1e-5)  # Set minimum gradient threshold
                        color = plt.cm.viridis(grad / max_grad)
                        edge = (f"layer_{i}_node_{j}", f"layer_{i+1}_node_{k}")
                        if edge in self.G.edges:
                            self.G.edges[edge]['weight'] = weight  # Update the edge weight
                            self.G.edges[edge]['color'] = color
                        node_weights[f"layer_{i}_node_{j}"] = max(node_weights[f"layer_{i}_node_{j}"], abs(weight))
                        node_weights[f"layer_{i+1}_node_{k}"] = max(node_weights[f"layer_{i+1}_node_{k}"], abs(weight))

        node_colors = [plt.cm.coolwarm(weight / max(node_weights.values())) if weight != 0 else plt.cm.coolwarm(0) for node, weight in node_weights.items()]
        edge_colors = [self.G[u][v]['color'] for u, v in self.G.edges]
        edge_weights = [self.G[u][v]['weight'] for u, v in self.G.edges]

        # Update the positions in case they changed
        self.pos = self.fixed_position_layout(self.G)

        # Draw the network with smaller nodes and increased spacing
        nx.draw(self.G, self.pos, with_labels=False, node_size=50, node_color=node_colors, edge_color=edge_colors, ax=ax, edge_cmap=plt.cm.viridis)
        if self.display_edge_labels:
            edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
            nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels, ax=ax)

        ax.set_title('Neural Network Visualization')
        self.network_canvas.draw()

    def toggle_edge_labels(self, state):
        self.display_edge_labels = state == Qt.Checked
        self.update_network(self.thread.model, self.thread.epochs, self.thread.epochs, {})

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec_())