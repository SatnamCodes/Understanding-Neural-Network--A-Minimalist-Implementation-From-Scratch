# Understanding Neural Networks: A Minimalist Implementation From Scratch

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-## Acknowledgments

- The Wisconsin Breast Cancer Dataset from the UCI Machine Learning Repository
- Educational inspiration from various deep learning courses and resources
- The NumPy and scikit-learn communities for their excellent documentation

## Contact

Author: SatnamCodes  
Repository: [Understanding-Neural-Network--A-Minimalist-Implementation-From-Scratch](https://github.com/SatnamCodes/Understanding-Neural-Network--A-Minimalist-Implementation-From-Scratch)

---

This project demonstrates that understanding neural networks doesn't require complex frameworks - just solid mathematical foundations and clear implementation!)](https://scikit-learn.org/)

This is a comprehensive educational project that implements neural networks from scratch using only NumPy. The focus is on binary classification of breast cancer diagnosis, and the repository demonstrates fundamental neural network concepts through systematic experimentation with different activation functions, loss functions, and training configurations.

## Dataset

This project uses the Wisconsin Breast Cancer Dataset, which is a well-known machine learning dataset. The dataset contains:

- 569 samples of breast cancer tumor measurements
- 30 numerical features derived from digitized images of breast mass cell nuclei
- Binary classification target where samples are labeled as either:
  - M (Malignant): 212 samples (37.3%)
  - B (Benign): 357 samples (62.7%)

### Feature Categories

The 30 features are organized into three statistical measures for each of 10 basic measurements:

| Statistic | Description |
|-----------|-------------|
| `_mean` | Mean value |
| `_se` | Standard error |
| `_worst` | Worst (largest) value |

The ten basic measurements are:
1. radius - Distance from center to perimeter points
2. texture - Standard deviation of gray-scale values
3. perimeter - Nucleus perimeter
4. area - Nucleus area
5. smoothness - Local variation in radius lengths
6. compactness - (perimeter² / area) - 1.0
7. concavity - Severity of concave portions
8. concave points - Number of concave portions
9. symmetry - Nucleus symmetry
10. fractal dimension - "Coastline approximation" - 1

## Project Structure

```
Understanding-Neural-Network--A-Minimalist-Implementation-From-Scratch/
├── Data/
│   └── data.csv                    # Wisconsin Breast Cancer Dataset
├── Experiments/                    # Systematic experimentation
│   ├── Initial Structure/          # Template/incomplete implementation
│   ├── ReLu_bce/                  # ReLU + Binary Cross-Entropy (1k epochs)
│   ├── ReLu_bce_10kEpochs/        # ReLU + Binary Cross-Entropy (10k epochs)
│   ├── ReLu_mse/                  # ReLU + Mean Squared Error (1k epochs)
│   ├── ReLu_mse_10kEpochs/        # ReLU + Mean Squared Error (10k epochs)
│   ├── Sigmoid_bce/               # Sigmoid + Binary Cross-Entropy (1k epochs)
│   ├── Sigmoid_bce_10kEpochs/     # Sigmoid + Binary Cross-Entropy (10k epochs)
│   ├── Sigmoid_mse/               # Sigmoid + Mean Squared Error (1k epochs)
│   └── Sigmoid_mse_10kEpochs/     # Sigmoid + Mean Squared Error (10k epochs)
├── Visual Analysis/               # Training loss visualization plots
├── requirements.txt              # Python dependencies
├── results.csv                   # Experimental results (to be populated)
└── README.md                     # This file
```

## Neural Network Architecture

The neural network used in this project has a simple but effective design:
- Input Layer: 30 neurons (one for each feature)
- Hidden Layer: 10 neurons
- Output Layer: 1 neuron (for binary classification)
- Architecture: Fully connected feedforward network

### Mathematical Implementation

The network implements standard forward and backward propagation algorithms.

#### Forward Propagation
```
Z₁ = X·W₁ + B₁
A₁ = activation_function(Z₁)
Z₂ = A₁·W₂ + B₂
A₂ = sigmoid(Z₂)
```

#### Backward Propagation
```
dZ₂ = A₂ - Y
dW₂ = (A₁ᵀ·dZ₂) / m
dB₂ = sum(dZ₂) / m
dA₁ = dZ₂·W₂ᵀ
dZ₁ = dA₁ * activation_derivative(Z₁)
dW₁ = (Xᵀ·dZ₁) / m
dB₁ = sum(dZ₁) / m
```

### Activation Functions

The project implements two popular activation functions to compare their performance.

#### ReLU (Rectified Linear Unit)
```python
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)
```

#### Sigmoid
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)
```

### Loss Functions

Two different loss functions are implemented to study their impact on training.

#### Binary Cross-Entropy (BCE)
```python
def binary_cross_entropy(y_pred, y_true):
    return -np.mean(y_true * np.log(y_pred + 1e-8) + 
                    (1 - y_true) * np.log(1 - y_pred + 1e-8))
```

#### Mean Squared Error (MSE)
```python
def mean_squared_error(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)
```

## Experimental Design

This project takes a systematic approach to exploring how different components affect neural network performance. The experiments examine three key factors:

### Activation Functions
- ReLU: A non-linear function that helps avoid the vanishing gradient problem
- Sigmoid: Provides smooth, bounded output between 0 and 1

### Loss Functions
- Binary Cross-Entropy: The theoretically optimal choice for binary classification tasks
- Mean Squared Error: An alternative approach that may show different convergence patterns

### Training Duration
- 1,000 epochs: Standard training duration for comparison
- 10,000 epochs: Extended training to observe long-term convergence behavior

### Experiment Matrix

| Experiment | Hidden Activation | Loss Function | Epochs | Purpose |
|------------|------------------|---------------|--------|---------|
| ReLu_bce | ReLU | Binary Cross-Entropy | 1,000 | Baseline optimal configuration |
| ReLu_bce_10kEpochs | ReLU | Binary Cross-Entropy | 10,000 | Extended training analysis |
| ReLu_mse | ReLU | Mean Squared Error | 1,000 | Alternative loss comparison |
| ReLu_mse_10kEpochs | ReLU | Mean Squared Error | 10,000 | Extended MSE training |
| Sigmoid_bce | Sigmoid | Binary Cross-Entropy | 1,000 | Traditional activation comparison |
| Sigmoid_bce_10kEpochs | Sigmoid | Binary Cross-Entropy | 10,000 | Extended sigmoid training |
| Sigmoid_mse | Sigmoid | Mean Squared Error | 1,000 | MSE with sigmoid activation |
| Sigmoid_mse_10kEpochs | Sigmoid | Mean Squared Error | 10,000 | Extended sigmoid+MSE training |

## Getting Started

### Prerequisites

You'll need the following installed on your system:
- Python 3.8 or higher
- NumPy
- pandas
- scikit-learn
- matplotlib

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/viraajsharma/Understanding-Neural-Network--A-Minimalist-Implementation-From-Scratch.git
   cd Understanding-Neural-Network--A-Minimalist-Implementation-From-Scratch
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running Experiments

Navigate to any experiment directory and run the main script:

```bash
cd Experiments/ReLu_bce
python main.py
```

This will:
1. Load and preprocess the breast cancer dataset
2. Initialize the neural network with specified configuration
3. Train the network and display loss progress
4. Evaluate final test accuracy
5. Display loss curve visualization

### Data Preprocessing

The preprocessing pipeline includes several important steps handled by the `preprocess.py` module:
- Loading data from the CSV file
- Handling missing values appropriately
- Encoding labels (M=1 for malignant, B=0 for benign)
- Separating features from target variables
- Splitting data into training and testing sets (80/20 ratio)
- Standardizing features using StandardScaler to ensure all features have similar scales

### Example Output

```
Epoch 0, Loss: 0.8234
Epoch 100, Loss: 0.4567
Epoch 200, Loss: 0.3421
...
Epoch 900, Loss: 0.1234
Test Accuracy: 94.74%
```

## Results and Analysis

### Expected Performance

The systematic experimentation reveals interesting patterns:

- ReLU with Binary Cross-Entropy typically achieves the highest accuracy (90-95%)
- Sigmoid with Binary Cross-Entropy shows good performance with smoother convergence
- MSE variants often display different convergence patterns compared to BCE
- Extended training (10k epochs) may improve performance but can also lead to overfitting

### Visual Analysis

The Visual Analysis directory contains loss curve plots for each experiment, showing:
- How training loss progresses over epochs
- Differences in convergence rates between configurations
- Optimal stopping points to avoid overfitting

## Learning Objectives

Working through this project will help you understand:

1. Neural Network Fundamentals
   - How forward and backward propagation work
   - The mechanics of gradient descent optimization
   - Different weight initialization strategies

2. Activation Function Impact
   - Key differences between ReLU and Sigmoid
   - How activation functions affect gradient flow
   - The importance of non-linearity in neural networks

3. Loss Function Selection
   - Why binary cross-entropy works well for classification
   - How MSE performs as an alternative approach
   - The concept of loss landscapes

4. Training Dynamics
   - How to analyze convergence behavior
   - Recognizing signs of overfitting
   - Understanding hyperparameter sensitivity

5. Implementation Skills
   - Efficient NumPy matrix operations
   - Writing vectorized computations
   - Designing object-oriented machine learning code

## Customization

### Modify Network Architecture

```python
# Change hidden layer size
nn = NeuralNetwork(input_size=30, hidden_size=20, output_size=1)

# Adjust learning rate
nn = NeuralNetwork(learning_rate=0.001, epochs=5000)
```

### Adding New Experiments

To create your own experiments:
1. Create a new experiment directory
2. Copy and modify the `main.py` and `preprocess.py` files
3. Implement your desired changes (new activation functions, loss functions, etc.)
4. Run the experiment and analyze the results

### Extending the Network

There are many ways to build upon this foundation:
- Add multiple hidden layers to create deeper networks
- Implement different optimizers like Adam or RMSprop
- Add regularization techniques such as L1, L2, or dropout
- Experiment with different weight initialization schemes

## Educational Value

This repository is particularly useful for:

- Students who are learning neural network fundamentals
- Educators who want to teach machine learning concepts with clear examples
- Practitioners who need to understand implementation details
- Researchers exploring different activation and loss function combinations

## Contributing

Contributions are welcome! Here are some areas where the project could be improved:

- Additional activation functions like Tanh, Leaky ReLU, or ELU
- More sophisticated optimizers beyond basic gradient descent
- Regularization techniques to prevent overfitting
- Advanced architectures or adaptations for different problem types
- Enhanced visualization tools for better analysis
- More comprehensive result analysis and comparison tools

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Wisconsin Breast Cancer Dataset from UCI Machine Learning Repository
- Educational inspiration from various deep learning courses
- NumPy and scikit-learn communities for excellent documentation

## Contact

**Author**: Viraaj Sharma 
**Repository**: [Understanding-Neural-Network--A-Minimalist-Implementation-From-Scratch](https://github.com/viraajsharma/Understanding-Neural-Network--A-Minimalist-Implementation-From-Scratch)

---

*This project demonstrates that understanding neural networks doesn't require complex frameworks - just solid mathematical foundations and clear implementation!*