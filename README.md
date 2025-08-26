# Fashion MNIST GAN with PyTorch Lightning

## 📋 Project Overview

This project implements a **Generative Adversarial Network (GAN)** using PyTorch Lightning to generate synthetic fashion images from the Fashion MNIST dataset. The project demonstrates advanced deep learning concepts including adversarial training, neural network architectures, and the practical applications of GANs in data augmentation for computer vision tasks.

## 🎯 Objectives

- Implement a robust GAN architecture for fashion image generation
- Train the model to generate realistic fashion items (clothing, shoes, accessories)
- Analyze the impact of GAN-generated data on CNN performance
- Demonstrate proficiency in modern deep learning frameworks and best practices

## 🏗️ Architecture

### Generator Network
- **Input**: Random noise vector (latent dimension: 100)
- **Architecture**: Linear layer → ConvTranspose2d layers → Conv2d output
- **Output**: 28x28 grayscale fashion images
- **Activation**: ReLU for hidden layers

### Discriminator Network
- **Input**: 28x28 grayscale images (real or generated)
- **Architecture**: Conv2d layers → Dropout → Fully connected layers
- **Output**: Binary classification (real vs. fake)
- **Activation**: ReLU for hidden layers, Sigmoid for output

## 🛠️ Technologies Used

- **Python 3.7+**
- **PyTorch**: Deep learning framework
- **PyTorch Lightning**: High-level PyTorch wrapper for clean, scalable code
- **Torchvision**: Computer vision utilities and datasets
- **Matplotlib**: Visualization and plotting
- **Fashion MNIST**: Dataset of clothing images

## 📊 Dataset

**Fashion MNIST** consists of:
- 70,000 grayscale images (28x28 pixels)
- 10 categories of fashion items
- Training set: 60,000 images
- Test set: 10,000 images
- Categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

## 🚀 Installation & Setup

### Prerequisites
```bash
pip install torch torchvision pytorch-lightning matplotlib
```

### For Google Colab
```python
!pip install pytorch-lightning
```

### Local Installation
1. Clone the repository
2. Install dependencies from requirements (ensure all packages listed above are installed)
3. Ensure GPU access for optimal performance

## 💻 Usage

### Running the Training
```python
python GAN_Assignement_Pytorch.py
```

### Jupyter Notebook
Open and run `GAN_Assignment_Pytorch.ipynb` for interactive execution

### Key Parameters
- **Epochs**: 20 (adjustable for improved accuracy)
- **Learning Rate**: 0.0002
- **Batch Size**: 128
- **Latent Dimension**: 100

### Viewing Results
After training completion, visualize generated images:
```python
model.plot_imgs()
```

## 📈 Results & Analysis

The project includes comprehensive analysis of:
- **Model Performance**: Training loss progression for both generator and discriminator
- **Image Quality**: Visual assessment of generated fashion items
- **Data Augmentation Impact**: Analysis of how GAN-generated data affects CNN classifier performance

### Key Findings
- Enhanced dataset diversity through synthetic image generation
- Improved CNN robustness when trained with augmented data
- Cost-effective approach to expanding limited datasets
- Potential applications in fashion recommendation systems

## 🔬 Research Applications

This project demonstrates practical applications in:
- **Data Augmentation**: Expanding limited datasets for better model training
- **Fashion Industry**: Generating diverse clothing designs and styles
- **Computer Vision**: Improving classification models through synthetic data
- **Research & Development**: Understanding adversarial learning dynamics

## ⚡ Performance Optimization

- **GPU Acceleration**: Utilizes GPU for efficient training
- **PyTorch Lightning**: Implements best practices for scalable deep learning
- **Modular Design**: Clean separation between data loading, model architecture, and training logic

## 📁 Project Structure

```
├── GAN_Assignement_Pytorch.py          # Main Python implementation
├── GAN_Assignment_Pytorch.ipynb        # Jupyter notebook version
├── Impact of GAN-Generated Data on CNN.txt  # Research analysis
└── README.md                           # Project documentation
```

## 🤝 Contributing

Feel free to:
- Report issues or bugs
- Suggest improvements
- Submit pull requests
- Provide feedback on model performance

## 📧 Contact

**Ahmed Tarek**
- GitHub: [GssHunterAI](https://github.com/GssHunterAI)

## 📝 License

This project is part of the WE Advanced AI Training program and is available for educational and research purposes.

---

*This project showcases advanced machine learning techniques and demonstrates proficiency in modern deep learning frameworks for generative modeling and computer vision applications.*
