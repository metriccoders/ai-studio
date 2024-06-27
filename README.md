# AI Studio by Metric Coders

## Overview
**AI Studio** is a GitHub repository created by **Metric Coders**. It provides a web-based user interface (UI) for the **scikit-learn** library, enabling users to create and train various machine learning (ML) models by configuring different hyperparameters through the UI. The platform is designed to simplify the process of building and experimenting with ML models, making it accessible for users with varying levels of expertise in machine learning. This application is built using **Streamlit** to provide an interactive and user-friendly experience.

## Features
### Current Features
- **Streamlit UI for Scikit-learn**: Users can easily create ML models using a graphical interface.
- **Hyperparameter Tuning**: Adjust and experiment with different hyperparameters to optimize model performance.
- **Preloaded Datasets**: Access to a variety of datasets provided by scikit-learn, facilitating immediate experimentation and learning.
- **Model Export**: Functionality to export trained models for deployment and further use.
- **Custom Datasets**: Users can upload and use their own datasets for model training.

### Upcoming Features
- **Data Cleaning and Preprocessing**: Tools for data cleaning and preprocessing will be integrated to prepare datasets for model training.
- **Support for LLMs**: Expansion to include large language models (LLMs), broadening the scope of the platform to natural language processing tasks.

## Getting Started
### Prerequisites
- Python 3.6+
- Scikit-learn library
- Streamlit

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/MetricCoders/AI-Studio.git
    ```
2. Navigate to the project directory:
    ```bash
    cd AI-Studio
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application
To start the Streamlit application, use the following command:
```bash
streamlit run Home.py
```
This will launch the AI Studio web UI. Open your web browser and navigate to the URL provided in the terminal to start using the platform.

## Usage
### Creating a Model
1. **Select a Dataset**: Choose from a variety of preloaded datasets.
2. **Choose an Algorithm**: Select the machine learning algorithm you wish to use (e.g., linear regression, decision trees, etc.).
3. **Set Hyperparameters**: Configure the hyperparameters for your chosen algorithm.
4. **Train the Model**: Click the "Train" button to build and train your model.

### Evaluating a Model
After training, AI Studio provides performance metrics and visualizations to help you evaluate the effectiveness of your model. Use these insights to refine your model by adjusting hyperparameters or selecting different algorithms.

## Contributing
Contributions are welcome! If you have suggestions for improvements or have identified issues, please open an issue or submit a pull request. Follow the guidelines in `CONTRIBUTING.md` to get started.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions, feedback, or support, please contact the Metric Coders team via email at info@metriccoders.com.

---

**AI Studio by Metric Coders** aims to democratize machine learning by providing an intuitive platform for model creation and experimentation. With upcoming features and continuous improvements, it strives to be a comprehensive tool for both novice and experienced data scientists.
