
# AI Studio

This repository provides a user-friendly web interface for exploring and using scikit-learn (sklearn) machine learning models. The UI is built using [Streamlit](https://streamlit.io/), making it easy to visualize data, train models, and evaluate their performance without writing any code. It is available as [AI Studio](https://metriccoders-sklearn-studio-home-zegcs8.streamlit.app/).

## Features

- **Data Upload:** Upload your own dataset in CSV format. (yet to be implemented)
- **Data Visualization:** Visualize the dataset with various charts and plots. (yet to be implemented)
- **Model Training:** Select and train different scikit-learn models. (Partial. Just basic for now)
- **Model Evaluation:** Evaluate model performance with metrics and visualizations. (Partial. Just basic for now)
- **Export Results:** Export trained models and evaluation results. (yet to be implemented)

## Demo

You can check out the live demo here: [AI Studio](https://metriccoders-sklearn-studio-home-zegcs8.streamlit.app/) (link to the deployed Streamlit app).

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**

   ```sh
   git clone https://github.com/metriccoders/sklearn-studio.git
   cd sklearn-studio
   ```

2. **Create a virtual environment:**

   ```sh
   python -m venv venv
   ```

3. **Activate the virtual environment:**

   - On Windows:
     ```sh
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```sh
     source venv/bin/activate
     ```

4. **Install the required dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

5. **Run the Streamlit app:**

   ```sh
   streamlit run Home.py
   ```

6. Open your web browser and go to `http://localhost:8501` to view the app.

## Usage

### Upload Dataset
(yet to be implemented)
- Click on the "Upload CSV" button to upload your dataset.
- The dataset should be in CSV format with a header row.

### Data Visualization
(yet to be implemented)
- Use the sidebar to select the columns you want to visualize.
- Choose from different types of plots like histograms, scatter plots, and box plots.

### Model Training
(In progress...)
- Select a machine learning model from the sidebar (e.g., RandomForestClassifier, LogisticRegression).
- Configure the model parameters as needed.
- Click the "Train Model" button to train the model on the uploaded dataset.

### Model Evaluation
(In progress...)
- View the performance metrics such as accuracy, precision, recall, and F1-score.
- Visualize the confusion matrix and ROC curve.

### Export Results
(yet to be implemented)
- Download the trained model and evaluation results.

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- [scikit-learn](https://scikit-learn.org/)
- [Streamlit](https://streamlit.io/)
