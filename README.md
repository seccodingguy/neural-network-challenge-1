# Student Loan Risk with Deep Learning - by Mark Wireman

This project explores the prediction of student loan repayment success using a deep learning model implemented with TensorFlow's Keras API. The dataset used in this project contains various features related to student loan risk factors, and the goal is to classify whether a student is likely to repay their loan (`credit_ranking`).

---

## Project Workflow

The notebook is organized into the following main steps:

### 1. Data Preparation
- The dataset `student-loans.csv` is loaded into a Pandas DataFrame for analysis. 
- The columns are reviewed to define the features (`X`) and the target variable (`y`).
  - **Features (`X`)**: All columns except `credit_ranking`.
  - **Target (`y`)**: The `credit_ranking` column.

### 2. Data Preprocessing
- The dataset is split into training and testing sets using `train_test_split` from Scikit-learn.
- The features are standardized using `StandardScaler` to ensure all input values are on a similar scale for better model performance.

### 3. Model Creation
- A deep neural network model is built using TensorFlow's Keras `Sequential` API.
  - The model consists of:
    - **Input Layer**: Matches the number of features.
    - **Two Hidden Layers** with `relu` activation:
      - First hidden layer: 11 neurons.
      - Second hidden layer: 8 neurons.
    - **Output Layer**: A single neuron with a `sigmoid` activation function for binary classification.
  - The model is compiled using:
    - Loss function: `binary_crossentropy`.
    - Optimizer: `adam`.
    - Metric: `accuracy`.

### 4. Model Training
- The model is trained for 50 epochs using the training dataset.
- The loss and accuracy of the model during training are monitored.

### 5. Model Evaluation
- The trained model's performance is evaluated using the test dataset.
- Key metrics include:
  - **Loss**: 0.5246
  - **Accuracy**: 74.25%

### 6. Making Predictions
- Predictions are made on the test dataset using the trained model.
- The predictions are saved into a DataFrame and rounded to binary values (`0` or `1`).

### 7. Model Export
- The trained model is saved to a `.keras` file (`student_loans.keras`) for future use or deployment.

### 8. Classification Report
- A classification report is generated to review the model's precision, recall, F1-score, and support for each class.

---

## Key Files

- **Dataset**: `student-loans.csv` (loaded from a remote URL).
- **Model File**: `student_loans.keras` (saved trained model).

---

## Libraries and Frameworks Used

- **Python Libraries**:
  - `pandas` (for data manipulation and analysis).
  - `tensorflow` (for building and training the deep learning model).
  - `sklearn` (for data preprocessing, splitting, and evaluation).
- **Framework**: TensorFlow's Keras API.

---

## Results

- The deep learning model achieved an accuracy of **74.25%** on the test dataset.
- The classification report shows the performance of the model for both classes (`0` and `1`):
  - **Precision**: 0.71 for class `0` and 0.78 for class `1`.
  - **Recall**: 0.77 for class `0` and 0.72 for class `1`.
  - **F1-Score**: 0.74 for class `0` and 0.75 for class `1`.

---

## How to Use

1. Clone the repository or download the notebook file.
2. Install the required Python libraries:
   ```bash
   pip install pandas tensorflow scikit-learn