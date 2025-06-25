# Smart Data Preprocessor

## How to Run This Product

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd SmartDataPreprocessor/SmartDataPreprocessor
   ```
2. **Install dependencies:**
   - It is recommended to use a virtual environment (venv or conda).
   - Install required packages:
     ```sh
     pip install -r requirements.txt
     ```
     *(If requirements.txt is not present, install: Flask, pandas, numpy, scikit-learn)*
     ```sh
     pip install flask pandas numpy scikit-learn
     ```
3. **Run the application:**
   ```sh
   python main.py
   ```
   - The app will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000)
4. **Open your browser:**
   - Go to [http://127.0.0.1:5000](http://127.0.0.1:5000)
   - Upload your CSV, select options, preprocess, and download your cleaned data!

---

## Overview

Smart Data Preprocessor is a full-stack AI-powered web application that automates machine learning data preprocessing. Users can upload raw CSV datasets and receive intelligently processed data ready for ML model training. The application automatically detects column types, handles missing values, scales numerical features, encodes categorical variables, and now provides advanced options for missing value handling and column selection.

## Key Features (2024 Update)

- **Automatic Column Type Detection**: Identifies numerical, categorical, datetime, and constant columns.
- **Missing Value Handling**: 
  - Users can choose to **impute** missing values (mean for numerical, mode for categorical) or **delete rows** containing any NaN values.
  - The app displays the total number of missing values and affected rows before preprocessing.
  - If all rows would be dropped, a clear error is shown.
- **Target and Serial/ID Column Selection**:
  - After upload, users can select which column is the target (to predict) and which is a serial/ID column (optional).
  - These columns are excluded from all preprocessing (scaling, encoding, imputation) and added back to the output as-is.
- **Preprocessing Pipeline**: Uses scikit-learn's ColumnTransformer for:
  - Scaling numerical features (StandardScaler)
  - Encoding categorical features (OneHotEncoder)
  - Imputation or row deletion as chosen by the user
- **Detailed Preprocessing Summary**: Shows a preview of the processed data, a summary of all steps, and warnings if rows were dropped.
- **Modern, Responsive UI**: Built with Bootstrap, Font Awesome, and custom JavaScript for a smooth workflow.

## System Architecture

### Backend
- **Framework**: Flask (Python 3.11)
- **Core Engine**: `DataCleaner` class in `automl_engine.py`
- **File Handling**: Secure uploads, size limits, and safe storage

### Frontend
- **Template Engine**: Jinja2 with Bootstrap-based UI
- **JavaScript**: Handles file upload, user options, and dynamic updates
- **User Workflow**:
  1. Upload CSV file
  2. See dataset info and missing value stats
  3. Select target and serial/ID columns
  4. Choose how to handle missing values (impute or delete)
  5. Run preprocessing and preview/download results

## Data Flow

1. **File Upload**: User uploads CSV file through web interface
2. **Validation**: Server validates file type, size, and format
3. **Type Detection**: AutoML engine analyzes and classifies all columns
4. **User Options**: User selects target/serial columns and NaN handling strategy
5. **Preprocessing**: Automated cleaning pipeline processes the data
6. **Summary Generation**: System creates detailed preprocessing report and warnings
7. **Output**: User receives processed data preview and download link

## Example Workflow

1. **Upload**: Drag and drop or select your CSV file.
2. **Review**: Instantly see how many missing values and which columns are present.
3. **Select Columns**: Choose your target (prediction) column and serial/ID column (if any).
4. **Handle NaNs**: Decide whether to impute missing values or delete rows with NaNs.
5. **Preprocess**: Click "Run Preprocessing" to clean your data. If rows are dropped, you'll see a warning.
6. **Download**: Preview the cleaned data and download the processed CSV for ML modeling.

## External Dependencies

- **Flask**: Web framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Preprocessing tools
- **Bootstrap**: Responsive UI
- **Font Awesome**: Icons


## Usage Notes
- For best results, ensure your CSV has clear column headers.
- If you have a serial/ID column, select it so it is not altered during preprocessing.
- If all rows have missing values, choose imputation or clean your data before upload.
