import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import logging
from automl_engine import DataCleaner
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'csv'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Global variable to store the data cleaner instance
data_cleaner = None
current_file_info = {}

def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with file upload interface."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and return basic file information."""
    global current_file_info
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Try to read and validate the CSV
        try:
            df = pd.read_csv(filepath)
            if df.empty:
                os.remove(filepath)
                return jsonify({'error': 'Uploaded CSV file is empty'}), 400
            
            # Store file information
            current_file_info = {
                'filename': filename,
                'filepath': filepath,
                'original_name': file.filename,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist()
            }
            
            logger.info(f"File uploaded successfully: {filename}, Shape: {df.shape}")
            
            return jsonify({
                'success': True,
                'message': f'File uploaded successfully! Found {len(df)} rows and {len(df.columns)} columns.',
                'file_info': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': df.columns.tolist()[:10]  # Show first 10 column names
                }
            })
            
        except Exception as e:
            # Remove file if CSV reading failed
            if os.path.exists(filepath):
                os.remove(filepath)
            logger.error(f"Error reading CSV file: {e}")
            return jsonify({'error': f'Invalid CSV file: {str(e)}'}), 400
            
    except Exception as e:
        logger.error(f"Error in upload_file: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/preprocess', methods=['POST'])
def preprocess_data():
    """Process the uploaded data and return preview with metadata."""
    global data_cleaner, current_file_info
    
    try:
        # Check if file was uploaded
        if not current_file_info or not os.path.exists(current_file_info['filepath']):
            return jsonify({'error': 'No file uploaded or file not found'}), 400
        
        # Read the uploaded CSV
        df = pd.read_csv(current_file_info['filepath'])
        logger.info(f"Starting preprocessing for file: {current_file_info['filename']}")
        
        # Initialize data cleaner and process the data
        data_cleaner = DataCleaner()
        X_transformed, feature_names = data_cleaner.fit_transform(df)
        
        # Convert transformed data back to DataFrame for easier handling
        processed_df = pd.DataFrame(X_transformed, columns=feature_names)
        
        # Save processed data
        processed_filename = f"processed_{current_file_info['filename']}"
        processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        processed_df.to_csv(processed_filepath, index=False)
        
        # Update current file info
        current_file_info['processed_filename'] = processed_filename
        current_file_info['processed_filepath'] = processed_filepath
        
        # Get preprocessing summary
        summary = data_cleaner.get_preprocessing_summary()
        
        # Prepare preview data (first 10 rows)
        preview_data = processed_df.head(10).round(4)  # Round to 4 decimal places
        preview_dict = preview_data.to_dict('records')
        
        # Prepare response
        response_data = {
            'success': True,
            'message': 'Data preprocessing completed successfully!',
            'preview': {
                'columns': feature_names,
                'data': preview_dict,
                'total_rows': len(processed_df)
            },
            'summary': {
                'original_shape': [current_file_info['rows'], current_file_info['columns']],
                'processed_shape': [len(processed_df), len(feature_names)],
                'column_types': summary['column_types'],
                'dropped_columns': summary['dropped_columns'],
                'preprocessing_stats': summary['preprocessing_summary']
            }
        }
        
        logger.info(f"Preprocessing completed. Original shape: {df.shape}, "
                   f"Processed shape: {processed_df.shape}")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in preprocess_data: {e}")
        return jsonify({'error': f'Preprocessing failed: {str(e)}'}), 500

@app.route('/download')
def download_processed():
    """Download the processed CSV file."""
    try:
        if not current_file_info or 'processed_filepath' not in current_file_info:
            return jsonify({'error': 'No processed file available'}), 400
        
        processed_filepath = current_file_info['processed_filepath']
        
        if not os.path.exists(processed_filepath):
            return jsonify({'error': 'Processed file not found'}), 404
        
        # Return the file as attachment
        return send_file(
            processed_filepath,
            as_attachment=True,
            download_name=f"processed_{current_file_info['original_name']}",
            mimetype='text/csv'
        )
        
    except Exception as e:
        logger.error(f"Error in download_processed: {e}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/reset', methods=['POST'])
def reset_session():
    """Reset the current session and clean up temporary files."""
    global current_file_info, data_cleaner
    
    try:
        # Clean up uploaded file
        if current_file_info and 'filepath' in current_file_info:
            if os.path.exists(current_file_info['filepath']):
                os.remove(current_file_info['filepath'])
                logger.info(f"Removed uploaded file: {current_file_info['filepath']}")
        
        # Clean up processed file
        if current_file_info and 'processed_filepath' in current_file_info:
            if os.path.exists(current_file_info['processed_filepath']):
                os.remove(current_file_info['processed_filepath'])
                logger.info(f"Removed processed file: {current_file_info['processed_filepath']}")
        
        # Reset global variables
        current_file_info = {}
        data_cleaner = None
        
        return jsonify({'success': True, 'message': 'Session reset successfully'})
        
    except Exception as e:
        logger.error(f"Error in reset_session: {e}")
        return jsonify({'error': f'Reset failed: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error occurred'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
