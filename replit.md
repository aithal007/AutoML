# Smart Data Preprocessor

## Overview

Smart Data Preprocessor is a full-stack AI-powered web application that automates machine learning data preprocessing. Users can upload raw CSV datasets and receive intelligently processed data ready for ML model training. The application automatically detects column types, handles missing values, scales numerical features, and encodes categorical variables.

## System Architecture

### Backend Architecture
- **Framework**: Flask web framework with Python 3.11
- **Core Engine**: Custom AutoML preprocessing engine (`automl_engine.py`)
- **Web Server**: Gunicorn WSGI server for production deployment
- **File Handling**: Werkzeug for secure file uploads with size limits (16MB)

### Frontend Architecture
- **Template Engine**: Jinja2 templates with Bootstrap-based UI
- **CSS Framework**: Bootstrap with custom dark theme
- **JavaScript**: Vanilla JavaScript for interactive functionality
- **Icons**: Font Awesome for UI icons

### Data Processing Architecture
- **Column Detection**: Automatic classification of numerical, categorical, datetime, and constant columns
- **Preprocessing Pipeline**: Scikit-learn ColumnTransformer with StandardScaler and OneHotEncoder
- **Missing Value Strategy**: Mean imputation for numerical, mode for categorical data

## Key Components

### DataCleaner Class (`automl_engine.py`)
- **Purpose**: Core preprocessing engine that handles all data transformations
- **Features**: 
  - Automatic column type detection
  - Constant column removal
  - Missing value imputation
  - Feature scaling and encoding
  - Preprocessing summary generation

### Flask Application (`app.py`)
- **Purpose**: Web server handling file uploads and processing requests
- **Key Routes**:
  - `/` - Main upload interface
  - `/upload` - File upload handling
  - Additional routes for processing and download (implementation ongoing)

### Frontend Interface
- **Upload Form**: Secure file upload with validation
- **Progress Tracking**: Visual feedback during processing
- **Results Display**: Processed data preview and download options

## Data Flow

1. **File Upload**: User uploads CSV file through web interface
2. **Validation**: Server validates file type, size, and format
3. **Type Detection**: AutoML engine analyzes and classifies all columns
4. **Preprocessing**: Automated cleaning pipeline processes the data
5. **Summary Generation**: System creates detailed preprocessing report
6. **Output**: User receives processed data preview and download link

## External Dependencies

### Core Libraries
- **Flask**: Web framework for API and template rendering
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing operations
- **Scikit-learn**: Machine learning preprocessing tools
- **Werkzeug**: WSGI utilities and security

### UI Libraries
- **Bootstrap**: Responsive CSS framework with dark theme
- **Font Awesome**: Icon library for enhanced UI

## Deployment Strategy

### Environment Configuration
- **Runtime**: Python 3.11 with Nix package management
- **Database**: PostgreSQL ready (configured but not yet implemented)
- **SSL**: OpenSSL support for secure connections

### Production Deployment
- **Server**: Gunicorn with auto-scaling deployment target
- **Port**: Application runs on port 5000
- **Process Management**: Configurable worker processes with port reuse
- **Development**: Hot reload enabled for development workflow

### File Storage
- **Upload Directory**: `uploads/` for temporary CSV storage
- **Processed Directory**: `processed/` for cleaned data output
- **Security**: Secure filename handling and file type validation

## Changelog
- June 24, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.