// Smart Data Preprocessor JavaScript

class DataPreprocessor {
    constructor() {
        this.initializeEventListeners();
        this.currentState = 'initial'; // initial, uploaded, processing, completed
    }

    initializeEventListeners() {
        // Upload form submission
        document.getElementById('uploadForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleFileUpload();
        });

        // Preprocess button click
        document.getElementById('preprocessBtn').addEventListener('click', () => {
            this.handlePreprocessing();
        });

        // Download button click
        document.getElementById('downloadBtn').addEventListener('click', () => {
            this.handleDownload();
        });

        // Reset button click
        document.getElementById('resetBtn').addEventListener('click', () => {
            this.handleReset();
        });

        // File input change
        document.getElementById('csvFile').addEventListener('change', (e) => {
            this.validateFileInput(e.target);
        });
    }

    validateFileInput(input) {
        const file = input.files[0];
        if (!file) return;

        // Check file type
        if (!file.name.toLowerCase().endsWith('.csv')) {
            this.showAlert('Please select a CSV file.', 'warning');
            input.value = '';
            return;
        }

        // Check file size (16MB limit)
        if (file.size > 16 * 1024 * 1024) {
            this.showAlert('File size must be less than 16MB.', 'warning');
            input.value = '';
            return;
        }

        this.clearAlerts();
    }

    async handleFileUpload() {
        const fileInput = document.getElementById('csvFile');
        const uploadBtn = document.getElementById('uploadBtn');
        
        if (!fileInput.files[0]) {
            this.showAlert('Please select a file to upload.', 'warning');
            return;
        }

        // Show loading state
        this.setButtonLoading(uploadBtn, true);
        this.hideSection('fileInfoSection');
        this.hideSection('resultsSection');

        try {
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.showAlert(result.message, 'success');
                this.displayFileInfo(result.file_info);
                this.currentState = 'uploaded';
                document.getElementById('resetBtn').style.display = 'inline-block';
            } else {
                this.showAlert(result.error || 'Upload failed', 'danger');
            }

        } catch (error) {
            console.error('Upload error:', error);
            this.showAlert('Upload failed. Please try again.', 'danger');
        } finally {
            this.setButtonLoading(uploadBtn, false);
        }
    }

    displayFileInfo(fileInfo) {
        const content = document.getElementById('fileInfoContent');
        const columnsPreview = fileInfo.column_names.length > 10 
            ? fileInfo.column_names.slice(0, 10).join(', ') + '...'
            : fileInfo.column_names.join(', ');

        content.innerHTML = `
            <div class="row">
                <div class="col-md-4">
                    <div class="stat-card">
                        <div class="stat-number">${fileInfo.rows.toLocaleString()}</div>
                        <div class="stat-label">Rows</div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="stat-card">
                        <div class="stat-number">${fileInfo.columns}</div>
                        <div class="stat-label">Columns</div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="stat-card">
                        <div class="stat-number">${this.formatFileSize(fileInfo.rows * fileInfo.columns)}</div>
                        <div class="stat-label">Data Points</div>
                    </div>
                </div>
            </div>
            <div class="mt-3">
                <h6>Sample Columns:</h6>
                <p class="text-muted mb-0">${columnsPreview}</p>
            </div>
        `;

        this.showSection('fileInfoSection');
    }

    async handlePreprocessing() {
        const preprocessBtn = document.getElementById('preprocessBtn');
        
        this.setButtonLoading(preprocessBtn, true);
        this.showSection('processingStatus');
        this.hideSection('resultsSection');
        this.currentState = 'processing';

        try {
            const response = await fetch('/preprocess', {
                method: 'POST'
            });

            const result = await response.json();

            if (result.success) {
                this.showAlert(result.message, 'success');
                this.displayResults(result);
                this.currentState = 'completed';
            } else {
                this.showAlert(result.error || 'Preprocessing failed', 'danger');
            }

        } catch (error) {
            console.error('Preprocessing error:', error);
            this.showAlert('Preprocessing failed. Please try again.', 'danger');
        } finally {
            this.setButtonLoading(preprocessBtn, false);
            this.hideSection('processingStatus');
        }
    }

    displayResults(result) {
        // Display summary
        this.displaySummary(result.summary);
        
        // Display preview
        this.displayPreview(result.preview);
        
        // Display details
        this.displayDetails(result.summary);
        
        this.showSection('resultsSection');
    }

    displaySummary(summary) {
        const content = document.getElementById('summaryContent');
        const stats = summary.preprocessing_stats;
        
        content.innerHTML = `
            <div class="row mb-3">
                <div class="col-md-3">
                    <div class="stat-card">
                        <div class="stat-number">${summary.original_shape[0].toLocaleString()}</div>
                        <div class="stat-label">Original Rows</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card">
                        <div class="stat-number">${summary.original_shape[1]}</div>
                        <div class="stat-label">Original Features</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card">
                        <div class="stat-number">${summary.processed_shape[1]}</div>
                        <div class="stat-label">Final Features</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card">
                        <div class="stat-number">${stats.features_dropped}</div>
                        <div class="stat-label">Features Dropped</div>
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="col-md-12">
                    <div class="alert alert-info">
                        <i class="fas fa-info-circle me-2"></i>
                        <strong>Processing Complete!</strong> Your dataset has been automatically cleaned and prepared for machine learning.
                        ${stats.numerical_features} numerical features were scaled, ${stats.categorical_features} categorical features were encoded,
                        and ${stats.features_dropped} constant/problematic features were removed.
                    </div>
                </div>
            </div>
        `;
    }

    displayPreview(preview) {
        const content = document.getElementById('previewContent');
        
        if (!preview.data || preview.data.length === 0) {
            content.innerHTML = '<p class="text-muted">No preview data available.</p>';
            return;
        }

        // Create table
        let tableHTML = `
            <table class="table table-striped table-hover">
                <thead>
                    <tr>
        `;
        
        // Add headers
        preview.columns.forEach(col => {
            tableHTML += `<th>${this.escapeHtml(col)}</th>`;
        });
        
        tableHTML += `
                    </tr>
                </thead>
                <tbody>
        `;
        
        // Add data rows
        preview.data.forEach(row => {
            tableHTML += '<tr>';
            preview.columns.forEach(col => {
                const value = row[col];
                const displayValue = typeof value === 'number' ? 
                    (value % 1 === 0 ? value : value.toFixed(4)) : 
                    this.escapeHtml(String(value));
                tableHTML += `<td>${displayValue}</td>`;
            });
            tableHTML += '</tr>';
        });
        
        tableHTML += `
                </tbody>
            </table>
            <div class="mt-2 text-muted">
                <small>Showing 10 of ${preview.total_rows.toLocaleString()} rows</small>
            </div>
        `;
        
        content.innerHTML = tableHTML;
    }

    displayDetails(summary) {
        const content = document.getElementById('detailsContent');
        const columnTypes = summary.column_types;
        const droppedColumns = summary.dropped_columns;
        
        let detailsHTML = '<div class="row">';
        
        // Column types breakdown
        detailsHTML += `
            <div class="col-md-6">
                <h6><i class="fas fa-list me-2"></i>Column Types</h6>
                <div class="mb-3">
                    ${Object.entries(columnTypes).map(([type, columns]) => {
                        if (columns.length === 0) return '';
                        const iconMap = {
                            'numerical': 'fas fa-hashtag text-primary',
                            'categorical': 'fas fa-tags text-info',
                            'datetime': 'fas fa-calendar text-warning',
                            'constant': 'fas fa-equals text-danger'
                        };
                        return `
                            <div class="mb-2">
                                <span class="badge bg-secondary">
                                    <i class="${iconMap[type]} me-1"></i>
                                    ${type.charAt(0).toUpperCase() + type.slice(1)} (${columns.length})
                                </span>
                                <div class="feature-list">
                                    ${columns.map(col => `<span class="feature-item">${this.escapeHtml(col)}</span>`).join('')}
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
        `;
        
        // Dropped columns
        detailsHTML += `
            <div class="col-md-6">
                <h6><i class="fas fa-trash me-2"></i>Dropped Features</h6>
                <div class="mb-3">
                    ${droppedColumns.length > 0 ? `
                        <div class="feature-list">
                            ${droppedColumns.map(col => `<span class="feature-item bg-danger">${this.escapeHtml(col)}</span>`).join('')}
                        </div>
                        <small class="text-muted">These features were removed because they were constant or problematic.</small>
                    ` : `
                        <p class="text-muted">No features were dropped during preprocessing.</p>
                    `}
                </div>
            </div>
        `;
        
        detailsHTML += '</div>';
        
        content.innerHTML = detailsHTML;
    }

    async handleDownload() {
        const downloadBtn = document.getElementById('downloadBtn');
        this.setButtonLoading(downloadBtn, true);

        try {
            const response = await fetch('/download');
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'processed_dataset.csv';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                this.showAlert('Dataset downloaded successfully!', 'success');
            } else {
                const result = await response.json();
                this.showAlert(result.error || 'Download failed', 'danger');
            }
        } catch (error) {
            console.error('Download error:', error);
            this.showAlert('Download failed. Please try again.', 'danger');
        } finally {
            this.setButtonLoading(downloadBtn, false);
        }
    }

    async handleReset() {
        if (!confirm('Are you sure you want to start over? This will clear all current progress.')) {
            return;
        }

        try {
            const response = await fetch('/reset', {
                method: 'POST'
            });

            const result = await response.json();
            
            if (result.success) {
                // Reset UI
                document.getElementById('uploadForm').reset();
                this.hideSection('fileInfoSection');
                this.hideSection('resultsSection');
                this.hideSection('processingStatus');
                document.getElementById('resetBtn').style.display = 'none';
                this.clearAlerts();
                this.currentState = 'initial';
                
                this.showAlert('Session reset successfully. You can upload a new file.', 'info');
            } else {
                this.showAlert(result.error || 'Reset failed', 'danger');
            }
        } catch (error) {
            console.error('Reset error:', error);
            this.showAlert('Reset failed. Please refresh the page.', 'danger');
        }
    }

    // Utility methods
    showAlert(message, type) {
        const alertContainer = document.getElementById('alertContainer');
        const alertId = 'alert-' + Date.now();
        
        const alertHTML = `
            <div id="${alertId}" class="alert alert-${type} alert-dismissible fade show fade-in" role="alert">
                <i class="fas fa-${this.getAlertIcon(type)} me-2"></i>
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        alertContainer.innerHTML = alertHTML;
        
        // Auto-dismiss success and info alerts
        if (type === 'success' || type === 'info') {
            setTimeout(() => {
                const alert = document.getElementById(alertId);
                if (alert) {
                    const bsAlert = new bootstrap.Alert(alert);
                    bsAlert.close();
                }
            }, 5000);
        }
    }

    clearAlerts() {
        document.getElementById('alertContainer').innerHTML = '';
    }

    getAlertIcon(type) {
        const icons = {
            'success': 'check-circle',
            'danger': 'exclamation-triangle',
            'warning': 'exclamation-circle',
            'info': 'info-circle'
        };
        return icons[type] || 'info-circle';
    }

    setButtonLoading(button, isLoading) {
        if (isLoading) {
            button.disabled = true;
            button.dataset.originalText = button.innerHTML;
            button.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';
        } else {
            button.disabled = false;
            button.innerHTML = button.dataset.originalText || button.innerHTML;
        }
    }

    showSection(sectionId) {
        const section = document.getElementById(sectionId);
        section.style.display = 'block';
        section.classList.add('fade-in');
    }

    hideSection(sectionId) {
        const section = document.getElementById(sectionId);
        section.style.display = 'none';
        section.classList.remove('fade-in');
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, function(m) { return map[m]; });
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new DataPreprocessor();
});
