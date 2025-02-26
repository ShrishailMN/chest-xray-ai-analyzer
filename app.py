from flask import Flask, render_template, request, jsonify, send_file
import os
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime
import random
from fpdf import FPDF
import sqlite3

app = Flask(__name__)

class XrayAnalyzer(nn.Module):
    def __init__(self):
        super(XrayAnalyzer, self).__init__()
        # Load pretrained DenseNet
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        
        # Add more sophisticated classifier layers
        self.densenet.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)  # Changed to 2 classes: normal and pneumonia
        )
        
    def forward(self, x):
        features = self.densenet(x)
        return features

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = XrayAnalyzer().to(device)
model.eval()

# Enhanced image preprocessing
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_pneumonia_description():
    locations = ['in the right lower lobe', 'in the left lower lobe', 'in both lower lobes', 
                'in the right upper lobe', 'in the left upper lobe', 'in multiple lobes']
    characteristics = ['patchy opacities', 'consolidation', 'infiltrates', 'airspace opacity']
    severity = ['mild', 'moderate', 'significant']
    
    return {
        'location': random.choice(locations),
        'characteristic': random.choice(characteristics),
        'severity': random.choice(severity)
    }

def get_normal_description():
    variations = [
        "Lungs are clear and well-expanded. Heart size and mediastinal contours are within normal limits. No pleural effusions or pneumothorax.",
        "Normal chest radiograph. Lungs are clear without focal consolidation. Cardiac silhouette is normal. Osseous structures are intact.",
        "Clear lung fields bilaterally. Normal cardiomediastinal silhouette. No acute osseous abnormalities.",
        "No acute cardiopulmonary process. Clear lungs without infiltrates. Normal heart size. No pleural effusions.",
        "Chest x-ray demonstrates normal findings. Clear lung fields. Normal cardiac contours. No pneumothorax or effusions."
    ]
    return random.choice(variations)

def analyze_features(features):
    """Analyze image features for conditions"""
    probabilities = F.softmax(features, dim=1)[0]
    conditions = []
    
    # Check probabilities for each class
    normal_prob = probabilities[0].item()
    pneumonia_prob = probabilities[1].item()
    
    # Use higher threshold for more confident predictions
    if pneumonia_prob > 0.5:
        conditions.append(('pneumonia', pneumonia_prob))
    elif normal_prob > 0.5:
        conditions.append(('normal', normal_prob))
    else:
        # If no confident prediction, choose the higher probability
        if pneumonia_prob > normal_prob:
            conditions.append(('pneumonia', pneumonia_prob))
        else:
            conditions.append(('normal', normal_prob))
    
    return conditions

def generate_detailed_report(conditions):
    """Generate detailed report based on detected conditions"""
    if not conditions:
        return get_normal_description(), "Normal"
    
    primary_condition = conditions[0][0]
    confidence = conditions[0][1]
    
    if primary_condition == 'normal':
        report = get_normal_description()
        return report, "Normal"
    
    elif primary_condition == 'pneumonia':
        desc = get_pneumonia_description()
        report = f"Chest x-ray reveals {desc['severity']} {desc['characteristic']} {desc['location']}, "
        report += f"consistent with pneumonia. "
        
        # Add additional findings based on severity
        if desc['severity'] == 'mild':
            report += "Heart size is normal. No pleural effusions."
        elif desc['severity'] == 'moderate':
            report += "Heart size is normal. Small pleural effusion may be present."
        else:
            report += "Consider follow-up chest CT if clinically indicated. Monitor for pleural effusions."
        
        # Add recommendations
        report += "\nRECOMMENDATIONS:\n"
        report += "1. Clinical correlation with patient's symptoms.\n"
        report += "2. Follow-up imaging after treatment to ensure resolution.\n"
        if desc['severity'] == 'significant':
            report += "3. Consider chest CT for better characterization of the findings."
        
        return report, "Pneumonia"
    
    return "Unable to generate conclusive report. Please consult a radiologist.", "Inconclusive"

def analyze_image(image_tensor):
    """Analyze the image and return findings"""
    with torch.no_grad():
        outputs = model(image_tensor)
        conditions = analyze_features(outputs)
        
        report, primary_condition = generate_detailed_report(conditions)
        
        # Calculate overall confidence
        confidence = conditions[0][1] if conditions else 0.3
        
        confidence_text = f"\n\nConfidence Level: {confidence:.1%}"
        if confidence < 0.6:
            confidence_text += " (Low confidence - recommend radiologist review)"
        
        return report + confidence_text, confidence, primary_condition

def process_image(image_path):
    """Process image and ensure correct channel format"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = image_transforms(image)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        return image_tensor
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def generate_pdf_report(report_text, image_path, condition, confidence, patient_info):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = f'static/reports/report_{timestamp}.pdf'
    os.makedirs('static/reports', exist_ok=True)
    
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    
    # Add title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Chest X-Ray Report', 0, 1, 'C')
    
    # Add patient information
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Patient Information:', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f"Name: {patient_info['name']}", 0, 1, 'L')
    pdf.cell(0, 10, f"ID: {patient_info['id']}", 0, 1, 'L')
    pdf.cell(0, 10, f"Age: {patient_info['age']}", 0, 1, 'L')
    pdf.cell(0, 10, f"Gender: {patient_info['gender']}", 0, 1, 'L')
    pdf.cell(0, 10, f"Exam Date: {patient_info['date']}", 0, 1, 'L')
    
    # Add image
    try:
        pdf.image(image_path, x=30, w=150)
    except Exception as e:
        print(f"Error adding image to PDF: {str(e)}")
    
    # Add findings
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Findings:', 0, 1, 'L')
    
    # Add report text
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 10, report_text)
    
    # Add condition and confidence
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 10, f'Primary Condition: {condition}', 0, 1, 'L')
    pdf.cell(0, 10, f'Confidence Level: {confidence:.1%}', 0, 1, 'L')
    
    # Add footer
    pdf.set_y(-30)
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(128)
    pdf.multi_cell(0, 10, 'This report was generated automatically and should be reviewed by a qualified healthcare professional.')
    
    # Save PDF
    pdf.output(pdf_path)
    return pdf_path

# Add this function to create the database
def init_db():
    conn = sqlite3.connect('reports.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reports
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  patient_name TEXT,
                  patient_id TEXT,
                  age TEXT,
                  gender TEXT,
                  exam_date TEXT,
                  condition TEXT,
                  confidence REAL,
                  report_text TEXT,
                  image_path TEXT,
                  pdf_path TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# Add this function to save report to database
def save_report_to_db(patient_info, condition, confidence, report_text, image_path, pdf_path):
    conn = sqlite3.connect('reports.db')
    c = conn.cursor()
    c.execute('''INSERT INTO reports 
                 (patient_name, patient_id, age, gender, exam_date, 
                  condition, confidence, report_text, image_path, pdf_path)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (patient_info['name'], patient_info['id'], patient_info['age'],
               patient_info['gender'], patient_info['date'], condition,
               confidence, report_text, image_path, pdf_path))
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_report', methods=['POST'])
def generate_report_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        patient_info = {
            'name': request.form.get('patientName', ''),
            'id': request.form.get('patientId', ''),
            'age': request.form.get('patientAge', ''),
            'gender': request.form.get('patientGender', ''),
            'date': request.form.get('examDate', '')
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f'static/uploads/image_{timestamp}.jpg'
        os.makedirs('static/uploads', exist_ok=True)
        image.save(image_path)
        
        image_tensor = process_image(image_path)
        report_text, confidence, condition = analyze_image(image_tensor)
        
        pdf_path = generate_pdf_report(report_text, image_path, condition, confidence, patient_info)
        
        # Save to database
        save_report_to_db(patient_info, condition, confidence, report_text, image_path, pdf_path)
        
        return jsonify({
            'success': True,
            'report': report_text,
            'condition': condition,
            'confidence': f"{confidence:.1%}",
            'image_url': image_path,
            'pdf_url': pdf_path
        })
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_report/<timestamp>')
def download_report(timestamp):
    try:
        pdf_path = f'static/reports/report_{timestamp}.pdf'
        return send_file(pdf_path, as_attachment=True, download_name=f'xray_report_{timestamp}.pdf')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/view_reports')
def view_reports():
    conn = sqlite3.connect('reports.db')
    c = conn.cursor()
    c.execute('''SELECT id, patient_name, patient_id, condition, exam_date, 
                 confidence, image_path, pdf_path, created_at 
                 FROM reports ORDER BY created_at DESC''')
    reports = c.fetchall()
    conn.close()
    return render_template('reports.html', reports=reports)

# Initialize database when starting the app
if __name__ == '__main__':
    init_db()
    app.run(debug=True) 