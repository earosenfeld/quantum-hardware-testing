import pandas as pd
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.linecharts import HorizontalLineChart
import matplotlib.pyplot as plt
import numpy as np
import io

def create_temperature_plot(temperatures, setpoint, title):
    """Create a temperature plot using matplotlib."""
    plt.figure(figsize=(8, 4))
    time = np.arange(len(temperatures)) * 0.1  # Time in seconds
    plt.plot(time, temperatures, 'b-', label='Temperature', linewidth=1)
    plt.axhline(y=setpoint, color='r', linestyle='--', label='Setpoint')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.title(title)
    plt.legend()
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

def generate_csv_report(results, output_file):
    """Generate a CSV report of test results."""
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

def generate_pdf_report(test_results, test_duration, setpoint, output_file):
    """Generate a detailed PDF report with improved aesthetics."""
    doc = SimpleDocTemplate(
        output_file,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Custom styles
    styles = getSampleStyleSheet()
    custom_styles = {
        'Title': ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2C3E50')
        ),
        'Subtitle': ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=20,
            textColor=colors.HexColor('#34495E')
        ),
        'Body': ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            textColor=colors.HexColor('#2C3E50')
        )
    }
    
    # Build the document
    story = []
    
    # Title
    story.append(Paragraph("Quantum Hardware Testing", custom_styles['Title']))
    story.append(Paragraph(f"Test Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", custom_styles['Subtitle']))
    story.append(Spacer(1, 20))
    
    # Summary
    story.append(Paragraph("Test Summary", custom_styles['Subtitle']))
    summary_data = [
        ["Total Tests", str(test_results['summary']['total_tests'])],
        ["Passed Tests", str(test_results['summary']['passed_tests'])],
        ["Pass Rate", f"{test_results['summary']['passed_tests']/test_results['summary']['total_tests']*100:.1f}%"],
        ["Total Duration", f"{test_duration:.1f} seconds"],
        ["Target Temperature", f"{setpoint}K"]
    ]
    summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ECF0F1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2C3E50')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#BDC3C7'))
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # Detailed Results
    story.append(Paragraph("Detailed Test Results", custom_styles['Subtitle']))
    for scenario in test_results['scenarios']:
        story.append(Paragraph(scenario['scenario_name'], custom_styles['Subtitle']))
        story.append(Paragraph(scenario['description'], custom_styles['Body']))
        
        # Add temperature plots for PID tests
        if scenario['scenario_name'] == "PID Controller Test" and 'data' in scenario:
            # Step Response Plot
            if 'step_response' in scenario['data']:
                buf = create_temperature_plot(
                    scenario['data']['step_response'],
                    setpoint,
                    'Step Response Test'
                )
                img = Image(buf, width=6*inch, height=3*inch)
                story.append(img)
                story.append(Spacer(1, 10))
            
            # Disturbance Rejection Plot
            if 'disturbance_rejection' in scenario['data']:
                buf = create_temperature_plot(
                    scenario['data']['disturbance_rejection'],
                    setpoint,
                    'Disturbance Rejection Test'
                )
                img = Image(buf, width=6*inch, height=3*inch)
                story.append(img)
                story.append(Spacer(1, 10))
        
        # Create results table
        data = [['Test', 'Result', 'Details']]
        for result in scenario['results']:
            status_color = colors.HexColor('#27AE60') if result['result'] == 'PASS' else colors.HexColor('#E74C3C')
            data.append([
                result['test'],
                Paragraph(f'<font color="{status_color}">{result["result"]}</font>', custom_styles['Body']),
                result['details']
            ])
        
        results_table = Table(data, colWidths=[2*inch, 1*inch, 3*inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#BDC3C7')),
            ('ALIGN', (2, 1), (2, -1), 'LEFT')
        ]))
        story.append(results_table)
        story.append(Spacer(1, 20))
    
    # Build the PDF
    doc.build(story) 