import os
import logging
import io
from typing import Dict, Any, Optional
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from googleapiclient.http import MediaFileUpload, MediaInMemoryUpload

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFDocumentHandler:
    """Class to convert JSON question data to a formatted PDF document and upload to Google Drive"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        # Modify existing styles
        self.styles['Heading1'].fontSize = 14
        self.styles['Heading1'].spaceAfter = 12
        
        self.styles['Heading2'].fontSize = 12
        self.styles['Heading2'].spaceAfter = 10
        
        self.styles['Normal'].fontSize = 11
        self.styles['Normal'].spaceAfter = 6
        
        # Add new styles with unique names
        self.styles.add(ParagraphStyle(
            name='Question',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            fontName='Helvetica-Bold'
        ))
        self.styles.add(ParagraphStyle(
            name='Answer',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=10,
            leftIndent=20
        ))
        
    def format_multiple_choice(self, questions):
        """Format multiple choice questions"""
        elements = []
        
        if not questions:
            return elements
            
        elements.append(Paragraph("Multiple Choice Questions", self.styles['Heading1']))
        elements.append(Spacer(1, 12))
        
        for i, question in enumerate(questions, 1):
            question_text = f"Question {i}: {question.get('question', 'N/A')}"
            elements.append(Paragraph(question_text, self.styles['Question']))
            
            options = question.get('options', [])
            if options:
                for j, option in enumerate(options):
                    option_letter = chr(65 + j)  # A, B, C, D...
                    option_text = f"{option_letter}. {option}"
                    elements.append(Paragraph(option_text, self.styles['Normal']))
            
            correct_answer = question.get('correct_answer')
            if correct_answer is not None:
                if isinstance(correct_answer, int) and 0 <= correct_answer < len(options):
                    correct_letter = chr(65 + correct_answer)
                    answer_text = f"Correct Answer: {correct_letter}"
                else:
                    answer_text = f"Correct Answer: {correct_answer}"
                elements.append(Paragraph(answer_text, self.styles['Answer']))
                    
            elements.append(Spacer(1, 12))  # Add spacing between questions
            
        return elements
    
    def format_short_answer(self, questions):
        """Format short answer questions"""
        elements = []
        
        if not questions:
            return elements
            
        elements.append(Paragraph("Short Answer Questions", self.styles['Heading1']))
        elements.append(Spacer(1, 12))
        
        for i, question in enumerate(questions, 1):
            question_text = f"Question {i}: {question.get('question', 'N/A')}"
            elements.append(Paragraph(question_text, self.styles['Question']))
            
            sample_answer = question.get('sample_answer')
            if sample_answer:
                elements.append(Paragraph("Sample Answer:", self.styles['Normal']))
                elements.append(Paragraph(sample_answer, self.styles['Answer']))
                
            elements.append(Spacer(1, 12))  # Add spacing between questions
            
        return elements
    
    def format_long_answer(self, questions):
        """Format long answer questions"""
        elements = []
        
        if not questions:
            return elements
            
        elements.append(Paragraph("Long Answer Questions", self.styles['Heading1']))
        elements.append(Spacer(1, 12))
        
        for i, question in enumerate(questions, 1):
            question_text = f"Question {i}: {question.get('question', 'N/A')}"
            elements.append(Paragraph(question_text, self.styles['Question']))
            
            sample_answer = question.get('sample_answer')
            if sample_answer:
                elements.append(Paragraph("Sample Answer:", self.styles['Normal']))
                elements.append(Paragraph(sample_answer, self.styles['Answer']))
                
            elements.append(Spacer(1, 12))  # Add spacing between questions
            
        return elements
    
    def format_case_studies(self, case_studies):
        """Format case study questions"""
        elements = []
        
        if not case_studies:
            return elements
            
        elements.append(Paragraph("Case Studies", self.styles['Heading1']))
        elements.append(Spacer(1, 12))
        
        for i, case in enumerate(case_studies, 1):
            elements.append(Paragraph(f"Case Study {i}", self.styles['Heading2']))
            
            scenario = case.get('scenario')
            if scenario:
                elements.append(Paragraph(scenario, self.styles['Normal']))
            
            questions = case.get('questions', [])
            for j, question in enumerate(questions, 1):
                question_text = f"Question {j}: {question.get('question', 'N/A')}"
                elements.append(Paragraph(question_text, self.styles['Question']))
                
                sample_answer = question.get('sample_answer')
                if sample_answer:
                    elements.append(Paragraph("Sample Answer:", self.styles['Normal']))
                    elements.append(Paragraph(sample_answer, self.styles['Answer']))
            
            elements.append(Spacer(1, 12))  # Add spacing between case studies
            
        return elements
    
    def generate_pdf_document(self, json_data: Dict[str, Any]) -> Optional[io.BytesIO]:
        """Generate a PDF document from the JSON question data and return as BytesIO object"""
        try:
            # Create BytesIO object to store PDF in memory
            buffer = io.BytesIO()
            
            # Create the document
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            elements = []
            
            # Add title
            elements.append(Paragraph("Generated Exam Questions", self.styles['Title']))
            elements.append(Spacer(1, 12))
            
            # Add introduction
            elements.append(Paragraph("This document contains automatically generated exam questions based on the provided content.", self.styles['Normal']))
            elements.append(Spacer(1, 12))
            
            # Format each section
            elements.extend(self.format_multiple_choice(json_data.get('multiple_choice', [])))
            elements.extend(self.format_short_answer(json_data.get('short_answer', [])))
            elements.extend(self.format_long_answer(json_data.get('long_answer', [])))
            elements.extend(self.format_case_studies(json_data.get('case_studies', [])))
            
            # Build the PDF
            doc.build(elements)
            
            # Reset the buffer position to the beginning
            buffer.seek(0)
            logger.info("PDF document successfully generated in memory")
            return buffer
            
        except Exception as e:
            logger.error(f"Error generating PDF document: {e}")
            return None

    def upload_pdf_to_drive(self, drive_service, pdf_buffer: io.BytesIO, filename: str, 
                           folder_id: str) -> Optional[str]:
        """Upload the PDF to Google Drive in the specified folder"""
        try:
            file_metadata = {
                'name': filename,
                'mimeType': 'application/pdf',
                'parents': [folder_id]
            }
            
            # Create media object from BytesIO buffer
            media = MediaInMemoryUpload(
                pdf_buffer.getvalue(),
                mimetype='application/pdf',
                resumable=True
            )
            
            # Upload the file
            file = drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            file_id = file.get('id')
            if file_id:
                logger.info(f"PDF successfully uploaded to Google Drive with ID: {file_id}")
                return file_id
            else:
                logger.error("File upload failed: No file ID returned")
                return None
                
        except Exception as e:
            logger.error(f"Error uploading PDF to Google Drive: {e}")
            return None
            
    def generate_and_upload_pdf(self, drive_service, json_data: Dict[str, Any], 
                               filename: str, folder_id: str) -> Optional[str]:
        """Generate PDF and upload to Google Drive in one operation"""
        try:
            # Generate PDF in memory
            pdf_buffer = self.generate_pdf_document(json_data)
            if not pdf_buffer:
                logger.error("Failed to generate PDF document")
                return None
                
            # Upload to Google Drive
            file_id = self.upload_pdf_to_drive(drive_service, pdf_buffer, filename, folder_id)
            if not file_id:
                logger.error("Failed to upload PDF to Google Drive")
                return None
                
            # Close the buffer
            pdf_buffer.close()
            
            return file_id
            
        except Exception as e:
            logger.error(f"Error in generate and upload process: {e}")
            return None
