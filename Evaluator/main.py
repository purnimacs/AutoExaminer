import os
import argparse
import json
from typing import Dict, Any, List, Optional
import re

# Import modules
from ocr import AzureOCR
from google_drive import authenticate_google_drive, find_answer_keys, download_file, list_files_in_folder, upload_results_to_drive
from answer_key_extractor import AnswerKeyExtractor
from answer_evaluator import AnswerEvaluator
import config

class AnswerSheetProcessor:
    def __init__(self, azure_key=None, azure_endpoint=None, gemini_key=None):
        """
        Initialize the answer sheet processing system.
        
        Args:
            azure_key: Azure OCR API key (optional, defaults to environment variable)
            azure_endpoint: Azure OCR endpoint (optional, defaults to environment variable)
            gemini_key: Gemini API key (optional, defaults to config)
        """
        # Initialize OCR client
        self.azure_key = azure_key or config.AZURE_VISION_KEY  # Changed from os.environ.get("AZURE_OCR_KEY")
        self.azure_endpoint = azure_endpoint or config.AZURE_VISION_ENDPOINT
        
        if not self.azure_key or not self.azure_endpoint:
            print("Warning: Azure OCR credentials not provided. OCR functionality will be limited.")
        else:
            self.ocr_client = AzureOCR(self.azure_key, self.azure_endpoint)
        
        # Initialize Gemini evaluator
        self.gemini_key = gemini_key or config.GEMINI_API_KEY
        self.evaluator = AnswerEvaluator(self.gemini_key)
        
        # Authenticate with Google Drive
        try:
            self.drive_service = authenticate_google_drive()
            print("Successfully authenticated with Google Drive")
        except Exception as e:
            print(f"Error authenticating with Google Drive: {e}")
            self.drive_service = None
        
        # Create temp directory if it doesn't exist
        os.makedirs(config.TEMP_DIR, exist_ok=True)
    
    def process_answer_sheets(self, answer_sheets_folder_id=None, answer_key_folder_id=None, results_folder_id=None):
        """
        Process all answer sheets in the specified Google Drive folder.
        
        Args:
            answer_sheets_folder_id: Google Drive folder ID containing answer sheets
            answer_key_folder_id: Google Drive folder ID containing answer keys
            results_folder_id: Google Drive folder ID to store results
        """
        # Use config values if not provided
        answer_sheets_folder_id = answer_sheets_folder_id or config.ANSWER_SHEETS_FOLDER_ID
        answer_key_folder_id = answer_key_folder_id or config.ANSWER_KEY_FOLDER_ID
        results_folder_id = results_folder_id or config.RESULTS_FOLDER_ID
        
        if not self.drive_service:
            print("Google Drive service not initialized. Cannot process sheets.")
            return
        
        # Step 1: Find and download answer key
        print("\n=== Finding Answer Keys ===")
        answer_keys = find_answer_keys(self.drive_service, answer_key_folder_id)
        
        if not answer_keys:
            print("No answer keys found. Aborting.")
            return
        
        # Let user select which answer key to use if multiple are found
        if len(answer_keys) > 1:
            print("\nMultiple answer keys found. Please select one:")
            for i, key in enumerate(answer_keys):
                print(f"{i+1}. {key['name']}")
            
            selection = input("\nEnter number: ")
            try:
                selected_key = answer_keys[int(selection) - 1]
            except (ValueError, IndexError):
                print("Invalid selection. Using the first answer key.")
                selected_key = answer_keys[0]
        else:
            selected_key = answer_keys[0]
        
        print(f"\nUsing answer key: {selected_key['name']}")
        
        # Download the selected answer key
        answer_key_path = os.path.join(config.TEMP_DIR, selected_key['name'])
        download_file(self.drive_service, selected_key['id'], answer_key_path)
        
        # Extract structured answer key
        print("\n=== Extracting Answer Key ===")
        extractor = AnswerKeyExtractor(answer_key_path)
        answer_key_data = extractor.extract_answers()
        
        # Save extracted answer key for reference
        answer_key_json_path = os.path.join(config.TEMP_DIR, "extracted_answer_key.json")
        extractor.save_to_json(answer_key_json_path)
        
        # Step 2: List answer sheets to process
        print("\n=== Finding Answer Sheets ===")
        answer_sheets = list_files_in_folder(self.drive_service, answer_sheets_folder_id)
        
        if not answer_sheets:
            print("No answer sheets found. Aborting.")
            return
        
        # Process each answer sheet
        all_results = []
        for i, sheet in enumerate(answer_sheets):
            print(f"\n=== Processing Answer Sheet {i+1}/{len(answer_sheets)}: {sheet['name']} ===")
            
            # Download the answer sheet
            sheet_path = os.path.join(config.TEMP_DIR, sheet['name'])
            download_file(self.drive_service, sheet['id'], sheet_path)
            
            # Extract student ID or name from filename
            student_id = os.path.splitext(sheet['name'])[0]
            
            # Process based on file type
            file_extension = os.path.splitext(sheet_path)[1].lower()
            
            if file_extension in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.pdf']:
                # Use OCR for image or PDF files
                if not hasattr(self, 'ocr_client'):
                    print("OCR client not initialized. Skipping this sheet.")
                    continue
                
                print("Performing OCR on answer sheet...")
                try:
                    ocr_text = self.ocr_client.recognize_handwriting(sheet_path)
                    
                    # Save full OCR output for reference and debugging
                    full_ocr_output_path = os.path.join(config.TEMP_DIR, f"{student_id}_full_ocr.txt")
                    with open(full_ocr_output_path, 'w', encoding='utf-8') as f:
                        f.write(ocr_text)
                    
                    # Extract answers from OCR text
                    student_answers = self.extract_answers_from_ocr(ocr_text)
                    
                    # Save extracted answers for debugging
                    answers_output_path = os.path.join(config.TEMP_DIR, f"{student_id}_extracted_answers.json")
                    with open(answers_output_path, 'w', encoding='utf-8') as f:
                        json.dump(student_answers, f, indent=2)
                    
                    print(f"Extracted {len(student_answers)} answers from OCR")
                    
                except Exception as e:
                    print(f"Error during OCR processing: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            else:
                # For other file types, we might need different processing
                print(f"Unsupported file type: {file_extension}. Skipping.")
                continue
            
            # Evaluate student answers
            print("Evaluating answers...")
            try:
                evaluation_results = self.evaluator.evaluate_answers(student_answers, answer_key_data)
                
                # Add student identifier
                evaluation_results["student_id"] = student_id
                all_results.append(evaluation_results)
                
                # Save individual evaluation results
                results_json_path = os.path.join(config.TEMP_DIR, f"{student_id}_evaluation.json")
                with open(results_json_path, 'w') as f:
                    json.dump(evaluation_results, f, indent=2)
                
                # Also save as CSV for easier viewing
                results_csv_path = os.path.join(config.TEMP_DIR, f"{student_id}_evaluation.csv")
                self.evaluator.generate_csv_results(evaluation_results, results_csv_path)
                
                # Upload results to Google Drive
                if results_folder_id:
                    print("Uploading results to Google Drive...")
                    upload_results_to_drive(
                        self.drive_service, 
                        results_folder_id, 
                        results_csv_path, 
                        f"{student_id}_results.csv"
                    )
                
                print(f"Evaluation complete: Score {evaluation_results['summary']['total_score']}/{evaluation_results['summary']['total_possible']} ({evaluation_results['summary']['percentage']}%)")
                
            except Exception as e:
                print(f"Error during evaluation: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Generate and upload consolidated results
        if all_results:
            self._generate_consolidated_results(all_results, results_folder_id)
    
    def extract_answers_from_ocr(self, ocr_text):
        """
        Extract question numbers and answers from OCR text.
        
        Args:
            ocr_text (str): Text extracted using OCR
            
        Returns:
            Dict[str, str]: Dictionary mapping question numbers to answers
        """
        answers = {}
        current_question = None
        current_answer_lines = []
        
        # Split text into lines for processing
        lines = ocr_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line starts a new question
            # Look for patterns like "1.", "1)", "Question 1", etc.
            question_match = re.search(r'^(?:Question\s*)?(\d+)[\.\)]?\s*$', line)
            
            if question_match:
                # If we were processing a previous question, save it
                if current_question and current_answer_lines:
                    answers[current_question] = '\n'.join(current_answer_lines).strip()
                
                # Start a new question
                current_question = question_match.group(1)
                current_answer_lines = []
            elif current_question:
                # Add this line to the current answer
                current_answer_lines.append(line)
        
        # Don't forget the last question
        if current_question and current_answer_lines:
            answers[current_question] = '\n'.join(current_answer_lines).strip()
        
        # Print for debugging purposes
        for q_num, answer in answers.items():
            print(f"Question {q_num}, length: {len(answer)} chars, preview: {answer[:50]}...")
        
        return answers
    
    def _generate_consolidated_results(self, all_results: List[Dict[str, Any]], results_folder_id: str) -> None:
        """
        Generate consolidated results for all processed answer sheets.
        
        Args:
            all_results: List of evaluation results for each student
            results_folder_id: Google Drive folder ID to store results
        """
        print("\n=== Generating Consolidated Results ===")
        
        # Create consolidated CSV
        consolidated_csv_path = os.path.join(config.TEMP_DIR, "consolidated_results.csv")
        
        import csv
        with open(consolidated_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            header = ['Student ID', 'Total Score', 'Total Possible', 'Percentage']
            
            # Add question-specific headers - assume all students have the same questions
            if all_results and 'questions' in all_results[0]:
                for q_num in sorted(all_results[0]['questions'].keys(), key=lambda x: int(x)):
                    header.append(f'Q{q_num}')
            
            writer.writerow(header)
            
            # Write data for each student
            for result in all_results:
                row = [
                    result.get('student_id', 'Unknown'),
                    result['summary']['total_score'],
                    result['summary']['total_possible'],
                    f"{result['summary']['percentage']}%"
                ]
                
                # Add question-specific scores
                if 'questions' in result:
                    for q_num in sorted(result['questions'].keys(), key=lambda x: int(x)):
                        q_data = result['questions'][q_num]
                        row.append(f"{q_data['score']}/{q_data['max_score']}")
                
                writer.writerow(row)
        
        print(f"Consolidated results saved to {consolidated_csv_path}")
        
        # Upload consolidated results to Google Drive
        if self.drive_service and results_folder_id:
            print("Uploading consolidated results to Google Drive...")
            upload_results_to_drive(
                self.drive_service,
                results_folder_id,
                consolidated_csv_path,
                "consolidated_results.csv"
            )


def main():
    """
    Main function to run the answer sheet processing system.
    """
    parser = argparse.ArgumentParser(description='Answer Sheet Evaluation System')
    parser.add_argument('--azure-key', help='Azure OCR API key')
    parser.add_argument('--azure-endpoint', help='Azure OCR endpoint')
    parser.add_argument('--gemini-key', help='Gemini API key')
    parser.add_argument('--answer-sheets', help='Google Drive folder ID containing answer sheets')
    parser.add_argument('--answer-keys', help='Google Drive folder ID containing answer keys')
    parser.add_argument('--results', help='Google Drive folder ID to store results')
    
    args = parser.parse_args()
    
    # Create and run the processor
    processor = AnswerSheetProcessor(
        azure_key=args.azure_key,
        azure_endpoint=args.azure_endpoint,
        gemini_key=args.gemini_key
    )
    
    processor.process_answer_sheets(
        answer_sheets_folder_id=args.answer_sheets,
        answer_key_folder_id=args.answer_keys,
        results_folder_id=args.results
    )


if __name__ == "__main__":
    main()