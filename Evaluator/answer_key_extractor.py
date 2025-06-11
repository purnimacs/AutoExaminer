import os
import re
import fitz  # PyMuPDF
import json
from typing import Dict, List, Any, Tuple, Optional


class AnswerKeyExtractor:
    def __init__(self, file_path: str):
        """
        Initialize the answer key extractor with the path to the answer key PDF.
        
        Args:
            file_path (str): Path to the answer key PDF file
        """
        self.file_path = file_path
        self.answer_key = {}
        
    def extract_answers(self) -> Dict[str, Any]:
        """
        Extract answers from the PDF file and structure them for LLM evaluation.
        
        Returns:
            Dict containing structured answer key information
        """
        # Check if file exists
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Answer key file not found: {self.file_path}")
            
        # Check file extension
        _, ext = os.path.splitext(self.file_path)
        if ext.lower() != '.pdf':
            raise ValueError(f"Expected PDF file, got {ext} instead")
        
        # Open the PDF file
        doc = fitz.open(self.file_path)
        text = ""
        
        # Extract text from all pages
        for page in doc:
            text += page.get_text()
        
        # Process the extracted text
        structured_answers = self._parse_answer_key(text)
        
        return structured_answers
    
    def _parse_answer_key(self, text: str) -> Dict[str, Any]:
        """
        Parse the extracted text to identify questions and answers.
        
        Args:
            text (str): The extracted text from the PDF
            
        Returns:
            Dict containing structured answer key information
        """
        # Initialize the result dictionary
        result = {
            "questions": {},
            "metadata": {
                "total_questions": 0,
                "total_marks": 0
            }
        }
        
        # Split text into lines for processing
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        current_question = None
        current_subquestion = None
        answer_text_buffer = []
        import re
        # Regular expressions for pattern matching
        question_pattern = re.compile(r'^(\d+)\.(.*)$')
        # Modified MCQ pattern to require brackets after the letter choice
        mcq_pattern = re.compile(r'^(\d+)\.\s*([A-D])\s*\((.*)\)$')
        subquestion_pattern = re.compile(r'^([a-z])\)(.*)|^([a-z]\))(.*)$')
        marks_pattern = re.compile(r'\((\d+)\)$|\((\d+)\s*marks?\)$')
        
        for line in lines:
            # Try to match an MCQ answer pattern (e.g., "1. A (explanation)" or "1.A(explanation)")
            mcq_match = mcq_pattern.match(line)
            if mcq_match:
                q_num = mcq_match.group(1)
                answer = mcq_match.group(2)
                explanation = mcq_match.group(3).strip()
                
                # Look for marks pattern at the end
                marks_match = marks_pattern.search(explanation)
                marks = int(marks_match.group(1) or marks_match.group(2)) if marks_match else 1
                
                result["questions"][q_num] = {
                    "type": "mcq",
                    "answer": answer,
                    "marks": marks,
                    "full_text": line
                }
                result["metadata"]["total_questions"] += 1
                result["metadata"]["total_marks"] += marks
                current_question = q_num
                current_subquestion = None
                answer_text_buffer = [explanation]
                continue
            
            # Check for question number pattern (e.g., "13.")
            q_match = question_pattern.match(line)
            if q_match:
                # If we were processing a previous question, save its buffer
                if current_question and answer_text_buffer:
                    if current_question in result["questions"]:
                        if current_subquestion:
                            if "subquestions" in result["questions"][current_question]:
                                result["questions"][current_question]["subquestions"][current_subquestion]["answer_text"] = '\n'.join(answer_text_buffer)
                        else:
                            result["questions"][current_question]["answer_text"] = '\n'.join(answer_text_buffer)
                
                q_num = q_match.group(1)
                rest_of_line = q_match.group(2).strip()
                
                # Look for marks pattern at the end
                marks_match = marks_pattern.search(rest_of_line)
                marks = int(marks_match.group(1) or marks_match.group(2)) if marks_match else 1
                
                # Create new question entry
                result["questions"][q_num] = {
                    "type": "descriptive",
                    "marks": marks,
                    "key_points": [],
                    "full_text": line
                }
                result["metadata"]["total_questions"] += 1
                result["metadata"]["total_marks"] += marks
                current_question = q_num
                current_subquestion = None
                answer_text_buffer = [rest_of_line]
                continue
            
            # Check for subquestion pattern (e.g., "a)" or "a)")
            sub_match = subquestion_pattern.match(line)
            if sub_match and current_question:
                # Save buffer for previous subquestion if any
                if current_subquestion and answer_text_buffer:
                    if "subquestions" in result["questions"][current_question]:
                        result["questions"][current_question]["subquestions"][current_subquestion]["answer_text"] = '\n'.join(answer_text_buffer)
                
                sub_letter = sub_match.group(1) or sub_match.group(3)
                rest_of_line = (sub_match.group(2) or sub_match.group(4)).strip()
                
                # Look for marks pattern at the end
                marks_match = marks_pattern.search(rest_of_line)
                marks = int(marks_match.group(1) or marks_match.group(2)) if marks_match else 1
                
                # Initialize subquestions dictionary if not exists
                if "subquestions" not in result["questions"][current_question]:
                    result["questions"][current_question]["subquestions"] = {}
                    result["questions"][current_question]["type"] = "composite"
                
                # Add the subquestion
                result["questions"][current_question]["subquestions"][sub_letter] = {
                    "marks": marks,
                    "key_points": [],
                    "full_text": line
                }
                
                # Update total marks
                result["metadata"]["total_marks"] += marks - result["questions"][current_question]["marks"]
                result["questions"][current_question]["marks"] = 0  # To avoid double counting
                
                current_subquestion = sub_letter
                answer_text_buffer = [rest_of_line]
                continue
            
            # If none of the above, append to the current answer text buffer
            if current_question:
                answer_text_buffer.append(line)
        
        # Process the last question/subquestion buffer
        if current_question and answer_text_buffer:
            if current_subquestion and "subquestions" in result["questions"][current_question]:
                result["questions"][current_question]["subquestions"][current_subquestion]["answer_text"] = '\n'.join(answer_text_buffer)
            else:
                result["questions"][current_question]["answer_text"] = '\n'.join(answer_text_buffer)
        
        # Extract key points from answer texts
        self._extract_key_points(result)
        
        return result
    
    def _extract_key_points(self, result: Dict[str, Any]) -> None:
        """
        Extract key points from answer texts for LLM evaluation.
        
        Args:
            result (Dict): The structured answer key dictionary
        """
        for q_num, q_data in result["questions"].items():
            if q_data["type"] == "mcq":
                # MCQ doesn't need key points extraction
                continue
                
            elif q_data["type"] == "descriptive":
                if "answer_text" in q_data:
                    # Simple key point extraction by sentences or comma-separated parts
                    answer_text = q_data["answer_text"]
                    points = [p.strip() for p in re.split(r'[.;](?=\s|$)', answer_text) if p.strip()]
                    q_data["key_points"] = points
            
            elif q_data["type"] == "composite":
                for sub_letter, sub_data in q_data["subquestions"].items():
                    if "answer_text" in sub_data:
                        # Extract key points for each subquestion
                        answer_text = sub_data["answer_text"]
                        points = [p.strip() for p in re.split(r'[.;](?=\s|$)', answer_text) if p.strip()]
                        sub_data["key_points"] = points
    
    def save_to_json(self, output_path: str) -> None:
        """
        Save the extracted answer key to a JSON file.
        
        Args:
            output_path (str): Path to save the JSON file
        """
        if not self.answer_key:
            self.answer_key = self.extract_answers()
            
        with open(output_path, 'w') as f:
            json.dump(self.answer_key, f, indent=2)
        
        print(f"Answer key saved to {output_path}")
        
    def get_answer_key(self) -> Dict[str, Any]:
        """
        Get the structured answer key.
        
        Returns:
            Dict containing the structured answer key
        """
        if not self.answer_key:
            self.answer_key = self.extract_answers()
            
        return self.answer_key
