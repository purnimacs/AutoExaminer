import os
import json
import google.generativeai as genai
from typing import Dict, Any, List, Tuple
import re
class AnswerEvaluator:
    def __init__(self, api_key: str):
        """
        Initialize the answer evaluator with a Gemini API key.
        
        Args:
            api_key (str): Gemini API key
        """
        self.api_key = api_key
        genai.configure(api_key=api_key)
        
        # Configure the model
        generation_config = {
            "temperature": 0.2,  # Lower temperature for more consistent outputs
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }
        
        # Get Gemini Pro model
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=generation_config
        )
    
    def evaluate_answers(self, 
                         student_answers: Dict[str, str], 
                         answer_key: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate student answers against the answer key using a single API call.
        
        Args:
            student_answers (Dict): Dictionary of student answers {question_num: answer_text}
            answer_key (Dict): Structured answer key
            
        Returns:
            Dict containing evaluation results with scores and feedback
        """
        results = {
            "questions": {},
            "summary": {
                "total_score": 0,
                "total_possible": answer_key["metadata"]["total_marks"],
                "percentage": 0
            }
        }
        
        # First, handle MCQ questions separately as they don't need LLM
        mcq_results = {}
        descriptive_questions = []
        
        for q_num, q_data in answer_key["questions"].items():
            # Skip if student didn't answer this question
            if q_num not in student_answers:
                results["questions"][q_num] = {
                    "score": 0,
                    "max_score": q_data["marks"] if q_data["type"] != "composite" else sum(sub["marks"] for sub in q_data["subquestions"].values()),
                    "feedback": "Question not attempted",
                    "answer_text": "",
                }
                continue
                
            student_answer = student_answers[q_num]
            
            # Handle MCQ questions directly
            if q_data["type"] == "mcq":
                score, feedback = self._evaluate_mcq(student_answer, q_data)
                mcq_results[q_num] = {
                    "score": score,
                    "max_score": q_data["marks"],
                    "feedback": feedback,
                    "answer_text": student_answer
                }
                results["summary"]["total_score"] += score
            elif q_data["type"] == "descriptive":
                descriptive_questions.append({
                    "q_num": q_num,
                    "type": "descriptive",
                    "student_answer": student_answer,
                    "answer_key_data": q_data
                })
            elif q_data["type"] == "composite":
                # For composite questions, collect all subquestions
                subquestion_answers = self._parse_subquestions(student_answer)
                
                for sub_letter, sub_data in q_data["subquestions"].items():
                    sub_answer = subquestion_answers.get(sub_letter, student_answer)
                    descriptive_questions.append({
                        "q_num": q_num,
                        "sub_letter": sub_letter,
                        "type": "composite_sub",
                        "student_answer": sub_answer,
                        "answer_key_data": sub_data,
                        "partial": sub_letter not in subquestion_answers
                    })
        
        # Now evaluate all descriptive questions in one API call if there are any
        if descriptive_questions:
            descriptive_results = self._batch_evaluate_descriptive(descriptive_questions)
            
            # Process descriptive results
            for q_num, eval_result in descriptive_results.items():
                if "_" in q_num:  # This is a subquestion
                    main_q, sub_letter = q_num.split("_")
                    if main_q not in results["questions"]:
                        results["questions"][main_q] = {
                            "score": 0,
                            "max_score": 0,
                            "subquestions": {},
                            "answer_text": student_answers[main_q]
                        }
                    
                    results["questions"][main_q]["subquestions"][sub_letter] = {
                        "score": eval_result["score"],
                        "max_score": eval_result["max_score"],
                        "feedback": eval_result["feedback"]
                    }
                    
                    results["questions"][main_q]["score"] += eval_result["score"]
                    results["questions"][main_q]["max_score"] += eval_result["max_score"]
                    results["summary"]["total_score"] += eval_result["score"]
                else:
                    results["questions"][q_num] = eval_result
                    results["summary"]["total_score"] += eval_result["score"]
        
        # Add MCQ results to final results
        for q_num, mcq_result in mcq_results.items():
            results["questions"][q_num] = mcq_result
        
        # Calculate percentage
        if results["summary"]["total_possible"] > 0:
            results["summary"]["percentage"] = round(
                (results["summary"]["total_score"] / results["summary"]["total_possible"]) * 100, 2
            )
        
        return results
    
    def _batch_evaluate_descriptive(self, questions: List[Dict]) -> Dict[str, Dict]:
        """
        Evaluate multiple descriptive questions in a single API call.
        
        Args:
            questions: List of question dictionaries containing question info
            
        Returns:
            Dictionary of evaluation results {question_id: result_dict}
        """
        if not questions:
            return {}
        
        # Build a single prompt for all questions
        prompt = """
        You are an expert evaluator for student answers. Your task is to grade multiple answers and provide constructive feedback.
        
        EVALUATION CRITERIA:
        - Accuracy: Does the answer contain correct information?
        - Completeness: Does the answer address all key points?
        - Clarity: Is the answer clear and well-structured?
        - Leniency: Be lenient in scoring and give them partial marks if they included key points in the answer.
        
        Below are the questions and answers to evaluate. Please provide scores and feedback for each.
        
        """
        
        for i, q in enumerate(questions):
            q_id = q["q_num"]
            if q["type"] == "composite_sub":
                q_id = f"{q_id}_{q['sub_letter']}"
            
            # Get key points
            if "key_points" in q["answer_key_data"] and q["answer_key_data"]["key_points"]:
                key_points_text = "\n".join([f"- {point}" for point in q["answer_key_data"]["key_points"]])
            else:
                key_points_text = q["answer_key_data"].get("answer_text", "No model answer provided")
            
            max_score = q["answer_key_data"]["marks"]
            
            prompt += f"""
            QUESTION {i+1} (ID: {q_id}):
            
            MODEL ANSWER KEY POINTS:
            {key_points_text}
            
            STUDENT ANSWER:
            {q["student_answer"]}
            
            TOTAL MARKS AVAILABLE: {max_score}
            
            """
            
            if q["type"] == "composite_sub" and q.get("partial", False):
                prompt += "Note: The student answer might contain information about multiple parts of a question. Focus only on evaluating the specific part related to the key points provided.\n"
        
        prompt += """
        Return your evaluation in this exact JSON format:
        {
            "question_id_1": {
                "score": [numerical score],
                "max_score": [maximum possible score],
                "feedback": "[concise feedback explaining the score]"
            },
            "question_id_2": {
                "score": [numerical score],
                "max_score": [maximum possible score],
                "feedback": "[concise feedback explaining the score]"
            },
            ...and so on for each question
        }
        
        Use the question IDs exactly as provided in the QUESTION sections above.
        """
        
        # Make a single API call
        try:
            response = self.model.generate_content(prompt)
            
            # Extract JSON from response
            json_match = re.search(r'({.*})', response.text, re.DOTALL)
            
            if json_match:
                try:
                    evaluation = json.loads(json_match.group(1))
                    
                    # Process and validate results
                    results = {}
                    for q in questions:
                        q_id = q["q_num"]
                        if q["type"] == "composite_sub":
                            q_id = f"{q_id}_{q['sub_letter']}"
                        
                        max_score = q["answer_key_data"]["marks"]
                        
                        if q_id in evaluation:
                            score = min(float(evaluation[q_id]["score"]), max_score)  # Ensure score doesn't exceed max
                            results[q_id] = {
                                "score": score,
                                "max_score": max_score,
                                "feedback": evaluation[q_id]["feedback"],
                                "answer_text": q["student_answer"]
                            }
                        else:
                            # Fallback for missing questions
                            results[q_id] = self._estimate_score_with_result(
                                q["student_answer"], 
                                q["answer_key_data"], 
                                max_score
                            )
                    
                    return results
                    
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error parsing LLM response: {e}")
                    # Fallback to individual evaluation
                    return self._fallback_individual_evaluation(questions)
            else:
                print("Could not extract JSON from LLM response")
                return self._fallback_individual_evaluation(questions)
                
        except Exception as e:
            print(f"Error during batch LLM evaluation: {e}")
            return self._fallback_individual_evaluation(questions)
    
    def _fallback_individual_evaluation(self, questions: List[Dict]) -> Dict[str, Dict]:
        """
        Fallback method to evaluate questions individually using keyword matching.
        """
        results = {}
        
        for q in questions:
            q_id = q["q_num"]
            if q["type"] == "composite_sub":
                q_id = f"{q_id}_{q['sub_letter']}"
            
            max_score = q["answer_key_data"]["marks"]
            result = self._estimate_score_with_result(
                q["student_answer"], 
                q["answer_key_data"], 
                max_score
            )
            results[q_id] = result
        
        return results
    
    def _evaluate_mcq(self, student_answer: str, answer_key_data: Dict[str, Any]) -> Tuple[int, str]:
        """
        Evaluate a multiple choice question.
        
        Args:
            student_answer (str): Student's answer text
            answer_key_data (Dict): Answer key data for this question
            
        Returns:
            Tuple of (score, feedback)
        """
        correct_answer = answer_key_data["answer"]
        max_score = answer_key_data["marks"]
        
        # Extract the student's selected option (A, B, C, D)
        # Looking for patterns like "A", "A)", "(A)", "A."
        option_match = re.search(r'[(\s]?([A-Da-d])[)\s\.]?', student_answer)

        # And then make sure to uppercase the extracted option for comparison
        if option_match:
            student_option = option_match.group(1).upper()  # Convert to uppercase for comparison
            if student_option == correct_answer:
                return max_score, "Correct answer."
            else:
                return 0, f"Incorrect answer. You selected {student_option}, but the correct answer is {correct_answer}."
        else:
            return 0, f"Could not identify a clear option selection. The correct answer is {correct_answer}."
    
    def _parse_subquestions(self, answer_text: str) -> Dict[str, str]:
        """
        Parse a student answer to identify subquestion responses.
        
        Args:
            answer_text (str): Student's full answer text
            
        Returns:
            Dict mapping subquestion letters to answer text
        """
        subquestions = {}
        
        # Split by subquestion markers
        lines = answer_text.split('\n')
        current_sub = None
        current_text = []
        
        for line in lines:
            # Check for subquestion markers like "a)", "b." etc.
            sub_match = re.match(r'^[(\s]*([a-z])[)\s\.](.*)$', line.strip())
            
            if sub_match:
                # If we were collecting text for a previous subquestion, save it
                if current_sub and current_text:
                    subquestions[current_sub] = '\n'.join(current_text).strip()
                
                # Start new subquestion
                current_sub = sub_match.group(1).lower()
                current_text = [sub_match.group(2).strip()]
            elif current_sub:
                # Continue collecting text for current subquestion
                current_text.append(line)
                
        # Save the last subquestion
        if current_sub and current_text:
            subquestions[current_sub] = '\n'.join(current_text).strip()
            
        return subquestions
    
    def _estimate_score_with_result(self, 
                                   student_answer: str, 
                                   answer_key_data: Dict[str, Any],
                                   max_score: int) -> Dict[str, Any]:
        """
        Estimate score based on keyword matching and return result dictionary.
        
        Args:
            student_answer (str): Student's answer text
            answer_key_data (Dict): Answer key data for this question
            max_score (int): Maximum possible score
            
        Returns:
            Dict with score, max_score, feedback, and answer_text
        """
        score, feedback = self._estimate_score(student_answer, answer_key_data, max_score)
        
        return {
            "score": score,
            "max_score": max_score,
            "feedback": feedback,
            "answer_text": student_answer
        }
    
    def _estimate_score(self, 
                       student_answer: str, 
                       answer_key_data: Dict[str, Any],
                       max_score: int) -> Tuple[float, str]:
        """
        Fallback method to estimate score based on keyword matching.
        
        Args:
            student_answer (str): Student's answer text
            answer_key_data (Dict): Answer key data for this question
            max_score (int): Maximum possible score
            
        Returns:
            Tuple of (score, feedback)
        """
        # Simple keyword matching as a fallback
        student_answer = student_answer.lower()
        
        key_points = answer_key_data.get("key_points", [])
        if not key_points and "answer_text" in answer_key_data:
            # Extract basic key points from answer text if not provided
            key_points = [p.strip().lower() for p in re.split(r'[.;](?=\s|$)', answer_key_data["answer_text"]) if p.strip()]
        
        matched_points = 0
        
        for point in key_points:
            # Look for key words and phrases in the student answer
            keywords = [word.strip().lower() for word in re.findall(r'\b\w{4,}\b', point) if len(word) > 3]
            if not keywords:
                continue
                
            # Count how many keywords are present in the student answer
            matches = sum(1 for keyword in keywords if keyword in student_answer)
            if matches / len(keywords) > 0.5:  # If more than half of keywords are present
                matched_points += 1
        
        if not key_points:
            score = 0
            feedback = "Unable to evaluate answer due to missing reference points."
        else:
            # Calculate score based on percentage of matched key points
            score = round((matched_points / len(key_points)) * max_score, 1)
            
            if matched_points == len(key_points):
                feedback = "Answer addresses all key points correctly."
            elif matched_points > len(key_points) / 2:
                feedback = "Answer addresses many key points but is missing some important details."
            else:
                feedback = "Answer is missing several key points or contains inaccuracies."
        
        return score, feedback
    
    def save_results_to_json(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save evaluation results to a JSON file.
        
        Args:
            results (Dict): Evaluation results
            output_path (str): Path to save the JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Evaluation results saved to {output_path}")
    
    def generate_csv_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Convert results to CSV format and save.
        
        Args:
            results (Dict): Evaluation results
            output_path (str): Path to save the CSV file
        """
        import csv
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['Question', 'Score', 'Max Score', 'Percentage', 'Feedback'])
            
            # Write data for each question
            for q_num, q_data in results["questions"].items():
                if "subquestions" in q_data:
                    # Write the main question as a header
                    writer.writerow([f"Question {q_num}", q_data["score"], q_data["max_score"], 
                                    f"{round((q_data['score']/q_data['max_score'])*100, 1)}%", ""])
                    
                    # Write each subquestion
                    for sub_letter, sub_data in q_data["subquestions"].items():
                        writer.writerow([f"    {q_num}{sub_letter}", sub_data["score"], sub_data["max_score"],
                                        f"{round((sub_data['score']/sub_data['max_score'])*100, 1)}%", 
                                        sub_data["feedback"]])
                else:
                    writer.writerow([f"Question {q_num}", q_data["score"], q_data["max_score"],
                                    f"{round((q_data['score']/q_data['max_score'])*100, 1)}%", 
                                    q_data["feedback"]])
            
            # Write summary
            writer.writerow([])
            writer.writerow(['TOTAL', results["summary"]["total_score"], 
                            results["summary"]["total_possible"],
                            f"{results['summary']['percentage']}%", ""])
            
        print(f"CSV results saved to {output_path}")