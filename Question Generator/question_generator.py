from langchain.prompts import PromptTemplate
import json
import logging
import re

class QuestionGenerator:
    def __init__(self, llm):
        """Initialize the QuestionGenerator with an LLM."""
        self.llm = llm
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Simplified prompt template
        self.unified_template = PromptTemplate(
            template="""
            You are a CBSE high school science teacher generating {difficulty} difficulty questions.

            Content: {context}

            REQUIREMENTS:
            - Generate {mcq_count} multiple choice questions
            - Generate {short_count} short answer questions
            - Generate {long_count} long answer questions
            - Generate {case_count} case study questions

            Provide response in STRICT JSON format:
            {{
                "multiple_choice": [
                    {{
                        "question": "Question text",
                        "options": ["A) Option1", "B) Option2", "C) Option3", "D) Option4"],
                        "correct_answer": "A",
                        "explanation": "Explanation"
                    }}
                ],
                "short_answer": [
                    {{
                        "question": "Question text",
                        "suggested_answer": "Response outline"
                    }}
                ],
                "long_answer": [
                    {{
                        "question": "Complex question",
                        "expected_analysis": ["Key points"]
                    }}
                ],
                "case_studies": [
                    {{
                        "scenario": "Detailed scenario",
                        "questions": [
                            {{
                                "question": "Analytical question",
                                "expected_analysis": "Key points"
                            }}
                            {{
                                "question": "Next Analytical question",
                                "expected_analysis": "Key points"
                            }}
                            {{
                                "question": "Next Analytical question",
                                "expected_analysis": "Key points"
                            }}
                        ]
                    }}
                ]
            }}
            """,
            input_variables=["context", "difficulty", "mcq_count", "short_count", "long_count", "case_count"]
        )
        
        # Initialize chain
        self.unified_chain = self.unified_template | self.llm

    def _parse_json(self, content):
        """
        Enhanced JSON parsing with multiple fallback strategies.
        """
        # Remove code block markers and extra whitespace
        content = content.replace('```json', '').replace('```', '').strip()

        # List of parsing strategies to try
        parsing_strategies = [
            # 1. Direct JSON parsing
            lambda x: json.loads(x),
            
            # 2. Regex extraction with minimal cleaning
            lambda x: json.loads(re.search(r'\{.*"multiple_choice".*\}', x, re.DOTALL).group(0)),
            
            # 3. Replace common problematic characters
            lambda x: json.loads(x.replace('\n', '').replace('\t', ''))
        ]

        # Try each parsing strategy
        for strategy in parsing_strategies:
            try:
                questions = strategy(content)
                
                # Validate key sections exist
                if all(key in questions for key in ['multiple_choice', 'short_answer', 'long_answer', 'case_studies']):
                    return questions
            except Exception as e:
                self.logger.info(f"Parsing strategy failed: {str(e)}")
                continue

        # If all strategies fail, log the content for debugging
        self.logger.error(f"Failed to parse JSON. Content: {content}")
        return {"error": "Could not parse questions"}

    def generate_questions(self, context, difficulty="standard", mcq_count=10, short_count=5, long_count=3, case_count=3):
        """Generate questions with enhanced parsing."""
        # Validate difficulty
        if difficulty.lower() not in ["standard", "difficult"]:
            self.logger.warning(f"Invalid difficulty '{difficulty}'. Defaulting to 'standard'.")
            difficulty = "standard"

        try:
            # Invoke the LLM
            result = self.unified_chain.invoke({
                "context": context,
                "difficulty": difficulty.lower(),
                "mcq_count": mcq_count,
                "short_count": short_count,
                "long_count": long_count,
                "case_count": case_count
            })

            # Extract content (handles different LLM response types)
            content = result.content if hasattr(result, 'content') else str(result)

            # Parse JSON
            questions = self._parse_json(content)

            # Check for parsing error
            if 'error' in questions:
                return questions

            # Validate and limit question counts
            questions['multiple_choice'] = questions.get('multiple_choice', [])[:mcq_count]
            questions['short_answer'] = questions.get('short_answer', [])[:short_count]
            questions['long_answer'] = questions.get('long_answer', [])[:long_count]
            questions['case_studies'] = questions.get('case_studies', [])[:case_count]

            return questions

        except Exception as e:
            self.logger.error(f"Question generation error: {str(e)}")
            return {"error": str(e)}
