import os
import json
import nltk
import logging
import time
import re
from typing import Optional, List, Dict, Any
from preprocessing import extract_text_from_pdf, clean_text_nltk, get_drive_service, get_pdf_from_drive, download_pdf
from rate_limited_llm import get_rate_limited_llm
from question_generator import QuestionGenerator
from pdf_handler import PDFDocumentHandler
from llm_handler import setup_vector_store, create_qa_chain, get_response

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExamGenerator:
    def __init__(self, google_api_key: str, folder_id: str, output_folder_id: str):
        self.google_api_key = google_api_key
        self.folder_id = folder_id  # Folder to look for PDFs
        self.output_folder_id = output_folder_id  # Folder to save output PDFs
        self.difficulty = "standard"  # Default difficulty, will be updated from filename
        self.drive_service = None
        self.llm = None
        self.vector_store = None
        self.qa_chain = None
        self.question_generator = None
        self.pdf_handler = None
        self.pdf_name = None  # Will store the name of the downloaded PDF

    def initialize_components(self) -> bool:
        """Initialize all required components with improved error handling"""
        try:
            # Initialize NLTK
            nltk_resources = ['punkt', 'stopwords', 'wordnet']
            for resource in nltk_resources:
                try:
                    nltk.download(resource, quiet=True)
                except Exception as e:
                    logger.error(f"Failed to download NLTK resource {resource}: {e}")
                    return False

            # Initialize Google Drive service
            self.drive_service = get_drive_service()
            if not self.drive_service:
                logger.error("Failed to initialize Google Drive service")
                return False
            
            # Initialize Rate-Limited LLM
            self.llm = get_rate_limited_llm(self.google_api_key)
            if not self.llm:
                logger.error("Failed to initialize LLM")
                return False
            
            # Initialize Question Generator with rate-limited LLM
            self.question_generator = QuestionGenerator(self.llm)
            
            # Initialize PDF Handler
            self.pdf_handler = PDFDocumentHandler()
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False

    def get_pdf_info_from_drive(self, pdf_id):
        """Get PDF filename from Drive using its ID"""
        try:
            file = self.drive_service.files().get(fileId=pdf_id, fields='name').execute()
            self.pdf_name = file.get('name')
            logger.info(f"Retrieved PDF name: {self.pdf_name}")
            return self.pdf_name
        except Exception as e:
            logger.error(f"Error getting PDF name: {e}")
            return None
            
    @staticmethod
    def extract_difficulty_from_filename(pdf_name):
        try:
            # Check if filename contains the '#' delimiter
            if '#' in pdf_name:
                # Split the filename by '#' and get the part after '#'
                parts = pdf_name.split('#')
                if len(parts) >= 2:
                    # Get the difficulty part (remove .pdf extension if present)
                    difficulty = parts[1].lower().replace('.pdf', '')
                    
                    # Validate difficulty
                    if difficulty in ["standard", "difficult"]:
                        logging.info(f"Extracted difficulty '{difficulty}' from filename '{pdf_name}'")
                        return difficulty
                    else:
                        logging.warning(f"Invalid difficulty '{difficulty}' in filename, using 'standard'")
                
            logging.info(f"No valid difficulty found in filename '{pdf_name}', using 'standard'")
            return "standard"  # Default
        
        except Exception as e:
            logging.error(f"Error extracting difficulty from filename: {e}")
            return "standard"  # Default to standard difficulty

    def update_difficulty_from_filename(self):
        """Update difficulty setting from PDF filename"""
        if not self.pdf_name:
            logger.error("Cannot update difficulty: PDF name is not set")
            return False
            
        try:
            # Extract difficulty from filename
            difficulty = self.extract_difficulty_from_filename(self.pdf_name)
            
            # Update the difficulty setting
            self.difficulty = difficulty
            logger.info(f"Updated difficulty to '{self.difficulty}' from filename")
            return True
            
        except Exception as e:
            logger.error(f"Error updating difficulty from filename: {e}")
            return False

    def process_pdf(self, pdf_id: str) -> Optional[str]:
        """Process PDF with improved error handling and cleanup"""
        download_path = f"downloaded_pdf_{pdf_id}.pdf"
        
        try:
            # Download PDF with timeout and retry
            if not download_pdf(self.drive_service, pdf_id, download_path):
                logger.error("Failed to download PDF")
                return None
                
            # Extract text with timeout
            extracted_text = extract_text_from_pdf(download_path)
            if not extracted_text:
                logger.error("Failed to extract text from PDF")
                return None
                
            # Clean text
            cleaned_text = clean_text_nltk(extracted_text)
            if not cleaned_text:
                logger.error("Failed to clean extracted text")
                return None
                
            logger.info("PDF processed successfully")
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return None
            
        finally:
            # Always try to clean up the downloaded file
            try:
                if os.path.exists(download_path):
                    os.remove(download_path)
            except Exception as e:
                logger.warning(f"Could not delete temporary file {download_path}: {e}")

    def create_vector_store(self, cleaned_text: str) -> Optional[List[str]]:
        """Create vector store using LangChain instead of direct ChromaDB"""
        try:
            # Import setup_vector_store function
            from llm_handler import setup_vector_store
            
            # Create vector store
            logger.info("Creating vector store with LangChain...")
            self.vector_store = setup_vector_store(cleaned_text, persist_directory="./chroma_langchain")
            
            if not self.vector_store:
                logger.error("Failed to create vector store")
                return None
                
            # Get the texts from the vector store if possible
            if hasattr(self.vector_store, "_collection") and hasattr(self.vector_store._collection, "get"):
                try:
                    # Try to retrieve all documents from the collection
                    all_docs = self.vector_store._collection.get()
                    if all_docs and "documents" in all_docs:
                        texts = all_docs["documents"]
                        logger.info(f"Retrieved {len(texts)} text chunks from vector store")
                        return texts
                except Exception as e:
                    logger.warning(f"Could not retrieve texts from vector store: {e}")
            
            # If we can't get texts from vector store, split the text manually
            if isinstance(cleaned_text, str):
                # Split into chunks
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=150,
                    length_function=len,
                )
                texts = text_splitter.split_text(cleaned_text)
                logger.info(f"Split text into {len(texts)} chunks")
                return texts
            else:
                return cleaned_text
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return None

    def generate_questions(self, texts: List[str]) -> Optional[Dict[str, Any]]:
        """Generate questions using vector store retrieval instead of raw text"""
        try:
            logger.info(f"Generating questions using vector store with {self.difficulty} difficulty")
            
            # Create a vector store with the texts if it doesn't exist yet
            if not hasattr(self, 'vector_store'):
                # Import required components
                from llm_handler import setup_vector_store, create_qa_chain
                
                logger.info("Setting up vector store for question generation")
                self.vector_store = setup_vector_store(texts)
                
                # Create QA chain with the vector store
                self.qa_chain = create_qa_chain(self.llm, self.vector_store)
            
            # Define key topics to query based on our content
            topic_queries = self.extract_key_topics(texts)
            
            # Limit the number of topics for processing
            topic_queries = topic_queries[:8]  # Use at most 8 topics to avoid excessive API calls
            
            logger.info(f"Identified {len(topic_queries)} key topics for question generation")
            
            # Initialize empty questions structure
            questions = {
                "multiple_choice": [],
                "short_answer": [],
                "long_answer": [],
                "case_studies": []
            }
            
            # Process each topic to generate relevant questions
            for i, topic in enumerate(topic_queries):
                logger.info(f"Processing topic {i+1}/{len(topic_queries)}: {topic[:50]}...")
                
                # Add delay between topics to avoid rate limits
                if i > 0:
                    time.sleep(5)
                
                # Generate specific context for this topic by querying the vector store
                context = self.retrieve_topic_context(topic)
                
                # Generate questions for this context with difficulty level
                topic_questions = self.question_generator.generate_questions(
                    context, 
                    difficulty=self.difficulty,
                    mcq_count=3,  # Smaller counts per topic
                    short_count=2,
                    long_count=1,
                    case_count=1
                )
                
                # Merge questions from this topic
                for key in questions.keys():
                    if key in topic_questions and isinstance(topic_questions[key], list):
                        questions[key].extend(topic_questions[key])
            
            # Add difficulty metadata
            questions["metadata"] = {"difficulty": self.difficulty}
            
            # Limit questions to the required counts
            questions = self.limit_questions_count(questions)
            return questions
            
        except Exception as e:
            logger.error(f"Error generating questions from embeddings: {e}")
            return None
    def extract_key_topics(self, texts: List[str]) -> List[str]:
        """Extract key topics from texts for targeted question generation"""
        try:
            # Combine texts with spacing to avoid merging issues
            combined_text = " ".join(texts)
            
            # Use LLM to identify key topics
            prompt = f"""
            Analyze the following text and identify 5-8 key topics that would be good for
            generating exam questions. For each topic, provide a brief description that would
            help retrieve relevant information about that topic from a vector database.
            
            Text: {combined_text[:6000]}  # Use first 6000 chars to avoid token limits
            
            Format each topic as a question or search query that would retrieve relevant information.
            """
             
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract topics using simple parsing
            topics = []
            for line in content.split('\n'):
                line = line.strip()
                # Look for numbered/bulleted items or topic headers
                if re.match(r'^[\d\-\*\.]+\s+', line) or ':' in line:
                    # Clean up the line and add to topics
                    topic = re.sub(r'^[\d\-\*\.]+\s+', '', line)
                    topic = topic.split(':')[-1].strip() if ':' in topic else topic
                    if topic and len(topic) > 10:  # Avoid too short queries
                        topics.append(topic)
            
            if not topics:
                # Backup approach: split into chunks as topics
                topics = [text[:300] for text in texts[:8]]
            
            return topics
        
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return texts[:5]  # Return first few chunks as fallback

    def retrieve_topic_context(self, topic: str) -> str:
        """Retrieve relevant context for a topic using the vector store"""
        try:
            from llm_handler import get_response
            
            # Query the vector store for this topic
            response = get_response(self.qa_chain, topic)
            
            if response and "answer" in response:
                # Get the answer and source documents
                answer = response["answer"]
                source_docs = response.get("source_documents", [])
                
                # Combine answer with source document contents
                context = answer + "\n\n"
                for i, doc in enumerate(source_docs):
                    if hasattr(doc, "page_content"):
                        context += f"Document {i+1}: {doc.page_content}\n\n"
                
                return context
            else:
                # If retrieval failed, just return the topic
                return topic
        
        except Exception as e:
            logger.error(f"Error retrieving topic context: {e}")
            return topic
    def limit_questions_count(self, questions: Dict[str, List]) -> Dict[str, List]:
        """Limit the number of questions to specified counts"""
        target_counts = {
            "multiple_choice": 7,  
            "short_answer": 5,     
            "long_answer": 3,      
            "case_studies": 1     
        }
        
        # Ensure each case study has exactly 3 questions
        if "case_studies" in questions:
            for case_study in questions["case_studies"]:
                if "questions" in case_study:
                    # If there are more than 3 questions, keep only the first 3
                    if len(case_study["questions"]) > 3:
                        case_study["questions"] = case_study["questions"][:3]                    
                # Ensure scenario is at least 6 lines
                if "scenario" in case_study:
                    scenario_lines = case_study["scenario"].count('\n') + 1
                    if scenario_lines < 6:
                        # Make the scenario longer by adding more context
                        case_study["scenario"] += "\n\n" + "Additional context information for this scenario: "
                        if "background_info" in case_study and len(case_study["background_info"]) > 0:
                            case_study["scenario"] += " " + ". ".join(case_study["background_info"])
                        else:
                            case_study["scenario"] += " Students should consider various factors when analyzing this situation."
        
        # Simply limit questions to the specified counts
        for question_type, count in target_counts.items():
            if question_type in questions:
                questions[question_type] = questions[question_type][:count]
        
        return questions

    def save_questions(self, questions: Dict[str, Any], filename: str = "generated_questions.json") -> bool:
        """Save questions with backup and validation"""
        try:
            # Check if questions is None or empty
            if not questions:
                logger.error("No questions to save")
                return False
                
            # Create backup of existing file if it exists
            if os.path.exists(filename):
                backup_filename = f"{filename}.backup"
                os.rename(filename, backup_filename)
            
            # Create a minimal structure if missing keys
            for key in ["multiple_choice", "short_answer", "long_answer", "case_studies"]:
                if key not in questions:
                    questions[key] = []
            
            # Ensure metadata with difficulty is present
            if "metadata" not in questions:
                questions["metadata"] = {"difficulty": self.difficulty}
            elif "difficulty" not in questions["metadata"]:
                questions["metadata"]["difficulty"] = self.difficulty
            
            # Save new questions
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(questions, f, indent=2, ensure_ascii=False)
            
            # Validate saved file
            with open(filename, 'r', encoding='utf-8') as f:
                loaded_questions = json.load(f)
                if loaded_questions:
                    logger.info(f"Questions successfully saved locally: {filename}")
                    logger.info(f"Generated {len(loaded_questions.get('multiple_choice', []))} MCQs, " +
                              f"{len(loaded_questions.get('short_answer', []))} short answers, " +
                              f"{len(loaded_questions.get('long_answer', []))} long answers, " +
                              f"{len(loaded_questions.get('case_studies', []))} case studies")
                    logger.info(f"Difficulty level: {loaded_questions.get('metadata', {}).get('difficulty', self.difficulty)}")
                    return True
                else:
                    logger.error("Validation failed: saved questions appear to be empty")
                    return False
            
        except Exception as e:
            logger.error(f"Error saving questions: {e}")
            return False

    def save_questions_to_drive(self, questions: Dict[str, Any], filename: str = None) -> Optional[str]:
        """Generate PDF and save directly to Google Drive"""
        try:
            if not questions:
                logger.error("No questions to save")
                return None
            
            # Use the original PDF name if no filename is provided
            if filename is None:
                if not self.pdf_name:
                    logger.error("Original PDF name not available")
                    return None
                
                filename = self.pdf_name
            
            # Save locally for backup (use JSON version of the filename)
            json_filename = os.path.splitext(filename)[0] + '.json'
            self.save_questions(questions, json_filename)
            
            # Generate PDF and upload directly to Drive
            file_id = self.pdf_handler.generate_and_upload_pdf(
                drive_service=self.drive_service,
                json_data=questions,
                filename=filename,
                folder_id=self.output_folder_id
            )
            
            if file_id:
                logger.info(f"PDF successfully uploaded to Google Drive: {filename} (ID: {file_id})")
                # Get viewable link
                try:
                    # Make the file viewable by anyone with the link
                    permission = {
                        'type': 'anyone',
                        'role': 'reader',
                        'allowFileDiscovery': False
                    }
                    self.drive_service.permissions().create(
                        fileId=file_id,
                        body=permission
                    ).execute()
                    
                    # Get the file's web viewable link
                    file = self.drive_service.files().get(
                        fileId=file_id,
                        fields='webViewLink'
                    ).execute()
                    
                    link = file.get('webViewLink')
                    if link:
                        logger.info(f"PDF viewable at: {link}")
                except Exception as e:
                    logger.warning(f"Could not set public permissions: {e}")
                
                return file_id
            else:
                logger.error("Failed to upload PDF to Google Drive")
                return None
                
        except Exception as e:
            logger.error(f"Error saving questions to Google Drive: {e}")
            return None
    def run(self) -> bool:
        """Main execution flow using LangChain vector store for embedding-based question generation"""
        try:
            logger.info("Starting exam generation process")
            
            # Initialize components (now without ChromaDB)
            if not self.initialize_components():
                logger.error("Component initialization failed")
                return False

            # Get PDF from Drive
            pdf_id = get_pdf_from_drive(self.drive_service, self.folder_id)
            if not pdf_id:
                logger.error("No PDF found in specified folder")
                return False
                
            # Get PDF name
            if not self.get_pdf_info_from_drive(pdf_id):
                logger.error("Failed to get PDF name")
                return False
                
            # Update difficulty from filename
            self.update_difficulty_from_filename()
            logger.info(f"Using difficulty level: {self.difficulty}")

            # Process PDF
            cleaned_text = self.process_pdf(pdf_id)
            if not cleaned_text:
                logger.error("PDF processing failed")
                return False

            # Create vector store using LangChain (replaces generate_and_store_embeddings)
            texts = self.create_vector_store(cleaned_text)
            if not texts:
                logger.error("Vector store creation failed")
                return False
                
            # Import regex for topic extraction
            import re
            
            # Create QA chain with the vector store
            from llm_handler import create_qa_chain
            self.qa_chain = create_qa_chain(self.llm, self.vector_store)
            
            # Generate questions using embeddings-based approach
            questions = self.generate_questions(texts)
            if not questions:
                logger.error("Question generation failed")
                return False

            # Save questions locally as JSON (for backup)
            json_filename = f"{self.pdf_name.replace('.pdf', '')}.json"
            self.save_questions(questions, json_filename)
            
            # Save questions as PDF directly to Google Drive
            file_id = self.save_questions_to_drive(questions, self.pdf_name)
            if not file_id:
                logger.error("Failed to save PDF to Google Drive")
                return False

            logger.info(f"Exam generation completed successfully with {self.difficulty} difficulty")
            return True
            
        except Exception as e:
            logger.error(f"Critical error in exam generation process: {e}")
            return False
            
    @staticmethod
    def get_timestamp():
        """Get current timestamp for metadata"""
        from datetime import datetime
        return datetime.now().isoformat()

# Modified main function
if __name__ == '__main__':
    # Configuration
    GOOGLE_API_KEY = #add ur api key
    SOURCE_FOLDER_ID = #add ur folder id
    OUTPUT_FOLDER_ID =#add ur folder id
    # Set up logging with more detailed format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    
    # Create and run exam generator with difficulty from filename
    exam_generator = ExamGenerator(
        GOOGLE_API_KEY, 
        SOURCE_FOLDER_ID, 
        OUTPUT_FOLDER_ID
    )
    
    try:
        success = exam_generator.run()
        
        if success:
            print(f"✅ Exam generation with {exam_generator.difficulty} difficulty completed successfully")
            print(f"📝 Questions have been saved locally to 'generated_questions_{exam_generator.difficulty.capitalize()}.json' for backup")
            print("📄 PDF document has been uploaded to Google Drive")
        else:
            print("❌ Exam generation failed. Check logs for details")
    except Exception as e:
        print(f"❌ Critical error: {e}")
        logger.critical(f"Unhandled exception in main: {e}", exc_info=True)
