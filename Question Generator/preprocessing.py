# preprocessing.py
import os
import logging
import re
import io
import string
from typing import Optional, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set NLTK data path and ensure resources
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

# Import NLTK and related modules
import nltk
nltk.data.path.append(nltk_data_dir)

# Import required NLTK components
try:
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except ImportError as e:
    logger.error(f"Failed to import NLTK components: {e}")
    raise

# Import Google Drive related modules
try:
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.http import MediaIoBaseDownload
except ImportError as e:
    logger.error(f"Failed to import Google API modules: {e}")
    raise

# Import PDF handling module
try:
    import fitz  # PyMuPDF
except ImportError as e:
    logger.error(f"Failed to import PyMuPDF (fitz): {e}")
    raise

# Define the scopes for Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive']

def ensure_nltk_resources():
    """Download required NLTK resources if not already available"""
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
            logger.info(f"NLTK resource '{resource}' is available")
        except Exception as e:
            logger.error(f"Error downloading NLTK resource {resource}: {e}")
            raise

# Call this function immediately to ensure resources are available
ensure_nltk_resources() 

def get_drive_service():
    """
    Authenticate and create Google Drive service.
    Returns:
        service: Google Drive service object or None if authentication fails
    """
    creds = None
    
    try:
        # Check for existing token
        if os.path.exists('token.json'):
            try:
                creds = Credentials.from_authorized_user_file('token.json', SCOPES)
                
                # Refresh token if expired
                if creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                elif not creds.valid:
                    os.remove('token.json')
                    creds = None
            except Exception:
                # If any error occurs with the token, remove it and start fresh
                if os.path.exists('token.json'):
                    os.remove('token.json')
                creds = None

        # Start fresh OAuth flow if needed
        if not creds:
            if not os.path.exists('credentials.json'):
                logger.error("credentials.json not found")
                return None
            
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            
            # Save the new token
            with open('token.json', 'w') as token:
                token.write(creds.to_json())

        # Build and test the service
        service = build('drive', 'v3', credentials=creds)
        service.files().list(pageSize=1).execute()  # Test API connection
        logger.info("Successfully authenticated with Google Drive")
        return service

    except Exception as e:
        logger.error(f"Error in authentication: {e}")
        return None

def get_pdf_from_drive(service, folder_id: str) -> Optional[str]:
    """
    Get the first PDF file from specified Google Drive folder.
    Args:
        service: Google Drive service object
        folder_id: ID of the Google Drive folder
    Returns:
        str: PDF file ID or None if no PDF found
    """
    try:
        results = service.files().list(
            q=f"'{folder_id}' in parents and mimeType='application/pdf'",
            fields="nextPageToken, files(id, name)"
        ).execute()
        items = results.get('files', [])

        if not items:
            logger.warning(f"No PDFs found in folder '{folder_id}'")
            return None

        logger.info(f"Found PDF: {items[0]['name']}")
        return items[0]['id']
        
    except Exception as e:
        logger.error(f"Error getting PDF from Drive: {e}")
        return None

def download_pdf(service, pdf_id: str, download_path: str) -> bool:
    """
    Download PDF from Google Drive.
    Args:
        service: Google Drive service object
        pdf_id: ID of the PDF file
        download_path: Path to save the PDF
    Returns:
        bool: True if download successful, False otherwise
    """
    try:
        request = service.files().get_media(fileId=pdf_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
            logger.info(f"Download progress: {int(status.progress() * 100)}%")

        fh.seek(0)
        with open(download_path, 'wb') as f:
            f.write(fh.read())
            
        logger.info(f"PDF downloaded successfully to {download_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading PDF: {e}")
        return False

def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Extract text from PDF file.
    Args:
        pdf_path: Path to the PDF file
    Returns:
        str: Extracted text or None if extraction fails
    """
    try:
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return None
            
        doc = fitz.open(pdf_path)
        text = []
        
        for page_num, page in enumerate(doc):
            try:
                text.append(page.get_text())
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num}: {e}")
                continue
                
        doc.close()
        
        if not text:
            logger.warning("No text extracted from PDF")
            return None
            
        combined_text = " ".join(text)
        # Process text directly here
        return preprocess_text(combined_text)
        
    except Exception as e:
        logger.error(f"Error in PDF text extraction: {e}")
        return None

def preprocess_text(text: str) -> str:
    """
    Basic preprocessing of text - handles non-ASCII chars, URLs, and whitespace.
    Args:
        text: Input text string
    Returns:
        str: Preprocessed text
    """
    try:
        # Remove non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Replace multiple spaces, tabs, and newlines with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep periods for sentence structure
        text = re.sub(r'[^A-Za-z0-9\s\.]', '', text)
        
        # Normalize whitespace
        text = text.strip()
        
        return text.lower()
        
    except Exception as e:
        logger.error(f"Error in text preprocessing: {e}")
        return text

def clean_text_nltk(text: str) -> str:
    """
    Advanced text cleaning using NLTK with lemmatization and stopword removal.
    Args:
        text: Input text string
    Returns:
        str: Cleaned text
    """
    try:
        # Perform basic preprocessing first
        text = preprocess_text(text)
        
        # Split into sentences
        sentences = sent_tokenize(text)
        cleaned_sentences = []
        
        # Get stop words once
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        for sentence in sentences:
            # Tokenization
            tokens = word_tokenize(sentence)
            
            # Remove stop words and non-alphabetic tokens
            tokens = [
                token for token in tokens 
                if token not in stop_words and token.isalpha()
            ]
            
            # Lemmatization
            lemmas = [lemmatizer.lemmatize(token) for token in tokens]
            
            if lemmas:  # Only add non-empty sentences
                cleaned_sentences.append(" ".join(lemmas))
        
        # Join sentences with proper spacing
        cleaned_text = " ".join(cleaned_sentences)
        return cleaned_text
        
    except Exception as e:
        logger.error(f"Error in NLTK text cleaning: {e}")
        # Fall back to basic cleaning
        return text
