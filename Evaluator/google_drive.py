import os
import io
import json
import time
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
import pandas as pd

# Google Drive API setup
SCOPES = ['https://www.googleapis.com/auth/drive']

def authenticate_google_drive():
    """Authenticate with Google Drive API"""
    creds = None
    
    # Check if token.json file exists with stored credentials
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_info(json.loads(open('token.json').read()), SCOPES)
    
    # If credentials don't exist or are invalid, get new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save credentials for future use
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    return build('drive', 'v3', credentials=creds)

def find_answer_keys(service, folder_id):
    """Find answer key files in the specified folder"""
    # Search for files with "answer key" or similar in the name
    # Added 'trashed=false' to exclude deleted files
    query = f"'{folder_id}' in parents and (name contains 'answer' or name contains 'key' or name contains 'solution') and trashed=false"
    
    results = service.files().list(
        q=query,
        fields="nextPageToken, files(id, name, mimeType)"
    ).execute()
    
    answer_keys = results.get('files', [])
    
    if not answer_keys:
        print("No answer key files found in the specified folder.")
    else:
        print(f"Found {len(answer_keys)} potential answer key files:")
        for i, file in enumerate(answer_keys):
            print(f"{i+1}. {file['name']} ({file['mimeType']})")
    
    return answer_keys

def download_answerkey(service, file_id, output_path):
    """Download a file from Google Drive to the specified path"""
    # Get file metadata to get the name and mime type
    file_metadata = service.files().get(fileId=file_id, fields='name, mimeType').execute()
    file_name = file_metadata['name']
    mime_type = file_metadata['mimeType']
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Handle Google Docs/Sheets/Slides by exporting them
    if mime_type == 'application/vnd.google-apps.document':
        request = service.files().export_media(fileId=file_id, mimeType='application/pdf')
        if not output_path.endswith('.pdf'):
            output_path += '.pdf'
    elif mime_type == 'application/vnd.google-apps.spreadsheet':
        request = service.files().export_media(fileId=file_id, mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        if not output_path.endswith('.xlsx'):
            output_path += '.xlsx'
    else:
        request = service.files().get_media(fileId=file_id)
    
    file = io.BytesIO()
    downloader = MediaIoBaseDownload(file, request)
    
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Downloading {file_name}: {int(status.progress() * 100)}%")
    
    file.seek(0)
    
    # Save the file locally
    with open(output_path, 'wb') as f:
        f.write(file.read())
    
    print(f"Downloaded file to: {os.path.abspath(output_path)}")
    return output_path

# Alias for download_file to maintain compatibility
download_file = download_answerkey

def list_files_in_folder(service, folder_id):
    """List all files in a Google Drive folder"""
    # Added 'trashed=false' to exclude deleted files
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="nextPageToken, files(id, name, mimeType)"
    ).execute()
    
    files = results.get('files', [])
    
    if not files:
        print("No files found in the specified folder.")
    else:
        print(f"Found {len(files)} files:")
        for i, file in enumerate(files):
            print(f"{i+1}. {file['name']} ({file['mimeType']})")
    
    return files

def batch_download_answer_keys(service, folder_id, output_dir):
    """Download all answer keys from a folder"""
    answer_keys = find_answer_keys(service, folder_id)
    
    if not answer_keys:
        return []
    
    downloaded_files = []
    for key in answer_keys:
        file_path = os.path.join(output_dir, key['name'])
        download_answerkey(service, key['id'], file_path)
        downloaded_files.append({
            'id': key['id'],
            'name': key['name'],
            'local_path': file_path
        })
    
    return downloaded_files

def upload_csv_to_drive(service, folder_id, csv_path, title=None):
    """
    Upload a CSV file to Google Drive as a spreadsheet
    
    Args:
        service: Google Drive service instance
        folder_id: ID of the folder to upload to
        csv_path: Path to the CSV file
        title: Optional title for the spreadsheet (default: filename + timestamp)
        
    Returns:
        ID of the uploaded file
    """
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Use provided title or generate one
    if not title:
        base_name = os.path.basename(csv_path)
        file_name = os.path.splitext(base_name)[0]
        title = f"{file_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    
    # Upload file to Google Drive
    file_metadata = {
        'name': title,
        'parents': [folder_id],
        'mimeType': 'application/vnd.google-apps.spreadsheet'
    }
    
    media = MediaFileUpload(csv_path, mimetype='text/csv', resumable=True)
    
    print(f"Uploading {csv_path} to Google Drive...")
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    
    print(f"Uploaded to Google Drive as spreadsheet: {title}")
    
    return file.get('id')

def upload_results_to_drive(drive_service, folder_id, csv_path, filename=None):
    """
    Upload a CSV file to Google Drive in the specified folder.
    
    Args:
        drive_service: Authenticated Google Drive service
        folder_id: ID of the folder to upload to
        csv_path: Path to the CSV file
        filename: Optional custom filename to use in Google Drive
    
    Returns:
        ID of the uploaded file
    """
    if filename is None:
        filename = os.path.basename(csv_path)
    
    file_metadata = {
        'name': filename,
        'parents': [folder_id],
        'mimeType': 'application/vnd.google-apps.spreadsheet'  # Convert to Google Sheets
    }
    
    media = MediaFileUpload(
        csv_path,
        mimetype='text/csv',
        resumable=True
    )
    
    file = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()
    
    print(f"File uploaded with ID: {file.get('id')}")
    return file.get('id')