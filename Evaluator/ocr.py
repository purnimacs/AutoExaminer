import requests
import time
from typing import Dict  # Add this import

class AzureOCR:
    def __init__(self, api_key, endpoint):
        self.api_key = api_key
        self.endpoint = endpoint
    
    def recognize_handwriting(self, image_path):
        """
        Use Azure Computer Vision API to recognize handwriting in the image
        """
        # Endpoint for Read API
        read_api_url = f"{self.endpoint}/vision/v3.2/read/analyze"
        
        # Prepare headers and parameters
        headers = {
            'Ocp-Apim-Subscription-Key': self.api_key,
            'Content-Type': 'application/octet-stream'
        }
        
        # Read the image file
        with open(image_path, 'rb') as image_data:
            image_bytes = image_data.read()
        
        # Call the API
        response = requests.post(read_api_url, headers=headers, data=image_bytes)
        
        if response.status_code != 202:
            raise Exception(f"Request failed with status code {response.status_code}: {response.text}")
        
        # Get the operation location (URL where the result will be)
        operation_location = response.headers["Operation-Location"]
        
        # Poll the operation to get the results
        polling_interval_seconds = 1
        max_polling_attempts = 60
        
        for i in range(max_polling_attempts):
            time.sleep(polling_interval_seconds)
            get_response = requests.get(
                operation_location, 
                headers={"Ocp-Apim-Subscription-Key": self.api_key}
            )
            result = get_response.json()
            
            if "status" in result and result["status"] == "succeeded":
                break
        
        # Extract text from the response
        extracted_text = ""
        if "analyzeResult" in result and "readResults" in result["analyzeResult"]:
            for read_result in result["analyzeResult"]["readResults"]:
                for line in read_result["lines"]:
                    extracted_text += line["text"] + "\n"
        
        return extracted_text
    
    
    