import requests
import nltk
nltk.download('punkt')

# Define the URL of your Flask API
api_url = "https://model-api-pem5.onrender.com/predict"

# Sample text for testing
sample_texts = ["You people should improve in the item's you ", "Hate this", "kindly improve this", "I just don't like this"]

# Create a dictionary with the text
for sample_text in sample_texts:
    data = {"text": sample_text}

    # Send a POST request to the API
    response = requests.post(api_url, json=data)

    # Check the response status code
    if response.status_code == 200:
        # Print the JSON response
        print(response.json())
    else:
        # Print an error message if the request was not successful
        print(f"Error: {response.status_code}, {response.text}")
