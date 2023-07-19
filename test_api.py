import requests
import json
import base64

# Open an image file and encode it as base64
with open("asset/palmar_example.jpg", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

# Prepare the data to send
data = {
    'base64_image': encoded_image
}

# Convert to JSON
payload = json.dumps(data)

# Send the request
response = requests.post("http://127.0.0.1:8000/predict", data=payload)
status_code = response.status_code
json_response = response.json()
prediction = json_response['prediction']
# Print the response
print("response status: {}, prediction: {}".format(status_code, prediction))
