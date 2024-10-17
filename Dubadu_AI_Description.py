from flask import Flask, request, jsonify
import g4f
from flask_cors import CORS
import logging
import base64
import io
import requests
import json
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.cloud import storage
import uuid
from datetime import datetime
import time
import random

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# GCS and Vertex AI setup
bucket_name = "dubadu_ai_photo"
project_id = "ornate-hangar-436219-g2"

#Функція завантаження зображень в Google Cloud Storage
def upload_to_gcs_from_memory(bucket_name, file_data, destination_blob_name, project_id):
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_file(file_data, content_type='image/jpeg')
    return f"gs://{bucket_name}/{destination_blob_name}"

#Функція генерації унікального ID для зображення
def generate_unique_filename():
    unique_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"images/{unique_id}_{timestamp}.jpg"

def process_image(base64_image, bucket_name, project_id):
    image_data = base64.b64decode(base64_image)
    file_data = io.BytesIO(image_data)
    destination_blob_name = generate_unique_filename()
    gcs_uri = upload_to_gcs_from_memory(bucket_name, file_data, destination_blob_name, project_id)
    return gcs_uri

#Функція генерації тексту за заданими зображеннями завдяки Gemini 1.5
def generate_image_description(gcs_uri):
    # Ініціалізація Vertex AI
    retries=5
    vertexai.init(project=project_id, location="us-central1")
    model = GenerativeModel("gemini-1.5-flash-002")

    backoff_time = 1

    for attempt in range(retries):
        try:
            response = model.generate_content(
                [
                    Part.from_uri(
                        gcs_uri,
                        mime_type="image/jpeg",
                    ),
                    "Describe the photo in text format. The text should be complete, concise, and to the point. Output only the description without any additional comments!",
                ]
            )
            return response.text

        except Exception as e:
            if "429" in str(e):
                logging.error(f"Quota exceeded. Attempt {attempt + 1} of {retries}. Retrying in {backoff_time} seconds.")
                time.sleep(backoff_time)
                backoff_time *= 2
            else:
                logging.error(f"Error processing image: {str(e)}")
                break

    return "Error: Could not generate image description due to repeated quota errors."

# Create description with chatGPT
@app.route('/improve-description', methods=['POST'])
def improve_description():
    data = request.json
    logging.debug(f"Received data: {data}")

    # Параметри витягнуті з JSON
    description = data.get('description', '')
    rooms = data.get('rooms', '')
    kitchen_area = data.get('kitchen_area', '')
    house_area = data.get('house_area', '')
    floors = data.get('floors', '')
    property_type = data.get('property_type', '')
    images_base64 = data.get('images', [])

    image_descriptions = []

    # Генерація тексту на основі зображення
    for idx, image_base64 in enumerate(images_base64):
        try:
            gcs_uri = process_image(image_base64, bucket_name, project_id)
        
            image_caption = generate_image_description(gcs_uri)
        
            if image_caption:
                image_descriptions.append(f"Image {idx + 1}: {image_caption}")
            
        except Exception as e:
            logging.error(f"Error processing image {idx + 1}: {str(e)}")

    image_descriptions_for_ai = "\n".join(image_descriptions)
    print(image_descriptions_for_ai)

    # Побудова промпту для створення якісного текстового опису нерухомості
    if description:
        prompt = f"Hi, you are a very talented copywriter in the real estate field, and I need your help writing a beautiful listing description. Here is a property description:\n\n{description}\n\n"
        prompt += "Please improve this description by making it more engaging, structured, and detailed. "
        prompt += "Use the following additional property details to enhance the content:\n"
        
        if image_descriptions:
            prompt += "These are the descriptions of the images that go with the real estate ad:\n"
            for description in image_descriptions:
                prompt += f"- {description}\n"
    else:
        prompt = f"Hi, you are a very talented copywriter in the real estate field, and I need your help writing a beautiful listing description.\n"
        prompt += "Please create an engaging, structured, and appealing description of the property based on the following details:\n"
        
        if image_descriptions:
            prompt += "These are the descriptions of the images that go with the real estate ad ( Use the image descriptions only to improve the real estate description without explicit indication ):\n"
            for description in image_descriptions:
                prompt += f"- {description}\n"

    # Параметри нерухомості, витягнуті з JSON-формату
    if rooms:
        prompt += f"- Number of rooms: {rooms}\n"
    if kitchen_area:
        prompt += f"- Kitchen area: {kitchen_area} m²\n"
    if house_area:
        prompt += f"- House area: {house_area} m²\n"
    if floors:
        prompt += f"- Number of floors: {floors}\n"
    if property_type:
        prompt += f"- Location: {property_type}\n"

    prompt += "\nAim for a response around 500 words, presented in a readable and attractive format. Always output only the description itself without any comments please!"

    print(prompt)

    try:
        # Генерація зображення
        response = g4f.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=800, 
            temperature=0.7,
        )
        return jsonify({"improved_description": response})

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)