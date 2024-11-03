from flask import Flask, request, jsonify
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

app = Flask(__name__)

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

client = QdrantClient(host="localhost", port=6333)
collection_name = "image_collection"

if not client.collection_exists(collection_name):
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=512, distance="Cosine")
    )

def get_image_embedding(image):
    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding.squeeze().cpu().numpy()

def add_image_to_qdrant(image, product_id):
    vector = get_image_embedding(image)
    point = PointStruct(id=product_id, vector=vector)
    client.upsert(collection_name=collection_name, points=[point])

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    image = Image.open(file).convert("RGB")
    current_count = client.count(collection_name).count
    product_id = current_count + 1
    add_image_to_qdrant(image, product_id)
    
    return jsonify({"message": "Image uploaded successfully", "product_id": product_id}), 201

@app.route('/search', methods=['POST'])
def search_images():
    if 'image' not in request.files:
        return jsonify({"error": "No query image part"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    query_image = Image.open(file).convert("RGB")
    
    query_vector = get_image_embedding(query_image)
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=5,
        score_threshold=0.7
    )
    results = [{'product_id': hit.id, 'score': hit.score} for hit in search_result]
    
    return jsonify({"results": results}), 200

if __name__ == '__main__':
    app.run(debug=True)