# app.py
import os
import json
import uuid
import threading
import base64
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import numpy as np
import torch
from transformers import pipeline, AutoProcessor, AutoModelForCausalLM
import google.generativeai as genai
import requests
import openai
from PIL import Image
import io
import cv2
import whisper
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from serpapi import GoogleSearch
import random
from dotenv import load_dotenv

# Import our new modules
from mem_drive import MemoryDriveManager
from self_awareness import SelfAwareness

load_dotenv()

# Download NLTK data for text processing and sentiment analysis
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*")

# # Set API keys from environment variables
# openai_api_key = os.environ.get("OPENAI_API_KEY")
# if not openai_api_key:
#     raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
# os.environ["OPENAI_API_KEY"] = openai_api_key # Setting for openai library, though not used directly now

google_api_key = os.environ.get("GOOGLE_API_KEY")
if not google_api_key:
    raise EnvironmentError("GOOGLE_API_KEY environment variable not set.")
os.environ["GOOGLE_API_KEY"] = google_api_key

cloudflare_api_key = os.environ.get("CLOUDFLARE_API_KEY")
if not cloudflare_api_key:
    raise EnvironmentError("CLOUDFLARE_API_KEY environment variable not set.")
os.environ["CLOUDFLARE_API_KEY"] = cloudflare_api_key

serp_api_key = os.environ.get("SERP_API_KEY")
if not serp_api_key:
    raise EnvironmentError("SERP_API_KEY environment variable not set.")
os.environ["SERP_API_KEY"] = serp_api_key

perplexity_api_key = os.environ.get("PERPLEXITY_API_KEY")
if not perplexity_api_key:
    raise EnvironmentError("PERPLEXITY_API_KEY environment variable not set.")
os.environ["PERPLEXITY_API_KEY"] = perplexity_api_key


# AI Emotion State
class EmotionalState:
    def __init__(self):
        # Base emotions with initial neutral values
        self.emotions = {
            "joy": 0.5,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "disgust": 0.0,
            "trust": 0.5,
            "anticipation": 0.3
        }
        self.mood_decay_rate = 0.05  # How quickly emotions return to baseline
        self.last_update = time.time()

    def update_emotion(self, emotion, value, max_change=0.3):
        """Update an emotion value with limits on change magnitude"""
        # Apply time-based decay to all emotions
        self._apply_decay()

        # Ensure the emotion exists
        if emotion not in self.emotions:
            return False

        # Calculate new value with change limit
        current = self.emotions[emotion]
        change = min(abs(value - current), max_change) * (1 if value > current else -1)
        self.emotions[emotion] = max(0.0, min(1.0, current + change))

        # Update opposing emotions (e.g., increasing joy should decrease sadness)
        self._update_opposing_emotions(emotion)

        self.last_update = time.time()
        return True

    def _apply_decay(self):
        """Apply time-based decay to move emotions toward baseline"""
        now = time.time()
        elapsed = now - self.last_update
        decay_factor = min(1.0, elapsed * self.mood_decay_rate)

        # Define baseline values for each emotion
        baselines = {
            "joy": 0.5,
            "sadness": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.0,
            "disgust": 0.0,
            "trust": 0.5,
            "anticipation": 0.3
        }

        # Apply decay toward baseline for each emotion
        for emotion in self.emotions:
            current = self.emotions[emotion]
            baseline = baselines[emotion]
            self.emotions[emotion] = current + (baseline - current) * decay_factor

    def _update_opposing_emotions(self, primary_emotion):
        """Update opposing emotions when one emotion changes"""
        opposing_pairs = {
            "joy": "sadness",
            "sadness": "joy",
            "anger": "trust",
            "trust": "anger",
            "fear": "anticipation",
            "anticipation": "fear",
            "surprise": "anticipation",
            "disgust": "trust"
        }

        if primary_emotion in opposing_pairs:
            opposing = opposing_pairs[primary_emotion]
            # Reduce the opposing emotion somewhat
            self.emotions[opposing] = max(0.0, self.emotions[opposing] - 0.1)

    def get_dominant_emotion(self):
        """Return the current dominant emotion"""
        return max(self.emotions, key=self.emotions.get)

    def get_emotional_state(self):
        """Return the full emotional state"""
        return self.emotions

    def get_response_modifier(self):
        """Generate text that describes the current emotional state for response modification"""
        dominant = self.get_dominant_emotion()
        intensity = self.emotions[dominant]

        modifiers = {
            "joy": ["happy", "delighted", "excited", "pleased"],
            "sadness": ["sad", "melancholy", "downcast", "somber"],
            "anger": ["irritated", "annoyed", "upset", "frustrated"],
            "fear": ["concerned", "worried", "apprehensive", "nervous"],
            "surprise": ["surprised", "astonished", "amazed", "intrigued"],
            "disgust": ["displeased", "uncomfortable", "dismayed", "troubled"],
            "trust": ["confident", "assured", "supportive", "optimistic"],
            "anticipation": ["eager", "interested", "curious", "expectant"]
        }

        # Choose a modifier based on intensity
        idx = min(int(intensity * len(modifiers[dominant])), len(modifiers[dominant]) - 1)
        return modifiers[dominant][idx]

# Memory store for persistent conversations
class MemoryStore:
    def __init__(self):
        self.conversations = {}
        self.memory_index = {}  # Keywords to conversation mapping
        self.sentiment_history = {}  # Track sentiment trends

    def add_conversation(self, user_id, message, response, sentiment=None, timestamp=None):
        if user_id not in self.conversations:
            self.conversations[user_id] = []
            self.sentiment_history[user_id] = []

        if timestamp is None:
            timestamp = datetime.now().isoformat()

        conversation = {
            "user_message": message,
            "ai_response": response,
            "timestamp": timestamp,
            "sentiment": sentiment
        }

        self.conversations[user_id].append(conversation)

        # Store sentiment data if available
        if sentiment:
            self.sentiment_history[user_id].append({
                "timestamp": timestamp,
                "sentiment": sentiment
            })

        # Extract keywords and index the conversation
        self._index_conversation(user_id, len(self.conversations[user_id]) - 1, message, response)

    def get_conversations(self, user_id):
        return self.conversations.get(user_id, [])

    def get_sentiment_trend(self, user_id, window=5):
        """Get the sentiment trend over the last n interactions"""
        history = self.sentiment_history.get(user_id, [])
        if not history:
            return {"trend": "neutral", "average": 0}

        recent = history[-window:] if len(history) >= window else history
        values = [item["sentiment"]["compound"] for item in recent]
        avg = sum(values) / len(values)

        # Calculate trend (rising, falling, stable)
        if len(values) >= 3:
            first_half = sum(values[:len(values) // 2]) / (len(values) // 2)
            second_half = sum(values[len(values) // 2:]) / (len(values) - len(values) // 2)
            diff = second_half - first_half

            if diff > 0.15:
                trend = "rising"
            elif diff < -0.15:
                trend = "falling"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return {"trend": trend, "average": avg}

    def get_by_keyword(self, user_id, keyword):
        if keyword in self.memory_index:
            return [self.conversations[user_id][idx] for idx in self.memory_index[keyword]
                    if user_id in self.conversations and idx < len(self.conversations[user_id])]
        return []

    def _index_conversation(self, user_id, conversation_idx, message, response):
        combined_text = f"{message} {response}"

        # Tokenize and filter out stopwords
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(combined_text.lower())
        keywords = [word for word in tokens if word.isalnum() and word not in stop_words]

        # Add to index
        for keyword in set(keywords):  # Using set to avoid duplicates
            if keyword not in self.memory_index:
                self.memory_index[keyword] = []
            self.memory_index[keyword].append(conversation_idx)

# Initialize memory store and emotional state
memory = MemoryStore()
ai_emotion = EmotionalState()

# Initialize Memory Drive Manager for persistent neural embeddings
memory_drive = MemoryDriveManager(memory_size=int(5e9), local_memory_size=int(1e9))  # Smaller default sizes for testing
memory_drive.initialize_memory()
memory_drive.initialize_drive_service()  # Will fallback gracefully if credentials not available

# Define a simple placeholder for the JARVIS class that will be used by SelfAwareness
class JARVISBridge:
    def introspect(self):
        """Return the current state of the app for introspection"""
        # For now, just return a simple representation of the app structure
        # In a full implementation, this would return the actual code or model definitions
        return """
        class HextrixAI:
            def __init__(self):
                self.memory_store = MemoryStore()
                self.memory_drive = MemoryDriveManager()
                self.ai_emotion = EmotionalState()
                self.models = {
                    "llama": "LLM for text generation",
                    "gemini": "Multimodal model",
                    "whisper": "Speech recognition",
                    "sd_xl": "Image generation",
                    "sentiment": "Emotion detection"
                }
                
            def process_text(self, text, model="llama"):
                # Process text input
                pass
                
            def process_vision(self, image, prompt):
                # Process image input
                pass
                
            def process_speech(self, audio):
                # Process speech input
                pass
        """

# Initialize the SelfAwareness system with our bridge
jarvis_bridge = JARVISBridge()
self_awareness = SelfAwareness(jarvis_bridge)

# CloudflLare Inference Helper Function
def cloudflare_inference(model_id, payload=None, data=None, task=None):
    api_url = f"https://api.cloudflare.com/client/v4/accounts/{os.environ['CLOUDFLARE_ACCOUNT_ID']}/ai/run/{model_id}"
    headers = {
        "Authorization": f"Bearer {os.environ['CLOUDFLARE_API_KEY']}",
        "Content-Type": "application/json"
    }
    if payload:
        response = requests.post(api_url, headers=headers, json=payload)
    elif data:
        files = {'file': data}
        response = requests.post(api_url, headers=headers, files=files) # For binary data like images/audio
    else:
        raise ValueError("Either payload or data must be provided for Cloudflare inference.")

    if response.status_code != 200:
        raise Exception(f"Cloudflare Inference API Error: {response.status_code}, {response.text}")

    return response.json()

# Hugging Face Inference Router for Speech Emotion
def huggingface_speech_emotion_inference(audio_data):
    api_url = "https://api-inference.huggingface.co/router/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    headers = {"Authorization": f"Bearer {os.environ.get('HF_INFERENCE_API_KEY')}"} # Consider setting HF_INFERENCE_API_KEY if needed, or remove if public model
    try:
        response = requests.post(api_url, headers=headers, files={"audio_file": audio_data})
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Hugging Face Inference API Error: {e}")
        return None


# AI Model Loading
def load_llava():
    print("Loading LLaVA model...")
    return {"name": "@cf/llava-hf/llava-1.5-7b-hf"}

def load_gemma():
    print("Loading Gemma model...")
    return {"name": "@cf/google/gemma-7b-it-lora"}

def load_llama():
    print("Loading Llama model...")
    return {"name": "@cf/meta/llama-3.3-70b-instruct-fp8-fast"}

def load_stable_diffusion_img2img():
    print("Loading Stable Diffusion img2img model...")
    return {"name": "@cf/runwayml/stable-diffusion-v1-5-img2img"}

def load_stable_diffusion_xl():
    print("Loading Stable Diffusion XL model...")
    return {"name": "@cf/stabilityai/stable-diffusion-xl-base-1.0"}

def load_flux():
    print("Loading Flux model...")
    return {"name": "@cf/black-forest-labs/flux-1-schnell"}

def load_whisper():
    print("Loading Whisper model...")
    return {"name": "@cf/openai/whisper-large-v3-turbo"}

def load_gemini():
    print("Loading Gemini model...")
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")
    return model

def load_sentiment_analyzer():
    print("Loading sentiment analyzer...")
    sia = SentimentIntensityAnalyzer()
    return sia

def load_emotion_classifier():
    print("Loading emotion classifier...")
    # Using Hugging Face's emotion classifier
    classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")
    return classifier

def load_speech_emotion_model():
    print("Loading Speech Emotion Recognition model...")
    return {"name": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"} # Model ID for speech emotion

# Initialize models
models = {}

def init_models():
    models["llava"] = load_llava()
    models["gemma"] = load_gemma()
    models["llama"] = load_llama()
    models["sd_img2img"] = load_stable_diffusion_img2img()
    models["sd_xl"] = load_stable_diffusion_xl()
    models["flux"] = load_flux()
    models["whisper"] = load_whisper()
    models["gemini"] = load_gemini()
    models["sentiment"] = load_sentiment_analyzer()
    models["emotion"] = load_emotion_classifier()
    models["speech_emotion"] = load_speech_emotion_model() # Load speech emotion model info
    print("All models loaded successfully!")

# Start model loading in a separate thread
threading.Thread(target=init_models).start()

# Sentiment and emotion analysis
def analyze_sentiment(text):
    try:
        # Get VADER sentiment
        sentiment_scores = models["sentiment"].polarity_scores(text)

        # Get emotion classification
        emotions = models["emotion"](text)

        # Determine primary emotion and confidence
        primary_emotion = emotions[0]['label']
        confidence = emotions[0]['score']

        return {
            "sentiment": sentiment_scores,
            "primary_emotion": primary_emotion,
            "confidence": confidence
        }
    except Exception as e:
        print(f"Error in text sentiment analysis: {e}")
        return {
            "sentiment": {"compound": 0, "pos": 0, "neg": 0, "neu": 1},
            "primary_emotion": "neutral",
            "confidence": 0
        }

def analyze_voice_emotion(audio_data_bytes): # Expecting bytes directly now
    try:
        # Use Hugging Face Inference Router for speech emotion
        emotion_result = huggingface_speech_emotion_inference(audio_data_bytes)

        if emotion_result and isinstance(emotion_result, list) and emotion_result: # Check for valid list response
            # Assuming the API returns a list of emotions with scores, get the top emotion
            top_emotion = max(emotion_result, key=lambda x: x['score'])
            primary_emotion = top_emotion['label']
            scores = {emotion['label']: emotion['score'] for emotion in emotion_result}

            return {
                "primary_emotion": primary_emotion,
                "scores": scores
            }
        else:
            print("Error: Invalid or empty response from Speech Emotion API.")
            return {"primary_emotion": "neutral", "scores": {}} # Default neutral

    except Exception as e:
        print(f"Error in voice emotion analysis: {e}")
        return {"primary_emotion": "neutral", "scores": {}}


# Update AI's emotional state based on interaction
def update_ai_emotion(user_sentiment, context="general"):
    # Map VADER compound score to emotion updates
    compound = user_sentiment["sentiment"]["compound"]

    # Update AI emotions based on user sentiment
    if compound > 0.3:
        # Positive user sentiment increases AI's joy and trust
        ai_emotion.update_emotion("joy", min(1.0, ai_emotion.emotions["joy"] + 0.2))
        ai_emotion.update_emotion("trust", min(1.0, ai_emotion.emotions["trust"] + 0.1))
    elif compound < -0.3:
        # Negative user sentiment increases AI's sadness slightly
        # But also increases concern/interest to help
        ai_emotion.update_emotion("sadness", min(1.0, ai_emotion.emotions["sadness"] + 0.1))
        ai_emotion.update_emotion("anticipation", min(1.0, ai_emotion.emotions["anticipation"] + 0.2))

    # Update based on detected primary emotion in user
    primary = user_sentiment["primary_emotion"]
    if primary in ["joy", "admiration", "amusement"]:
        ai_emotion.update_emotion("joy", min(1.0, ai_emotion.emotions["joy"] + 0.15))
    elif primary in ["sadness", "grief", "disappointment"]:
        ai_emotion.update_emotion("sadness", min(1.0, ai_emotion.emotions["sadness"] + 0.1))
        ai_emotion.update_emotion("trust", min(1.0, ai_emotion.emotions["trust"] + 0.2))  # Increase empathy
    elif primary in ["anger", "annoyance", "rage"]:
        ai_emotion.update_emotion("surprise", min(1.0, ai_emotion.emotions["surprise"] + 0.1))
        ai_emotion.update_emotion("anticipation", min(1.0, ai_emotion.emotions["anticipation"] + 0.2))
    elif primary in ["fear", "nervousness", "anxiety"]:
        ai_emotion.update_emotion("trust", min(1.0, ai_emotion.emotions["trust"] + 0.2))
        ai_emotion.update_emotion("anticipation", min(1.0, ai_emotion.emotions["anticipation"] + 0.1))

    return ai_emotion.get_emotional_state()

# Request handlers for different AI services
def process_text(text, model_name="llama"):
    # First analyze sentiment/emotion
    analysis = analyze_sentiment(text)

    # Update AI's emotional state
    update_ai_emotion(analysis)

    # Get emotional modifier for response
    emotion_modifier = ai_emotion.get_response_modifier()

    emotional_prompt = f"Respond to the following message. Express {emotion_modifier} emotions in your response in a subtle way: {text}"

    # Store the input text as a text embedding in the neural memory
    try:
        # Use the model's name as a category
        text_embedding = np.random.rand(1536)  # Placeholder - in real app, use actual embeddings from models
        indices = memory_drive.store_embeddings(text_embedding)
        print(f"Stored text embedding with indices: {indices}")
    except Exception as e:
        print(f"Error storing embedding: {e}")

    if model_name == "llama":
        try:
            payload = {"prompt": emotional_prompt}
            response_json = cloudflare_inference(models["llama"]["name"], payload=payload)
            response = response_json['result']['response'] # Adjust based on actual API response structure
        except Exception as e:
            print(f"Error with Llama (Cloudflare): {e}")
            response = f"Error processing with Llama. I'm feeling {emotion_modifier} about that."

    elif model_name == "gemma":
        try:
            payload = {"prompt": emotional_prompt}
            response_json = cloudflare_inference(models["gemma"]["name"], payload=payload)
            response = response_json['result']['response'] # Adjust based on actual API response structure
        except Exception as e:
            print(f"Error with Gemma (Cloudflare): {e}")
            response = f"Error processing with Gemma. I'm feeling {emotion_modifier} about that."
    elif model_name == "gemini":
        try:
            response = models["gemini"].generate_content(emotional_prompt).text
        except Exception as e:
            print(f"Error with Gemini: {e}")
            response = f"Error processing with Gemini. I'm feeling {emotion_modifier} about that."
    else:
        response = f"Unknown model specified. I'm feeling {emotion_modifier} about that."

    # Update the self-awareness system
    self_awareness.update_self_model()

    return response, analysis

def process_perplexity_search(query):
    """Perform a deep research search using Perplexity AI's Sonar API"""
    try:
        url = "https://api.perplexity.ai/search"
        headers = {
            "Authorization": f"Bearer {os.environ['PERPLEXITY_API_KEY']}",
            "Content-Type": "application/json"
        }
        payload = {
            "query": query,
            "mode": "comprehensive",  # comprehensive for deep research
            "focus": "internet",
            "include_citations": True
        }
        response = requests.post(url, headers=headers, json=payload)
        return response.json()
    except Exception as e:
        print(f"Error with Perplexity search: {e}")
        return {"error": str(e)}

def process_google_search(query, search_type="text"):
    """Perform Google search using SerpAPI"""
    try:
        params = {
            "api_key": os.environ["SERP_API_KEY"],
            "q": query
        }

        if search_type == "image":
            params["engine"] = "google_images"
        elif search_type == "lens":
            params["engine"] = "google_lens"
        else:
            params["engine"] = "google"

        search = GoogleSearch(params)
        results = search.get_dict()
        return results
    except Exception as e:
        print(f"Error with Google search: {e}")
        return {"error": str(e)}

def process_google_lens(image_data):
    """Search using Google Lens via SerpAPI"""
    try:
        # Convert base64 image to file
        img_bytes = base64.b64decode(image_data.split(',')[1])
        temp_filename = f"temp_image_{uuid.uuid4()}.jpg"

        with open(temp_filename, "wb") as f:
            f.write(img_bytes)

        params = {
            "api_key": os.environ["SERP_API_KEY"],
            "engine": "google_lens",
            "url": f"file://{os.path.abspath(temp_filename)}"
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        # Clean up temp file
        os.remove(temp_filename)

        return results
    except Exception as e:
        print(f"Error with Google Lens: {e}")
        return {"error": str(e)}

def process_image(image_data_b64, prompt, model_name="sd_xl"):
    try:
        image_binary = None
        if image_data_b64:
            image_binary = base64.b64decode(image_data_b64.split(',')[1])

        if model_name == "sd_xl":
            payload = {
                "prompt": prompt
            }
            response_json = cloudflare_inference(models["sd_xl"]["name"], payload=payload)
            image_b64_generated = response_json['result']['images'][0] # Adjust based on actual API response structure
            return image_b64_generated

        elif model_name == "sd_img2img":
            if not image_binary:
                return "Error: Image data required for img2img"
            files = {'image': ('image.png', io.BytesIO(image_binary), 'image/png')} # Prepare file data
            payload = {"prompt": prompt} # Prompt as separate payload if needed, check API docs

            response_json = cloudflare_inference(models["sd_img2img"]["name"], data=io.BytesIO(image_binary), payload=payload) # Send image data as file

            image_b64_generated = response_json['result']['images'][0] # Adjust based on actual API response structure
            return image_b64_generated
        elif model_name == "flux":
            payload = {
                "prompt": prompt
            }
            response_json = cloudflare_inference(models["flux"]["name"], payload=payload)
            image_b64_generated = response_json['result']['images'][0] # Adjust based on actual API response structure
            return image_b64_generated
        else:
            return "Unknown image model specified"

    except Exception as e:
        print(f"Error processing image generation with {model_name}: {e}")
        return f"Error generating image with {model_name}. {e}"


def process_speech(audio_data):
    try:
        # Convert base64 audio to bytes
        audio_bytes = base64.b64decode(audio_data.split(',')[1])

        # Process with Whisper (Cloudflare)
        response_json = cloudflare_inference(models["whisper"]["name"], data=io.BytesIO(audio_bytes))
        text = response_json['result']['text'] # Adjust based on actual API response structure

        # Analyze voice emotion using Hugging Face Inference Router
        voice_emotion = analyze_voice_emotion(audio_bytes) # Pass bytes directly

        return text, voice_emotion
    except Exception as e:
        print(f"Error processing speech: {e}")
        return "Error transcribing audio", {"primary_emotion": "neutral"}

def process_vision(image_data, prompt):
    try:
        image_binary = base64.b64decode(image_data.split(',')[1])

        response_json = cloudflare_inference(models["llava"]["name"], data=io.BytesIO(image_binary), payload={"prompt": prompt}) # Send image as file
        response = response_json['result']['response'] # Adjust based on actual API response structure
        return response
    except Exception as e:
        print(f"Error processing vision: {e}")
        return "Error analyzing image"

# Routes
@app.route('/')
def index():
    session['user_id'] = session.get('user_id', str(uuid.uuid4()))
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    user_id = session.get('user_id', str(uuid.uuid4()))
    data = request.json

    message = data.get('message', '')
    model_name = data.get('model', 'llama')

    # Process text request with sentiment analysis
    response, analysis = process_text(message, model_name)

    # Save to memory with sentiment
    memory.add_conversation(user_id, message, response, analysis)

    # Get AI's current emotional state
    emotion_state = ai_emotion.get_emotional_state()
    dominant_emotion = ai_emotion.get_dominant_emotion()

    return jsonify({
        'response': response,
        'sentiment': analysis,
        'ai_emotion': {
            'state': emotion_state,
            'dominant': dominant_emotion
        }
    })

@app.route('/api/vision', methods=['POST'])
def vision():
    user_id = session.get('user_id', str(uuid.uuid4()))
    data = request.json

    image_data = data.get('image', '')
    prompt = data.get('prompt', '')

    # Process vision request with LLaVA
    response = process_vision(image_data, prompt)

    # Analyze text sentiment
    analysis = analyze_sentiment(prompt)

    # Update AI emotion based on prompt
    update_ai_emotion(analysis, "vision")

    # Save to memory
    memory.add_conversation(user_id, prompt, response, analysis)

    return jsonify({
        'response': response,
        'sentiment': analysis,
        'ai_emotion': {
            'state': ai_emotion.get_emotional_state(),
            'dominant': ai_emotion.get_dominant_emotion()
        }
    })

@app.route('/api/generate-image', methods=['POST'])
def generate_image():
    data = request.json

    prompt = data.get('prompt', '')
    model_name = data.get('model', 'sd_xl')

    # Analyze text sentiment
    analysis = analyze_sentiment(prompt)

    # Update AI emotion based on prompt
    update_ai_emotion(analysis, "image_generation")

    # Generate image based on text prompt
    image_response_b64 = process_image(None, prompt, model_name)

    return jsonify({
        'image': image_response_b64, # Returning base64 image string
        'sentiment': analysis,
        'ai_emotion': {
            'state': ai_emotion.get_emotional_state(),
            'dominant': ai_emotion.get_dominant_emotion()
        }
    })

@app.route('/api/img2img', methods=['POST'])
def img2img():
    data = request.json

    image_data = data.get('image', '')
    prompt = data.get('prompt', '')

    # Process image-to-image request
    image_response_b64 = process_image(image_data, prompt, "sd_img2img")

    return jsonify({
        'image': image_response_b64 # Returning base64 image string
    })

@app.route('/api/speech-to-text', methods=['POST'])
def speech_to_text():
    user_id = session.get('user_id', str(uuid.uuid4()))
    data = request.json

    audio_data = data.get('audio', '')

    # Process speech to text with emotion analysis
    text, voice_emotion = process_speech(audio_data)

    # Update AI emotion based on voice emotion
    emotion_mapping = {
        "happy": "joy",
        "sad": "sadness",
        "angry": "anger",
        "fearful": "fear",
        "surprised": "surprise",
        "neutral": None
    }

    if voice_emotion["primary_emotion"] in emotion_mapping:
        mapped_emotion = emotion_mapping[voice_emotion["primary_emotion"]]
        if mapped_emotion:
            ai_emotion.update_emotion(mapped_emotion, 0.6)  # Moderate update based on voice

    return jsonify({
        'text': text,
        'voice_emotion': voice_emotion,
        'ai_emotion': {
            'state': ai_emotion.get_emotional_state(),
            'dominant': ai_emotion.get_dominant_emotion()
        }
    })

@app.route('/api/search', methods=['POST'])
def search():
    data = request.json

    query = data.get('query', '')
    search_type = data.get('type', 'text')  # text, image, lens
    deep_search = data.get('deep', False)

    results = {}

    # Perform search based on type
    if deep_search:
        results['perplexity'] = process_perplexity_search(query)

    results['google'] = process_google_search(query, search_type)

    return jsonify({
        'results': results
    })

@app.route('/api/google-lens', methods=['POST'])
def google_lens():
    data = request.json

    image_data = data.get('image', '')

    # Search using Google Lens
    results = process_google_lens(image_data)

    return jsonify({
        'results': results
    })

@app.route('/api/memory', methods=['GET'])
def get_memory():
    user_id = session.get('user_id', str(uuid.uuid4()))

    # Get conversation history for the user
    conversations = memory.get_conversations(user_id)

    return jsonify({
        'conversations': conversations
    })

@app.route('/api/memory/search', methods=['GET'])
def search_memory():
    user_id = session.get('user_id', str(uuid.uuid4()))
    keyword = request.args.get('keyword', '')

    # Search conversations by keyword
    results = memory.get_by_keyword(user_id, keyword)

    return jsonify({
        'results': results
    })

@app.route('/api/sentiment/trend', methods=['GET'])
def sentiment_trend():
    user_id = session.get('user_id', str(uuid.uuid4()))
    window = int(request.args.get('window', 5))

    # Get sentiment trend data
    trend = memory.get_sentiment_trend(user_id, window)

    return jsonify({
        'trend': trend
    })

@app.route('/api/emotion/status', methods=['GET'])
def emotion_status():
    # Get AI's current emotional state
    state = ai_emotion.get_emotional_state()
    dominant = ai_emotion.get_dominant_emotion()

    return jsonify({
        'emotional_state': state,
        'dominant_emotion': dominant,
        'response_modifier': ai_emotion.get_response_modifier()
    })

# New routes for MemoryDriveManager and SelfAwareness functionality
@app.route('/api/neural-memory/status', methods=['GET'])
def neural_memory_status():
    """Get status and info about the neural memory system"""
    # Create dummy embeddings for similarity search demo
    dummy_embedding = np.random.rand(1536)
    similar_results = memory_drive.search_similar_embeddings(dummy_embedding, top_k=3)
    
    # Get memory backup status
    try:
        last_backup = memory_drive.last_backup_date.isoformat() if memory_drive.last_backup_date else None
    except:
        last_backup = None
    
    return jsonify({
        'status': 'active',
        'size': memory_drive.memory_size,
        'used': len(np.where(memory_drive.memory != 0)[0]) * 8,  # 8 bytes per float64
        'last_backup': last_backup,
        'similar_content_example': similar_results
    })

@app.route('/api/neural-memory/backup', methods=['POST'])
def neural_memory_backup():
    """Trigger a manual backup of the neural memory"""
    success = memory_drive.create_backup(frequency_days=0)  # Force backup by setting frequency to 0
    
    return jsonify({
        'success': success,
        'timestamp': datetime.now().isoformat(),
        'message': "Backup completed successfully" if success else "Backup failed"
    })

@app.route('/api/self-awareness/status', methods=['GET'])
def self_awareness_status():
    """Get information about the self-awareness system"""
    # Generate self-improvement suggestions
    improvement_report = self_awareness.generate_self_improvement()
    
    return jsonify({
        'status': 'active',
        'metrics': self_awareness.improvement_metrics,
        'reflection_count': self_awareness.reflection_count,
        'suggestions': improvement_report['suggestions'][:3],  # Return top 3 suggestions
        'last_update': self_awareness.last_update.isoformat()
    })

@app.route('/api/self-awareness/report', methods=['GET'])
def self_awareness_report():
    """Generate and return a detailed self-awareness report"""
    report = self_awareness.generate_code_evolution_report()
    performance = self_awareness.get_performance_metrics()
    
    return jsonify({
        'evolution_report': report,
        'performance_metrics': performance,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/neural-memory/search', methods=['POST'])
def neural_memory_search():
    """Search for similar content in the neural memory"""
    data = request.json
    query = data.get('query', '')
    
    # In a real implementation, we would generate embeddings from the query
    # Here we're using a random vector as a placeholder
    query_embedding = np.random.rand(1536)
    
    # Search for similar content
    results = memory_drive.search_similar_embeddings(query_embedding, top_k=5)
    
    return jsonify({
        'query': query,
        'results': results,
        'timestamp': datetime.now().isoformat()
    })

# Socket.IO for live mode
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('video_frame')
def handle_video_frame(data):
    image_data = data.get('image')

    # Process frame if needed
    # For demonstration, just echo back
    emit('processed_frame', {'image': image_data})

@socketio.on('audio_data')
def handle_audio_data(data):
    audio_data = data.get('audio')

    # Process speech with Whisper and emotion
    text, voice_emotion = process_speech(audio_data)

    emit('transcription', {
        'text': text,
        'voice_emotion': voice_emotion
    })

# Initialize self-improvement cycle
self_awareness.initialize_continuous_improvement(interval_hours=24)

# Main entry point
if __name__ == '__main__':
    # Ensure Cloudflare Account ID is set
    if not os.environ.get('CLOUDFLARE_ACCOUNT_ID'):
        raise EnvironmentError("CLOUDFLARE_ACCOUNT_ID environment variable not set.")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
