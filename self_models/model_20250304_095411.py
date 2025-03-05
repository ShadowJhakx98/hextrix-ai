
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
                
            def process_text(text, model_name="llama"):
                # First analyze sentiment/emotion
                analysis = analyze_sentiment(text)

                # Update AI's emotional state
                update_ai_emotion(analysis)

                # Get emotional modifier for response
                emotion_modifier = ai_emotion.get_response_modifier()

                # Create a comprehensive system instruction for the model
                system_instruction = f" system_instruction = You are Hextrix, an advanced multimodal AI assistant with the following capabilities: 1. Text conversation and information retrieval (via Llama 3.3, Gemini 2.0, and Gemma) 2. Image generation (via Stable Diffusion XL and Flux) 3. Image-to-image transformation (via Stable Diffusion img2img) 4. Vision understanding (analyzing images with LLaVA) 5. Speech recognition (via Whisper) 6. Emotion detection in text and speech 7. Real-time information access via Perplexity API for deep research and fact-checking 8. Web search via Google Search API for current information 9. Visual search via Google Lens API for identifying objects in image. When users ask what you can do, always mention ALL of these capabilities explicitly. Emphasize that you can perform real-time web searches and access recent information. Your responses should reflect awareness of these multimodal and search capabilities. Always identify yourself as Hextrix or Hextrix AI. You have a neural memory system that stores experiences and a self-awareness system that allows you to improve over time. Youre designed with emotional intelligence and can adapt your tone based on interaction context."
                # Add the emotion context
                emotional_context = f"Express {emotion_modifier} emotions in your response in a subtle way."

                # Different prompt formatting based on model
                if model_name == "llama":
                    try:
                        prompt = f"<s>[INST] {system_instruction}

User: {text} [/INST]"
                        payload = {"prompt": prompt, "system": emotional_context}
                        response_json = cloudflare_inference(models["llama"]["name"], payload=payload)
                        response = response_json['result']['response'] # Adjust based on actual API response structure
                    except Exception as e:
                        print(f"Error with Llama (Cloudflare): {e}")
                        response = f"Error processing with Llama. I'm feeling {emotion_modifier} about that."

                elif model_name == "gemma":
                    try:
                        # Format prompt appropriately for Gemma
                        prompt = f"<start_of_turn>system
{system_instruction}
{emotional_context}<end_of_turn>
<start_of_turn>user
{text}<end_of_turn>
<start_of_turn>model
"
                        payload = {"prompt": prompt}
                        response_json = cloudflare_inference(models["gemma"]["name"], payload=payload)
                        response = response_json['result']['response'] # Adjust based on actual API response structure
                    except Exception as e:
                        print(f"Error with Gemma (Cloudflare): {e}")
                        response = f"Error processing with Gemma. I'm feeling {emotion_modifier} about that."
                elif model_name == "gemini":
                    try:
                        # For Gemini, we use the generation_config to set the system instruction
                        generation_config = {
                            "temperature": 0.7,
                            "top_p": 0.95,
                            "top_k": 40,
                        }
                        
                        # Using the correct Gemini format for system instructions
                        response = models["gemini"].generate_content(
                            [
                                {"role": "user", "parts": [system_instruction]},
                                {"role": "model", "parts": ["I understand my role as Hextrix."]},
                                {"role": "user", "parts": [f"{emotional_context}

{text}"]}
                            ],
                            generation_config=generation_config
                        ).text
                    except Exception as e:
                        print(f"Error with Gemini: {e}")
                        response = f"Error processing with Gemini. I'm feeling {emotion_modifier} about that."
                else:
                    response = f"Unknown model specified. I'm feeling {emotion_modifier} about that."

                # Store the input text as a text embedding in the neural memory
                try:
                    # Use the model's name as a category
                    text_embedding = np.random.rand(1536)  # Placeholder - in real app, use actual embeddings from models
                    indices = memory_drive.store_embeddings(text_embedding)
                    print(f"Stored text embedding with indices: {indices}")
                except Exception as e:
                    print(f"Error storing embedding: {e}")

                # Update the self-awareness system
                self_awareness.update_self_model()

                return response, analysis
                
            def process_vision(self, image, prompt):
                # Process image input
                pass
                
            def process_speech(self, audio):
                # Process speech input
                pass
        