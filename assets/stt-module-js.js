/**
 * Speech-to-Text Module
 * 
 * Note: This JavaScript implementation would typically use the Web Speech API
 * or another speech recognition service compatible with browsers.
 */

class STTHandlerError extends Error {
  /**
   * Custom exception for STTHandler errors.
   */
  constructor(message) {
    super(message);
    this.name = "STTHandlerError";
  }
}

class STTHandler {
  /**
   * Handles speech-to-text functionalities.
   */
  constructor(language = "en-US") {
    // Check if SpeechRecognition is available in the browser
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
      console.error("Speech recognition not supported in this browser");
      throw new STTHandlerError("Speech recognition not supported");
    }
    
    this.recognizer = new SpeechRecognition();
    this.recognizer.lang = language;
    this.recognizer.continuous = false;
    this.recognizer.interimResults = false;
    this.language = language;
    
    console.log("STTHandler initialized");
  }

  /**
   * Converts speech from the microphone to text.
   * @returns {Promise<string>} - The recognized text
   */
  speech_to_text() {
    return new Promise((resolve, reject) => {
      console.log("Listening...");
      
      this.recognizer.onresult = (event) => {
        const text = event.results[0][0].transcript;
        console.log(`Recognized text: ${text}`);
        resolve(text);
      };
      
      this.recognizer.onerror = (event) => {
        console.error(`Speech recognition error: ${event.error}`);
        if (event.error === 'no-speech') {
          resolve(""); // No speech detected, return empty string
        } else {
          reject(new STTHandlerError(`Error during speech-to-text: ${event.error}`));
        }
      };
      
      try {
        this.recognizer.start();
      } catch (e) {
        console.error(`Failed to start speech recognition: ${e}`);
        reject(new STTHandlerError(`Failed to start speech recognition: ${e.message}`));
      }
    });
  }
}

export { STTHandler, STTHandlerError };
