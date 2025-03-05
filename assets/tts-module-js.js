/**
 * Text-to-Speech Module
 * 
 * This JavaScript implementation uses the Web Speech API's SpeechSynthesis
 * interface instead of gTTS which is used in the Python version.
 */

class TTSHandler {
  /**
   * Handles text-to-speech functionalities.
   * @param {string} outputDir - Directory for saving output (not used in browser version)
   * @param {string} voice - Voice identifier or language code
   */
  constructor(outputDir = "", voice = "en-US") {
    this.outputDir = outputDir; // Not used in browser implementation
    this.voice = voice;
    
    // Check if speech synthesis is available
    if (!window.speechSynthesis) {
      console.error("Speech synthesis not supported in this browser");
      throw new Error("Speech synthesis not supported");
    }
    
    console.log("TTSHandler initialized");
  }

  /**
   * Converts text to speech and plays it using Web Speech API.
   * @param {string} text - The text to convert to speech
   * @returns {Promise<void>}
   */
  text_to_speech(text) {
    return new Promise((resolve, reject) => {
      try {
        const utterance = new SpeechSynthesisUtterance(text);
        
        // Set the language
        utterance.lang = this.voice;
        
        // Try to find a voice that matches the language
        const voices = window.speechSynthesis.getVoices();
        const matchingVoice = voices.find(v => v.lang.includes(this.voice.split('-')[0]));
        
        if (matchingVoice) {
          utterance.voice = matchingVoice;
        }
        
        // Event handlers
        utterance.onend = () => {
          console.log("Text-to-speech processing complete.");
          resolve();
        };
        
        utterance.onerror = (event) => {
          console.error(`Error during text-to-speech processing: ${event.error}`);
          reject(new Error(`TTS error: ${event.error}`));
        };
        
        // Speak the text
        window.speechSynthesis.speak(utterance);
      } catch (e) {
        console.error(`Error during text-to-speech processing: ${e}`);
        reject(e);
      }
    });
  }
  
  /**
   * Get available voices
   * @returns {Array} - List of available voices
   */
  getAvailableVoices() {
    return window.speechSynthesis.getVoices();
  }
}

export { TTSHandler };
