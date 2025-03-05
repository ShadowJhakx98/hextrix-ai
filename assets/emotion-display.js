/**
 * Emotion Display Handler
 * Connects the EmotionEngine to the frontend UI
 */

import { EmotionalVector, EmotionEngine } from './emotions-js.js';

class EmotionDisplay {
  constructor() {
    this.emotionEngine = new EmotionEngine();
    this.currentMood = 'neutral';
    this.emotionUpdateInterval = null;
    this.emotionHistory = [];
    this.maxHistoryLength = 10;
  }

  /**
   * Initialize the emotion display system
   */
  initialize() {
    // Create emotion indicator in the UI
    this._createEmotionIndicator();
    
    // Start periodic updates
    this._startEmotionUpdates();
    
    // Add event listeners for user interactions
    this._addInteractionListeners();
  }

  /**
   * Create the emotion indicator element in the UI
   */
  _createEmotionIndicator() {
    const container = document.querySelector('#conversation-container');
    if (!container) return;
    
    // Create emotion status element if it doesn't exist
    if (!document.querySelector('#emotion-status')) {
      const emotionStatus = document.createElement('div');
      emotionStatus.id = 'emotion-status';
      emotionStatus.className = 'emotion-status';
      emotionStatus.innerHTML = `
        <div class="emotion-indicator">
          <span class="emotion-label">AI Emotional State:</span>
          <span class="emotion-value">Neutral</span>
        </div>
        <div class="emotion-details hidden">
          <div class="emotion-bars">
            <div class="emotion-bar">
              <span>Joy</span>
              <div class="bar-container"><div class="bar" data-emotion="joy" style="width: 50%"></div></div>
            </div>
            <div class="emotion-bar">
              <span>Sadness</span>
              <div class="bar-container"><div class="bar" data-emotion="sadness" style="width: 50%"></div></div>
            </div>
            <div class="emotion-bar">
              <span>Anger</span>
              <div class="bar-container"><div class="bar" data-emotion="anger" style="width: 50%"></div></div>
            </div>
            <div class="emotion-bar">
              <span>Fear</span>
              <div class="bar-container"><div class="bar" data-emotion="fear" style="width: 50%"></div></div>
            </div>
            <div class="emotion-bar">
              <span>Disgust</span>
              <div class="bar-container"><div class="bar" data-emotion="disgust" style="width: 50%"></div></div>
            </div>
            <div class="emotion-bar">
              <span>Surprise</span>
              <div class="bar-container"><div class="bar" data-emotion="surprise" style="width: 50%"></div></div>
            </div>
          </div>
          <div class="mood-meter">
            <span>Mood: </span>
            <span class="mood-value">Neutral</span>
          </div>
        </div>
      `;
      
      // Add toggle functionality
      const indicator = emotionStatus.querySelector('.emotion-indicator');
      indicator.addEventListener('click', () => {
        const details = emotionStatus.querySelector('.emotion-details');
        details.classList.toggle('hidden');
      });
      
      // Insert at the top of the conversation container
      container.insertBefore(emotionStatus, container.firstChild);
      
      // Add styles
      this._addEmotionStyles();
    }
  }

  /**
   * Add CSS styles for the emotion display
   */
  _addEmotionStyles() {
    if (!document.querySelector('#emotion-styles')) {
      const styleEl = document.createElement('style');
      styleEl.id = 'emotion-styles';
      styleEl.textContent = `
        .emotion-status {
          background-color: #f5f5f5;
          border-radius: 8px;
          padding: 10px;
          margin-bottom: 15px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .emotion-indicator {
          display: flex;
          justify-content: space-between;
          cursor: pointer;
          font-weight: bold;
        }
        
        .emotion-details {
          margin-top: 10px;
          padding-top: 10px;
          border-top: 1px solid #ddd;
        }
        
        .emotion-details.hidden {
          display: none;
        }
        
        .emotion-bar {
          display: flex;
          align-items: center;
          margin-bottom: 6px;
        }
        
        .emotion-bar span {
          width: 70px;
        }
        
        .bar-container {
          flex-grow: 1;
          background-color: #e0e0e0;
          height: 8px;
          border-radius: 4px;
          overflow: hidden;
        }
        
        .bar {
          height: 100%;
          background-color: #4a90e2;
          transition: width 0.5s ease;
        }
        
        .bar[data-emotion="joy"] { background-color: #FFD700; }
        .bar[data-emotion="sadness"] { background-color: #6495ED; }
        .bar[data-emotion="anger"] { background-color: #FF4500; }
        .bar[data-emotion="fear"] { background-color: #9370DB; }
        .bar[data-emotion="disgust"] { background-color: #32CD32; }
        .bar[data-emotion="surprise"] { background-color: #FF69B4; }
        
        .mood-meter {
          margin-top: 10px;
          font-style: italic;
        }
      `;
      document.head.appendChild(styleEl);
    }
  }

  /**
   * Start periodic emotion updates
   */
  _startEmotionUpdates() {
    // Clear any existing interval
    if (this.emotionUpdateInterval) {
      clearInterval(this.emotionUpdateInterval);
    }
    
    // Update emotions every 5 seconds with small random changes
    this.emotionUpdateInterval = setInterval(() => {
      this._simulateEmotionChanges();
      this.updateEmotionDisplay();
    }, 5000);
  }

  /**
   * Add event listeners for user interactions
   */
  _addInteractionListeners() {
    // Listen for form submissions
    const submitBtn = document.querySelector('#submit-btn');
    if (submitBtn) {
      submitBtn.addEventListener('click', () => {
        const userInput = document.querySelector('#user-input')?.value;
        if (userInput) {
          this._processUserInput(userInput);
        }
      });
    }
  }

  /**
   * Process user input to update AI emotions
   */
  _processUserInput(input) {
    // Simple sentiment analysis for demo purposes
    const positiveWords = ['happy', 'good', 'great', 'excellent', 'love', 'like', 'wonderful', 'amazing'];
    const negativeWords = ['sad', 'bad', 'terrible', 'awful', 'hate', 'dislike', 'horrible', 'angry'];
    
    const words = input.toLowerCase().split(/\s+/);
    
    let positiveCount = 0;
    let negativeCount = 0;
    
    words.forEach(word => {
      if (positiveWords.includes(word)) positiveCount++;
      if (negativeWords.includes(word)) negativeCount++;
    });
    
    // Create stimulus based on sentiment
    const stimulus = {};
    
    if (positiveCount > 0) {
      stimulus.joy = 0.2 * positiveCount;
      stimulus.valence = 0.2 * positiveCount;
    }
    
    if (negativeCount > 0) {
      stimulus.sadness = 0.2 * negativeCount;
      stimulus.valence = -0.2 * negativeCount;
    }
    
    // Add some arousal based on question marks and exclamation points
    const excitementLevel = (input.match(/[!?]/g) || []).length;
    if (excitementLevel > 0) {
      stimulus.arousal = 0.1 * excitementLevel;
      stimulus.surprise = 0.1 * excitementLevel;
    }
    
    // Update emotion engine
    if (Object.keys(stimulus).length > 0) {
      this.emotionEngine.update_emotion(stimulus);
      this.updateEmotionDisplay();
    }
  }

  /**
   * Simulate small changes in emotions over time
   */
  _simulateEmotionChanges() {
    // Small random changes to create a sense of "aliveness"
    const smallChange = () => (Math.random() * 0.1) - 0.05;
    
    const stimulus = {
      joy: smallChange(),
      sadness: smallChange(),
      anger: smallChange(),
      fear: smallChange(),
      disgust: smallChange(),
      surprise: smallChange(),
      valence: smallChange(),
      arousal: smallChange()
    };
    
    this.emotionEngine.update_emotion(stimulus);
    
    // Record emotion history
    this.emotionHistory.push({
      timestamp: new Date(),
      state: JSON.parse(JSON.stringify(this.emotionEngine.current_state))
    });
    
    // Limit history length
    if (this.emotionHistory.length > this.maxHistoryLength) {
      this.emotionHistory.shift();
    }
  }

  /**
   * Update the emotion display in the UI
   */
  updateEmotionDisplay() {
    const emotionStatus = document.querySelector('#emotion-status');
    if (!emotionStatus) return;
    
    const emotionValue = emotionStatus.querySelector('.emotion-value');
    const moodValue = emotionStatus.querySelector('.mood-value');
    
    // Get current emotional state
    const state = this.emotionEngine.current_state;
    
    // Update emotion bars
    Object.entries(state.base_emotions).forEach(([emotion, value]) => {
      const bar = emotionStatus.querySelector(`.bar[data-emotion="${emotion}"]`);
      if (bar) {
        bar.style.width = `${value * 100}%`;
      }
    });
    
    // Determine dominant emotion
    let dominantEmotion = 'neutral';
    let maxValue = 0.5; // Threshold
    
    Object.entries(state.base_emotions).forEach(([emotion, value]) => {
      if (value > maxValue) {
        maxValue = value;
        dominantEmotion = emotion;
      }
    });
    
    // Set mood text based on valence
    let moodText = 'Neutral';
    if (state.mood > 0.3) moodText = 'Positive';
    if (state.mood > 0.6) moodText = 'Very Positive';
    if (state.mood < -0.3) moodText = 'Negative';
    if (state.mood < -0.6) moodText = 'Very Negative';
    
    // Update UI
    emotionValue.textContent = dominantEmotion.charAt(0).toUpperCase() + dominantEmotion.slice(1);
    moodValue.textContent = moodText;
    
    // Store current mood
    this.currentMood = dominantEmotion;
  }

  /**
   * Get the current emotional state
   */
  getCurrentEmotionalState() {
    return this.emotionEngine.current_state;
  }

  /**
   * Get emotion history
   */
  getEmotionHistory() {
    return this.emotionHistory;
  }

  /**
   * Apply emotional contagion from external source
   */
  applyEmotionalContagion(externalEmotion, socialParams = {
    proximity: 0.5,
    authority: 0.3,
    similarity: 0.4
  }) {
    this.emotionEngine.apply_contagion(externalEmotion, socialParams);
    this.updateEmotionDisplay();
  }
}

// Create and export singleton instance
const emotionDisplay = new EmotionDisplay();
export default emotionDisplay;