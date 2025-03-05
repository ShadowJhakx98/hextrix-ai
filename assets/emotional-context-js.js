/**
 * Unified Emotional Context Manager
 */

import { EmotionEngine } from './emotions-js.js';

class EmotionalContext {
  constructor(window_size = 30) {
    this.modalities = {
      'audio': [],
      'gesture': [],
      'fusion': []
    };
    this.emotionEngine = new EmotionEngine();
    
    // Implement maxlen functionality for the deques
    Object.keys(this.modalities).forEach(key => {
      Object.defineProperty(this.modalities[key], 'push', {
        configurable: true,
        enumerable: false,
        writable: true,
        value: function(...items) {
          Array.prototype.push.apply(this, items);
          while (this.length > window_size) {
            this.shift();
          }
          return this.length;
        }
      });
    });
    
    this.weights = {
      'audio': 0.6,
      'gesture': 0.4
    };
  }

  update_modality(modality, state) {
    /**
     * Update emotional state for a specific modality
     */
    this.modalities[modality].push(state);
    this._update_fusion();

    // Update emotion engine with the new state
    const stimulus = {
      valence: state.valence,
      arousal: state.arousal,
      // Map valence to joy/sadness
      joy: Math.max(0, state.valence),
      sadness: Math.max(0, -state.valence)
    };
    this.emotionEngine.update_emotion(stimulus);
  }

  _update_fusion() {
    /**
     * Temporal fusion of emotional states
     */
    if (this.modalities['audio'].length === 0 || this.modalities['gesture'].length === 0) {
      return;
    }

    const audio_states = this.modalities['audio'].map(s => s.valence);
    const gesture_states = this.modalities['gesture'].map(s => s.valence);
    
    // Align temporal windows
    const min_len = Math.min(audio_states.length, gesture_states.length);
    const audio_subset = audio_states.slice(-min_len);
    const gesture_subset = gesture_states.slice(-min_len);
    
    const fused_valence = audio_subset.map((val, i) => 
      this.weights['audio'] * val + this.weights['gesture'] * gesture_subset[i]
    );
    
    // Apply exponential smoothing
    const smoothed = new Array(min_len).fill(0);
    const alpha = 0.2;
    
    for (let i = 0; i < min_len; i++) {
      smoothed[i] = alpha * fused_valence[i];
      if (i > 0) {
        smoothed[i] += (1 - alpha) * smoothed[i - 1];
      }
    }

    // Add to fusion modality
    this.modalities['fusion'].push(...smoothed.map(v => ({ valence: v, arousal: 0.5 })));
  }

  get_current_state() {
    /**
     * Get fused emotional state with temporal context and emotion engine state
     */
    const fusedState = this.modalities['fusion'].length === 0 ? 
      { valence: 0.5, arousal: 0.5 } : 
      {
        valence: this.modalities['fusion'].map(s => s.valence).reduce((a, b) => a + b, 0) / this.modalities['fusion'].length,
        arousal: this.modalities['fusion'].map(s => s.arousal).reduce((a, b) => a + b, 0) / this.modalities['fusion'].length
      };

    // Combine with emotion engine state
    return {
      ...fusedState,
      base_emotions: this.emotionEngine.current_state.base_emotions,
      mood: this.emotionEngine.current_state.mood,
      stability: this.emotionEngine.current_state.stability
    };
  }
}

export { EmotionalContext };
