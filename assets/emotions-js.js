/**
 * Multidimensional emotional state tracking with dynamic contagion modeling
 */

class EmotionalVector {
  constructor() {
    this.base_emotions = {
      'joy': 0.5,
      'sadness': 0.5,
      'anger': 0.5,
      'fear': 0.5,
      'disgust': 0.5,
      'surprise': 0.5
    };
    this.mood = 0.0;  // [-1,1] scale
    this.valence = 0.0;
    this.arousal = 0.0;
    this.stability = 1.0;
  }

  normalize() {
    /**
     * Constrain emotions to valid ranges
     */
    for (const k in this.base_emotions) {
      this.base_emotions[k] = Math.max(0, Math.min(1, this.base_emotions[k]));
    }
    this.mood = Math.max(-1, Math.min(1, this.mood));
    this.valence = Math.max(-1, Math.min(1, this.valence));
    this.arousal = Math.max(0, Math.min(1, this.arousal));
    this.stability = Math.max(0, Math.min(1, this.stability));
  }
}

class EmotionEngine {
  constructor() {
    this.current_state = new EmotionalVector();
    this.history = [];
    this.decay_rates = {
      'joy': 0.95,
      'sadness': 0.90,
      'anger': 0.85,
      'fear': 0.92,
      'disgust': 0.88,
      'surprise': 0.75
    };
    this.social_contagion_factors = this._init_contagion_model();
  }

  _init_contagion_model() {
    return {
      'emotional_congruence': 0.65,
      'social_proximity': 0.80,
      'authority_bias': 0.70,
      'group_dynamics': 0.75
    };
  }

  update_emotion(stimulus) {
    /**
     * Process emotional stimulus using differential equations
     */
    this.history.push([new Date(), this.current_state]);

    // Convert stimulus to emotion delta
    const delta = this._stimulus_to_delta(stimulus);

    // Solve emotional dynamics ODE
    const t = this._linspace(0, 1, 10);
    const y0 = [...Object.values(this.current_state.base_emotions),
                this.current_state.mood, 
                this.current_state.valence,
                this.current_state.arousal];

    const solution = this._odeint(this._emotion_ode.bind(this), y0, t, delta);

    // Update state with new values
    this._apply_ode_solution(solution[solution.length - 1]);
    this.current_state.normalize();
  }

  _stimulus_to_delta(stimulus) {
    /**
     * Convert stimulus description to emotion deltas
     */
    const delta = new Array(9).fill(0);
    const emotion_map = {
      'joy': 0,
      'sadness': 1,
      'anger': 2,
      'fear': 3,
      'disgust': 4,
      'surprise': 5,
      'mood': 6,
      'valence': 7,
      'arousal': 8
    };

    for (const [k, v] of Object.entries(stimulus)) {
      if (k in emotion_map) {
        delta[emotion_map[k]] = v * 0.1;  // Scale stimulus impact
      }
    }

    return delta;
  }

  _emotion_ode(y, t, delta) {
    /**
     * Differential equations governing emotional dynamics
     */
    const dydt = new Array(y.length).fill(0);
    const emotionKeys = Object.keys(this.current_state.base_emotions);

    // Base emotion dynamics
    for (let i = 0; i < 6; i++) {
      dydt[i] = (delta[i] - y[i]) * (1 - this.decay_rates[emotionKeys[i]]);
    }

    // Mood dynamics
    dydt[6] = (this._mean(y.slice(0, 6)) - 0.5) * 0.1 + delta[6];

    // Valence-arousal dynamics
    dydt[7] = (y[6] * 0.8 + delta[7]) * 0.5;
    dydt[8] = (this._std(y.slice(0, 6)) * 1.2 + delta[8]) * 0.3;

    return dydt;
  }

  _apply_ode_solution(solution) {
    /**
     * Map ODE solution back to emotional state
     */
    this.current_state.base_emotions = {
      'joy': solution[0],
      'sadness': solution[1],
      'anger': solution[2],
      'fear': solution[3],
      'disgust': solution[4],
      'surprise': solution[5]
    };
    this.current_state.mood = solution[6];
    this.current_state.valence = solution[7];
    this.current_state.arousal = solution[8];
  }

  apply_contagion(external_emotion, social_params) {
    /**
     * Model emotional contagion from external sources
     */
    const contagion_effect = new Array(6).fill(0);
    const weights = [
      social_params.proximity || 0.5,
      social_params.authority || 0.3,
      social_params.similarity || 0.4
    ];
    const total_weight = weights.reduce((a, b) => a + b, 0);
    const emotions = Object.keys(this.current_state.base_emotions);

    for (let i = 0; i < emotions.length; i++) {
      const emotion = emotions[i];
      const external_val = external_emotion[emotion] || 0.5;
      const difference = external_val - this.current_state.base_emotions[emotion];
      contagion_effect[i] = difference * total_weight * 0.1;
    }

    const stimulus = {};
    for (let i = 0; i < emotions.length; i++) {
      stimulus[emotions[i]] = contagion_effect[i];
    }
    
    this.update_emotion(stimulus);
  }

  get_emotional_synergy(other_emotion) {
    /**
     * Calculate emotional alignment with another entity
     */
    const vec_self = Object.values(this.current_state.base_emotions);
    const vec_other = Object.keys(this.current_state.base_emotions)
      .map(k => other_emotion[k] || 0.5);

    return this._dot(vec_self, vec_other) / 
           (this._norm(vec_self) * this._norm(vec_other));
  }

  // Helper numerical methods
  _linspace(start, stop, num) {
    const step = (stop - start) / (num - 1);
    return Array.from({length: num}, (_, i) => start + step * i);
  }

  _mean(arr) {
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  _std(arr) {
    const mean = this._mean(arr);
    return Math.sqrt(arr.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b, 0) / arr.length);
  }

  _dot(a, b) {
    return a.reduce((sum, val, i) => sum + val * b[i], 0);
  }

  _norm(a) {
    return Math.sqrt(this._dot(a, a));
  }

  _odeint(func, y0, t, args) {
    const dt = t[1] - t[0];
    const y = [y0];
    
    for (let i = 1; i < t.length; i++) {
      const dydt = func(y[i-1], t[i-1], args);
      const new_y = y[i-1].map((val, j) => val + dydt[j] * dt);
      y.push(new_y);
    }
    
    return y;
  }
}

class EmotionalMemory {
  constructor(decay_rate = 0.95) {
    this.memory_traces = [];
    this.decay_rate = decay_rate;
  }

  add_experience(emotion_state, context) {
    /**
     * Store emotional experience with context
     */
    this.memory_traces.push({
      'timestamp': new Date(),
      'emotion': emotion_state,
      'context': context,
      'strength': 1.0
    });
  }

  decay_memories() {
    /**
     * Apply temporal decay to emotional memories
     */
    for (const trace of this.memory_traces) {
      const age = (new Date() - trace.timestamp) / 3600000; // milliseconds to hours
      trace.strength *= Math.pow(this.decay_rate, age);
    }

    // Remove faded memories
    this.memory_traces = this.memory_traces.filter(t => t.strength > 0.1);
  }
}

export { EmotionalVector, EmotionEngine, EmotionalMemory };
