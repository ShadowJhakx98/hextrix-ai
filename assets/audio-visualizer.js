/**
 * Advanced Audio Visualizer
 * Provides real-time audio visualization for the Hextrix interface
 */

class AudioVisualizer {
  constructor(scene, position = { x: 0, y: 0, z: 0 }, size = { width: 4, height: 1 }) {
    this.scene = scene;
    this.position = position;
    this.size = size;
    this.audioContext = null;
    this.analyser = null;
    this.dataArray = null;
    this.source = null;
    this.visualizerMesh = null;
    this.audioBars = [];
    this.barCount = 32;
    this.isActive = false;
    this.visualizationMode = 'bars'; // 'bars', 'wave', or 'circle'
    this.colorMode = 'spectrum'; // 'spectrum', 'single', or 'emotion'
    this.baseColor = 0x4a7bff;
    this.emotionColor = 0x4a7bff;
  }

  /**
   * Initialize the audio visualizer
   */
  async initialize() {
    if (!window.THREE) {
      console.error('THREE.js is required but not loaded');
      return false;
    }

    try {
      // Create audio context
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      this.audioContext = new AudioContext();
      
      // Create analyser node
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 256;
      this.analyser.smoothingTimeConstant = 0.8;
      
      // Create data array for frequency data
      const bufferLength = this.analyser.frequencyBinCount;
      this.dataArray = new Uint8Array(bufferLength);
      
      // Create visualizer container
      this.createVisualizerContainer();
      
      // Create audio bars
      this.createAudioBars();
      
      return true;
    } catch (error) {
      console.error('Error initializing audio visualizer:', error);
      return false;
    }
  }

  /**
   * Create visualizer container
   */
  createVisualizerContainer() {
    // Create panel for visualizer
    const panelGeometry = new THREE.PlaneGeometry(this.size.width, this.size.height);
    const panelMaterial = new THREE.MeshPhongMaterial({
      color: 0x000000,
      emissive: 0x4a7bff,
      emissiveIntensity: 0.1,
      transparent: true,
      opacity: 0.3,
      side: THREE.DoubleSide
    });
    
    this.visualizerMesh = new THREE.Mesh(panelGeometry, panelMaterial);
    this.visualizerMesh.position.set(this.position.x, this.position.y, this.position.z);
    this.scene.add(this.visualizerMesh);
  }

  /**
   * Create audio bars for visualization
   */
  createAudioBars() {
    // Clear existing bars
    this.audioBars.forEach(bar => {
      if (bar && this.scene) {
        this.scene.remove(bar);
      }
    });
    this.audioBars = [];
    
    // Create new bars
    const barWidth = (this.size.width * 0.95) / this.barCount;
    const startX = this.position.x - (this.size.width / 2) + (barWidth / 2);
    
    for (let i = 0; i < this.barCount; i++) {
      const barGeometry = new THREE.BoxGeometry(barWidth * 0.8, 0.1, 0.1);
      const barMaterial = new THREE.MeshPhongMaterial({
        color: this.baseColor,
        emissive: this.baseColor,
        emissiveIntensity: 0.5,
        transparent: true,
        opacity: 0.8
      });
      
      const bar = new THREE.Mesh(barGeometry, barMaterial);
      const x = startX + (i * barWidth);
      bar.position.set(x, this.position.y, this.position.z + 0.1);
      this.scene.add(bar);
      this.audioBars.push(bar);
    }
  }

  /**
   * Connect to audio source
   * @param {MediaStream|AudioNode} source - Audio source to visualize
   */
  connectSource(source) {
    // Disconnect any existing source
    if (this.source) {
      this.source.disconnect();
    }
    
    try {
      if (source instanceof MediaStream) {
        // For microphone input
        this.source = this.audioContext.createMediaStreamSource(source);
      } else if (source instanceof AudioNode) {
        // For other audio nodes
        this.source = source;
      } else {
        console.error('Invalid audio source');
        return false;
      }
      
      // Connect source to analyser
      this.source.connect(this.analyser);
      this.isActive = true;
      return true;
    } catch (error) {
      console.error('Error connecting audio source:', error);
      return false;
    }
  }

  /**
   * Connect to audio element
   * @param {HTMLAudioElement} audioElement - Audio element to visualize
   */
  connectAudioElement(audioElement) {
    try {
      const source = this.audioContext.createMediaElementSource(audioElement);
      source.connect(this.audioContext.destination); // Connect to speakers
      return this.connectSource(source);
    } catch (error) {
      console.error('Error connecting audio element:', error);
      return false;
    }
  }

  /**
   * Start microphone input
   */
  async startMicrophone() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      return this.connectSource(stream);
    } catch (error) {
      console.error('Error accessing microphone:', error);
      return false;
    }
  }

  /**
   * Update visualization
   */
  update() {
    if (!this.isActive || !this.analyser || !this.dataArray) return;
    
    // Get frequency data
    this.analyser.getByteFrequencyData(this.dataArray);
    
    // Update visualization based on mode
    switch (this.visualizationMode) {
      case 'bars':
        this.updateBars();
        break;
      case 'wave':
        this.updateWave();
        break;
      case 'circle':
        this.updateCircle();
        break;
    }
  }

  /**
   * Update bar visualization
   */
  updateBars() {
    if (!this.audioBars.length) return;
    
    const barCount = this.audioBars.length;
    const step = Math.floor(this.dataArray.length / barCount);
    
    for (let i = 0; i < barCount; i++) {
      const value = this.dataArray[i * step] / 255;
      const bar = this.audioBars[i];
      
      if (bar) {
        // Update bar height
        bar.scale.y = value * 5 + 0.1;
        bar.position.y = this.position.y + (value * 2.5);
        
        // Update bar color based on color mode
        if (this.colorMode === 'spectrum') {
          const hue = (i / barCount) * 0.3 + 0.6; // Blue to purple range
          const color = new THREE.Color().setHSL(hue, 0.8, 0.5);
          bar.material.color.set(color);
          bar.material.emissive.set(color);
        } else if (this.colorMode === 'emotion') {
          bar.material.color.set(this.emotionColor);
          bar.material.emissive.set(this.emotionColor);
        }
        
        // Update emissive intensity based on value
        bar.material.emissiveIntensity = value * 0.5 + 0.3;
      }
    }
  }

  /**
   * Update wave visualization
   */
  updateWave() {
    // Implementation for wave visualization
    // This would create/update a line geometry based on audio data
    // For simplicity, we'll use the bars in a different configuration
    if (!this.audioBars.length) return;
    
    const barCount = this.audioBars.length;
    this.analyser.getByteTimeDomainData(this.dataArray);
    
    for (let i = 0; i < barCount; i++) {
      const index = Math.floor(i * (this.dataArray.length / barCount));
      const value = ((this.dataArray[index] / 128.0) - 1) * 0.5; // Convert to -0.5 to 0.5 range
      const bar = this.audioBars[i];
      
      if (bar) {
        // Position bars in a wave pattern
        bar.position.y = this.position.y + value;
        bar.scale.y = 0.1; // Fixed height
        
        // Update bar color
        if (this.colorMode === 'spectrum') {
          const hue = 0.6 + (value + 0.5) * 0.2; // Blue to purple based on position
          const color = new THREE.Color().setHSL(hue, 0.8, 0.5);
          bar.material.color.set(color);
          bar.material.emissive.set(color);
        }
      }
    }
  }

  /**
   * Update circle visualization
   */
  updateCircle() {
    // Implementation for circular visualization
    // This would arrange the bars in a circle pattern
    if (!this.audioBars.length) return;
    
    const barCount = this.audioBars.length;
    const radius = this.size.width / 3;
    
    for (let i = 0; i < barCount; i++) {
      const value = this.dataArray[i * Math.floor(this.dataArray.length / barCount)] / 255;
      const bar = this.audioBars[i];
      
      if (bar) {
        // Calculate position on circle
        const angle = (i / barCount) * Math.PI * 2;
        const x = this.position.x + Math.cos(angle) * radius;
        const y = this.position.y + Math.sin(angle) * radius;
        
        // Update bar position and rotation
        bar.position.set(x, y, this.position.z + 0.1);
        bar.rotation.z = angle + Math.PI / 2;
        
        // Update bar scale
        bar.scale.y = value * 2 + 0.5;
        
        // Update bar color
        if (this.colorMode === 'spectrum') {
          const hue = (i / barCount) * 0.8 + 0.6; // Full spectrum
          const color = new THREE.Color().setHSL(hue, 0.8, 0.5);
          bar.material.color.set(color);
          bar.material.emissive.set(color);
        }
      }
    }
  }

  /**
   * Set visualization mode
   * @param {string} mode - 'bars', 'wave', or 'circle'
   */
  setVisualizationMode(mode) {
    if (['bars', 'wave', 'circle'].includes(mode)) {
      this.visualizationMode = mode;
      
      // Recreate bars for circle mode as it needs different arrangement
      if (mode === 'circle') {
        this.createAudioBars();
      }
    }
  }

  /**
   * Set color mode
   * @param {string} mode - 'spectrum', 'single', or 'emotion'
   * @param {number} color - Base color for 'single' or 'emotion' mode
   */
  setColorMode(mode, color = 0x4a7bff) {
    if (['spectrum', 'single', 'emotion'].includes(mode)) {
      this.colorMode = mode;
      
      if (mode === 'single') {
        this.baseColor = color;
      } else if (mode === 'emotion') {
        this.emotionColor = color;
      }
      
      // Update all bars with new base color
      if (mode === 'single') {
        this.audioBars.forEach(bar => {
          bar.material.color.set(this.baseColor);
          bar.material.emissive.set(this.baseColor);
        });
      }
    }
  }

  /**
   * Set emotion color
   * @param {string} emotion - Emotion name
   */
  setEmotionColor(emotion) {
    let color;
    
    switch (emotion) {
      case 'happy':
        color = 0x00ff00; // Green
        break;
      case 'sad':
        color = 0x0000ff; // Blue
        break;
      case 'angry':
        color = 0xff0000; // Red
        break;
      case 'surprised':
        color = 0xffff00; // Yellow
        break;
      default:
        color = 0x4a7bff; // Default blue
    }
    
    this.emotionColor = color;
    
    if (this.colorMode === 'emotion') {
      this.audioBars.forEach(bar => {
        bar.material.color.set(color);
        bar.material.emissive.set(color);
      });
    }
  }

  /**
   * Get current audio amplitude
   * @returns {number} - Average amplitude (0-1)
   */
  getAmplitude() {
    if (!this.dataArray) return 0;
    
    let sum = 0;
    for (let i = 0; i < this.dataArray.length; i++) {
      sum += this.dataArray[i];
    }
    
    return sum / (this.dataArray.length * 255);
  }

  /**
   * Show/hide visualizer
   * @param {boolean} visible - Whether visualizer should be visible
   */
  setVisible(visible) {
    if (this.visualizerMesh) {
      this.visualizerMesh.visible = visible;
      this.audioBars.forEach(bar => bar.visible = visible);
    }
  }

  /**
   * Clean up resources
   */
  dispose() {
    // Disconnect audio source
    if (this.source) {
      this.source.disconnect();
      this.source = null;
    }
    
    // Close audio context
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
    
    // Remove visualizer elements from scene
    if (this.visualizerMesh && this.scene) {
      this.scene.remove(this.visualizerMesh);
    }
    
    // Remove audio bars from scene
    this.audioBars.forEach(bar => {
      if (bar && this.scene) {
        this.scene.remove(bar);
      }
    });
    
    // Clear arrays
    this.audioBars = [];
    this.dataArray = null;
  }
}