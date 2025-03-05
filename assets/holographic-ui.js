/**
 * Holographic UI Components
 * Provides 3D holographic elements for the Hextrix interface
 */

class HolographicUI {
    constructor() {
      this.scene = null;
      this.camera = null;
      this.renderer = null;
      this.headModel = null;
      this.audioAnalyser = null;
      this.audioContext = null;
      this.networkPanel = null;
      this.audioPanel = null;
      this.animationFrameId = null;
      this.initialized = false;
      this.lipSyncActive = false;
      this.networkData = {
        ip: 'Fetching...',
        downloadSpeed: '0 Mbps',
        uploadSpeed: '0 Mbps',
        latency: '0 ms'
      };
      this.audioBars = []; // Initialize audioBars in the constructor
    }
  
    /**
     * Initialize the holographic UI
     */
    async initialize() {
      if (this.initialized) return;
  
      // Load Three.js from CDN if not already loaded
      await this.loadDependencies();
  
      // Create container for holographic elements
      this.createContainer();
  
      // Setup Three.js scene
      this.setupScene();
  
      // Create holographic head
      this.createHolographicHead();
  
      // Create audio visualizer
      this.setupAudioVisualizer();
  
      // Create network status panel
      this.createNetworkPanel();
  
      // Start animation loop
      this.animate(); // Call the class's animate method
  
      // Fetch network information
      this.fetchNetworkInfo();
  
      this.initialized = true;
  
      // Add window resize handler
      window.addEventListener('resize', this.onWindowResize.bind(this));
    }
  
    /**
     * Load required dependencies
     */
    async loadDependencies() {
        try {
            // Import Three.js and its modules using ES modules
            const THREE = await import('https://cdn.skypack.dev/three@0.160.0');
            const { GLTFLoader } = await import('https://cdn.skypack.dev/three@0.160.0/examples/jsm/loaders/GLTFLoader.js');
            const { OrbitControls } = await import('https://cdn.skypack.dev/three@0.160.0/examples/jsm/controls/OrbitControls.js');
            
            // Assign THREE to window for global access
            window.THREE = THREE;
            window.THREE.GLTFLoader = GLTFLoader;
            window.THREE.OrbitControls = OrbitControls;
            
            return Promise.resolve();
        } catch (error) {
            console.error('Error loading Three.js dependencies:', error);
            return Promise.reject(error);
        }
      }
  
    /**
     * Create container for holographic elements
     */
    createContainer() {
      const container = document.createElement('div');
      container.id = 'holographic-container';
      container.style.position = 'fixed';
      container.style.top = '0';
      container.style.left = '0';
      container.style.width = '100%';
      container.style.height = '100%';
      container.style.pointerEvents = 'none';
      container.style.zIndex = '10';
      document.body.appendChild(container);
  
      this.container = container;
    }
  
    /**
     * Setup Three.js scene
     */
    setupScene() {
      const { innerWidth: width, innerHeight: height } = window;
  
      // Create scene
      this.scene = new THREE.Scene();
  
      // Create camera
      this.camera = new THREE.PerspectiveCamera(70, width / height, 0.1, 1000);
      this.camera.position.z = 5;
  
      // Create renderer with transparent background
      this.renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
      this.renderer.setSize(width, height);
      this.renderer.setClearColor(0x000000, 0);
      this.renderer.setPixelRatio(window.devicePixelRatio);
      this.container.appendChild(this.renderer.domElement);
  
      // Add ambient light
      const ambientLight = new THREE.AmbientLight(0x4a7bff, 0.5);
      this.scene.add(ambientLight);
  
      // Add directional light
      const directionalLight = new THREE.DirectionalLight(0x4a7bff, 1);
      directionalLight.position.set(0, 1, 2);
      this.scene.add(directionalLight);
    }
  
      /**
   * Create holographic head model
   */
    createHolographicHead() {
        const loader = new THREE.GLTFLoader();
        const holographicUI = this; // Capture 'this' for use inside the loader callback

        loader.load(
        // URL of the GLTF model - **Replace with your actual model path!**
        'scene.gltf', // <--- **IMPORTANT: REPLACE THIS PATH**

        // Callback when the model is loaded
        function (gltf) {
            holographicUI.headModel = gltf.scene; // Assign the loaded scene to headModel

            // Apply holographic material to the entire model (or parts of it)
            holographicUI.headModel.traverse((child) => {
            if (child.isMesh) {
                child.material = new THREE.MeshPhongMaterial({
                color: 0x4a7bff,
                emissive: 0x4a7bff,
                emissiveIntensity: 0.5,
                transparent: true,
                opacity: 0.8,
                wireframe: true, // Start with wireframe for the first style
                wireframeLinewidth: 2
                });
            }
            });

            holographicUI.headModel.scale.set(0.5, 0.5, 0.5); // Adjust scale as needed
            holographicUI.scene.add(holographicUI.headModel);
            holographicUI.headModel.visible = false; // Hidden by default

            // Create mouth geometry for lip sync - keep this part
            const mouthGeometry = new THREE.TorusGeometry(0.2, 0.05, 16, 100);
            const mouthMaterial = new THREE.MeshPhongMaterial({
            color: 0x4a7bff,
            emissive: 0x4a7bff,
            emissiveIntensity: 0.5,
            transparent: true,
            opacity: 0.9
            });

            holographicUI.mouth = new THREE.Mesh(mouthGeometry, mouthMaterial);
            holographicUI.mouth.position.set(0, -0.5, 1);
            holographicUI.mouth.rotation.x = Math.PI / 2;
            holographicUI.headModel.add(holographicUI.mouth);

        },
        // Optional: Callback for loading progress
        function (xhr) {
            console.log((xhr.loaded / xhr.total * 100) + '% loaded');
        },
        // Optional: Callback for errors
        function (error) {
            console.error('An error happened loading the GLTF model', error);
        }
        );
    }
    
    /**
     * Setup audio visualizer
     */
    setupAudioVisualizer() {
      // Create audio context
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      this.audioContext = new AudioContext();
  
      // Create audio panel
      const panelGeometry = new THREE.PlaneGeometry(4, 1);
      const panelMaterial = new THREE.MeshPhongMaterial({
        color: 0x4a7bff,
        emissive: 0x4a7bff,
        emissiveIntensity: 0.3,
        transparent: true,
        opacity: 0.5,
        side: THREE.DoubleSide
      });
  
      this.audioPanel = new THREE.Mesh(panelGeometry, panelMaterial);
      this.audioPanel.position.set(0, 2, 0);
      this.audioPanel.visible = false;
      this.scene.add(this.audioPanel);
  
      // Create audio bars
      const barCount = 32;
      const barWidth = 3.8 / barCount;
  
  
      for (let i = 0; i < barCount; i++) {
        const barGeometry = new THREE.BoxGeometry(barWidth * 0.8, 0.1, 0.1);
        const barMaterial = new THREE.MeshPhongMaterial({
          color: 0x4a7bff,
          emissive: 0x4a7bff,
          emissiveIntensity: 0.5,
          transparent: true,
          opacity: 0.8
        });
  
        const bar = new THREE.Mesh(barGeometry, barMaterial);
        const x = (i * barWidth) - 1.9 + (barWidth / 2);
        bar.position.set(x, 3, 0.1); // Updated Y position to 3 for top positioning
        this.scene.add(bar);
        this.audioBars.push(bar);
      }
    }
  
    /**
     * Create network status panel
     */
    createNetworkPanel() {
      // Create network panel
      const panelGeometry = new THREE.PlaneGeometry(3, 1.5);
      const panelMaterial = new THREE.MeshPhongMaterial({
        color: 0x4a7bff,
        emissive: 0x4a7bff,
        emissiveIntensity: 0.3,
        transparent: true,
        opacity: 0.5,
        side: THREE.DoubleSide
      });
  
      this.networkPanel = new THREE.Mesh(panelGeometry, panelMaterial);
      this.networkPanel.position.set(-3, 0, 0);
      this.networkPanel.rotation.y = Math.PI / 6;
      this.networkPanel.visible = false;
      this.scene.add(this.networkPanel);
  
      // Create canvas for network info text
      this.networkCanvas = document.createElement('canvas');
      this.networkCanvas.width = 512;
      this.networkCanvas.height = 256;
      this.networkContext = this.networkCanvas.getContext('2d');
  
      // Create texture from canvas
      this.networkTexture = new THREE.CanvasTexture(this.networkCanvas);
      this.networkPanel.material.map = this.networkTexture;
      this.networkPanel.material.needsUpdate = true;
  
      // Update network info display
      this.updateNetworkDisplay();
    }
  
    /**
     * Update network info display
     */
    updateNetworkDisplay() {
      const ctx = this.networkContext;
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  
      // Set background
      ctx.fillStyle = 'rgba(0, 20, 40, 0.5)';
      ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  
      // Draw border
      ctx.strokeStyle = '#4a7bff';
      ctx.lineWidth = 4;
      ctx.strokeRect(4, 4, ctx.canvas.width - 8, ctx.canvas.height - 8);
  
      // Draw header
      ctx.fillStyle = '#4a7bff';
      ctx.font = '24px Orbitron, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('NETWORK STATUS', ctx.canvas.width / 2, 40);
  
      // Draw network info
      ctx.font = '20px Orbitron, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(`IP: ${this.networkData.ip}`, 40, 90);
      ctx.fillText(`Download: ${this.networkData.downloadSpeed}`, 40, 130);
      ctx.fillText(`Upload: ${this.networkData.uploadSpeed}`, 40, 170);
      ctx.fillText(`Latency: ${this.networkData.latency}`, 40, 210);
  
      // Update texture
      this.networkTexture.needsUpdate = true;
    }
  
    /**
     * Fetch network information
     */
    async fetchNetworkInfo() {
      try {
        // Fetch public IP (using a free API)
        const ipResponse = await fetch('https://api.ipify.org?format=json');
        const ipData = await ipResponse.json();
        this.networkData.ip = ipData.ip;
  
        // Simulate network speed test (in a real app, use actual speed test API)
        // This is just for demonstration purposes
        this.networkData.downloadSpeed = `${(Math.random() * 100 + 50).toFixed(1)} Mbps`;
        this.networkData.uploadSpeed = `${(Math.random() * 50 + 10).toFixed(1)} Mbps`;
        this.networkData.latency = `${Math.floor(Math.random() * 50 + 10)} ms`;
  
        this.updateNetworkDisplay();
  
        // Update every 30 seconds
        setTimeout(() => this.fetchNetworkInfo(), 30000);
      } catch (error) {
        console.error('Error fetching network info:', error);
        this.networkData.ip = 'Error fetching';
        this.updateNetworkDisplay();
  
        // Retry after 10 seconds on error
        setTimeout(() => this.fetchNetworkInfo(), 10000);
      }
    }
  
    /**
     * Handle window resize
     */
    onWindowResize() {
      const { innerWidth: width, innerHeight: height } = window;
  
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(width, height);
    }
  
    /**
     * Animation loop
     */
    animate() {
      this.animationFrameId = requestAnimationFrame(this.animate.bind(this));
  
      // Pulse effect for head model
      if (this.headModel) {
        const pulseScale = 1 + Math.sin(Date.now() * 0.002) * 0.05;
        this.headModel.scale.set(pulseScale, pulseScale, pulseScale);
        // Gentle rotation for holographic effect
        this.headModel.rotation.y += 0.005;
      }
  
      // Gentle floating animation for network panel
      if (this.networkPanel) {
        this.networkPanel.position.y += Math.sin(Date.now() * 0.001) * 0.002;
      }
  
      // Animate audio bars if audio data is available
      if (this.audioAnalyser && this.audioBars.length > 0) {
        const dataArray = new Uint8Array(this.audioAnalyser.frequencyBinCount);
        this.audioAnalyser.getByteFrequencyData(dataArray);
  
        const barCount = this.audioBars.length;
        const step = Math.floor(dataArray.length / barCount);
  
        for (let i = 0; i < barCount; i++) {
          const value = dataArray[i * step] / 255;
          const bar = this.audioBars[i];
          bar.scale.y = value * 3 + 0.1;
          // Adjust animation to work with the new top position (y=3)
          bar.position.y = 3 + (value * 1.5);
          bar.material.emissiveIntensity = 0.3 + value * 0.4;
        }
      } else if (this.audioBars.length > 0) {
        // Enhanced audio bar animations when no audio data
        this.audioBars.forEach((bar, i) => {
          const value = Math.sin(Date.now() * 0.002 + i * 0.2) * 0.5 + 0.5;
          bar.scale.y = value * 2 + 0.1;
          // Adjust animation to work with the new top position (y=3)
          bar.position.y = 3 + (value * 1.5);
          bar.material.emissiveIntensity = 0.3 + value * 0.4;
        });
      }
  
      // Render scene
      this.renderer.render(this.scene, this.camera);
    }
  
    /**
     * Show/hide holographic head
     */
    toggleHead(visible) {
      if (this.headModel) {
        this.headModel.visible = visible;
      }
    }
  
    /**
     * Show/hide network panel
     */
    toggleNetworkPanel(visible) {
      if (this.networkPanel) {
        this.networkPanel.visible = visible;
      }
    }
  
    /**
     * Show/hide audio visualizer
     */
    toggleAudioVisualizer(visible) {
      if (this.audioPanel) {
        this.audioPanel.visible = visible;
        this.audioBars.forEach(bar => bar.visible = visible);
      }
    }
  
    /**
     * Animate lip sync based on audio amplitude
     */
    updateLipSync(amplitude) {
      if (!this.mouth || !this.lipSyncActive) return;
  
      // Scale mouth based on audio amplitude
      const scale = 1 + (amplitude * 2);
      this.mouth.scale.set(scale, 1, scale);
    }
  
    /**
     * Enable/disable lip sync animation
     */
    setLipSyncActive(active) {
      this.lipSyncActive = active;
      if (!active && this.mouth) {
        this.mouth.scale.set(1, 1, 1);
      }
    }
  
    /**
     * Clean up resources
     */
    dispose() {
      if (this.animationFrameId) {
        cancelAnimationFrame(this.animationFrameId);
      }
  
      if (this.audioContext) {
        this.audioContext.close();
      }
  
      if (this.container && this.container.parentNode) {
        this.container.parentNode.removeChild(this.container);
      }
  
      window.removeEventListener('resize', this.onWindowResize);
    }
  }