/**
 * Advanced Holographic Head Model
 * Provides a detailed female head model with enhanced lip-sync capabilities
 */

class AdvancedHeadModel {
  constructor(scene, position = { x: 0, y: 0, z: 0 }) {
    this.scene = scene;
    this.position = position;
    this.headModel = null;
    this.facialFeatures = {};
    this.morphTargets = {};
    this.lipSyncActive = false;
    this.emotionState = 'neutral';
    this.blinkInterval = null;
    this.loaded = false;
  }

  /**
   * Load the advanced head model
   */
  async load() {
    if (!window.THREE) {
      console.error('THREE.js is required but not loaded');
      return false;
    }

    try {
      // Create a more detailed head using combined geometries
      this.createDetailedHead();
      
      // Add facial features
      this.addFacialFeatures();
      
      // Setup morph targets for expressions
      this.setupMorphTargets();
      
      // Start blinking animation
      this.startBlinking();
      
      this.loaded = true;
      return true;
    } catch (error) {
      console.error('Error loading advanced head model:', error);
      return false;
    }
  }

  /**
   * Create a more detailed head model
   */
  createDetailedHead() {
    // Create head group
    this.headModel = new THREE.Group();
    this.headModel.position.set(this.position.x, this.position.y, this.position.z);
    
    // Create head geometry (more oval-shaped for female appearance)
    const headGeometry = new THREE.SphereGeometry(1, 32, 32);
    // Adjust vertices to create more oval shape
    const vertices = headGeometry.attributes.position.array;
    for (let i = 0; i < vertices.length; i += 3) {
      // Slightly elongate the head vertically
      vertices[i + 1] *= 1.1;
      // Slightly narrow the head horizontally
      vertices[i] *= 0.95;
      // Slightly flatten the head depth
      vertices[i + 2] *= 0.95;
    }
    headGeometry.attributes.position.needsUpdate = true;
    
    // Create holographic material with more subtle effect
    const headMaterial = new THREE.MeshPhongMaterial({
      color: 0x4a7bff,
      emissive: 0x4a7bff,
      emissiveIntensity: 0.2,
      transparent: true,
      opacity: 0.7,
      wireframe: true,
      wireframeLinewidth: 1
    });
    
    const headMesh = new THREE.Mesh(headGeometry, headMaterial);
    this.headModel.add(headMesh);
    
    // Add a solid inner head with lower opacity for depth
    const innerHeadGeometry = new THREE.SphereGeometry(0.95, 32, 32);
    const innerHeadMaterial = new THREE.MeshPhongMaterial({
      color: 0x4a7bff,
      emissive: 0x4a7bff,
      emissiveIntensity: 0.1,
      transparent: true,
      opacity: 0.3
    });
    
    const innerHeadMesh = new THREE.Mesh(innerHeadGeometry, innerHeadMaterial);
    this.headModel.add(innerHeadMesh);
    
    // Add to scene
    this.scene.add(this.headModel);
  }

  /**
   * Add facial features to the head model
   */
  addFacialFeatures() {
    // Create eyes
    const eyeGeometry = new THREE.SphereGeometry(0.12, 16, 16);
    const eyeMaterial = new THREE.MeshPhongMaterial({
      color: 0xffffff,
      emissive: 0x4a7bff,
      emissiveIntensity: 0.5,
      transparent: true,
      opacity: 0.9
    });
    
    // Left eye
    const leftEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
    leftEye.position.set(-0.3, 0.1, 0.75);
    this.headModel.add(leftEye);
    this.facialFeatures.leftEye = leftEye;
    
    // Right eye
    const rightEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
    rightEye.position.set(0.3, 0.1, 0.75);
    this.headModel.add(rightEye);
    this.facialFeatures.rightEye = rightEye;
    
    // Create pupils
    const pupilGeometry = new THREE.SphereGeometry(0.05, 16, 16);
    const pupilMaterial = new THREE.MeshPhongMaterial({
      color: 0x000000,
      emissive: 0x4a7bff,
      emissiveIntensity: 0.2,
      transparent: true,
      opacity: 0.9
    });
    
    // Left pupil
    const leftPupil = new THREE.Mesh(pupilGeometry, pupilMaterial);
    leftPupil.position.set(0, 0, 0.07);
    leftEye.add(leftPupil);
    this.facialFeatures.leftPupil = leftPupil;
    
    // Right pupil
    const rightPupil = new THREE.Mesh(pupilGeometry, pupilMaterial);
    rightPupil.position.set(0, 0, 0.07);
    rightEye.add(rightPupil);
    this.facialFeatures.rightPupil = rightPupil;
    
    // Create eyebrows
    const eyebrowGeometry = new THREE.BoxGeometry(0.25, 0.03, 0.05);
    const eyebrowMaterial = new THREE.MeshPhongMaterial({
      color: 0x4a7bff,
      emissive: 0x4a7bff,
      emissiveIntensity: 0.5,
      transparent: true,
      opacity: 0.9
    });
    
    // Left eyebrow
    const leftEyebrow = new THREE.Mesh(eyebrowGeometry, eyebrowMaterial);
    leftEyebrow.position.set(-0.3, 0.25, 0.75);
    leftEyebrow.rotation.z = 0.1;
    this.headModel.add(leftEyebrow);
    this.facialFeatures.leftEyebrow = leftEyebrow;
    
    // Right eyebrow
    const rightEyebrow = new THREE.Mesh(eyebrowGeometry, eyebrowMaterial);
    rightEyebrow.position.set(0.3, 0.25, 0.75);
    rightEyebrow.rotation.z = -0.1;
    this.headModel.add(rightEyebrow);
    this.facialFeatures.rightEyebrow = rightEyebrow;
    
    // Create nose
    const noseGeometry = new THREE.ConeGeometry(0.08, 0.2, 16);
    const noseMaterial = new THREE.MeshPhongMaterial({
      color: 0x4a7bff,
      emissive: 0x4a7bff,
      emissiveIntensity: 0.3,
      transparent: true,
      opacity: 0.7
    });
    
    const nose = new THREE.Mesh(noseGeometry, noseMaterial);
    nose.position.set(0, -0.1, 0.9);
    nose.rotation.x = -Math.PI / 2;
    this.headModel.add(nose);
    this.facialFeatures.nose = nose;
    
    // Create mouth with better shape for lip sync
    const mouthGeometry = new THREE.TorusGeometry(0.15, 0.04, 16, 100);
    const mouthMaterial = new THREE.MeshPhongMaterial({
      color: 0x4a7bff,
      emissive: 0x4a7bff,
      emissiveIntensity: 0.5,
      transparent: true,
      opacity: 0.9
    });
    
    const mouth = new THREE.Mesh(mouthGeometry, mouthMaterial);
    mouth.position.set(0, -0.4, 0.75);
    mouth.rotation.x = Math.PI / 2;
    this.headModel.add(mouth);
    this.facialFeatures.mouth = mouth;
    
    // Add teeth (visible during speech)
    const teethGeometry = new THREE.BoxGeometry(0.25, 0.04, 0.02);
    const teethMaterial = new THREE.MeshPhongMaterial({
      color: 0xffffff,
      emissive: 0x4a7bff,
      emissiveIntensity: 0.1,
      transparent: true,
      opacity: 0.7
    });
    
    const teeth = new THREE.Mesh(teethGeometry, teethMaterial);
    teeth.position.set(0, 0, 0);
    mouth.add(teeth);
    this.facialFeatures.teeth = teeth;
  }

  /**
   * Setup morph targets for facial expressions
   */
  setupMorphTargets() {
    // Store original positions for morphing
    this.morphTargets.neutral = {
      leftEyebrow: { position: { ...this.facialFeatures.leftEyebrow.position }, rotation: { ...this.facialFeatures.leftEyebrow.rotation } },
      rightEyebrow: { position: { ...this.facialFeatures.rightEyebrow.position }, rotation: { ...this.facialFeatures.rightEyebrow.rotation } },
      mouth: { scale: { ...this.facialFeatures.mouth.scale } }
    };
    
    // Happy expression
    this.morphTargets.happy = {
      leftEyebrow: { position: { x: -0.3, y: 0.28, z: 0.75 }, rotation: { x: 0, y: 0, z: 0.2 } },
      rightEyebrow: { position: { x: 0.3, y: 0.28, z: 0.75 }, rotation: { x: 0, y: 0, z: -0.2 } },
      mouth: { scale: { x: 1.2, y: 1, z: 0.8 } }
    };
    
    // Sad expression
    this.morphTargets.sad = {
      leftEyebrow: { position: { x: -0.3, y: 0.2, z: 0.75 }, rotation: { x: 0, y: 0, z: -0.2 } },
      rightEyebrow: { position: { x: 0.3, y: 0.2, z: 0.75 }, rotation: { x: 0, y: 0, z: 0.2 } },
      mouth: { scale: { x: 0.8, y: 1, z: 1.2 } }
    };
    
    // Surprised expression
    this.morphTargets.surprised = {
      leftEyebrow: { position: { x: -0.3, y: 0.35, z: 0.75 }, rotation: { x: 0, y: 0, z: 0 } },
      rightEyebrow: { position: { x: 0.3, y: 0.35, z: 0.75 }, rotation: { x: 0, y: 0, z: 0 } },
      mouth: { scale: { x: 0.7, y: 1, z: 1.5 } }
    };
    
    // Angry expression
    this.morphTargets.angry = {
      leftEyebrow: { position: { x: -0.3, y: 0.2, z: 0.75 }, rotation: { x: 0, y: 0, z: -0.3 } },
      rightEyebrow: { position: { x: 0.3, y: 0.2, z: 0.75 }, rotation: { x: 0, y: 0, z: 0.3 } },
      mouth: { scale: { x: 0.9, y: 1, z: 0.9 } }
    };
  }

  /**
   * Set the head's emotion state
   * @param {string} emotion - One of: neutral, happy, sad, surprised, angry
   */
  setEmotion(emotion) {
    if (!this.morphTargets[emotion]) {
      console.warn(`Unknown emotion: ${emotion}. Using neutral.`);
      emotion = 'neutral';
    }
    
    this.emotionState = emotion;
    const target = this.morphTargets[emotion];
    
    // Animate to the target expression
    this.animateToMorphTarget(target);
  }

  /**
   * Animate to a specific morph target
   */
  animateToMorphTarget(target) {
    // Animate eyebrows
    if (this.facialFeatures.leftEyebrow && target.leftEyebrow) {
      this.facialFeatures.leftEyebrow.position.x = target.leftEyebrow.position.x;
      this.facialFeatures.leftEyebrow.position.y = target.leftEyebrow.position.y;
      this.facialFeatures.leftEyebrow.position.z = target.leftEyebrow.position.z;
      this.facialFeatures.leftEyebrow.rotation.z = target.leftEyebrow.rotation.z;
    }
    
    if (this.facialFeatures.rightEyebrow && target.rightEyebrow) {
      this.facialFeatures.rightEyebrow.position.x = target.rightEyebrow.position.x;
      this.facialFeatures.rightEyebrow.position.y = target.rightEyebrow.position.y;
      this.facialFeatures.rightEyebrow.position.z = target.rightEyebrow.position.z;
      this.facialFeatures.rightEyebrow.rotation.z = target.rightEyebrow.rotation.z;
    }
    
    // Animate mouth
    if (this.facialFeatures.mouth && target.mouth) {
      this.facialFeatures.mouth.scale.x = target.mouth.scale.x;
      this.facialFeatures.mouth.scale.y = target.mouth.scale.y;
      this.facialFeatures.mouth.scale.z = target.mouth.scale.z;
    }
  }

  /**
   * Start blinking animation
   */
  startBlinking() {
    // Clear any existing interval
    if (this.blinkInterval) {
      clearInterval(this.blinkInterval);
    }
    
    // Set up blinking at random intervals
    this.blinkInterval = setInterval(() => {
      this.blink();
    }, Math.random() * 3000 + 2000); // Blink every 2-5 seconds
  }

  /**
   * Perform a single blink animation
   */
  blink() {
    if (!this.facialFeatures.leftEye || !this.facialFeatures.rightEye) return;
    
    // Store original scale
    const originalScaleY = this.facialFeatures.leftEye.scale.y;
    
    // Close eyes
    this.facialFeatures.leftEye.scale.y = 0.1;
    this.facialFeatures.rightEye.scale.y = 0.1;
    
    // Open eyes after a short delay
    setTimeout(() => {
      this.facialFeatures.leftEye.scale.y = originalScaleY;
      this.facialFeatures.rightEye.scale.y = originalScaleY;
    }, 150);
  }

  /**
   * Update lip sync based on audio amplitude
   * @param {number} amplitude - Audio amplitude (0-1)
   */
  updateLipSync(amplitude) {
    if (!this.facialFeatures.mouth || !this.lipSyncActive) return;
    
    // Scale mouth based on audio amplitude
    const mouthScale = 1 + (amplitude * 1.5);
    this.facialFeatures.mouth.scale.x = mouthScale;
    this.facialFeatures.mouth.scale.z = mouthScale;
    
    // Move teeth slightly based on mouth opening
    if (this.facialFeatures.teeth) {
      this.facialFeatures.teeth.position.z = amplitude * 0.05;
    }
  }

  /**
   * Enable/disable lip sync animation
   */
  setLipSyncActive(active) {
    this.lipSyncActive = active;
    
    // Reset mouth to neutral position when deactivated
    if (!active && this.facialFeatures.mouth) {
      const neutral = this.morphTargets.neutral.mouth.scale;
      this.facialFeatures.mouth.scale.set(neutral.x, neutral.y, neutral.z);
      
      if (this.facialFeatures.teeth) {
        this.facialFeatures.teeth.position.z = 0;
      }
    }
  }

  /**
   * Look at a specific point in 3D space
   * @param {THREE.Vector3} target - Point to look at
   */
  lookAt(target) {
    if (!this.facialFeatures.leftEye || !this.facialFeatures.rightEye) return;
    
    // Calculate direction from head to target
    const headPosition = new THREE.Vector3();
    this.headModel.getWorldPosition(headPosition);
    const direction = target.clone().sub(headPosition).normalize();
    
    // Limit eye movement range
    const maxEyeMove = 0.05;
    
    // Move pupils to look in the direction
    if (this.facialFeatures.leftPupil) {
      this.facialFeatures.leftPupil.position.x = direction.x * maxEyeMove;
      this.facialFeatures.leftPupil.position.y = direction.y * maxEyeMove;
    }
    
    if (this.facialFeatures.rightPupil) {
      this.facialFeatures.rightPupil.position.x = direction.x * maxEyeMove;
      this.facialFeatures.rightPupil.position.y = direction.y * maxEyeMove;
    }
  }

  /**
   * Clean up resources
   */
  dispose() {
    // Clear blinking interval
    if (this.blinkInterval) {
      clearInterval(this.blinkInterval);
      this.blinkInterval = null;
    }
    
    // Remove from scene
    if (this.scene && this.headModel) {
      this.scene.remove(this.headModel);
    }
    
    // Clear references
    this.headModel = null;
    this.facialFeatures = {};
    this.morphTargets = {};
  }
}