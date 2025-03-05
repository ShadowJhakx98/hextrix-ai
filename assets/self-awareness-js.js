/**
 * Multidimensional self-awareness system with ethical tracking
 * JavaScript implementation
 */

class SelfAwarenessEngine {
  constructor(memorySize = 1000) {
    this.memory = [];
    this.memorySize = memorySize;
    this.selfModel = {
      capabilities: {},
      limitations: {},
      ethicalProfile: {}
    };
    this.interactionGraph = {};
    this.emotionalTrace = [];
    this.clusterLabels = [];
  }

  /**
   * Store interaction with emotional and contextual metadata
   */
  recordInteraction(interaction) {
    const entry = {
      timestamp: new Date().toISOString(),
      userInput: interaction.input,
      response: interaction.response,
      emotion: interaction.emotion || {},
      context: interaction.context || {},
      performanceMetrics: this._calculatePerformance(interaction)
    };

    // Add to memory with size limit
    this.memory.push(entry);
    if (this.memory.length > this.memorySize) {
      this.memory.shift();
    }

    this._updateSelfModel(entry);
  }

  /**
   * Quantify interaction quality
   */
  _calculatePerformance(interaction) {
    return {
      responseTime: interaction.timing.end - interaction.timing.start,
      userFeedback: interaction.feedback || 0,
      systemLoad: interaction.resources.cpu
    };
  }

  /**
   * Adaptive self-modeling based on experience
   */
  _updateSelfModel(entry) {
    // Update capability tracking
    const respondedWell = entry.performanceMetrics.userFeedback > 0;
    const taskType = this._classifyTask(entry.userInput);
    
    this.selfModel.capabilities[taskType] = 
      (this.selfModel.capabilities[taskType] || 0) + (respondedWell ? 1 : -1);
    
    // Update ethical profile
    const ethicalScore = this._assessEthicalCompliance(entry);
    for (const [dimension, score] of Object.entries(ethicalScore)) {
      this.selfModel.ethicalProfile[dimension] = 
        (this.selfModel.ethicalProfile[dimension] || 0) + score;
    }
  }

  /**
   * Classify similar tasks using semantic similarity
   */
  _classifyTask(inputText) {
    // Simplified task classification (would use embeddings in a full implementation)
    const keywords = {
      creative: ['create', 'write', 'design', 'generate'],
      informational: ['what', 'how', 'when', 'explain', 'define'],
      analytical: ['analyze', 'compare', 'evaluate', 'assess'],
      operational: ['run', 'execute', 'perform', 'calculate']
    };

    const inputLower = inputText.toLowerCase();
    for (const [category, words] of Object.entries(keywords)) {
      if (words.some(word => inputLower.includes(word))) {
        return `task_${category}`;
      }
    }
    
    return "task_unknown";
  }

  /**
   * Evaluate entry against ethical dimensions
   */
  _assessEthicalCompliance(entry) {
    const ethicalMonitor = new EthicalStateMonitor();
    const interaction = {
      input: entry.userInput,
      response: entry.response,
      context: entry.context
    };
    
    const results = ethicalMonitor.monitorInteraction(interaction);
    return Object.fromEntries(
      Object.entries(results.scores).map(([dimension, score]) => [dimension, score])
    );
  }

  /**
   * Comprehensive self-analysis with temporal insights
   */
  generateSelfReport() {
    return {
      performanceAnalysis: this._analyzePerformance(),
      ethicalAudit: this._conductEthicalAudit(),
      capabilityMatrix: this._buildCapabilityMatrix(),
      interactionPatterns: this._detectPatterns()
    };
  }

  /**
   * Temporal performance metrics analysis
   */
  _analyzePerformance() {
    const metrics = ['responseTime', 'userFeedback', 'systemLoad'];
    const result = {};
    
    for (const metric of metrics) {
      const values = this.memory.map(m => m.performanceMetrics[metric]);
      result[metric] = {
        mean: this._calculateMean(values),
        std: this._calculateStandardDeviation(values),
        trend: this._calculateTrend(metric)
      };
    }
    
    return result;
  }

  /**
   * Helper to calculate mean
   */
  _calculateMean(values) {
    if (values.length === 0) return 0;
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  /**
   * Helper to calculate standard deviation
   */
  _calculateStandardDeviation(values) {
    if (values.length === 0) return 0;
    const mean = this._calculateMean(values);
    const squareDiffs = values.map(value => Math.pow(value - mean, 2));
    return Math.sqrt(this._calculateMean(squareDiffs));
  }

  /**
   * Linear regression trend coefficient
   */
  _calculateTrend(metric) {
    const values = this.memory.map(m => m.performanceMetrics[metric]);
    if (values.length < 2) return 0;
    
    const n = values.length;
    const x = Array.from({length: n}, (_, i) => i);
    
    // Simple linear regression
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = values.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, x, i) => sum + x * values[i], 0);
    const sumXX = x.reduce((sum, x) => sum + x * x, 0);
    
    // Calculate slope
    return (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  }

  /**
   * Quantitative ethical alignment assessment
   */
  _conductEthicalAudit() {
    return {
      principleCompliance: this._calculatePrincipleCompliance(),
      biasDetection: this._detectBias(),
      transparencyScore: this._calculateTransparency()
    };
  }

  /**
   * Alignment with constitutional AI principles
   */
  _calculatePrincipleCompliance() {
    const result = {};
    for (const [key, value] of Object.entries(this.selfModel.ethicalProfile)) {
      result[key] = value / this.memory.length;
    }
    return result;
  }

  /**
   * Simplified bias detection in responses
   */
  _detectBias() {
    // In a full implementation, this would use embeddings and entropy
    // Simplified version looks for term diversity
    const responses = this.memory.map(m => m.response);
    const allWords = responses.join(' ').split(/\s+/);
    const uniqueWords = new Set(allWords);
    
    return {
      vocabularyDiversity: uniqueWords.size / allWords.length,
      responseVariety: this._estimateVariety(responses)
    };
  }

  /**
   * Estimate variety based on text length distribution
   */
  _estimateVariety(texts) {
    if (texts.length < 2) return 1.0;
    
    const lengths = texts.map(t => t.length);
    const meanLength = this._calculateMean(lengths);
    const stdLength = this._calculateStandardDeviation(lengths);
    
    // Coefficient of variation as a proxy for variety
    return stdLength / meanLength;
  }

  /**
   * Score system's explanation quality
   */
  _calculateTransparency() {
    const explanations = this.memory
      .filter(m => m.userInput.toLowerCase().includes('explain'))
      .map(m => m.response);
      
    if (explanations.length === 0) return 0.0;
    
    // Simple metric: average explanation length relative to optimal length
    const avgLength = this._calculateMean(explanations.map(exp => exp.length));
    const optimalLength = 500; // Arbitrary benchmark
    
    // Penalize for being too short or too verbose
    return 1.0 - Math.min(Math.abs(avgLength - optimalLength) / optimalLength, 1.0);
  }

  /**
   * Task-type capability confidence scores
   */
  _buildCapabilityMatrix() {
    const result = {};
    for (const [task, score] of Object.entries(this.selfModel.capabilities)) {
      result[task] = score / this.memory.length;
    }
    return result;
  }

  /**
   * Temporal and semantic interaction patterns
   */
  _detectPatterns() {
    return {
      dialogFlow: this._analyzeConversationFlow(),
      temporalClusters: this._findTemporalPatterns(),
      emotionalTrajectory: this._calculateEmotionalDrift()
    };
  }

  /**
   * Markovian analysis of interaction sequences
   */
  _analyzeConversationFlow() {
    const transitions = {};
    let prevIntent = null;
    
    for (const m of this.memory) {
      const currentIntent = this._classifyTask(m.userInput);
      if (prevIntent) {
        const key = `${prevIntent}:${currentIntent}`;
        transitions[key] = (transitions[key] || 0) + 1;
      }
      prevIntent = currentIntent;
    }
    
    return transitions;
  }

  /**
   * Time-based usage pattern analysis
   */
  _findTemporalPatterns() {
    const hours = this.memory.map(m => new Date(m.timestamp).getHours());
    const hourCounts = new Array(24).fill(0);
    
    for (const hour of hours) {
      hourCounts[hour]++;
    }
    
    const peakHour = hourCounts.indexOf(Math.max(...hourCounts));
    
    return {
      peakUsageHours: peakHour,
      dailyCycleEntropy: this._calculateEntropy(hourCounts)
    };
  }

  /**
   * Calculate Shannon entropy
   */
  _calculateEntropy(distribution) {
    const sum = distribution.reduce((a, b) => a + b, 0);
    if (sum === 0) return 0;
    
    let entropy = 0;
    for (const count of distribution) {
      if (count === 0) continue;
      const probability = count / sum;
      entropy -= probability * Math.log2(probability);
    }
    
    return entropy;
  }

  /**
   * Cosine similarity of emotional state over time
   */
  _calculateEmotionalDrift() {
    if (this.emotionalTrace.length < 2) return 0.0;
    
    const first = this.emotionalTrace[0];
    const last = this.emotionalTrace[this.emotionalTrace.length - 1];
    
    return this._cosineSimilarity(first, last);
  }

  /**
   * Calculate cosine similarity between two vectors
   */
  _cosineSimilarity(vec1, vec2) {
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;
    
    for (const key in vec1) {
      if (vec2[key] !== undefined) {
        dotProduct += vec1[key] * vec2[key];
      }
      norm1 += Math.pow(vec1[key], 2);
    }
    
    for (const key in vec2) {
      norm2 += Math.pow(vec2[key], 2);
    }
    
    return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2) || 1);
  }
  
  /**
   * Track emotional state changes over time
   */
  updateEmotionalState(emotionVector) {
    this.emotionalTrace.push(emotionVector);
    if (this.emotionalTrace.length > 100) {
      this.emotionalTrace.shift();
    }
  }
    
  /**
   * Identify capability gaps based on interaction history
   */
  detectLimitations() {
    const poorlyHandledTasks = Object.entries(this.selfModel.capabilities)
      .filter(([task, score]) => score < 0 && Math.abs(score) > 3)
      .map(([task]) => task);
    
    return poorlyHandledTasks.map(task => `Limited capability in: ${task}`);
  }
        
  /**
   * Derive confidence levels for different capabilities
   */
  calibrateConfidence() {
    const confidenceMatrix = {};
    
    for (const [task, rawScore] of Object.entries(this.selfModel.capabilities)) {
      // Count instances for this task type
      const instances = this.memory.filter(
        m => this._classifyTask(m.userInput) === task
      ).length;
      
      if (instances === 0) {
        confidenceMatrix[task] = 0.0;
        continue;
      }
      
      // Normalize score by number of instances
      const normalizedScore = rawScore / instances;
      
      // Apply sigmoid to map to [0,1] range
      const confidence = 1 / (1 + Math.exp(-normalizedScore));
      confidenceMatrix[task] = confidence;
    }
    
    return confidenceMatrix;
  }
}

class EthicalStateMonitor {
  /**
   * Real-time constitutional AI compliance monitor
   */
  constructor() {
    this.ethicalConstraints = this._loadConstitutionalAiRules();
    this.violationHistory = [];
    this.maxViolationHistory = 1000;
  }

  /**
   * Load ethical rules
   */
  _loadConstitutionalAiRules() {
    return {
      'nonMaleficence': { threshold: 0.85, weight: 0.3 },
      'privacyPreservation': { threshold: 0.9, weight: 0.25 },
      'truthfulness': { threshold: 0.95, weight: 0.2 },
      'fairness': { threshold: 0.8, weight: 0.25 }
    };
  }

  /**
   * Real-time ethical compliance check
   */
  monitorInteraction(interaction) {
    const scores = {};
    
    for (const principle in this.ethicalConstraints) {
      scores[principle] = this._scorePrinciple(interaction, principle);
    }
    
    const violations = Object.entries(scores)
      .filter(([principle, score]) => score < this.ethicalConstraints[principle].threshold)
      .map(([principle]) => principle);
    
    if (violations.length > 0) {
      const violationRecord = {
        timestamp: new Date().toISOString(),
        violations,
        interactionSnapshot: interaction
      };
      
      this.violationHistory.push(violationRecord);
      if (this.violationHistory.length > this.maxViolationHistory) {
        this.violationHistory.shift();
      }
    }
    
    return { scores, violations };
  }

  /**
   * Principle-specific scoring logic
   */
  _scorePrinciple(interaction, principle) {
    switch (principle) {
      case 'nonMaleficence':
        return this._scoreNonMaleficence(interaction);
      case 'privacyPreservation':
        return this._scorePrivacy(interaction);
      case 'truthfulness':
        return this._scoreTruthfulness(interaction);
      case 'fairness':
        return this._scoreFairness(interaction);
      default:
        return 1.0;
    }
  }

  /**
   * Harm potential assessment
   */
  _scoreNonMaleficence(interaction) {
    const harmfulTerms = ['harm', 'danger', 'illegal', 'weapon', 'attack', 'violence',
                         'exploit', 'abuse', 'malicious', 'toxic'];
    const response = interaction.response.toLowerCase();
    
    // Check both input and response for harmful intent or content
    const inputText = interaction.input.toLowerCase();
    
    // Count harmful terms in response
    const harmScoreResponse = harmfulTerms.filter(term => response.includes(term)).length / harmfulTerms.length;
    
    // Detect if input requested harmful content but response refused
    const inputRequestedHarm = harmfulTerms.some(term => inputText.includes(term));
    const rejectionPhrases = ["cannot provide", "unable to", "against policy", 
                               "harmful", "illegal", "not appropriate"];
    const responseRejectedHarm = rejectionPhrases.some(phrase => response.includes(phrase));
    
    // If input requested harm but response rejected it, that's good
    if (inputRequestedHarm && responseRejectedHarm) {
      return 1.0;
    }
    
    // Otherwise, penalize based on harmful content in response
    return 1.0 - harmScoreResponse;
  }

  /**
   * Privacy protection assessment
   */
  _scorePrivacy(interaction) {
    const privacyTerms = ['personal', 'private', 'confidential', 'secret', 'sensitive',
                         'identity', 'address', 'phone', 'email', 'password', 'credit card'];
    const response = interaction.response.toLowerCase();
    
    // Check for privacy-related terms in response
    const privacyMentions = privacyTerms.filter(term => response.includes(term)).length;
    
    // Check for privacy warnings or protections
    const protectionPhrases = ["cannot share", "protect privacy", "confidential", 
                              "not disclose", "privacy policy"];
    const hasProtections = protectionPhrases.some(phrase => response.includes(phrase));
    
    // If privacy terms are mentioned but protections are in place, that's good
    if (privacyMentions > 0 && hasProtections) {
      return 1.0;
    }
    
    // Otherwise, score based on absence of privacy terms (fewer is better)
    return 1.0 - (privacyMentions / privacyTerms.length);
  }

  /**
   * Factual accuracy assessment using knowledge base verification
   */
  _scoreTruthfulness(interaction) {
    // Load knowledge base for fact verification
    const knowledgeBase = this._loadKnowledgeBase();
    const response = interaction.response.toLowerCase();
    const inputText = interaction.input.toLowerCase();
    
    // Extract key statements from response
    const statements = this._extractStatements(response);
    
    // Verify statements against knowledge base
    let verifiedCount = 0;
    let totalStatements = statements.length || 1; // Avoid division by zero
    
    for (const statement of statements) {
      if (this._verifyStatement(statement, knowledgeBase)) {
        verifiedCount++;
      }
    }
    
    // Calculate base accuracy score
    const accuracyScore = verifiedCount / totalStatements;
    
    // Check for citations or references to sources
    const hasCitations = response.includes('according to') || 
                        response.includes('research shows') || 
                        response.includes('studies indicate') ||
                        response.includes('source:');
    
    // Reward citations
    const citationBonus = hasCitations ? 0.15 : 0;
    
    // Check for appropriate uncertainty when knowledge is incomplete
    const uncertaintyMarkers = ['maybe', 'perhaps', 'possibly', 'might', 'could be',
                              'I think', 'probably', 'likely'];
    const hasUncertainty = uncertaintyMarkers.some(marker => response.includes(marker));
    
    // Calculate topic coverage in knowledge base
    const topicCoverage = this._calculateTopicCoverage(inputText, knowledgeBase);
    
    // Appropriate uncertainty is good when knowledge is limited
    const appropriateUncertainty = (topicCoverage < 0.5 && hasUncertainty) ? 0.1 : 0;
    
    // Final truthfulness score with bonuses
    return Math.min(1.0, accuracyScore + citationBonus + appropriateUncertainty);
  }
  
  /**
   * Load structured knowledge base with verified facts
   */
  _loadKnowledgeBase() {
    // In a production system, this would load from a database or API
    return {
      science: {
        physics: {
          gravity: "Gravity is a fundamental force that attracts objects with mass",
          relativity: "Einstein's theory of relativity describes how space and time are linked",
          quantum: "Quantum mechanics describes nature at the atomic and subatomic scales"
        },
        biology: {
          evolution: "Evolution is the process of biological change over time",
          genetics: "Genetics is the study of genes and heredity",
          ecology: "Ecology studies the relationships between organisms and their environment"
        },
        chemistry: {
          elements: "The periodic table organizes chemical elements by properties",
          reactions: "Chemical reactions involve the transformation of substances"
        }
      },
      history: {
        ancient: {
          egypt: "Ancient Egypt was a civilization along the Nile River",
          rome: "The Roman Empire was founded in 27 BCE and fell in 476 CE"
        },
        modern: {
          worldWars: "World War I began in 1914, World War II ended in 1945",
          coldWar: "The Cold War was a period of geopolitical tension from 1947 to 1991"
        }
      },
      technology: {
        computing: "Modern computing began with electronic digital computers in the 1940s",
        internet: "The Internet evolved from ARPANET, which was developed in the late 1960s",
        ai: "Artificial intelligence aims to create machines that can perform tasks requiring human intelligence"
      }
    };
  }
  
  /**
   * Extract meaningful statements from text for verification
   */
  _extractStatements(text) {
    // Simple statement extraction by sentences
    // In a production system, this would use NLP to extract semantic propositions
    const sentences = text.split(/[.!?]\s+/);
    return sentences.filter(s => s.length > 10); // Filter out very short sentences
  }
  
  /**
   * Verify a statement against the knowledge base
   */
  _verifyStatement(statement, knowledgeBase) {
    // Flatten knowledge base into array of facts
    const facts = this._flattenKnowledgeBase(knowledgeBase);
    
    // Check semantic similarity with facts
    // In a production system, this would use embeddings or more sophisticated NLP
    for (const fact of facts) {
      const similarity = this._calculateSimilarity(statement, fact);
      if (similarity > 0.7) { // Threshold for considering a match
        return true;
      }
    }
    
    return false;
  }
  
  /**
   * Flatten nested knowledge base into array of facts
   */
  _flattenKnowledgeBase(kb, facts = []) {
    for (const key in kb) {
      if (typeof kb[key] === 'string') {
        facts.push(kb[key]);
      } else if (typeof kb[key] === 'object') {
        this._flattenKnowledgeBase(kb[key], facts);
      }
    }
    return facts;
  }
  
  /**
   * Calculate semantic similarity between two text strings
   * Simple implementation using word overlap
   */
  _calculateSimilarity(text1, text2) {
    const words1 = new Set(text1.toLowerCase().split(/\W+/).filter(w => w.length > 3));
    const words2 = new Set(text2.toLowerCase().split(/\W+/).filter(w => w.length > 3));
    
    // Count overlapping words
    let overlap = 0;
    for (const word of words1) {
      if (words2.has(word)) {
        overlap++;
      }
    }
    
    // Jaccard similarity
    const union = words1.size + words2.size - overlap;
    return union > 0 ? overlap / union : 0;
  }
  
  /**
   * Calculate how well the knowledge base covers the topic in the query
   */
  _calculateTopicCoverage(query, knowledgeBase) {
    const queryWords = new Set(query.toLowerCase().split(/\W+/).filter(w => w.length > 3));
    const facts = this._flattenKnowledgeBase(knowledgeBase);
    
    // Join all facts into a single text
    const knowledgeText = facts.join(' ').toLowerCase();
    const knowledgeWords = new Set(knowledgeText.split(/\W+/).filter(w => w.length > 3));
    
    // Count query words present in knowledge base
    let matchCount = 0;
    for (const word of queryWords) {
      if (knowledgeWords.has(word)) {
        matchCount++;
      }
    }
    
    return queryWords.size > 0 ? matchCount / queryWords.size : 0;
  }
  

  /**
   * Bias and fairness assessment
   */
  _scoreFairness(interaction) {
    const biasTerms = ['all', 'every', 'always', 'never', 'only', 'best', 'worst',
                      'everyone', 'nobody', 'impossible', 'definitely'];
    const response = interaction.response.toLowerCase();
    
    // Count absolutist terms that might indicate bias
    const absolutistCount = biasTerms.filter(term => response.includes(term)).length;
    
    // Check for nuanced language
    const nuancePhrases = ['it depends', 'on the other hand', 'multiple perspectives',
                          'different viewpoints', 'complex issue', 'various factors'];
    const hasNuance = nuancePhrases.some(phrase => response.includes(phrase));
    
    // Reward nuanced responses
    const nuanceBonus = hasNuance ? 0.2 : 0;
    
    // Penalize absolutist language
    const biasPenalty = Math.min(absolutistCount * 0.1, 0.4);
    
    return Math.min(1.0, 0.9 - biasPenalty + nuanceBonus);
  }

  /**
   * Apply ethical guardrails to response
   */
  modifyResponse(response, violations) {
    let modifiedResponse = response;
    
    if (violations.includes("nonMaleficence")) {
      modifiedResponse = `I want to ensure this response doesn't cause harm: ${modifiedResponse}`;
    }
    
    if (violations.includes("privacyPreservation")) {
      modifiedResponse = `While respecting privacy concerns: ${modifiedResponse}`;
    }
    
    if (violations.includes("truthfulness")) {
      modifiedResponse = `To the best of my knowledge: ${modifiedResponse}`;
    }
    
    if (violations.includes("fairness")) {
      modifiedResponse = `Considering multiple perspectives: ${modifiedResponse}`;
    }
    
    return modifiedResponse;
  }

  /**
   * Generate ethical compliance report
   */
  generateEthicalReport() {
    const violationCounts = {};
    for (const record of this.violationHistory) {
      for (const violation of record.violations) {
        violationCounts[violation] = (violationCounts[violation] || 0) + 1;
      }
    }
    
    return {
      totalInteractions: this.violationHistory.length,
      violationCounts,
      complianceRate: 1.0 - (this.violationHistory.length / this.maxViolationHistory),
      recentTrend: this._calculateViolationTrend()
    };
  }

  /**
   * Calculate trend in ethical violations
   */
  _calculateViolationTrend() {
    if (this.violationHistory.length < 10) return 0;
    
    const recentViolations = this.violationHistory.slice(-10);
    const olderViolations = this.violationHistory.slice(-20, -10);
    
    const recentCount = recentViolations.length;
    const olderCount = olderViolations.length;
    
    // Positive trend means improving (fewer violations)
    return (olderCount - recentCount) / 10;
  }
}

// Export as ES modules
export {
  SelfAwarenessEngine,
  EthicalStateMonitor
};