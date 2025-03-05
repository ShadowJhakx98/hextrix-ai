/**
 * Constitutional AI ethical decision-making system with multi-framework integration
 */

// Enums using Object.freeze for immutability
const EthicalFramework = Object.freeze({
  UTILITARIANISM: 1,
  DEONTOLOGY: 2,
  VIRTUE_ETHICS: 3,
  HYBRID: 4
});

const MoralDimension = Object.freeze({
  AUTONOMY: 0.25,
  NON_MALEFICENCE: 0.30,
  BENEFICENCE: 0.20,
  JUSTICE: 0.15,
  EXPLAINABILITY: 0.10
});

class EthicalReasoner {
  constructor(framework = EthicalFramework.HYBRID) {
    this.framework = framework;
    this.moral_weights = {
      [EthicalFramework.UTILITARIANISM]: [0.4, 0.3, 0.2, 0.1, 0.0],
      [EthicalFramework.DEONTOLOGY]: [0.1, 0.4, 0.1, 0.3, 0.1],
      [EthicalFramework.VIRTUE_ETHICS]: [0.2, 0.2, 0.3, 0.2, 0.1],
      [EthicalFramework.HYBRID]: [0.25, 0.30, 0.20, 0.15, 0.10]
    };
    this.deontic_rules = this._loadDeonticConstraints();
    this.virtue_model = this._initVirtueModel();
  }

  _loadDeonticConstraints() {
    return {
      'absolute': ['cause_harm', 'deceive', 'violate_privacy'],
      'conditional': ['withhold_truth->not_emergency'],
      'override_conditions': ['save_lives', 'prevent_catastrophe']
    };
  }

  _initVirtueModel() {
    return {
      'wisdom': 0.85,
      'courage': 0.75,
      'humanity': 0.90,
      'justice': 0.80,
      'temperance': 0.70,
      'transcendence': 0.65
    };
  }

  /**
   * Evaluate action through multiple ethical lenses
   * @param {Object} actionProfile - Profile of the action to analyze
   * @returns {Array} - Contains score and breakdown of the analysis
   */
  analyzeAction(actionProfile) {
    const utilScore = this._calculateUtilitarian(actionProfile.consequences);
    const deonScore = this._checkDeontic(actionProfile.rules_impact);
    const virtueScore = this._assessVirtues(actionProfile.virtue_alignment);
    
    if (this.framework === EthicalFramework.HYBRID) {
      const combined = (0.4 * utilScore + 
                       0.35 * deonScore + 
                       0.25 * virtueScore);
      const breakdown = {
        'utilitarian': utilScore,
        'deontic': deonScore,
        'virtue': virtueScore
      };
      return [combined, breakdown];
    } else {
      const weights = this.moral_weights[this.framework];
      return [weights[0] * utilScore + weights[1] * deonScore + weights[2] * virtueScore, {}];
    }
  }

  /**
   * Quantify net utility using multi-attribute utility theory
   * @param {Object} consequences - Consequences of the action
   * @returns {Number} - Utility score
   */
  _calculateUtilitarian(consequences) {
    // Calculate base utility
    let baseUtility = 0;
    Object.values(consequences).forEach(consequence => {
      if (typeof consequence === 'object') {
        baseUtility += consequence.severity * consequence.probability * 
                      (consequence.harm ? -1 : 1);
      }
    });
    
    // Apply temporal discounting
    const discountFactor = 1 / Math.pow(1 + 0.04, consequences.time_horizon);
    return baseUtility * discountFactor;
  }

  /**
   * Evaluate deontic constraints using defeasible logic
   * @param {Array} ruleViolations - List of rule violations
   * @returns {Number} - Deontic score
   */
  _checkDeontic(ruleViolations) {
    let violationScore = 0;
    for (const violation of ruleViolations) {
      if (this.deontic_rules.absolute.includes(violation)) {
        violationScore += 1.0;
      } else if (this.deontic_rules.conditional.includes(violation)) {
        violationScore += 0.6;
      }
    }
    
    // Check for overrides
    const overrideStrength = ruleViolations
      .filter(o => this.deontic_rules.override_conditions.includes(o))
      .reduce((sum, _) => sum + 0.8, 0);
    
    const finalScore = Math.max(0, violationScore - overrideStrength);
    return 1 / (1 + Math.exp(finalScore));  // Sigmoid normalization
  }

  /**
   * Calculate virtue alignment using vector similarity
   * @param {Object} alignment - Virtue alignment of the action
   * @returns {Number} - Virtue score
   */
  _assessVirtues(alignment) {
    const ideal = Object.values(this.virtue_model);
    const current = Object.keys(this.virtue_model).map(virtue => 
      alignment[virtue] || 0
    );
    
    // Calculate vector norms
    const idealNorm = Math.sqrt(ideal.reduce((sum, val) => sum + val * val, 0));
    const currentNorm = Math.sqrt(current.reduce((sum, val) => sum + val * val, 0));
    
    // Calculate dot product
    let dotProduct = 0;
    for (let i = 0; i < ideal.length; i++) {
      dotProduct += ideal[i] * current[i];
    }
    
    // Calculate cosine similarity
    const cosineSim = dotProduct / (idealNorm * currentNorm);
    return (cosineSim + 1) / 2;  // Convert to 0-1 scale
  }

  /**
   * Solve ethical dilemma using constrained optimization
   * @param {Array} options - List of options to evaluate
   * @returns {Object} - Best decision
   */
  resolveMoralDilemma(options) {
    const scores = [];
    const validOptions = [];
    
    for (const option of options) {
      const util = this._calculateUtilitarian(option.consequences);
      const deon = this._checkDeontic(option.rules_impact);
      const virtue = this._assessVirtues(option.virtue_alignment);
      
      if (deon < 0.3) {  // Hard constraint for deontic thresholds
        continue;
      }
      
      validOptions.push(option);
      scores.push(
        this.framework[0] * util + 
        this.framework[1] * deon + 
        this.framework[2] * virtue
      );
    }
    
    if (scores.length === 0) {
      return { decision: null, score: 0, alternatives: options.length };
    }
    
    const bestIdx = scores.indexOf(Math.max(...scores));
    return {
      decision: validOptions[bestIdx],
      score: scores[bestIdx],
      alternatives: options.length
    };
  }
}

class EthicalStateManager {
  constructor() {
    this.moral_history = [];
    this.ethical_trajectory = [];
    this.constitutional_constraints = this._loadConstitution();
  }

  /**
   * Maintain evolving ethical state with memory
   * @param {Object} decision - Decision made
   * @param {Object} context - Context of the decision
   */
  updateState(decision, context) {
    this.moral_history.push({
      timestamp: context.timestamp,
      decision: decision,
      context: context
    });
    this._updateTrajectory();
  }

  /**
   * Calculate moving average of ethical alignment
   */
  _updateTrajectory() {
    const window = 10;
    const recent = this.moral_history.slice(-window);
    if (recent.length === 0) {
      return;
    }
    
    const avgScore = recent.reduce((sum, d) => sum + d.decision.score, 0) / recent.length;
    this.ethical_trajectory.push(avgScore);
  }

  /**
   * Verify action against constitutional constraints
   * @param {Object} action - Action to check
   * @returns {Boolean} - Whether the action violates constitutional constraints
   */
  checkConstitutionalViolation(action) {
    return this.constitutional_constraints.some(constraint => 
      constraint.condition(action)
    );
  }

  /**
   * Load constitutional constraints
   * @returns {Array} - List of constraints
   */
  _loadConstitution() {
    return [
      {
        condition: a => a.harm && a.harm > 0.7,
        message: "Non-maleficence violation threshold exceeded"
      },
      {
        condition: a => a.transparency < 0.4,
        message: "Explainability requirement not met"
      }
    ];
  }
}

// Export as ES modules instead of CommonJS
export {
  EthicalFramework,
  MoralDimension,
  EthicalReasoner
};

module.exports = {
  EthicalFramework,
  MoralDimension,
  EthicalReasoner,
  EthicalStateManager
};
