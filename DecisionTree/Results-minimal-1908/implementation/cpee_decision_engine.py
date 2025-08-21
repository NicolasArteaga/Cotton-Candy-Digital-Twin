#!/usr/bin/env python3
"""
CPEE Decision Engine for Cotton Candy Digital Twin
=================================================

This module implements the minimal decision tree model for integration with
Cloud Process Execution Engine (CPEE) to make real-time quality predictions
and process optimization decisions.

Features:
- Minimal feature decision tree implementation
- REST API endpoint for CPEE integration
- Real-time quality prediction
- Process parameter recommendations
- Logging and monitoring capabilities

Usage:
    python cpee_decision_engine.py
    
API Endpoints:
    POST /predict_quality - Predict cotton candy quality score
    POST /optimize_parameters - Get optimized process parameters
    GET /health - Health check endpoint
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from flask import Flask, request, jsonify
import numpy as np


@dataclass
class ProcessParameters:
    """Process parameters for cotton candy manufacturing."""
    iteration_since_maintenance: float
    wait_time: float
    cook_time: float
    cooldown_time: float
    duration_cc_flow: float
    baseline_env_EnvH: float
    baseline_env_EnvT: float
    before_turn_on_env_InH: float
    before_turn_on_env_InT: float
    before_turn_on_env_IrO: float
    before_turn_on_env_IrA: float


class MinimalDecisionTree:
    """
    Minimal Cotton Candy Decision Tree Implementation
    
    This class implements the exact decision tree rules from the minimal model
    for predicting cotton candy quality scores (0-100).
    """
    
    def __init__(self):
        """Initialize the decision tree with logging."""
        self.logger = logging.getLogger(__name__)
        self.model_info = {
            "name": "Minimal Cotton Candy Decision Tree",
            "features": 11,
            "performance": {
                "train_r2": 0.7879,
                "test_r2": -0.0640,
                "cv_r2_mean": -0.0015,
                "train_mae": 6.31,
                "test_mae": 13.0
            }
        }
        
    def predict_quality(self, params: ProcessParameters) -> float:
        """
        Predict cotton candy quality score using the minimal decision tree.
        
        Args:
            params: ProcessParameters object with all required features
            
        Returns:
            Predicted quality score (0-100)
        """
        try:
            # Implement the exact decision tree logic from the minimal model
            if params.before_turn_on_env_InH <= 35.86:
                if params.iteration_since_maintenance <= 21.50:
                    if params.before_turn_on_env_IrO <= 58.03:
                        return 10.67
                    else:  # before_turn_on_env_IrO > 58.03
                        return 4.00
                else:  # iteration_since_maintenance > 21.50
                    if params.wait_time <= 50.00:
                        return 46.67
                    else:  # wait_time > 50.00
                        if params.wait_time <= 65.00:
                            return 20.25
                        else:  # wait_time > 65.00
                            return 27.50
            else:  # before_turn_on_env_InH > 35.86
                if params.duration_cc_flow <= 66.20:
                    return 30.00
                else:  # duration_cc_flow > 66.20
                    if params.before_turn_on_env_InH <= 36.31:
                        return 51.67
                    else:  # before_turn_on_env_InH > 36.31
                        return 43.00
                        
        except Exception as e:
            self.logger.error(f"Error in quality prediction: {e}")
            raise
    
    def predict_with_confidence(self, params: ProcessParameters) -> Tuple[float, str, Dict[str, Any]]:
        """
        Predict quality with confidence level and decision path.
        
        Returns:
            Tuple of (predicted_score, confidence_level, decision_path)
        """
        score = self.predict_quality(params)
        
        # Determine confidence based on score range and model performance
        if score >= 40:
            confidence = "HIGH"
        elif score >= 20:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        # Create decision path for transparency
        decision_path = self._get_decision_path(params)
        
        return score, confidence, decision_path
    
    def _get_decision_path(self, params: ProcessParameters) -> Dict[str, Any]:
        """Get the decision path taken through the tree."""
        path = []
        
        # Track the decision path
        if params.before_turn_on_env_InH <= 35.86:
            path.append(f"before_turn_on_env_InH <= 35.86 ({params.before_turn_on_env_InH:.2f})")
            
            if params.iteration_since_maintenance <= 21.50:
                path.append(f"iteration_since_maintenance <= 21.50 ({params.iteration_since_maintenance:.1f})")
                
                if params.before_turn_on_env_IrO <= 58.03:
                    path.append(f"before_turn_on_env_IrO <= 58.03 ({params.before_turn_on_env_IrO:.2f})")
                else:
                    path.append(f"before_turn_on_env_IrO > 58.03 ({params.before_turn_on_env_IrO:.2f})")
            else:
                path.append(f"iteration_since_maintenance > 21.50 ({params.iteration_since_maintenance:.1f})")
                
                if params.wait_time <= 50.00:
                    path.append(f"wait_time <= 50.00 ({params.wait_time:.1f})")
                else:
                    path.append(f"wait_time > 50.00 ({params.wait_time:.1f})")
                    if params.wait_time <= 65.00:
                        path.append(f"wait_time <= 65.00 ({params.wait_time:.1f})")
                    else:
                        path.append(f"wait_time > 65.00 ({params.wait_time:.1f})")
        else:
            path.append(f"before_turn_on_env_InH > 35.86 ({params.before_turn_on_env_InH:.2f})")
            
            if params.duration_cc_flow <= 66.20:
                path.append(f"duration_cc_flow <= 66.20 ({params.duration_cc_flow:.1f})")
            else:
                path.append(f"duration_cc_flow > 66.20 ({params.duration_cc_flow:.1f})")
                if params.before_turn_on_env_InH <= 36.31:
                    path.append(f"before_turn_on_env_InH <= 36.31 ({params.before_turn_on_env_InH:.2f})")
                else:
                    path.append(f"before_turn_on_env_InH > 36.31 ({params.before_turn_on_env_InH:.2f})")
        
        return {
            "decision_path": path,
            "key_features_used": self._get_key_features_used(params)
        }
    
    def _get_key_features_used(self, params: ProcessParameters) -> Dict[str, float]:
        """Get the key features that were actually used in the decision."""
        if params.before_turn_on_env_InH <= 35.86:
            if params.iteration_since_maintenance <= 21.50:
                return {
                    "before_turn_on_env_InH": params.before_turn_on_env_InH,
                    "iteration_since_maintenance": params.iteration_since_maintenance,
                    "before_turn_on_env_IrO": params.before_turn_on_env_IrO
                }
            else:
                return {
                    "before_turn_on_env_InH": params.before_turn_on_env_InH,
                    "iteration_since_maintenance": params.iteration_since_maintenance,
                    "wait_time": params.wait_time
                }
        else:
            return {
                "before_turn_on_env_InH": params.before_turn_on_env_InH,
                "duration_cc_flow": params.duration_cc_flow
            }


class CPEEDecisionEngine:
    """
    Main decision engine for CPEE integration.
    
    Provides process optimization recommendations and quality predictions
    for the cotton candy digital twin.
    """
    
    def __init__(self):
        """Initialize the decision engine."""
        self.decision_tree = MinimalDecisionTree()
        self.logger = logging.getLogger(__name__)
        
        # Define optimal parameter ranges based on decision tree analysis
        self.optimal_ranges = {
            "wait_time": (45, 55),  # Sweet spot from tree analysis
            "before_turn_on_env_InH": (34, 36),  # Critical humidity range
            "iteration_since_maintenance": (15, 25),  # Optimal maintenance window
            "duration_cc_flow": (65, 70),  # Flow duration sweet spot
            "before_turn_on_env_IrO": (55, 60)  # Oxygen level range
        }
    
    def predict_quality(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict quality score for given process parameters.
        
        Args:
            input_data: Dictionary with process parameters
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Convert input to ProcessParameters object
            params = self._dict_to_process_params(input_data)
            
            # Get prediction with confidence
            score, confidence, decision_path = self.decision_tree.predict_with_confidence(params)
            
            # Determine quality category
            if score >= 40:
                quality_category = "EXCELLENT"
                recommendation = "Proceed with current parameters"
            elif score >= 25:
                quality_category = "GOOD"
                recommendation = "Consider minor adjustments"
            elif score >= 15:
                quality_category = "ACCEPTABLE"
                recommendation = "Review parameters, optimization recommended"
            else:
                quality_category = "POOR"
                recommendation = "Stop and adjust parameters immediately"
            
            return {
                "timestamp": datetime.now().isoformat(),
                "predicted_score": round(score, 2),
                "quality_category": quality_category,
                "confidence": confidence,
                "recommendation": recommendation,
                "decision_path": decision_path,
                "model_info": self.decision_tree.model_info
            }
            
        except Exception as e:
            self.logger.error(f"Error in quality prediction: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def optimize_parameters(self, current_params: Dict[str, Any], 
                          target_quality: float = 40.0) -> Dict[str, Any]:
        """
        Suggest optimized parameters to achieve target quality.
        
        Args:
            current_params: Current process parameters
            target_quality: Desired quality score (default: 40.0)
            
        Returns:
            Dictionary with optimization suggestions
        """
        try:
            optimizations = []
            params = self._dict_to_process_params(current_params)
            current_score = self.decision_tree.predict_quality(params)
            
            # Analyze key parameters and suggest improvements
            suggestions = {}
            
            # 1. Check humidity level (most critical)
            if params.before_turn_on_env_InH <= 35.86:
                if current_score < target_quality:
                    optimal_humidity = self.optimal_ranges["before_turn_on_env_InH"]
                    suggestions["before_turn_on_env_InH"] = {
                        "current": params.before_turn_on_env_InH,
                        "suggested": f"{optimal_humidity[0]}-{optimal_humidity[1]}",
                        "reason": "Increase humidity to improve quality",
                        "priority": "HIGH"
                    }
            
            # 2. Check maintenance cycle
            if params.iteration_since_maintenance > 21.50:
                if current_score < target_quality:
                    suggestions["iteration_since_maintenance"] = {
                        "current": params.iteration_since_maintenance,
                        "suggested": "< 21",
                        "reason": "Schedule maintenance to improve quality",
                        "priority": "MEDIUM"
                    }
            
            # 3. Check wait time
            if params.wait_time > 50 and params.iteration_since_maintenance > 21.50:
                optimal_wait = self.optimal_ranges["wait_time"]
                suggestions["wait_time"] = {
                    "current": params.wait_time,
                    "suggested": f"{optimal_wait[0]}-{optimal_wait[1]}",
                    "reason": "Optimize wait time for better quality",
                    "priority": "MEDIUM"
                }
            
            # 4. Check flow duration
            if params.duration_cc_flow <= 66.20 and current_score < target_quality:
                optimal_flow = self.optimal_ranges["duration_cc_flow"]
                suggestions["duration_cc_flow"] = {
                    "current": params.duration_cc_flow,
                    "suggested": f"{optimal_flow[0]}-{optimal_flow[1]}",
                    "reason": "Increase flow duration for better quality",
                    "priority": "MEDIUM"
                }
            
            return {
                "timestamp": datetime.now().isoformat(),
                "current_score": round(current_score, 2),
                "target_score": target_quality,
                "optimization_needed": current_score < target_quality,
                "suggestions": suggestions,
                "estimated_improvement": self._estimate_improvement(params, suggestions)
            }
            
        except Exception as e:
            self.logger.error(f"Error in parameter optimization: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _dict_to_process_params(self, data: Dict[str, Any]) -> ProcessParameters:
        """Convert dictionary to ProcessParameters object."""
        return ProcessParameters(
            iteration_since_maintenance=float(data.get('iteration_since_maintenance', 0)),
            wait_time=float(data.get('wait_time', 0)),
            cook_time=float(data.get('cook_time', 0)),
            cooldown_time=float(data.get('cooldown_time', 0)),
            duration_cc_flow=float(data.get('duration_cc_flow', 0)),
            baseline_env_EnvH=float(data.get('baseline_env_EnvH', 0)),
            baseline_env_EnvT=float(data.get('baseline_env_EnvT', 0)),
            before_turn_on_env_InH=float(data.get('before_turn_on_env_InH', 0)),
            before_turn_on_env_InT=float(data.get('before_turn_on_env_InT', 0)),
            before_turn_on_env_IrO=float(data.get('before_turn_on_env_IrO', 0)),
            before_turn_on_env_IrA=float(data.get('before_turn_on_env_IrA', 0))
        )
    
    def _estimate_improvement(self, current_params: ProcessParameters, 
                            suggestions: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate potential quality improvement from suggestions."""
        if not suggestions:
            return {"potential_improvement": 0, "confidence": "N/A"}
        
        # Simple heuristic based on decision tree structure
        potential_improvement = 0
        
        # High priority suggestions have more impact
        for param, suggestion in suggestions.items():
            if suggestion["priority"] == "HIGH":
                potential_improvement += 15
            elif suggestion["priority"] == "MEDIUM":
                potential_improvement += 8
            else:
                potential_improvement += 3
        
        return {
            "potential_improvement": min(potential_improvement, 30),  # Cap at 30 points
            "confidence": "MEDIUM" if len(suggestions) <= 2 else "HIGH"
        }


# Flask App for CPEE Integration
app = Flask(__name__)
decision_engine = CPEEDecisionEngine()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for CPEE monitoring."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Cotton Candy Decision Engine",
        "version": "1.0.0"
    })


@app.route('/predict_quality', methods=['POST'])
def predict_quality():
    """Predict cotton candy quality score."""
    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400
        
        result = decision_engine.predict_quality(input_data)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/optimize_parameters', methods=['POST'])
def optimize_parameters():
    """Get optimized process parameters."""
    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400
        
        target_quality = input_data.pop('target_quality', 40.0)
        result = decision_engine.optimize_parameters(input_data, target_quality)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the decision tree model."""
    return jsonify(decision_engine.decision_tree.model_info)


if __name__ == '__main__':
    print("🍭 Cotton Candy Decision Engine Starting...")
    print("📡 Endpoints available:")
    print("   POST /predict_quality - Predict quality score")
    print("   POST /optimize_parameters - Get optimization suggestions")
    print("   GET /health - Health check")
    print("   GET /model_info - Model information")
    print("🚀 Starting server on http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
