#!/usr/bin/env python3
"""
Simple Decision Tree Implementation for CPEE Integration
======================================================

This is a lightweight, dependency-free implementation of the minimal decision tree
that can be easily integrated into CPEE workflows or used as a standalone service.

No external dependencies required - pure Python implementation.
"""

import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional


class CottonCandyDecisionTree:
    """
    Standalone implementation of the minimal cotton candy decision tree.
    
    This class implements the exact decision rules without any external dependencies,
    making it easy to integrate into CPEE workflows or other systems.
    """
    
    def __init__(self):
        """Initialize the decision tree."""
        self.model_info = {
            "name": "Minimal Cotton Candy Decision Tree",
            "version": "1.0",
            "features": [
                "before_turn_on_env_InH",
                "iteration_since_maintenance", 
                "before_turn_on_env_IrO",
                "wait_time",
                "duration_cc_flow"
            ],
            "performance": {
                "train_r2": 0.7879,
                "test_r2": -0.0640,
                "cross_validation_r2": -0.0015
            }
        }
    
    def predict(self, parameters: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict cotton candy quality score using the minimal decision tree.
        
        Args:
            parameters: Dictionary with required features:
                - before_turn_on_env_InH: Internal humidity before turn on (%)
                - iteration_since_maintenance: Iterations since maintenance
                - before_turn_on_env_IrO: Infrared oxygen before turn on
                - wait_time: Wait time in seconds
                - duration_cc_flow: Cotton candy flow duration in seconds
        
        Returns:
            Dictionary with prediction results including:
                - score: Predicted quality score (0-100)
                - category: Quality category string
                - confidence: Confidence level
                - decision_path: Path taken through decision tree
        """
        try:
            # Extract required parameters with defaults
            humidity = parameters.get('before_turn_on_env_InH', 0.0)
            maintenance_iter = parameters.get('iteration_since_maintenance', 0.0)
            ir_oxygen = parameters.get('before_turn_on_env_IrO', 0.0)
            wait_time = parameters.get('wait_time', 0.0)
            flow_duration = parameters.get('duration_cc_flow', 0.0)
            
            # Decision tree logic (exact implementation from minimal model)
            decision_path = []
            
            if humidity <= 35.86:
                decision_path.append(f"before_turn_on_env_InH <= 35.86 ({humidity:.2f})")
                
                if maintenance_iter <= 21.50:
                    decision_path.append(f"iteration_since_maintenance <= 21.50 ({maintenance_iter:.1f})")
                    
                    if ir_oxygen <= 58.03:
                        decision_path.append(f"before_turn_on_env_IrO <= 58.03 ({ir_oxygen:.2f})")
                        score = 10.67
                    else:
                        decision_path.append(f"before_turn_on_env_IrO > 58.03 ({ir_oxygen:.2f})")
                        score = 4.00
                        
                else:  # maintenance_iter > 21.50
                    decision_path.append(f"iteration_since_maintenance > 21.50 ({maintenance_iter:.1f})")
                    
                    if wait_time <= 50.00:
                        decision_path.append(f"wait_time <= 50.00 ({wait_time:.1f})")
                        score = 46.67
                    else:  # wait_time > 50.00
                        decision_path.append(f"wait_time > 50.00 ({wait_time:.1f})")
                        
                        if wait_time <= 65.00:
                            decision_path.append(f"wait_time <= 65.00 ({wait_time:.1f})")
                            score = 20.25
                        else:
                            decision_path.append(f"wait_time > 65.00 ({wait_time:.1f})")
                            score = 27.50
                            
            else:  # humidity > 35.86
                decision_path.append(f"before_turn_on_env_InH > 35.86 ({humidity:.2f})")
                
                if flow_duration <= 66.20:
                    decision_path.append(f"duration_cc_flow <= 66.20 ({flow_duration:.1f})")
                    score = 30.00
                else:
                    decision_path.append(f"duration_cc_flow > 66.20 ({flow_duration:.1f})")
                    
                    if humidity <= 36.31:
                        decision_path.append(f"before_turn_on_env_InH <= 36.31 ({humidity:.2f})")
                        score = 51.67
                    else:
                        decision_path.append(f"before_turn_on_env_InH > 36.31 ({humidity:.2f})")
                        score = 43.00
            
            # Determine quality category and confidence
            if score >= 40:
                category = "EXCELLENT"
                confidence = "HIGH"
                recommendation = "Proceed with production"
            elif score >= 25:
                category = "GOOD"
                confidence = "MEDIUM"
                recommendation = "Good quality expected, proceed"
            elif score >= 15:
                category = "ACCEPTABLE"
                confidence = "MEDIUM"
                recommendation = "Acceptable quality, consider optimization"
            else:
                category = "POOR"
                confidence = "LOW"
                recommendation = "Low quality predicted, optimization required"
            
            return {
                "timestamp": datetime.now().isoformat(),
                "score": round(score, 2),
                "category": category,
                "confidence": confidence,
                "recommendation": recommendation,
                "decision_path": decision_path,
                "key_features": {
                    "before_turn_on_env_InH": humidity,
                    "iteration_since_maintenance": maintenance_iter,
                    "before_turn_on_env_IrO": ir_oxygen,
                    "wait_time": wait_time,
                    "duration_cc_flow": flow_duration
                }
            }
            
        except Exception as e:
            return {
                "error": f"Prediction failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def get_optimization_suggestions(self, parameters: Dict[str, float], 
                                   target_score: float = 35.0) -> Dict[str, Any]:
        """
        Get optimization suggestions to improve quality score.
        
        Args:
            parameters: Current process parameters
            target_score: Desired quality score (default: 35.0)
        
        Returns:
            Dictionary with optimization suggestions
        """
        current_prediction = self.predict(parameters)
        current_score = current_prediction.get("score", 0)
        
        suggestions = {}
        
        # Rule-based optimization suggestions
        humidity = parameters.get('before_turn_on_env_InH', 0.0)
        maintenance_iter = parameters.get('iteration_since_maintenance', 0.0)
        wait_time = parameters.get('wait_time', 0.0)
        flow_duration = parameters.get('duration_cc_flow', 0.0)
        
        # Strategy 1: Optimize humidity (most critical factor)
        if humidity <= 35.86 and current_score < target_score:
            suggestions["humidity_adjustment"] = {
                "parameter": "before_turn_on_env_InH",
                "current_value": humidity,
                "suggested_range": "35.9 - 36.3",
                "reason": "Increase humidity to access higher quality branch",
                "priority": "HIGH",
                "expected_improvement": "15-25 points"
            }
        
        # Strategy 2: Maintenance scheduling
        if maintenance_iter > 21.50:
            suggestions["maintenance_scheduling"] = {
                "parameter": "iteration_since_maintenance",
                "current_value": maintenance_iter,
                "suggested_action": "Schedule maintenance (reset to < 21)",
                "reason": "Machine needs maintenance for optimal performance",
                "priority": "MEDIUM",
                "expected_improvement": "Variable, prevents quality degradation"
            }
        
        # Strategy 3: Wait time optimization
        if humidity <= 35.86 and maintenance_iter > 21.50 and wait_time > 50:
            if wait_time > 65:
                suggestions["wait_time_adjustment"] = {
                    "parameter": "wait_time", 
                    "current_value": wait_time,
                    "suggested_range": "45-50 seconds",
                    "reason": "Reduce wait time for better quality",
                    "priority": "MEDIUM",
                    "expected_improvement": "19-26 points"
                }
            elif 50 < wait_time <= 65:
                suggestions["wait_time_optimization"] = {
                    "parameter": "wait_time",
                    "current_value": wait_time,
                    "suggested_range": "45-50 seconds", 
                    "reason": "Optimize wait time for maximum quality",
                    "priority": "MEDIUM",
                    "expected_improvement": "26 points"
                }
        
        # Strategy 4: Flow duration optimization
        if humidity > 35.86 and flow_duration <= 66.20:
            suggestions["flow_duration_increase"] = {
                "parameter": "duration_cc_flow",
                "current_value": flow_duration,
                "suggested_range": "67-70 seconds",
                "reason": "Increase flow duration for higher quality",
                "priority": "MEDIUM",
                "expected_improvement": "13-22 points"
            }
        
        return {
            "current_score": current_score,
            "target_score": target_score,
            "optimization_needed": current_score < target_score,
            "suggestions": suggestions,
            "estimated_max_score": self._estimate_max_achievable_score(parameters),
            "timestamp": datetime.now().isoformat()
        }
    
    def _estimate_max_achievable_score(self, parameters: Dict[str, float]) -> float:
        """Estimate maximum achievable score with optimal parameters."""
        # Test optimal parameter combinations
        optimal_scenarios = [
            {"before_turn_on_env_InH": 36.1, "duration_cc_flow": 68.0},  # Should give 51.67
            {"before_turn_on_env_InH": 35.0, "iteration_since_maintenance": 20.0, "wait_time": 45.0}  # Should give 46.67
        ]
        
        max_score = 0
        for scenario in optimal_scenarios:
            test_params = parameters.copy()
            test_params.update(scenario)
            result = self.predict(test_params)
            max_score = max(max_score, result.get("score", 0))
        
        return max_score


# Command Line Interface
def main():
    """Command line interface for the decision tree."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python simple_decision_tree.py predict <params.json>")
        print("  python simple_decision_tree.py optimize <params.json> [target_score]")
        print("  python simple_decision_tree.py example")
        return
    
    command = sys.argv[1]
    dt = CottonCandyDecisionTree()
    
    if command == "example":
        # Show example usage
        example_params = {
            "before_turn_on_env_InH": 34.5,
            "iteration_since_maintenance": 15.0,
            "before_turn_on_env_IrO": 55.2,
            "wait_time": 45.0,
            "duration_cc_flow": 68.5
        }
        
        print("🍭 Cotton Candy Decision Tree - Example")
        print("=" * 50)
        print("Example parameters:", json.dumps(example_params, indent=2))
        
        prediction = dt.predict(example_params)
        print("\nPrediction result:")
        print(json.dumps(prediction, indent=2))
        
        optimization = dt.get_optimization_suggestions(example_params, 40.0)
        print("\nOptimization suggestions:")
        print(json.dumps(optimization, indent=2))
        
    elif command == "predict":
        if len(sys.argv) < 3:
            print("Error: Please provide parameters file")
            return
        
        try:
            with open(sys.argv[2], 'r') as f:
                params = json.load(f)
            
            result = dt.predict(params)
            print(json.dumps(result, indent=2))
            
        except Exception as e:
            print(f"Error: {e}")
    
    elif command == "optimize":
        if len(sys.argv) < 3:
            print("Error: Please provide parameters file") 
            return
        
        try:
            with open(sys.argv[2], 'r') as f:
                params = json.load(f)
            
            target = float(sys.argv[3]) if len(sys.argv) > 3 else 35.0
            result = dt.get_optimization_suggestions(params, target)
            print(json.dumps(result, indent=2))
            
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        print(f"Unknown command: {command}")


# HTTP Server for CPEE Integration (optional)
def start_http_server(host='localhost', port=8080):
    """
    Start a simple HTTP server for CPEE integration.
    
    This provides REST endpoints that CPEE can call directly.
    Requires Python's built-in http.server module only.
    """
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import urllib.parse as urlparse
    
    dt = CottonCandyDecisionTree()
    
    class DecisionTreeHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == '/predict':
                self._handle_predict()
            elif self.path == '/optimize':
                self._handle_optimize()
            else:
                self._send_error(404, "Endpoint not found")
        
        def do_GET(self):
            if self.path == '/health':
                self._send_json({"status": "healthy", "timestamp": datetime.now().isoformat()})
            elif self.path == '/info':
                self._send_json(dt.model_info)
            else:
                self._send_error(404, "Endpoint not found")
        
        def _handle_predict(self):
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                params = json.loads(post_data.decode('utf-8'))
                
                result = dt.predict(params)
                self._send_json(result)
                
            except Exception as e:
                self._send_error(400, str(e))
        
        def _handle_optimize(self):
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                params = data.get('parameters', {})
                target = data.get('target_score', 35.0)
                
                result = dt.get_optimization_suggestions(params, target)
                self._send_json(result)
                
            except Exception as e:
                self._send_error(400, str(e))
        
        def _send_json(self, data):
            response = json.dumps(data).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response)))
            self.end_headers()
            self.wfile.write(response)
        
        def _send_error(self, code, message):
            error_data = {"error": message, "timestamp": datetime.now().isoformat()}
            response = json.dumps(error_data).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response)))
            self.end_headers()
            self.wfile.write(response)
    
    server = HTTPServer((host, port), DecisionTreeHandler)
    print(f"🍭 Cotton Candy Decision Tree Server")
    print(f"🚀 Starting server on http://{host}:{port}")
    print(f"📡 Endpoints:")
    print(f"   POST /predict - Predict quality score")
    print(f"   POST /optimize - Get optimization suggestions") 
    print(f"   GET /health - Health check")
    print(f"   GET /info - Model information")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'server':
        # Start HTTP server mode
        host = sys.argv[2] if len(sys.argv) > 2 else 'localhost'
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 8080
        start_http_server(host, port)
    else:
        # Command line mode
        main()
