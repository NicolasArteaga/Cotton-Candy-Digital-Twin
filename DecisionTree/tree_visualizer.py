#!/usr/bin/env python3
"""
Cotton Candy Decision Tree - Complete Tree Visualization
========================================================
This script creates comprehensive visualizations of the trained decision tree.
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
import seaborn as sns
from pathlib import Path
import numpy as np

class CottonCandyTreeVisualizer:
    def __init__(self, model_path="cotton_candy_digital_twin.joblib", 
                 features_path="/Users/nicolas/Desktop/Cotton-Candy-Digital-Twin/Data_Collection/xy/features_X.csv"):
        """Initialize the tree visualizer."""
        self.model_path = Path(model_path)
        self.features_path = Path(features_path)
        self.model = None
        self.feature_names = None
        self.load_model_and_data()
        
    def load_model_and_data(self):
        """Load the trained model and feature names."""
        print("🔍 Loading trained model and features...")
        
        # Load the trained model
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
            print(f"   ✅ Model loaded from: {self.model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load feature names
        if self.features_path.exists():
            features_df = pd.read_csv(self.features_path)
            self.feature_names = list(features_df.columns)
            print(f"   ✅ Features loaded: {len(self.feature_names)} features")
        else:
            raise FileNotFoundError(f"Features file not found: {self.features_path}")
    
    def create_detailed_tree_plot(self, figsize=(20, 15), save_path="complete_decision_tree.png"):
        """Create a detailed tree plot with all nodes and splits."""
        print("🌳 Creating detailed decision tree visualization...")
        
        plt.figure(figsize=figsize)
        
        # Create the tree plot
        plot_tree(self.model, 
                 feature_names=self.feature_names,
                 filled=True,
                 rounded=True,
                 fontsize=8,
                 max_depth=None,  # Show full tree
                 class_names=None,
                 label='all',
                 impurity=True,
                 proportion=False)
        
        max_depth_str = str(self.model.max_depth) if self.model.max_depth else "Unlimited"
        plt.title("Cotton Candy Digital Twin - Complete Decision Tree\n" + 
                 f"Features: {len(self.feature_names)}, Max Depth: {max_depth_str}", 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✅ Detailed tree saved to: {save_path}")
        
        return plt.gcf()
    
    def create_horizontal_tree_plot(self, figsize=(25, 12), save_path="horizontal_decision_tree.png"):
        """Create a horizontal tree layout for better readability."""
        print("🌲 Creating horizontal decision tree layout...")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal tree plot
        plot_tree(self.model,
                 feature_names=self.feature_names,
                 filled=True,
                 rounded=True,
                 fontsize=10,
                 max_depth=None,
                 ax=ax,
                 impurity=True,
                 proportion=False)
        
        ax.set_title("Cotton Candy Digital Twin - Horizontal Tree Layout\n" + 
                    f"Predicting Cotton Candy Weight (g)", 
                    fontsize=18, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✅ Horizontal tree saved to: {save_path}")
        
        return fig
    
    def create_tree_structure_text(self, save_path="tree_structure.txt"):
        """Create a text representation of the complete tree structure."""
        print("📄 Creating text representation of tree structure...")
        
        try:
            # Get the text representation
            tree_rules = export_text(self.model, feature_names=self.feature_names, 
                                    max_depth=None, spacing=3, decimals=2, show_weights=True)
            
            # Add header information
            max_depth_str = str(self.model.max_depth) if self.model.max_depth else "Unlimited"
            header = f"""COTTON CANDY DECISION TREE - COMPLETE STRUCTURE
{'='*60}

Model Information:
  • Algorithm: Decision Tree Regressor
  • Features: {len(self.feature_names)}
  • Max Depth: {max_depth_str}
  • Min Samples Split: {self.model.min_samples_split}
  • Min Samples Leaf: {self.model.min_samples_leaf}
  • Target: Cotton Candy Weight (grams)

Tree Structure:
{'-'*40}

"""
            
            full_text = header + tree_rules
            
            # Save to file
            with open(save_path, 'w') as f:
                f.write(full_text)
            
            print(f"   ✅ Tree structure saved to: {save_path}")
            return full_text
            
        except Exception as e:
            print(f"   ⚠️ Error creating text structure: {e}")
            print("   Continuing with other visualizations...")
            return None
    
    def analyze_tree_complexity(self):
        """Analyze and display tree complexity metrics."""
        print("📊 Analyzing tree complexity...")
        
        # Get tree structure information
        tree = self.model.tree_
        
        complexity_info = {
            'Total Nodes': tree.node_count,
            'Leaf Nodes': sum(tree.children_left == -1),
            'Internal Nodes': sum(tree.children_left != -1),
            'Max Depth': tree.max_depth,
            'Features Used': len(set(tree.feature[tree.feature != -2])),
            'Total Features Available': len(self.feature_names)
        }
        
        print("\n🔍 TREE COMPLEXITY ANALYSIS:")
        print("   " + "="*40)
        for metric, value in complexity_info.items():
            print(f"   • {metric}: {value}")
        
        return complexity_info
    
    def create_feature_usage_plot(self, figsize=(12, 8), save_path="feature_usage_in_tree.png"):
        """Create a plot showing which features are used in the tree."""
        print("📈 Creating feature usage visualization...")
        
        tree = self.model.tree_
        feature_usage = {}
        
        # Count how many times each feature is used for splitting
        for i in range(tree.node_count):
            if tree.children_left[i] != -1:  # Internal node
                feature_idx = tree.feature[i]
                feature_name = self.feature_names[feature_idx]
                feature_usage[feature_name] = feature_usage.get(feature_name, 0) + 1
        
        if not feature_usage:
            print("   ⚠️  No internal nodes found in tree")
            return None
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        features = list(feature_usage.keys())
        counts = list(feature_usage.values())
        
        # Create bar plot
        bars = ax.bar(range(len(features)), counts, color='skyblue', alpha=0.7)
        ax.set_xlabel('Features Used in Tree', fontsize=12)
        ax.set_ylabel('Number of Splits', fontsize=12)
        ax.set_title('Feature Usage in Decision Tree\nHow many times each feature is used for splitting', 
                    fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels for readability
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✅ Feature usage plot saved to: {save_path}")
        
        return fig
    
    def create_node_depth_analysis(self, figsize=(10, 6), save_path="node_depth_analysis.png"):
        """Analyze the distribution of nodes by depth."""
        print("📊 Creating node depth analysis...")
        
        tree = self.model.tree_
        
        # Calculate depth for each node
        def get_node_depth(node_id, depth=0):
            depths = {node_id: depth}
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]
            
            if left_child != -1:
                depths.update(get_node_depth(left_child, depth + 1))
            if right_child != -1:
                depths.update(get_node_depth(right_child, depth + 1))
            
            return depths
        
        node_depths = get_node_depth(0)
        
        # Count nodes at each depth
        depth_counts = {}
        for depth in node_depths.values():
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Bar plot of nodes per depth
        depths = list(depth_counts.keys())
        counts = list(depth_counts.values())
        
        ax1.bar(depths, counts, color='lightcoral', alpha=0.7)
        ax1.set_xlabel('Tree Depth', fontsize=12)
        ax1.set_ylabel('Number of Nodes', fontsize=12)
        ax1.set_title('Nodes per Depth Level', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative plot
        cumulative_counts = np.cumsum(counts)
        ax2.plot(depths, cumulative_counts, marker='o', color='darkgreen', linewidth=2)
        ax2.fill_between(depths, cumulative_counts, alpha=0.3, color='lightgreen')
        ax2.set_xlabel('Tree Depth', fontsize=12)
        ax2.set_ylabel('Cumulative Nodes', fontsize=12)
        ax2.set_title('Cumulative Node Count', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Decision Tree Depth Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✅ Node depth analysis saved to: {save_path}")
        
        return fig
    
    def create_complete_visualization_suite(self):
        """Create all visualizations for the decision tree."""
        print("\n🎨 CREATING COMPLETE TREE VISUALIZATION SUITE")
        print("="*60)
        
        results = {}
        
        # Create all visualizations
        try:
            self.create_detailed_tree_plot()
            results['detailed_plot'] = True
        except Exception as e:
            print(f"   ⚠️ Error creating detailed plot: {e}")
            results['detailed_plot'] = False
        
        try:
            self.create_horizontal_tree_plot()
            results['horizontal_plot'] = True
        except Exception as e:
            print(f"   ⚠️ Error creating horizontal plot: {e}")
            results['horizontal_plot'] = False
        
        try:
            self.create_tree_structure_text()
            results['text_structure'] = True
        except Exception as e:
            print(f"   ⚠️ Error creating text structure: {e}")
            results['text_structure'] = False
        
        try:
            self.create_feature_usage_plot()
            results['feature_usage'] = True
        except Exception as e:
            print(f"   ⚠️ Error creating feature usage plot: {e}")
            results['feature_usage'] = False
        
        try:
            self.create_node_depth_analysis()
            results['depth_analysis'] = True
        except Exception as e:
            print(f"   ⚠️ Error creating depth analysis: {e}")
            results['depth_analysis'] = False
        
        # Analyze complexity
        try:
            complexity = self.analyze_tree_complexity()
            results['complexity'] = complexity
        except Exception as e:
            print(f"   ⚠️ Error analyzing complexity: {e}")
            results['complexity'] = None
        
        print("\n🎉 VISUALIZATION SUITE COMPLETE!")
        success_count = sum(1 for k, v in results.items() if v is True or (k == 'complexity' and v is not None))
        print(f"   Successfully created {success_count}/{len(results)} visualizations")
        print("   Files created in DecisionTree/ folder:")
        
        if results.get('detailed_plot'):
            print("   • complete_decision_tree.png - Full tree visualization")
        if results.get('horizontal_plot'):
            print("   • horizontal_decision_tree.png - Horizontal layout")
        if results.get('text_structure'):
            print("   • tree_structure.txt - Text representation")
        if results.get('feature_usage'):
            print("   • feature_usage_in_tree.png - Feature usage analysis")
        if results.get('depth_analysis'):
            print("   • node_depth_analysis.png - Tree depth analysis")
        
        return results

def main():
    """Main function to run the tree visualizer."""
    print("🍭 COTTON CANDY DECISION TREE VISUALIZER")
    print("="*50)
    
    try:
        # Create visualizer
        visualizer = CottonCandyTreeVisualizer()
        
        # Create all visualizations
        results = visualizer.create_complete_visualization_suite()
        
        print(f"\n✅ Visualization suite completed!")
        
        # Display complexity info if available
        if results.get('complexity'):
            complexity = results['complexity']
            print(f"   Tree has {complexity['Total Nodes']} total nodes")
            print(f"   Uses {complexity['Features Used']}/{complexity['Total Features Available']} available features")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
