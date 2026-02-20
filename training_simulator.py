#!/usr/bin/env python3
"""
Training Simulator: How Language Models Learn
Demonstrates pre-training, loss calculation, and backpropagation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════
# PART 1: THE PREDICTION GAME
# ═══════════════════════════════════════════════════════════════

def softmax(x):
    """Convert scores to probabilities"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def cross_entropy_loss(predictions, target_idx):
    """
    How wrong was the model?
    
    Cross-entropy loss measures the distance between:
    - What the model predicted (probability distribution)
    - What actually came next (one-hot encoded)
    
    Perfect prediction → Loss = 0
    Terrible prediction → Loss = infinity
    """
    # Avoid log(0)
    predictions = np.clip(predictions, 1e-10, 1.0)
    return -np.log(predictions[target_idx])

def simulate_prediction_before_training(sentence, vocab):
    """Show what a random (untrained) model predicts"""
    words = sentence.split()
    vocab_size = len(vocab)
    
    print("=" * 70)
    print("UNTRAINED MODEL: Random Predictions")
    print("=" * 70)
    
    total_loss = 0
    predictions_log = []
    
    for i in range(len(words) - 1):
        context = " ".join(words[:i+1])
        actual_next = words[i+1]
        
        # Random model: all words equally likely (no learning yet)
        random_scores = np.random.randn(vocab_size)
        probabilities = softmax(random_scores)
        
        # What did it predict?
        predicted_idx = np.argmax(probabilities)
        predicted_word = vocab[predicted_idx]
        
        # How confident was it in the CORRECT word?
        actual_idx = vocab.index(actual_next)
        confidence_in_correct = probabilities[actual_idx]
        
        # Calculate loss
        loss = cross_entropy_loss(probabilities, actual_idx)
        total_loss += loss
        
        predictions_log.append({
            'context': context,
            'predicted': predicted_word,
            'actual': actual_next,
            'confidence': confidence_in_correct,
            'loss': loss
        })
        
        print(f"\nContext: '{context}'")
        print(f"  Model predicts: '{predicted_word}' (random guess)")
        print(f"  Actual next word: '{actual_next}'")
        print(f"  Confidence in correct answer: {confidence_in_correct:.1%}")
        print(f"  Loss: {loss:.3f} {'😢 terrible!' if loss > 3 else ''}")
    
    avg_loss = total_loss / (len(words) - 1)
    print(f"\n{'='*70}")
    print(f"Average Loss: {avg_loss:.3f}")
    print(f"This is BAD. The model is basically guessing randomly.")
    print(f"{'='*70}\n")
    
    return predictions_log, avg_loss

def simulate_prediction_after_training(sentence, vocab):
    """Show what a trained model predicts"""
    words = sentence.split()
    vocab_size = len(vocab)
    
    print("=" * 70)
    print("TRAINED MODEL: Learned Predictions")
    print("=" * 70)
    
    # Simulate learned patterns (in reality these come from billions of examples)
    learned_patterns = {
        'The': ['cat', 'dog', 'mat'],
        'cat': ['sat', 'ran', 'jumped'],
        'sat': ['on', 'down', 'up'],
        'on': ['the', 'a'],
        'the': ['mat', 'chair', 'floor']
    }
    
    total_loss = 0
    predictions_log = []
    
    for i in range(len(words) - 1):
        context = " ".join(words[:i+1])
        actual_next = words[i+1]
        last_word = words[i]
        
        # Trained model: learned what typically comes next
        scores = np.random.randn(vocab_size) * 0.1  # Small random noise
        
        # Boost scores for likely next words (learned pattern)
        if last_word in learned_patterns:
            for likely_word in learned_patterns[last_word]:
                if likely_word in vocab:
                    idx = vocab.index(likely_word)
                    scores[idx] += 5.0  # Strong signal for learned patterns
        
        probabilities = softmax(scores)
        
        # What did it predict?
        predicted_idx = np.argmax(probabilities)
        predicted_word = vocab[predicted_idx]
        
        # Confidence in correct answer
        actual_idx = vocab.index(actual_next)
        confidence_in_correct = probabilities[actual_idx]
        
        # Calculate loss
        loss = cross_entropy_loss(probabilities, actual_idx)
        total_loss += loss
        
        predictions_log.append({
            'context': context,
            'predicted': predicted_word,
            'actual': actual_next,
            'confidence': confidence_in_correct,
            'loss': loss
        })
        
        print(f"\nContext: '{context}'")
        print(f"  Model predicts: '{predicted_word}'", 
              f"{'✓ CORRECT!' if predicted_word == actual_next else '✗ wrong'}")
        print(f"  Actual next word: '{actual_next}'")
        print(f"  Confidence in correct answer: {confidence_in_correct:.1%}")
        print(f"  Loss: {loss:.3f} {'✨ great!' if loss < 1 else '😐 okay' if loss < 2 else '😢 bad'}")
    
    avg_loss = total_loss / (len(words) - 1)
    print(f"\n{'='*70}")
    print(f"Average Loss: {avg_loss:.3f}")
    print(f"Much better! The model learned patterns from training data.")
    print(f"{'='*70}\n")
    
    return predictions_log, avg_loss

# ═══════════════════════════════════════════════════════════════
# PART 2: BACKPROPAGATION - HOW LEARNING HAPPENS
# ═══════════════════════════════════════════════════════════════

def visualize_backprop():
    """
    Show how errors propagate backward through the network
    to update parameters
    """
    print("\n" + "=" * 70)
    print("BACKPROPAGATION: The Learning Algorithm")
    print("=" * 70)
    print("""
The model makes a prediction → it's wrong → we need to fix it.
But HOW do we fix billions of parameters?

Answer: BACKPROPAGATION (gradient descent)

┌─────────────────────────────────────────────────────────────┐
│                    THE LEARNING LOOP                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. FORWARD PASS                                            │
│     Input: "The cat sat"                                    │
│     → Layer 1 → Layer 2 → ... → Layer 32                    │
│     Output: Predict "on" (wrong! it was "on")               │
│     Calculate loss: How wrong was I?                        │
│                                                             │
│  2. BACKWARD PASS (Backpropagation)                         │
│     Start at output: "I should have predicted 'on' higher"  │
│     ← Adjust Layer 32 parameters slightly                   │
│     ← Adjust Layer 31 parameters slightly                   │
│     ← ... all the way back to Layer 1                       │
│                                                             │
│  3. UPDATE PARAMETERS                                       │
│     New_Weight = Old_Weight - (learning_rate × gradient)    │
│                                                             │
│  4. REPEAT                                                  │
│     Do this TRILLIONS of times on different examples        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
""")

def simulate_gradient_descent():
    """Visualize how a parameter learns over time"""
    
    # Imagine one parameter learning to predict "cat" after "The"
    # True optimal value: 1.0
    # Starting value: random
    
    iterations = 100
    learning_rate = 0.1
    
    # Random start
    parameter_value = np.random.randn() * 2
    optimal_value = 1.0
    
    history = []
    loss_history = []
    
    for i in range(iterations):
        # Calculate error (simplified)
        error = parameter_value - optimal_value
        loss = error ** 2
        
        # Gradient: which direction should we move?
        gradient = 2 * error  # Derivative of (x - optimal)^2
        
        # Update parameter
        parameter_value -= learning_rate * gradient
        
        history.append(parameter_value)
        loss_history.append(loss)
    
    # Plot
    fig = plt.figure(figsize=(14, 5))
    fig.patch.set_facecolor('#0f0f1a')
    
    # Parameter evolution
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_facecolor('#0f0f1a')
    ax1.plot(history, color='#3498db', linewidth=2.5, label='Parameter Value')
    ax1.axhline(optimal_value, color='#2ecc71', linestyle='--', 
                linewidth=2, label='Optimal Value')
    ax1.scatter([0], [history[0]], color='#e74c3c', s=100, 
                zorder=5, label='Start (random)')
    ax1.scatter([len(history)-1], [history[-1]], color='#f39c12', 
                s=100, zorder=5, label='End (learned)')
    ax1.set_xlabel('Training Iterations', color='white', fontsize=11)
    ax1.set_ylabel('Parameter Value', color='white', fontsize=11)
    ax1.set_title('How a Single Parameter Learns', color='white', 
                  fontsize=12, pad=10)
    ax1.tick_params(colors='white')
    ax1.legend(facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#444444')
    ax1.grid(True, alpha=0.2, color='#444')
    
    # Loss decrease
    ax2 = plt.subplot(1, 2, 2)
    ax2.set_facecolor('#0f0f1a')
    ax2.plot(loss_history, color='#e74c3c', linewidth=2.5)
    ax2.fill_between(range(len(loss_history)), loss_history, 
                     alpha=0.3, color='#e74c3c')
    ax2.set_xlabel('Training Iterations', color='white', fontsize=11)
    ax2.set_ylabel('Loss (Error)', color='white', fontsize=11)
    ax2.set_title('Loss Decreases as Model Learns', color='white', 
                  fontsize=12, pad=10)
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#444444')
    ax2.grid(True, alpha=0.2, color='#444')
    
    plt.tight_layout()
    plt.savefig('/home/claude/gradient_descent.png', dpi=150, 
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("✓ Saved: gradient_descent.png")

# ═══════════════════════════════════════════════════════════════
# PART 3: THE SCALE OF TRAINING
# ═══════════════════════════════════════════════════════════════

def explain_training_scale():
    """Show the mind-boggling scale of modern training"""
    print("\n" + "=" * 70)
    print("THE SCALE OF PRE-TRAINING")
    print("=" * 70)
    print("""
Let's talk numbers. GPT-3 training:

┌────────────────────────────────────────────────────────────┐
│ TRAINING DATA                                              │
├────────────────────────────────────────────────────────────┤
│ • 300 BILLION tokens                                       │
│ • That's ~570,000 copies of "War and Peace"                │
│ • Or all of Wikipedia × 30                                 │
│ • Sources: Books, web, Reddit, Wikipedia, code             │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ COMPUTE                                                    │
├────────────────────────────────────────────────────────────┤
│ • 175 billion parameters to update                         │
│ • 10,000 GPUs running for ~1 month                         │
│ • Cost: Estimated $4-12 million                            │
│ • 3,640 petaflop-days (10²¹ operations)                    │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ THE PROCESS                                                │
├────────────────────────────────────────────────────────────┤
│ For EACH of 300 billion tokens:                            │
│   1. Forward pass through 96 layers                        │
│   2. Calculate loss                                        │
│   3. Backward pass (compute gradients)                     │
│   4. Update 175 billion parameters                         │
│                                                            │
│ This happens in BATCHES of ~3.2 million tokens at once    │
│ Takes weeks even with thousands of GPUs in parallel        │
└────────────────────────────────────────────────────────────┘

And this is just Phase 1!
After pre-training comes fine-tuning and RLHF...
""")

def visualize_training_phases():
    """Show the three phases of training"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0f0f1a')
    
    phases = [
        {
            'name': 'Pre-Training',
            'color': '#3498db',
            'data': 'Internet-scale text\n(trillions of tokens)',
            'goal': 'Learn language patterns',
            'duration': '~1 month',
            'cost': '$5-10M'
        },
        {
            'name': 'Supervised Fine-Tuning',
            'color': '#e67e22',
            'data': 'High-quality examples\n(~100K conversations)',
            'goal': 'Learn helpful responses',
            'duration': '~1-2 days',
            'cost': '$100K'
        },
        {
            'name': 'RLHF',
            'color': '#2ecc71',
            'data': 'Human preferences\n(~10-50K comparisons)',
            'goal': 'Align to human values',
            'duration': '~1-2 days',
            'cost': '$200K'
        }
    ]
    
    for ax, phase in zip(axes, phases):
        ax.set_facecolor('#1a1a2e')
        ax.axis('off')
        
        # Create box
        from matplotlib.patches import FancyBboxPatch
        rect = FancyBboxPatch((0.1, 0.1), 0.8, 0.8,
                              boxstyle="round,pad=0.05",
                              facecolor=phase['color'],
                              edgecolor='white',
                              linewidth=2,
                              alpha=0.3)
        ax.add_patch(rect)
        
        # Add text
        ax.text(0.5, 0.85, phase['name'], ha='center', va='center',
                fontsize=13, fontweight='bold', color='white')
        ax.text(0.5, 0.65, phase['data'], ha='center', va='center',
                fontsize=9, color='white')
        ax.text(0.5, 0.45, f"Goal: {phase['goal']}", ha='center', va='center',
                fontsize=9, color='#f39c12', style='italic')
        ax.text(0.5, 0.25, f"Time: {phase['duration']}", ha='center', va='center',
                fontsize=8, color='#aaaaaa')
        ax.text(0.5, 0.15, f"Cost: {phase['cost']}", ha='center', va='center',
                fontsize=8, color='#aaaaaa')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    plt.suptitle('The Three Phases of Training a Language Model',
                 color='white', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig('/home/claude/training_phases.png', dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("✓ Saved: training_phases.png")

# ═══════════════════════════════════════════════════════════════
# MAIN DEMONSTRATION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "🎓" * 30)
    print("TRAINING DEEP DIVE: How Language Models Learn")
    print("🎓" * 30 + "\n")
    
    # Simple vocabulary for demonstration
    vocab = ['The', 'cat', 'sat', 'on', 'the', 'mat', 'dog', 'ran', 
             'jumped', 'chair', 'floor', 'a', 'an']
    
    sentence = "The cat sat on the mat"
    
    print("\nWe'll use this simple sentence to demonstrate:")
    print(f"'{sentence}'")
    print(f"\nVocabulary size: {len(vocab)} words\n")
    
    # Show untrained vs trained
    print("\n" + "▼" * 70)
    simulate_prediction_before_training(sentence, vocab)
    
    print("\n" + "▼" * 70)
    simulate_prediction_after_training(sentence, vocab)
    
    # Explain backpropagation
    visualize_backprop()
    
    # Visualize learning
    print("\nGenerating gradient descent visualization...")
    simulate_gradient_descent()
    
    # Show training phases
    print("\nGenerating training phases overview...")
    visualize_training_phases()
    
    # Scale explanation
    explain_training_scale()
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
1. PREDICTION TASK: The model learns by predicting the next token
   over and over on trillions of examples.

2. LOSS FUNCTION: Measures how wrong the prediction was.
   Training = minimize this loss.

3. BACKPROPAGATION: The algorithm that figures out how to adjust
   billions of parameters to reduce loss.

4. SCALE MATTERS: You need:
   - Massive data (300B+ tokens)
   - Massive compute (10K+ GPUs)
   - Time (weeks)
   - Money (millions of dollars)

5. EMERGENCE: At sufficient scale, the model doesn't just memorize—
   it learns to reason, generalize, and solve new problems.

This is just PHASE 1 (pre-training).
Next: Fine-tuning and RLHF (how raw models become helpful assistants)
""")
    print("=" * 70)
    print("\n✓ Demonstration complete!\n")