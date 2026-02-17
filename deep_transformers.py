#!/usr/bin/env python3
"""
Deep Dive: Full Transformer Architecture Simulator
Simulates how layers build progressively deeper understanding
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

np.random.seed(42)

# ─────────────────────────────────────────────
# CORE MATH
# ─────────────────────────────────────────────

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def layer_norm(x):
    """Normalize to stabilize training"""
    return (x - x.mean()) / (x.std() + 1e-8)

def relu(x):
    """ReLU activation: max(0, x)"""
    return np.maximum(0, x)

def feed_forward(x, scale=1.2):
    """
    Feed Forward Network (FFN)
    Runs after attention in each transformer block.
    This is where 'memory' lives — facts the model learned.
    """
    # Expand (like neurons lighting up)
    hidden = relu(x * scale + np.random.randn(*x.shape) * 0.05)
    # Contract back
    output = hidden * 0.8
    return layer_norm(output)

def multi_head_attention(embeddings, n_heads=4):
    """
    Simulate multi-head attention
    Each head looks for different patterns
    """
    n_words, dim = embeddings.shape
    attention_outputs = []

    for head in range(n_heads):
        # Each head has different random projection (learned in real models)
        np.random.seed(head * 10)
        W_q = np.random.randn(dim, dim) * 0.1
        W_k = np.random.randn(dim, dim) * 0.1

        Q = embeddings @ W_q
        K = embeddings @ W_k

        # Scaled dot-product attention
        scores = Q @ K.T / np.sqrt(dim)
        weights = np.array([softmax(row) for row in scores])
        head_output = weights @ embeddings
        attention_outputs.append(head_output)

    # Combine all heads
    combined = np.mean(attention_outputs, axis=0)
    return layer_norm(combined), attention_outputs

# ─────────────────────────────────────────────
# TRANSFORMER BLOCK
# ─────────────────────────────────────────────

def transformer_block(embeddings, layer_num, n_heads=4):
    """
    One full transformer block:
    Input → Multi-Head Attention → Add & Norm → FFN → Add & Norm → Output
    """
    # Multi-Head Attention
    attn_output, head_outputs = multi_head_attention(embeddings, n_heads)

    # Residual Connection (Add & Norm) — crucial for deep networks!
    x = layer_norm(embeddings + attn_output)

    # Feed Forward Network
    ffn_output = feed_forward(x, scale=1.0 + layer_num * 0.05)

    # Another Residual Connection
    output = layer_norm(x + ffn_output)

    return output, head_outputs

# ─────────────────────────────────────────────
# WHAT EACH LAYER LEARNS (conceptual)
# ─────────────────────────────────────────────

LAYER_CONCEPTS = {
    1: {
        "name": "Surface Patterns",
        "learns": ["Basic syntax", "Part of speech", "Capitalization"],
        "example": "'cat' = noun, 'sat' = verb",
        "color": "#3498db"
    },
    2: {
        "name": "Local Grammar",
        "learns": ["Subject-verb pairs", "Noun phrases", "Verb phrases"],
        "example": "'cat sat' = subject-verb",
        "color": "#2ecc71"
    },
    3: {
        "name": "Semantic Roles",
        "learns": ["Who did what", "Object relationships", "Agent-patient"],
        "example": "cat=agent, mat=location",
        "color": "#e67e22"
    },
    4: {
        "name": "Coreference",
        "learns": ["What pronouns refer to", "Entity tracking", "Discourse"],
        "example": "'it' → 'cat' or 'mat'?",
        "color": "#9b59b6"
    },
    5: {
        "name": "World Knowledge",
        "learns": ["Facts about entities", "Common sense", "Context"],
        "example": "cats sit on things",
        "color": "#e74c3c"
    },
    6: {
        "name": "Abstract Meaning",
        "learns": ["Intent", "Tone", "High-level semantics"],
        "example": "Simple declarative scene",
        "color": "#1abc9c"
    },
}

# ─────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────

def plot_architecture_overview():
    """Draw the transformer architecture"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('#0f0f1a')
    fig.patch.set_facecolor('#0f0f1a')

    title_color = 'white'
    box_color = '#1a1a2e'

    ax.text(7, 9.5, 'The Transformer Architecture', fontsize=16,
            color=title_color, ha='center', va='center', fontweight='bold')

    # ── Input Tokens ──
    tokens = ['The', 'cat', 'sat', 'on', 'mat']
    colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#e74c3c']
    for i, (tok, col) in enumerate(zip(tokens, colors)):
        x = 1.5 + i * 2.2
        rect = mpatches.FancyBboxPatch((x-0.6, 0.3), 1.2, 0.7,
                                        boxstyle="round,pad=0.05",
                                        facecolor=col, edgecolor='white', linewidth=1)
        ax.add_patch(rect)
        ax.text(x, 0.65, tok, color='white', ha='center', va='center',
                fontsize=10, fontweight='bold')

    ax.text(7, 0.15, 'Input Tokens', color='#aaaaaa', ha='center', fontsize=9)

    # ── Embeddings ──
    for i in range(5):
        x = 1.5 + i * 2.2
        ax.annotate('', xy=(x, 1.6), xytext=(x, 1.05),
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.5))

    rect = mpatches.FancyBboxPatch((0.5, 1.6), 13, 0.7,
                                    boxstyle="round,pad=0.05",
                                    facecolor='#16213e', edgecolor='#3498db', linewidth=2)
    ax.add_patch(rect)
    ax.text(7, 1.95, '📐 Embeddings + Positional Encoding', color='#3498db',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # ── Transformer Blocks ──
    block_colors = ['#1a472a', '#2d1b69', '#4a1942', '#1b3a4b', '#3d2b1f', '#1a3a3a']
    border_colors = ['#2ecc71', '#9b59b6', '#e74c3c', '#3498db', '#e67e22', '#1abc9c']

    for i in range(6):
        y_base = 2.6 + i * 1.0
        col = block_colors[i]
        border = border_colors[i]
        layer_info = LAYER_CONCEPTS[i+1]

        # Block background
        rect = mpatches.FancyBboxPatch((0.5, y_base), 13, 0.85,
                                        boxstyle="round,pad=0.05",
                                        facecolor=col, edgecolor=border, linewidth=1.5)
        ax.add_patch(rect)

        # Layer number
        ax.text(1.2, y_base + 0.42, f'L{i+1}', color=border,
                ha='center', va='center', fontsize=9, fontweight='bold')

        # Components inside block
        components = ['Multi-Head\nAttention', 'Add &\nNorm', 'Feed\nForward', 'Add &\nNorm']
        comp_colors = ['#2980b9', '#27ae60', '#8e44ad', '#27ae60']
        for j, (comp, cc) in enumerate(zip(components, comp_colors)):
            cx = 2.5 + j * 2.5
            crect = mpatches.FancyBboxPatch((cx-0.8, y_base+0.1), 1.6, 0.65,
                                             boxstyle="round,pad=0.05",
                                             facecolor=cc, edgecolor='white',
                                             linewidth=0.8, alpha=0.85)
            ax.add_patch(crect)
            ax.text(cx, y_base+0.42, comp, color='white', ha='center',
                    va='center', fontsize=7.5)

        # What this layer learns
        ax.text(12.5, y_base + 0.42, layer_info['name'], color=border,
                ha='center', va='center', fontsize=8, style='italic')

        # Arrow to next block
        if i < 5:
            ax.annotate('', xy=(7, y_base+0.95), xytext=(7, y_base+0.88),
                        arrowprops=dict(arrowstyle='->', color='white', lw=1))

    # ── Output ──
    ax.annotate('', xy=(7, 9.0), xytext=(7, 8.55),
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
    rect = mpatches.FancyBboxPatch((3, 9.0), 8, 0.6,
                                    boxstyle="round,pad=0.05",
                                    facecolor='#7f0000', edgecolor='#ff6b6b', linewidth=2)
    ax.add_patch(rect)
    ax.text(7, 9.3, '🎯 Output: Next Token Probabilities', color='white',
            ha='center', va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('/home/claude/transformer_architecture.png', dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("✓ Saved: transformer_architecture.png")


def plot_layer_evolution():
    """Show how embeddings evolve through layers"""
    words = ['The', 'cat', 'sat', 'on', 'mat']
    n_words = len(words)
    dim = 8

    # Initial embeddings
    np.random.seed(99)
    embeddings = np.random.randn(n_words, dim)

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#0f0f1a')
    gs = GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

    n_layers = 6
    layer_outputs = [embeddings.copy()]
    current = embeddings.copy()

    for i in range(n_layers):
        current, _ = transformer_block(current, i+1)
        layer_outputs.append(current.copy())

    panel_layers = [0, 1, 2, 4, 5, 6]
    panel_titles = [
        'Raw Embeddings\n(Before any layer)',
        'After Layer 1\n(Surface patterns)',
        'After Layer 2\n(Grammar learned)',
        'After Layer 4\n(Coreference)',
        'After Layer 5\n(World knowledge)',
        'After Layer 6\n(Full meaning)',
    ]

    panel_colors = ['#1a1a2e', '#1a2e1a', '#1a1a2e', '#2e1a2e', '#2e1a1a', '#1a2e2e']

    for idx, (layer_idx, title) in enumerate(zip(panel_layers, panel_titles)):
        row = idx // 4
        col = idx % 4
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(panel_colors[idx])

        data = layer_outputs[layer_idx]
        im = ax.imshow(data, cmap='RdYlBu', aspect='auto',
                       vmin=-2, vmax=2)
        ax.set_yticks(range(n_words))
        ax.set_yticklabels(words, color='white', fontsize=9)
        ax.set_xticks([])
        ax.set_title(title, color='white', fontsize=9, pad=8)

        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')

    # Highlight differences
    ax_diff = fig.add_subplot(gs[1, 3])
    ax_diff.set_facecolor('#1a1a1a')
    diff = np.abs(layer_outputs[-1] - layer_outputs[0])
    im = ax_diff.imshow(diff, cmap='hot', aspect='auto')
    ax_diff.set_yticks(range(n_words))
    ax_diff.set_yticklabels(words, color='white', fontsize=9)
    ax_diff.set_xticks([])
    ax_diff.set_title('Total Change\n(bright = most transformed)', color='white',
                       fontsize=9, pad=8)
    for spine in ax_diff.spines.values():
        spine.set_edgecolor('#444444')

    fig.suptitle('How Word Representations Evolve Through Transformer Layers',
                 color='white', fontsize=14, fontweight='bold', y=1.01)

    plt.savefig('/home/claude/layer_evolution.png', dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("✓ Saved: layer_evolution.png")


def plot_residual_connections():
    """Show why residual connections are critical"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0f0f1a')

    for ax in axes:
        ax.set_facecolor('#0f0f1a')

    x = np.linspace(0, 10, 200)

    # Without residuals: gradient vanishes
    signal_no_res = np.exp(-x * 0.5) * np.sin(x * 2)
    axes[0].plot(x, signal_no_res, color='#e74c3c', lw=2.5)
    axes[0].axhline(0, color='#444444', lw=0.8)
    axes[0].fill_between(x, signal_no_res, alpha=0.2, color='#e74c3c')
    axes[0].set_title('❌ Without Residual Connections\nGradient vanishes — deep layers learn nothing',
                       color='white', fontsize=11)
    axes[0].set_xlabel('Layer Depth', color='white')
    axes[0].set_ylabel('Signal Strength (Gradient)', color='white')
    axes[0].tick_params(colors='white')
    for spine in axes[0].spines.values():
        spine.set_edgecolor('#444444')

    # With residuals: signal stays strong
    signal_with_res = 0.8 + 0.15 * np.sin(x * 2) + np.random.randn(200) * 0.02
    axes[1].plot(x, signal_with_res, color='#2ecc71', lw=2.5)
    axes[1].axhline(0.8, color='#444444', lw=0.8, linestyle='--')
    axes[1].fill_between(x, signal_with_res, 0.8, alpha=0.2, color='#2ecc71')
    axes[1].set_title('✅ With Residual Connections\nGradient stays strong — all layers learn!',
                       color='white', fontsize=11)
    axes[1].set_xlabel('Layer Depth', color='white')
    axes[1].tick_params(colors='white')
    for spine in axes[1].spines.values():
        spine.set_edgecolor('#444444')

    formula_text = "Output = LayerNorm(x + Attention(x))\n← The '+x' is the residual connection"
    fig.text(0.5, -0.02, formula_text, ha='center', color='#f39c12',
             fontsize=11, fontfamily='monospace')

    plt.suptitle('Why Residual Connections Are Critical for Deep Learning',
                 color='white', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig('/home/claude/residual_connections.png', dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("✓ Saved: residual_connections.png")


def explain_feed_forward():
    """Explain the FFN as memory storage"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           THE FEED FORWARD NETWORK: WHERE MEMORY LIVES              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Each transformer block has TWO parts:                               ║
║                                                                      ║
║  1. ATTENTION  → "Who should I talk to?"                            ║
║     Handles relationships between words dynamically                  ║
║                                                                      ║
║  2. FEED FORWARD NETWORK (FFN) → "What do I know about this?"       ║
║     Stores factual knowledge learned during training                 ║
║                                                                      ║
║  The FFN is 4x BIGGER than the attention layer.                     ║
║  In GPT-3: 12,288 → 49,152 → 12,288 dimensions                     ║
║                                                                      ║
║  Think of it like:                                                   ║
║  ┌─────────────────────────────────────────────────────────┐        ║
║  │ Attention: "I'm processing the word 'Paris'"           │        ║
║  │ FFN neuron 4,521: *activates strongly*                 │        ║
║  │ FFN output: adds knowledge about France, Eiffel Tower  │        ║
║  │             Romance, fashion, history...               │        ║
║  └─────────────────────────────────────────────────────────┘        ║
║                                                                      ║
║  Research has found specific neurons that store:                     ║
║  - "Paris is in France" (factual)                                    ║
║  - "Python uses indentation" (coding knowledge)                      ║
║  - "Shakespeare wrote in verse" (literary knowledge)                 ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

def explain_scaling():
    """Show the shocking impact of scale"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                   THE SCALING LAWS: WHY SIZE MATTERS                ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Model         │ Parameters  │ Emergent Abilities                   ║
║  ──────────────┼─────────────┼──────────────────────────────        ║
║  GPT-1 (2018)  │ 117M        │ Basic text completion               ║
║  GPT-2 (2019)  │ 1.5B        │ Coherent paragraphs                 ║
║  GPT-3 (2020)  │ 175B        │ Few-shot learning, reasoning        ║
║  GPT-4 (2023)  │ ~1.8T est.  │ Multimodal, expert-level tasks     ║
║  Claude 3 Opus │ Unknown     │ Nuanced reasoning, long context     ║
║                                                                      ║
║  The SHOCKING DISCOVERY: Emergent Abilities                          ║
║                                                                      ║
║  Nobody programmed GPT-3 to do arithmetic. It just... could.       ║
║  Nobody taught GPT-4 to reason about chemistry. It emerged.         ║
║                                                                      ║
║  At certain scale thresholds, new abilities appear suddenly —       ║
║  like a phase transition in physics.                                 ║
║                                                                      ║
║  Small model (1B params):  Cannot do multi-step math               ║
║  Medium model (10B params): Struggles with multi-step math          ║
║  Large model (100B params): Suddenly masters multi-step math! ✨    ║
║                                                                      ║
║  This is called "emergence" — and it's not fully understood yet.   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🏗️  Deep Transformer Architecture Simulator\n")
    print("Generating visualizations...\n")

    plot_architecture_overview()
    plot_layer_evolution()
    plot_residual_connections()
    explain_feed_forward()
    explain_scaling()

    print("\n" + "=" * 70)
    print("ALL COMPONENTS EXPLAINED")
    print("=" * 70)
    print("""
A transformer block = 4 key operations:

  [Input]
     ↓
  Multi-Head Attention   → Words look at each other
     ↓
  Add & Norm (Residual)  → Don't forget original meaning
     ↓
  Feed Forward Network   → Apply stored knowledge
     ↓
  Add & Norm (Residual)  → Stabilize the signal
     ↓
  [Richer Output]

Stack this 32-96 times = A modern LLM!
""")
    print("✓ All visualizations saved!\n")