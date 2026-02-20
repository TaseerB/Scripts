#!/usr/bin/env python3
"""
Modern Gen AI Techniques: The Complete Production Guide
Covers RAG, Fine-tuning, Agents, and Optimization (2024-2025)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.gridspec import GridSpec
import json

# ═══════════════════════════════════════════════════════════════
# TECHNIQUE 1: RAG (RETRIEVAL AUGMENTED GENERATION)
# ═══════════════════════════════════════════════════════════════

def explain_rag():
    """Explain RAG and when to use it"""
    print("=" * 70)
    print("TECHNIQUE 1: RAG (RETRIEVAL AUGMENTED GENERATION)")
    print("=" * 70)
    print("""
THE PROBLEM:
LLMs have a knowledge cutoff. They don't know about:
- Your company's internal documents
- Recent events after training
- Private/proprietary information
- Real-time data (stock prices, weather, news)

SOLUTION 1 (BAD): Fine-tune the model on new data
❌ Expensive ($$$)
❌ Slow (days/weeks)
❌ Data gets "baked in" (can't easily update)
❌ Requires lots of data

SOLUTION 2 (GOOD): RAG - Give the model access to external knowledge
✓ Fast to implement (hours)
✓ Cheap (just API calls)
✓ Easy to update (change the database)
✓ Works with small amounts of data

╔══════════════════════════════════════════════════════════════╗
║                    HOW RAG WORKS                             ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Step 1: INDEX YOUR KNOWLEDGE                               ║
║  ────────────────────────────────────────────────────────── ║
║  • Split documents into chunks (~500 words each)            ║
║  • Convert each chunk into an embedding (vector)            ║
║  • Store embeddings in a vector database                    ║
║                                                              ║
║  Step 2: RETRIEVAL (when user asks a question)              ║
║  ────────────────────────────────────────────────────────── ║
║  • Convert user query into an embedding                     ║
║  • Search vector DB for most similar chunks                 ║
║  • Retrieve top 3-5 most relevant chunks                    ║
║                                                              ║
║  Step 3: AUGMENTATION                                       ║
║  ────────────────────────────────────────────────────────── ║
║  • Inject retrieved chunks into the prompt                  ║
║  • Send to LLM with original query                          ║
║                                                              ║
║  Step 4: GENERATION                                         ║
║  ────────────────────────────────────────────────────────── ║
║  • LLM generates answer using provided context              ║
║  • Can cite specific chunks                                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

EXAMPLE:

User Query: "What's our company's vacation policy?"

Without RAG:
LLM: "I don't have access to your company's specific policies..."

With RAG:
1. Query embedding: [0.23, 0.87, -0.45, ...]
2. Vector DB finds similar: "Employee handbook, Section 4.2: Vacation"
3. Retrieved text: "Employees receive 15 days of PTO annually..."
4. Augmented prompt:
   "Context: Employees receive 15 days of PTO annually...
    Question: What's our company's vacation policy?
    Answer:"
5. LLM: "According to your employee handbook, you receive 15 days 
    of paid time off annually..."

KEY INSIGHT: The LLM doesn't "learn" the information. It just uses
what you give it in the prompt. Next query? Retrieves fresh info.
""")

def rag_architecture():
    """Visualize RAG architecture"""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#0f0f1a')
    
    ax = plt.subplot(1, 1, 1)
    ax.set_facecolor('#0f0f1a')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(8, 11.5, 'RAG Architecture: Knowledge Retrieval Pipeline',
            ha='center', fontsize=14, color='white', fontweight='bold')
    
    # Step 1: Documents
    docs = ['Doc 1\nVacation\nPolicy', 'Doc 2\nExpense\nRules', 'Doc 3\nRemote\nWork']
    for i, doc in enumerate(docs):
        x = 1 + i * 1.8
        rect = FancyBboxPatch((x-0.4, 9), 0.8, 1.2,
                              boxstyle="round,pad=0.05",
                              facecolor='#3498db', edgecolor='white',
                              linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, 9.6, doc, ha='center', va='center',
                fontsize=8, color='white')
    
    ax.text(2.8, 8.5, '① Documents', ha='center',
            fontsize=10, color='#3498db', fontweight='bold')
    
    # Arrow down
    ax.annotate('', xy=(2.8, 7.8), xytext=(2.8, 8.3),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    
    # Step 2: Chunking + Embedding
    rect = FancyBboxPatch((0.5, 6.5), 4.6, 1.2,
                          boxstyle="round,pad=0.1",
                          facecolor='#2c3e50', edgecolor='#e67e22',
                          linewidth=2)
    ax.add_patch(rect)
    ax.text(2.8, 7.4, 'Chunk & Embed', ha='center',
            fontsize=10, color='white', fontweight='bold')
    ax.text(2.8, 6.9, 'Text → Vectors', ha='center',
            fontsize=8, color='#aaa')
    
    ax.text(2.8, 6.0, '② Embedding Model', ha='center',
            fontsize=10, color='#e67e22', fontweight='bold')
    
    # Arrow down
    ax.annotate('', xy=(2.8, 5.3), xytext=(2.8, 5.8),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    
    # Step 3: Vector Database
    rect = FancyBboxPatch((0.5, 3.8), 4.6, 1.4,
                          boxstyle="round,pad=0.1",
                          facecolor='#16a085', edgecolor='white',
                          linewidth=2, alpha=0.4)
    ax.add_patch(rect)
    ax.text(2.8, 4.8, '🗄️ Vector Database', ha='center',
            fontsize=10, color='white', fontweight='bold')
    
    # Show some vectors
    for i in range(3):
        ax.text(1.2 + i*1.6, 4.2, f'[{np.random.rand():.2f}...]',
                fontsize=7, color='#2ecc71', family='monospace')
    
    ax.text(2.8, 3.3, '③ Indexed Knowledge', ha='center',
            fontsize=10, color='#2ecc71', fontweight='bold')
    
    # User Query (right side)
    rect = FancyBboxPatch((10, 9), 4, 1.2,
                          boxstyle="round,pad=0.1",
                          facecolor='#9b59b6', edgecolor='white',
                          linewidth=2)
    ax.add_patch(rect)
    ax.text(12, 9.6, '❓ User Query\n"Vacation policy?"',
            ha='center', va='center', fontsize=9, color='white')
    
    # Arrow down
    ax.annotate('', xy=(12, 7.8), xytext=(12, 8.8),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    
    # Embed query
    rect = FancyBboxPatch((10, 6.5), 4, 1.2,
                          boxstyle="round,pad=0.1",
                          facecolor='#2c3e50', edgecolor='#e67e22',
                          linewidth=2)
    ax.add_patch(rect)
    ax.text(12, 7.1, 'Embed Query', ha='center',
            fontsize=10, color='white', fontweight='bold')
    
    # Arrow to vector DB (search)
    arrow = FancyArrowPatch((10, 4.5), (5.2, 4.5),
                           arrowstyle='<->', mutation_scale=20,
                           color='#f39c12', linewidth=3)
    ax.add_patch(arrow)
    ax.text(7.6, 5.0, '④ Similarity Search', ha='center',
            fontsize=9, color='#f39c12', fontweight='bold')
    
    # Retrieved chunks
    rect = FancyBboxPatch((10, 2.8), 4, 1.4,
                          boxstyle="round,pad=0.1",
                          facecolor='#27ae60', edgecolor='white',
                          linewidth=2, alpha=0.4)
    ax.add_patch(rect)
    ax.text(12, 3.8, '📄 Top 3 Chunks', ha='center',
            fontsize=10, color='white', fontweight='bold')
    ax.text(12, 3.2, 'Most relevant\ndocument sections',
            ha='center', fontsize=7, color='#aaa')
    
    # Arrow down
    ax.annotate('', xy=(12, 2.0), xytext=(12, 2.6),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    
    # LLM
    rect = FancyBboxPatch((8.5, 0.3), 7, 1.6,
                          boxstyle="round,pad=0.1",
                          facecolor='#c0392b', edgecolor='white',
                          linewidth=2, alpha=0.5)
    ax.add_patch(rect)
    ax.text(12, 1.4, '🤖 Large Language Model', ha='center',
            fontsize=11, color='white', fontweight='bold')
    ax.text(12, 0.8, 'Context: [chunks] + Query: [question]',
            ha='center', fontsize=8, color='#aaa', family='monospace')
    
    ax.text(12, -0.2, '⑤ Generate Answer with Context', ha='center',
            fontsize=10, color='#e74c3c', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/claude/rag_architecture.png', dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("✓ Saved: rag_architecture.png")

def explain_rag_challenges():
    """Discuss RAG challenges"""
    print("\n" + "-" * 70)
    print("RAG CHALLENGES & SOLUTIONS")
    print("-" * 70)
    print("""
CHALLENGE 1: Chunk Size
────────────────────────────────────────────────────────────
Too small (100 words): Loses context, misses connections
Too large (2000 words): Irrelevant info, wastes tokens

✓ SOLUTION: 
  - Typically 300-500 words
  - Overlap chunks by 50-100 words
  - Experiment with your specific domain

CHALLENGE 2: Retrieval Quality
────────────────────────────────────────────────────────────
Semantic search isn't perfect. Sometimes retrieves irrelevant docs.

✓ SOLUTIONS:
  - Hybrid search (semantic + keyword)
  - Reranking (retrieve 20, rerank to top 3)
  - Query expansion (rephrase query multiple ways)
  - Metadata filtering (date, author, category)

CHALLENGE 3: Context Length Limits
────────────────────────────────────────────────────────────
Models have token limits. Can't fit entire database in context.

✓ SOLUTIONS:
  - Retrieve top K most relevant (typically K=3-5)
  - Hierarchical retrieval (coarse → fine)
  - Use long-context models (Claude: 200K tokens)
  - Compress retrieved text

CHALLENGE 4: Hallucination Despite RAG
────────────────────────────────────────────────────────────
Model still makes stuff up even with correct context.

✓ SOLUTIONS:
  - Prompt engineering: "Only use provided context"
  - Citation enforcement: "Quote the source"
  - Confidence scores: "Say 'I don't know' if uncertain"
  - Post-processing: Verify facts against retrieved docs

CHALLENGE 5: Cost
────────────────────────────────────────────────────────────
Every query = embedding API call + LLM API call

✓ SOLUTIONS:
  - Cache common queries
  - Batch processing
  - Use cheaper embedding models
  - Self-host embedding model
  - Smart chunking to reduce LLM token usage
""")

# ═══════════════════════════════════════════════════════════════
# TECHNIQUE 2: FINE-TUNING STRATEGIES
# ═══════════════════════════════════════════════════════════════

def explain_finetuning():
    """Explain when and how to fine-tune"""
    print("\n" + "=" * 70)
    print("TECHNIQUE 2: FINE-TUNING STRATEGIES")
    print("=" * 70)
    print("""
WHEN TO FINE-TUNE vs. WHEN TO USE RAG:

╔══════════════════════════════════════════════════════════════╗
║                 RAG vs FINE-TUNING DECISION                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  USE RAG WHEN:                                               ║
║  • You need fresh/changing information                       ║
║  • You have a knowledge base to query                        ║
║  • You want to cite sources                                  ║
║  • You need transparency in where answers come from          ║
║  • Budget is limited                                         ║
║                                                              ║
║  USE FINE-TUNING WHEN:                                       ║
║  • You need a specific style/tone                            ║
║  • You want to teach new behaviors/patterns                  ║
║  • You need consistent formatting                            ║
║  • You have many examples of input → output                  ║
║  • Latency matters (no retrieval step)                       ║
║                                                              ║
║  USE BOTH WHEN:                                              ║
║  • Complex domain with both style and knowledge needs        ║
║  • Fine-tune for behavior, RAG for facts                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

EXAMPLES:

Scenario 1: Customer support chatbot for your SaaS product
→ RAG: Product docs, FAQs, troubleshooting guides
→ Fine-tuning: Company tone, specific response patterns

Scenario 2: Medical diagnosis assistant
→ RAG: Latest research papers, drug databases
→ Fine-tuning: Medical terminology, diagnosis patterns

Scenario 3: Code generation in your company's style
→ Fine-tuning: Your company's coding conventions, patterns
→ RAG: Internal libraries, API documentation

FINE-TUNING METHODS (2024-2025):

┌──────────────────────────────────────────────────────────────┐
│ 1. FULL FINE-TUNING (Traditional)                            │
├──────────────────────────────────────────────────────────────┤
│ Update ALL parameters in the model                           │
│                                                              │
│ Cost: $$$$$ (Very expensive)                                 │
│ Time: Days/weeks                                             │
│ Data: 10K-100K+ examples needed                              │
│ Use: Only for major custom models                            │
│                                                              │
│ Example: OpenAI doesn't even offer this for GPT-4            │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ 2. LoRA (Low-Rank Adaptation) ⭐ RECOMMENDED                 │
├──────────────────────────────────────────────────────────────┤
│ Freeze original model, add small "adapter" layers            │
│                                                              │
│ Cost: $$ (10-100x cheaper than full fine-tuning)             │
│ Time: Hours                                                   │
│ Data: 100-10K examples                                        │
│ Memory: ~1/3 of original model                                │
│                                                              │
│ How it works:                                                │
│ Instead of updating W (huge matrix), update:                 │
│ W_new = W_frozen + A × B                                     │
│ where A and B are small matrices                             │
│                                                              │
│ Example: Fine-tune GPT-3.5 with LoRA = ~$50 instead of $5K  │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ 3. QLoRA (Quantized LoRA) ⭐⭐ BEST FOR RESEARCH             │
├──────────────────────────────────────────────────────────────┤
│ LoRA + quantization (4-bit) = Fine-tune on consumer GPU!     │
│                                                              │
│ Cost: $ (Can run on single GPU)                              │
│ Time: Hours                                                   │
│ Memory: 1/4 of LoRA (can fine-tune 65B model on 1x A100)    │
│                                                              │
│ Trade-off: Slight quality decrease vs full fine-tuning       │
│ Sweet spot: Best balance of cost/quality/accessibility       │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ 4. PREFIX TUNING / PROMPT TUNING                             │
├──────────────────────────────────────────────────────────────┤
│ Only learn a "soft prompt" (embedding vectors)               │
│                                                              │
│ Cost: $ (Cheapest)                                           │
│ Data: 50-1000 examples                                        │
│ Limitation: Less powerful than LoRA                          │
│                                                              │
│ Use case: Very limited compute, simple tasks                 │
└──────────────────────────────────────────────────────────────┘
""")

def visualize_lora():
    """Visualize how LoRA works"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0f0f1a')
    
    for ax in axes:
        ax.set_facecolor('#0f0f1a')
        ax.axis('off')
    
    # Full fine-tuning
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(5, 9, 'Full Fine-Tuning', ha='center',
            fontsize=12, color='white', fontweight='bold')
    
    # Large weight matrix
    rect = FancyBboxPatch((2, 4), 6, 4,
                          boxstyle="round,pad=0.1",
                          facecolor='#e74c3c', edgecolor='white',
                          linewidth=2, alpha=0.5)
    ax.add_patch(rect)
    ax.text(5, 6, 'Weights: W\n175B parameters\nALL UPDATED', ha='center',
            va='center', fontsize=10, color='white', fontweight='bold')
    
    ax.text(5, 2.5, '💰 Cost: $5,000+', ha='center',
            fontsize=10, color='#e74c3c')
    ax.text(5, 1.8, '⏱️ Time: Days', ha='center',
            fontsize=10, color='#e74c3c')
    ax.text(5, 1.1, '💾 Memory: 700GB+', ha='center',
            fontsize=10, color='#e74c3c')
    
    # LoRA
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(5, 9, 'LoRA (Low-Rank Adaptation)', ha='center',
            fontsize=12, color='white', fontweight='bold')
    
    # Frozen weights
    rect = FancyBboxPatch((2, 4), 6, 4,
                          boxstyle="round,pad=0.1",
                          facecolor='#95a5a6', edgecolor='white',
                          linewidth=2, alpha=0.3)
    ax.add_patch(rect)
    ax.text(5, 6, 'W (FROZEN)\n175B parameters\nNOT updated', ha='center',
            va='center', fontsize=9, color='#aaa')
    
    # Small adapter matrices
    rect = FancyBboxPatch((1, 2.5), 1.5, 1,
                          boxstyle="round,pad=0.05",
                          facecolor='#2ecc71', edgecolor='white',
                          linewidth=2)
    ax.add_patch(rect)
    ax.text(1.75, 3, 'A\n(small)', ha='center', va='center',
            fontsize=8, color='white')
    
    ax.text(2.8, 3, '×', ha='center', va='center',
            fontsize=14, color='white')
    
    rect = FancyBboxPatch((3.3, 2.5), 1.5, 1,
                          boxstyle="round,pad=0.05",
                          facecolor='#3498db', edgecolor='white',
                          linewidth=2)
    ax.add_patch(rect)
    ax.text(4.05, 3, 'B\n(small)', ha='center', va='center',
            fontsize=8, color='white')
    
    ax.text(5.2, 3, '= ΔW', ha='center', va='center',
            fontsize=12, color='#f39c12', fontweight='bold')
    
    ax.text(7.5, 3, 'Only ~1M\nparameters!', ha='center', va='center',
            fontsize=9, color='#2ecc71')
    
    ax.text(5, 1.5, '💰 Cost: $50', ha='center',
            fontsize=10, color='#2ecc71')
    ax.text(5, 0.9, '⏱️ Time: Hours', ha='center',
            fontsize=10, color='#2ecc71')
    ax.text(5, 0.3, '💾 Memory: ~200GB', ha='center',
            fontsize=10, color='#2ecc71')
    
    plt.suptitle('LoRA: 100x Cheaper, 10x Faster, Same Quality',
                 color='white', fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('/home/claude/lora_comparison.png', dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("✓ Saved: lora_comparison.png")

# ═══════════════════════════════════════════════════════════════
# TECHNIQUE 3: AGENTS & TOOL USE
# ═══════════════════════════════════════════════════════════════

def explain_agents():
    """Explain AI agents and tool use"""
    print("\n" + "=" * 70)
    print("TECHNIQUE 3: AI AGENTS & TOOL USE")
    print("=" * 70)
    print("""
THE EVOLUTION:

2020: LLMs can chat
2022: LLMs can write code
2023: LLMs can USE TOOLS → AI Agents are born
2024: Agents can chain multiple tools, plan, reflect

╔══════════════════════════════════════════════════════════════╗
║                    WHAT IS AN AGENT?                         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  An AI that can:                                             ║
║  1. Take actions (not just generate text)                    ║
║  2. Use tools (APIs, databases, calculators, code)           ║
║  3. Make decisions (which tool? when? how?)                  ║
║  4. Iterate (try → observe → adjust → retry)                 ║
║                                                              ║
║  Example:                                                    ║
║  User: "Find me the cheapest flight to Tokyo next month     ║
║         and book it."                                        ║
║                                                              ║
║  Agent:                                                      ║
║  1. Calls flight search API                                  ║
║  2. Compares prices across airlines                          ║
║  3. Checks user's calendar for availability                  ║
║  4. Calls booking API                                        ║
║  5. Sends confirmation email                                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

TOOL USE / FUNCTION CALLING:

The foundation of agents is giving LLMs access to tools.

┌──────────────────────────────────────────────────────────────┐
│ HOW IT WORKS                                                 │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│ 1. Define available tools:                                   │
│    {                                                         │
│      "name": "web_search",                                   │
│      "description": "Search the web for information",        │
│      "parameters": {                                         │
│        "query": "string"                                     │
│      }                                                       │
│    }                                                         │
│                                                              │
│ 2. LLM decides to use a tool:                                │
│    "I should search for current weather in Tokyo"           │
│    → Generates: {"tool": "web_search",                       │
│                  "args": {"query": "Tokyo weather"}}        │
│                                                              │
│ 3. Your code executes the tool:                              │
│    result = web_search("Tokyo weather")                     │
│    → "Currently 18°C, partly cloudy..."                     │
│                                                              │
│ 4. Send result back to LLM:                                  │
│    "Tool result: Currently 18°C, partly cloudy..."          │
│    LLM: "The weather in Tokyo is currently 18°C..."        │
│                                                              │
└──────────────────────────────────────────────────────────────┘

AGENT ARCHITECTURES:

┌──────────────────────────────────────────────────────────────┐
│ 1. ReAct (Reasoning + Acting)                                │
├──────────────────────────────────────────────────────────────┤
│ Loop:                                                        │
│   Thought: "I need to search for this"                       │
│   Action: web_search("query")                                │
│   Observation: "Found: ..."                                  │
│   Thought: "Now I should calculate..."                       │
│   Action: calculator(123 + 456)                              │
│   Observation: "Result: 579"                                 │
│   Thought: "I have enough info to answer"                    │
│   Final Answer: "..."                                        │
│                                                              │
│ Strengths: Transparent reasoning, good for debugging         │
│ Weakness: Can get stuck in loops, expensive (many LLM calls) │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ 2. Plan-and-Execute                                          │
├──────────────────────────────────────────────────────────────┤
│ Step 1: Create a plan                                        │
│   "To book a flight, I need to:                              │
│    1. Search flights                                         │
│    2. Compare prices                                         │
│    3. Check calendar                                         │
│    4. Book the flight"                                       │
│                                                              │
│ Step 2: Execute each step                                    │
│   [Actually calls the tools]                                 │
│                                                              │
│ Strengths: More efficient, better for complex tasks          │
│ Weakness: Less adaptable if plan fails                       │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ 3. Reflexion (Self-Reflection)                               │
├──────────────────────────────────────────────────────────────┤
│ Try → Evaluate → Learn from failure → Retry                  │
│                                                              │
│ Example:                                                     │
│ Attempt 1: "I'll search for 'best pizza'"                   │
│ Result: Too generic, unhelpful                               │
│ Reflection: "I should be more specific about location"       │
│ Attempt 2: "I'll search for 'best pizza in Brooklyn'"       │
│ Result: Success!                                             │
│                                                              │
│ Strengths: Learns from mistakes, improves over time          │
│ Weakness: More LLM calls, needs good reflection prompts      │
└──────────────────────────────────────────────────────────────┘

REAL AGENT EXAMPLES (2024-2025):

✓ Coding agents (Cursor, Devin, Claude Code)
  • Write code, run tests, debug, iterate
  
✓ Research agents (Perplexity, Claude with search)
  • Search web, synthesize info, cite sources
  
✓ Data analysis agents (ChatGPT Code Interpreter)
  • Read CSV, plot graphs, run statistics
  
✓ Customer service agents (many companies)
  • Query knowledge base, create tickets, escalate
  
✓ Task automation (Zapier AI, n8n with AI)
  • Chain multiple tools, complex workflows
""")

def visualize_agent_loop():
    """Visualize the agent decision loop"""
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#0f0f1a')
    
    ax = plt.subplot(1, 1, 1)
    ax.set_facecolor('#0f0f1a')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    ax.text(7, 11.5, 'AI Agent: The ReAct Loop',
            ha='center', fontsize=14, color='white', fontweight='bold')
    
    # Central Agent
    rect = FancyBboxPatch((5, 5), 4, 2,
                          boxstyle="round,pad=0.1",
                          facecolor='#9b59b6', edgecolor='white',
                          linewidth=2, alpha=0.6)
    ax.add_patch(rect)
    ax.text(7, 6.2, '🤖 LLM Agent', ha='center',
            fontsize=11, color='white', fontweight='bold')
    ax.text(7, 5.6, 'Thinks & Decides', ha='center',
            fontsize=9, color='#aaa')
    
    # Tools around it
    tools = [
        {'name': '🔍 Web Search', 'x': 2, 'y': 9, 'color': '#3498db'},
        {'name': '🧮 Calculator', 'x': 12, 'y': 9, 'color': '#e67e22'},
        {'name': '📧 Email API', 'x': 2, 'y': 2, 'color': '#2ecc71'},
        {'name': '💾 Database', 'x': 12, 'y': 2, 'color': '#e74c3c'},
    ]
    
    for tool in tools:
        rect = FancyBboxPatch((tool['x']-1, tool['y']-0.5), 2, 1,
                              boxstyle="round,pad=0.08",
                              facecolor=tool['color'], edgecolor='white',
                              linewidth=1.5, alpha=0.5)
        ax.add_patch(rect)
        ax.text(tool['x'], tool['y'], tool['name'], ha='center', va='center',
                fontsize=9, color='white')
        
        # Bidirectional arrows
        if tool['x'] < 7:  # Left side
            arrow = FancyArrowPatch((tool['x']+1.2, tool['y']), (5, 6),
                                   arrowstyle='<->', mutation_scale=15,
                                   color='white', linewidth=1.5, alpha=0.7)
        else:  # Right side
            arrow = FancyArrowPatch((tool['x']-1.2, tool['y']), (9, 6),
                                   arrowstyle='<->', mutation_scale=15,
                                   color='white', linewidth=1.5, alpha=0.7)
        ax.add_patch(arrow)
    
    # Example trace
    trace_y = 0.3
    trace_text = """
Example Trace:
1. Thought: "I need to find current weather" → Action: web_search("Tokyo weather")
2. Observation: "18°C, cloudy" → Thought: "Now convert to Fahrenheit"
3. Action: calculator(18 * 9/5 + 32) → Observation: "64.4°F"
4. Thought: "I have the answer" → Final Answer: "It's 64.4°F in Tokyo"
"""
    ax.text(7, trace_y, trace_text, ha='center', va='bottom',
            fontsize=8, color='#f39c12', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/home/claude/agent_loop.png', dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("✓ Saved: agent_loop.png")

# ═══════════════════════════════════════════════════════════════
# TECHNIQUE 4: OPTIMIZATION (COST & SPEED)
# ═══════════════════════════════════════════════════════════════

def explain_optimization():
    """Explain production optimization techniques"""
    print("\n" + "=" * 70)
    print("TECHNIQUE 4: OPTIMIZATION (COST & SPEED)")
    print("=" * 70)
    print("""
In production, two things matter: COST and LATENCY.
Here's how to optimize both:

╔══════════════════════════════════════════════════════════════╗
║                    COST OPTIMIZATION                         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║ 1. MODEL SELECTION                                           ║
║    • GPT-4: $30/1M tokens (slow, expensive, best quality)    ║
║    • GPT-3.5: $0.50/1M tokens (fast, cheap, good quality)    ║
║    • Claude Sonnet: $3/1M tokens (fast, mid-cost, great)     ║
║    • Claude Haiku: $0.25/1M tokens (fastest, cheapest)       ║
║                                                              ║
║    Strategy: Use the smallest model that works               ║
║    Example: GPT-4 for complex reasoning, GPT-3.5 for simple  ║
║                                                              ║
║ 2. PROMPT ENGINEERING                                        ║
║    Bad: 2000 token prompt → $0.06 per call (GPT-4)           ║
║    Good: 200 token prompt → $0.006 per call                  ║
║    Savings: 10x cheaper!                                     ║
║                                                              ║
║    Techniques:                                               ║
║    • Remove unnecessary examples                             ║
║    • Use concise instructions                                ║
║    • Cache common prefixes (see caching below)               ║
║                                                              ║
║ 3. CACHING (Anthropic's Prompt Caching) ⭐                   ║
║    Cache parts of your prompt that don't change.             ║
║                                                              ║
║    Example:                                                  ║
║    Prompt structure:                                         ║
║    [Large system instructions - 5000 tokens] ← CACHE THIS    ║
║    [Company knowledge base - 10000 tokens]   ← CACHE THIS    ║
║    [User query - 50 tokens]                  ← Fresh input   ║
║                                                              ║
║    First call: Pay for all 15,050 tokens                     ║
║    Subsequent calls: Pay for 50 tokens only!                 ║
║    Savings: Up to 90% reduction in cost                      ║
║                                                              ║
║ 4. BATCHING                                                  ║
║    Process multiple requests together when possible.         ║
║    • 50% cost reduction for batch API (OpenAI)               ║
║    • Trade-off: 24-hour turnaround time                      ║
║    • Good for: Data analysis, classification, etc.           ║
║                                                              ║
║ 5. OUTPUT TOKEN LIMITING                                     ║
║    Set max_tokens to what you actually need.                 ║
║    • Need a JSON object? max_tokens=200                      ║
║    • Need a summary? max_tokens=500                          ║
║    • Don't let model ramble (saves output token cost)        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║                    LATENCY OPTIMIZATION                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║ 1. MODEL SELECTION                                           ║
║    • Claude Haiku: ~3 seconds                                ║
║    • GPT-3.5 Turbo: ~5 seconds                               ║
║    • Claude Sonnet: ~8 seconds                               ║
║    • GPT-4: ~15 seconds                                      ║
║                                                              ║
║ 2. STREAMING                                                 ║
║    Instead of waiting for entire response:                   ║
║    • Stream tokens as they're generated                      ║
║    • User sees output immediately                            ║
║    • Feels 3-5x faster even though total time is same        ║
║                                                              ║
║    Implementation: Set stream=True in API call               ║
║                                                              ║
║ 3. PARALLEL PROCESSING                                       ║
║    Independent operations? Run them in parallel.             ║
║                                                              ║
║    Sequential (slow):                                        ║
║    1. Translate text → 5s                                    ║
║    2. Summarize text → 5s                                    ║
║    3. Extract keywords → 5s                                  ║
║    Total: 15 seconds                                         ║
║                                                              ║
║    Parallel (fast):                                          ║
║    Run all three simultaneously                              ║
║    Total: 5 seconds                                          ║
║                                                              ║
║ 4. PREFETCHING / PREDICTION                                  ║
║    Anticipate what users will ask next.                      ║
║    • Start generating likely follow-ups                      ║
║    • Cache results                                           ║
║    • If predicted correctly: instant response                ║
║                                                              ║
║ 5. QUANTIZATION (Self-hosting)                               ║
║    Run models in 4-bit or 8-bit precision.                   ║
║    • 4x less memory                                          ║
║    • 2-3x faster inference                                   ║
║    • Minimal quality loss                                    ║
║    • Tools: llama.cpp, bitsandbytes, GPTQ                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

ADVANCED: SPECULATIVE DECODING
────────────────────────────────────────────────────────────────
For self-hosted models:
• Small model generates draft tokens (fast)
• Large model verifies draft (batch operation)
• Keep correct tokens, reject wrong ones
• Result: 2-3x faster generation

State-of-the-art in 2024-2025.

CONTEXT WINDOW OPTIMIZATION
────────────────────────────────────────────────────────────────
Long context models (Claude: 200K tokens) are powerful but:
• More expensive
• Slower
• "Lost in the middle" problem

Strategies:
✓ Smart chunking (only include relevant parts)
✓ Summarization (condense older context)
✓ Sliding window (keep recent + summarized older)
✓ Hierarchical attention (coarse → fine)

REAL-WORLD COST EXAMPLE:
────────────────────────────────────────────────────────────────
Chatbot handling 1M requests/month:

Naive approach:
• GPT-4, 2000 token prompts
• Cost: $60,000/month

Optimized:
• GPT-3.5 for 80% of requests (simple queries)
• GPT-4 for 20% (complex queries)
• Prompt caching enabled
• Prompts reduced to 500 tokens
• Cost: $4,000/month

Savings: $56,000/month (93% reduction!)
""")

def visualize_cost_latency_tradeoff():
    """Visualize model selection tradeoffs"""
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    
    models = [
        {'name': 'Claude Haiku', 'cost': 0.25, 'latency': 3, 'quality': 7, 'color': '#3498db'},
        {'name': 'GPT-3.5', 'cost': 0.5, 'latency': 5, 'quality': 7.5, 'color': '#2ecc71'},
        {'name': 'Claude Sonnet', 'cost': 3, 'latency': 8, 'quality': 9, 'color': '#e67e22'},
        {'name': 'GPT-4', 'cost': 30, 'latency': 15, 'quality': 9.5, 'color': '#9b59b6'},
        {'name': 'Claude Opus', 'cost': 15, 'latency': 12, 'quality': 9.7, 'color': '#e74c3c'},
    ]
    
    for model in models:
        # Size represents quality
        size = (model['quality'] ** 2) * 100
        ax.scatter(model['latency'], model['cost'], s=size,
                  alpha=0.6, color=model['color'], edgecolors='white', linewidth=2)
        
        # Label
        offset_y = -0.3 if model['cost'] < 10 else 0.3
        ax.text(model['latency'], model['cost'] + offset_y, model['name'],
               ha='center', fontsize=10, color='white', fontweight='bold')
    
    ax.set_xlabel('Latency (seconds)', color='white', fontsize=12)
    ax.set_ylabel('Cost ($/1M tokens)', color='white', fontsize=12)
    ax.set_title('Model Selection: Cost vs Latency vs Quality\n(bubble size = quality)',
                color='white', fontsize=13, fontweight='bold', pad=15)
    
    ax.set_yscale('log')
    ax.grid(True, alpha=0.2, color='#444')
    ax.tick_params(colors='white')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    
    # Add annotations
    ax.annotate('Sweet Spot\nfor most apps', xy=(5, 0.5),
               xytext=(8, 1.5), fontsize=9, color='#f39c12',
               arrowprops=dict(arrowstyle='->', color='#f39c12', lw=2))
    
    ax.annotate('Use for\ncomplex tasks', xy=(15, 30),
               xytext=(11, 20), fontsize=9, color='#f39c12',
               arrowprops=dict(arrowstyle='->', color='#f39c12', lw=2))
    
    plt.tight_layout()
    plt.savefig('/home/claude/cost_latency_tradeoff.png', dpi=150,
               bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print("✓ Saved: cost_latency_tradeoff.png")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "🚀" * 30)
    print("MODERN GEN AI TECHNIQUES: PRODUCTION GUIDE")
    print("🚀" * 30 + "\n")
    
    explain_rag()
    rag_architecture()
    explain_rag_challenges()
    
    explain_finetuning()
    visualize_lora()
    
    explain_agents()
    visualize_agent_loop()
    
    explain_optimization()
    visualize_cost_latency_tradeoff()
    
    print("\n" + "=" * 70)
    print("PRODUCTION CHECKLIST")
    print("=" * 70)
    print("""
Before deploying an LLM application to production:

☐ Security
  ✓ Input validation (prevent prompt injection)
  ✓ Output filtering (catch harmful content)
  ✓ Rate limiting (prevent abuse)
  ✓ API key rotation
  
☐ Monitoring
  ✓ Log all requests/responses
  ✓ Track latency metrics
  ✓ Monitor costs per user/endpoint
  ✓ Set up alerts for anomalies
  
☐ Quality
  ✓ Human evaluation on sample queries
  ✓ A/B testing different approaches
  ✓ Regression testing (save test cases)
  ✓ Red-teaming (try to break it)
  
☐ Cost Management
  ✓ Budget alerts
  ✓ Caching strategy
  ✓ Model selection per use case
  ✓ Rate limits per user tier
  
☐ User Experience
  ✓ Streaming for perceived speed
  ✓ Loading states
  ✓ Error messages (fallback responses)
  ✓ Citation/source attribution
  
☐ Compliance
  ✓ Data retention policy
  ✓ GDPR compliance (user data)
  ✓ Terms of service for AI usage
  ✓ Content moderation
""")
    print("=" * 70)
    print("\n✓ Advanced Techniques Complete!\n")