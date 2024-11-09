# MicroAttention
**A parameter-light micro-attention mechanism**<br>
![Architecture Schematic](https://github.com/DisOOM/MicroAttention/blob/main/MicroAttention.png)
A plug-and-play lightweight attention component for sequence modeling.
## Overview
MicroAttention is built on the hypothesis that the most generalized form of attention can be expressed as generating a contextual vector for each time step, which is weighted and aggregated in a well-justified manner.

In causal modeling, this form can be computed recursively by accumulating all past weighted V sums and score sums, with context vectors generated through dynamic normalization by dividing by the score sum.

### Key Features

- 🔋 **Lightweight**: Minimizes all non-essential components 
- ⚡ **Efficient Inference**: Only maintains one scalar and one vector
- 🔌 **No Position Encoding Required**
- ⚙️ **Flexible**: Supports both parallel training and recursive inference

## Technical Details

### Core Ideas

In MicroAttention, I minimize all non-essential components where:
- Input `x` directly plays the role of both v and k
- Query (q) consists of multiple learnable global semantic vectors forming the score matrix, which somewhat compensates for the lack of dynamically generated k, q, and v

We chose ReLU activation for three main advantages:
1. Non-negativity
2. Preserves original activation scale, making certain tokens stand out
3. Simple computation and numerical stability

These characteristics allow it to achieve effects closer to softmax during dynamic normalization.

The original score vector is summed to synthesize the scores of p queries, which further polarizes the token score distribution.

The differential computation serves multiple purposes:
1. Acts as a form of residual connection
2. Encodes position information to some degree
3. Provides a gating effect - producing fewer activations when current representations are similar to historical states

---
**Note**: This is an after-school experimental implementation aimed at exploring lightweight attention mechanisms. All feedback and discussions are welcome!
