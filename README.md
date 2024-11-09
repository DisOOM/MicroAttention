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

Here's the complete implementation of MicroAttention:

```python
class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.p = args.p
        
        # Scoring matrix
        self.ScoresMatrix = nn.Parameter(torch.randn(self.p, self.dim))
        nn.init.xavier_uniform_(self.Matrix)
        self.relu = nn.ReLU()
        
        # Output projection
        self.out_proj = nn.Linear(self.dim, self.dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        
        # 1. Calculate semantic scores [B, S, P], ReLU for polarized positive output
        scores = self.relu(torch.einsum('bsd,pd->bsp', x, self.Matrix))
        
        # 2. Normalize over P dimension to get weights for each position [B, S, 1]
        scores_normalized = scores.sum(dim=-1, keepdim=True)
        
        # 3. Calculate weighted representation [B, S, D]
        weighted = x * scores_normalized
        
        # 4. Cumsum over sequence dimension [B, S, D]
        cum_weighted = torch.cumsum(weighted, dim=1)
        cum_scores = torch.cumsum(scores_normalized, dim=1)  # [B, S, 1]
        
        # 5. Calculate attention [B, S, D]
        attn = cum_weighted / (cum_scores + 1e-9)
        
        # 6. Calculate difference
        diff = x - attn
        
        return self.out_proj(diff)
```

---
**Note**: This is an after-school experimental implementation aimed at exploring lightweight attention mechanisms. All feedback and discussions are welcome!
