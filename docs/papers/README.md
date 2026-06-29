# Solver Literature

This folder keeps the research references used to judge the solver design. Store
open-access PDFs in `pdfs/`; for closed or project-hosted references, keep a
stable citation/link here instead of checking in a copied file.

## Core References

| Topic | Reference | Local file | Why it matters |
|---|---|---|---|
| AlphaZero loop | Silver et al., "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (2017), arXiv:1712.01815 | `pdfs/alphazero-1712.01815.pdf` | Generate -> train -> evaluate loop; policy/value model behind search. |
| AlphaGo Zero | Silver et al., "Mastering the game of Go without human knowledge" (Nature, 2017) | Link only: https://www.nature.com/articles/nature24270 | Self-play without human data; useful as design lineage, not a direct fit for simultaneous imperfect tactical states. |
| DeepStack | Moravcik et al., "DeepStack: Expert-Level Artificial Intelligence in Heads-Up No-Limit Poker" (Science, 2017), arXiv:1701.01724 | `pdfs/deepstack-1701.01724.pdf` | Depth-limited continual re-solving with a learned value function at the frontier. |
| Libratus | Brown and Sandholm, "Superhuman AI for heads-up no-limit poker: Libratus beats top professionals" (Science, 2018) | Link only: https://www.science.org/doi/10.1126/science.aao1733 | Blueprint strategy plus endgame solving; directly relevant to subgame-resolve hardening. |
| ReBeL | Brown et al., "Combining Deep Reinforcement Learning and Search for Imperfect-Information Games" (2020), arXiv:2007.13544 | `pdfs/rebel-2007.13544.pdf` | Public-belief/state value learning and search in imperfect-information games. |
| CFR | Zinkevich et al., "Regret Minimization in Games with Incomplete Information" (NeurIPS, 2007) | `pdfs/cfr-neurips-2007.pdf` | Original CFR/exploitability foundation. |
| CFR+ | Tammelin, "Solving Large Imperfect Information Games Using CFR+" (2014), arXiv:1407.5042 | `pdfs/cfr-plus-1407.5042.pdf` | Faster regret-minimization baseline for large imperfect-information games. |
| Simultaneous-move MCTS | Lisý et al., "Monte Carlo Tree Search in Simultaneous Move Games" (2013) | Link only: https://arxiv.org/abs/1301.0421 | Justifies average root strategies over unstable final pure vertices. |
| Stockfish / NNUE | Stockfish NNUE documentation | Link only: https://official-stockfish.github.io/docs/stockfish-wiki/Stockfish-NNUE.html | Search/evaluator separation and fast incremental neural evaluation. |
| Local unknown | `duel1712.pdf` | `pdfs/duel1712.pdf` | Moved from repo root; metadata was not identifiable locally and should be renamed once identified. |

## Reading Order

1. CFR and simultaneous-move MCTS: understand the exact/search core.
2. AlphaZero and AlphaGo Zero: understand the generate/train/evaluate loop.
3. DeepStack, Libratus, ReBeL: understand why local subgame re-solving is the
   correct next pressure point.
4. Stockfish/NNUE: use as a systems analogy for fast evaluation behind search,
   not as a claim that this repo is chess-engine equivalent.
