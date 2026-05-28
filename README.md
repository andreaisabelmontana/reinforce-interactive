# REINFORCE — Interactive RL Playground

Every core Reinforcement Learning concept, rendered as a live demo you can poke, tweak, and watch learn in real time.

**Live site:** https://andreaisabelmontana.github.io/reinforce-interactive/

## What's inside

| # | Page | What you can do |
|---|------|-----------------|
| 01 | [Multi-Armed Bandits](bandits.html) | Compare ε-greedy, UCB, Thompson sampling, gradient bandits, optimistic init. Watch regret accumulate. |
| 02 | [MDPs & Gridworld](mdp.html) | Build a gridworld by clicking — place walls, goals, pits. See transition probabilities for every action. |
| 03 | [Dynamic Programming](dp.html) | Value iteration vs policy iteration. Watch v* and π* converge on the grid. |
| 04 | [MC & TD Learning](td.html) | Q-Learning, SARSA, Expected SARSA, Monte Carlo control on cliff-walk and frozen lake. |
| 05 | [Function Approximation](function-approx.html) | Polynomial, RBF, tile coding, Fourier, neural net features on a 1-D regression task. |
| 06 | [REINFORCE](policy-gradient.html) | Vanilla policy gradient with optional baseline. Toggle the baseline and see variance change. |
| 07 | [Actor-Critic (A2C)](actor-critic.html) | Side-by-side: REINFORCE vs A2C. Same env, same hyperparams, different variance. |
| 08 | [DQN](dqn.html) | Neural Q-network on Mountain Car. Replay buffer, target network, ε-decay. |
| 09 | [Evolutionary RL](evolution.html) | A genetic algorithm evolves a linear CartPole controller. No gradients. |
| 10 | [Multi-Agent (Axelrod)](multi-agent.html) | Round-robin iterated prisoner's dilemma. Watch Tit-for-Tat and friends. |

## How it's built

100% vanilla JavaScript, no frameworks, no build step. Every algorithm runs in your browser at full speed. Canvas 2D for all rendering.

- `js/common.js` — shared utilities (RNG, softmax, drawing helpers, animation loop)
- `js/gridworld.js` — shared MDP engine used by mdp/dp/td pages
- `js/<page>.js` — one file per demo

## Running locally

Any static file server works:

```bash
cd reinforce-interactive
python -m http.server 8000
# open http://localhost:8000
```

## License

MIT.
