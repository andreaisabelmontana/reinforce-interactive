# REINFORCE — Interactive RL Playground

Every core Reinforcement Learning concept, rendered as a live demo you can poke, tweak, and watch learn in real time. The algorithms behind the demos are extracted into framework-free ES modules under `src/` and covered by a unit-test suite — the pages import those modules, so the tested code is the code that runs.

**Live site:** https://andreaisabelmontana.github.io/reinforce-interactive/

## The pages

| # | Page | What you can do |
|---|------|-----------------|
| 01 | [Multi-Armed Bandits](bandits.html) | Compare ε-greedy, UCB, Thompson sampling, gradient bandits, optimistic init. Watch regret accumulate. |
| 02 | [MDPs & Gridworld](mdp.html) | Build a gridworld by clicking — place walls, goals, pits. See transition probabilities for every action. |
| 03 | [Dynamic Programming](dp.html) | Value iteration vs policy iteration. Watch v* and π* converge on the grid. |
| 04 | [MC & TD Learning](td.html) | Q-Learning, SARSA, Expected SARSA, Monte Carlo control on cliff-walk and frozen lake. |
| 05 | [Function Approximation](function-approx.html) | Polynomial, RBF, tile coding, Fourier, neural-net features on a 1-D regression task. |
| 06 | [REINFORCE](policy-gradient.html) | Vanilla policy gradient with optional baseline. Toggle the baseline and see variance change. |
| 07 | [Actor-Critic (A2C)](actor-critic.html) | Side-by-side: REINFORCE vs A2C. Same env, same hyperparams, different variance. |
| 08 | [DQN](dqn.html) | Neural Q-network on Mountain Car. Replay buffer, target network, ε-decay. |
| 09 | [Evolutionary RL](evolution.html) | A genetic algorithm evolves a linear CartPole controller. No gradients. |
| 10 | [Multi-Agent](multi-agent.html) | Round-robin iterated prisoner's dilemma. Watch Tit-for-Tat and friends. |

## Tested algorithm cores (`src/`)

The numerical heart of the bandit, MDP, DP, TD and function-approximation demos lives in small, DOM-free modules. They take a seedable PRNG so every run is reproducible.

| Module | Contents |
|--------|----------|
| `src/rng.js` | `mulberry32` seedable PRNG + `RNG` (uniform / gaussian / integer / categorical), `argmax`, `argmaxTies`, `softmax` |
| `src/gridworld.js` | `Gridworld` MDP: states `(r,c)`, 4 actions, full transition distribution `p(s′,r\|s,a)`, sampling, slip, walls, goal/pit terminals, cliff teleport |
| `src/dp.js` | `valueIteration`, `policyIteration`, `actionValue`, `greedyPolicy` |
| `src/td.js` | `trainEpisode` / `train` for Q-learning, SARSA, Expected SARSA; `policyFromQ`, rollout helpers |
| `src/bandit.js` | k-armed Gaussian `Bandit` with ε-greedy / UCB / greedy / optimistic-init selection and cumulative-regret tracking |
| `src/function-approx.js` | poly / RBF / tile / Fourier feature bases + `LinearVFA` semi-gradient SGD value approximator |

The browser pages import these directly (`dp.html`, `mdp.html`, `td.html`, `bandits.html`, `function-approx.html` load their scripts as `type="module"`), so the demos and the tests exercise the same code.

## What the tests prove

`test/` uses Node's built-in runner (`node:test` + `node:assert`, **no npm dependencies**). With a seeded PRNG the assertions are deterministic:

- **Gridworld dynamics** — clear moves pay the step reward and land on the intended cell; walls and grid edges block movement; the goal is an absorbing terminal that pays the goal reward; the cliff applies a large penalty and teleports to start without terminating; slip spreads probability over intended + perpendicular moves that sum to 1; sampling matches the distribution (~80% intended at slip 0.2 over 40k draws).
- **Dynamic programming** — value iteration converges and the result satisfies the Bellman optimality equation (residual `< 1e-6`); on a deterministic corridor it matches the closed-form optimal value and policy; policy iteration produces the same V and π in no more outer iterations.
- **TD control** — Q-learning and SARSA both beat the uniform-random policy and the learning curve trends upward; Q-learning's greedy action at the start state matches the DP-optimal action.
- **Bandits** — ε-greedy and UCB pull the best arm a majority of the time after enough rounds; average regret shrinks over time (sublinear cumulative regret); averaged over many random testbeds ε-greedy beats pure greedy, and UCB has lower average regret than greedy.
- **Function approximation** — RBF and Fourier linear regression drive grid MSE below 0.05; one tile is active per tiling; the semi-gradient update reduces error toward its target.

### Running the tests

```bash
node --test
```

```
✔ epsilon-greedy pulls the best arm a majority of the time
✔ UCB pulls the best arm a majority of the time
✔ epsilon-greedy beats greedy on average across many testbeds
✔ average regret shrinks over time (sublinear cumulative regret)
✔ UCB achieves lower average regret than greedy
✔ value iteration converges and satisfies the Bellman optimality equation
✔ value iteration recovers the analytic optimum on a deterministic corridor
✔ policy iteration agrees with value iteration
✔ feature bases return the expected dimensions
✔ a single tile is active per tiling
✔ linear update reduces error toward a constant target
✔ RBF regression fits the target function well
✔ fourier regression also drives MSE down
✔ an untrained model has higher error than a trained one
✔ a clear move yields the step reward and the intended neighbour
✔ moving off the grid keeps the agent in place
✔ walls block movement
✔ the goal is terminal and pays the goal reward
✔ slip spreads probability over intended + perpendicular moves and sums to 1
✔ cliff: large penalty, teleport to start, not terminal
✔ sampling respects the transition distribution
✔ mulberry32 is deterministic for a fixed seed
✔ different seeds produce different streams
✔ RNG.random stays in [0,1) and RNG.int in [0,n)
✔ RNG.normal has roughly zero mean and unit variance
✔ argmax and softmax behave
✔ Q-learning beats the random policy on the tiny gridworld
✔ Q-learning approaches the DP-optimal policy at the start state
✔ SARSA also improves over random
✔ learning curve trends upward (later returns beat early returns)
ℹ tests 30
ℹ pass 30
ℹ fail 0
```

## Running the site

No build step. Any static file server works (ES-module imports need to be served over HTTP, not `file://`):

```bash
python -m http.server 8000
# open http://localhost:8000
```

## How it's built

100% vanilla JavaScript, no frameworks, no bundler. Canvas 2D for rendering.

- `src/` — framework-free, unit-tested algorithm cores (above)
- `js/common.js` — shared browser utilities (drawing helpers, animation loop)
- `js/gridworld.js` — re-exports the tested `Gridworld` model and adds the canvas drawing + click-to-edit helpers
- `js/ui.js` — the `Loop` animation helper for the module pages
- `js/<page>.js` — one rendering/controls file per demo

## License

MIT.
