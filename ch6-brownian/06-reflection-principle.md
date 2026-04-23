# 06. 반사원리(Reflection Principle)와 최대값

## 🎯 핵심 질문

- **반사원리** $\mathbb{P}(M_t \geq a) = 2\mathbb{P}(B_t \geq a)$ ($M_t = \max_{s \leq t} B_s$) — 왜 팩터 2인가?
- 첫 도달시각 $\tau_a := \inf\{t : B_t = a\}$의 분포는 무엇이고 역가우시안(inverse Gaussian)인 이유?
- Joint 분포 $(M_t, B_t)$는 어떻게 되는가?
- 장벽 옵션(barrier option) 가격에서 이 원리의 직접 응용?

---

## 🔍 왜 이 결과가 AI에서 중요한가

**Barrier Options Pricing**: "First hitting" 사건이 핵심. ML-based option pricing에서 반사원리 기반 closed-form과 NN training data.

**Optimal Stopping in RL**: Termination time이 first hitting — reward / policy 결정의 이론적 기반.

**Extreme Value Theory**: Maximum 분포가 sequence model (RNN, Transformer)의 extreme event prediction.

**Diffusion Model의 generation variance**: Reverse process의 경로 범함수 (sup, integral) 분석.

---

## 📐 수학적 선행 조건

- [Ch6-01 ~ Ch6-05](./01-axiomatic-definition.md): BM 공리, 이차변분
- [Ch1-04](../ch1-foundations/04-filtration.md): Stopping time, 강한 Markov property
- [Ch5-03](../ch5-martingale/03-optional-stopping.md): Optional stopping

---

## 📖 직관적 이해

### 반사원리의 아이디어

$M_t := \max_{s \leq t} B_s \geq a$라는 것은 "BM이 시각 $t$ 전에 $a$를 적어도 한 번 건드림". 첫 건드림 시각 $\tau_a$에서:
- BM이 $\tau_a$까지 $a$ 도달.
- $\tau_a$ 이후 두 가지 대칭 경로 가능:
  - 경로 A: 원래대로 계속 진행 → $B_t$가 어디든지 가능
  - 경로 B: $\tau_a$ 시점에서 **$a$에 대해 반사** → $2a - B_t$ 위치.

**대칭성** (강한 Markov property + BM의 역대칭): 두 경로 확률 같음.

**결과**:
$\mathbb{P}(M_t \geq a) = \mathbb{P}(M_t \geq a, B_t \geq a) + \mathbb{P}(M_t \geq a, B_t < a) = 2 \mathbb{P}(B_t \geq a)$ (첫 항은 자명, 두번째는 반사 대칭).

> **비유**: 집에서 친구 집까지 랜덤 워크로 걸어가는데, 도중에 카페에 들렀는지 모르겠음. 카페 들른 모든 경로 = (카페 들른 뒤 친구 집 도착) ∪ (카페 들른 뒤 친구 집 반대 방향 반사 도착). 두 경우 확률 같음.

### 첫 도달시각 $\tau_a$

$\tau_a := \inf\{t \geq 0 : B_t = a\}$.

**분포**:
$\mathbb{P}(\tau_a \leq t) = \mathbb{P}(M_t \geq a) = 2\mathbb{P}(B_t \geq a) = 2(1 - \Phi(a/\sqrt t))$.

**밀도**: 
$f_{\tau_a}(t) = \frac{a}{\sqrt{2\pi t^3}} e^{-a^2/(2t)}$, $t > 0$.

이 분포가 **Lévy 분포** (또는 inverse Gaussian의 특수 경우).

**특성**: $\mathbb{E}[\tau_a] = \infty$ (heavy tail) — "언젠가 $a$에 도달하지만 평균 시간 무한" (Ch5-03 문제 2의 continuous 버전).

---

## ✏️ 엄밀한 정의

### 정의 6.1 — Maximum Process

$M_t := \max_{s \in [0, t]} B_s, \quad M_0 = 0$.

$M_t \geq B_t$ always. $M_t$ non-decreasing, continuous.

### 정의 6.2 — First Hitting Time

$\tau_a := \inf\{t \geq 0 : B_t = a\}$ for $a \neq 0$.

$\tau_a$는 stopping time (continuous BM + closed set). A.s. finite (Ch6-03 Donsker + recurrence of 1D BM on bounded interval).

### 정의 6.3 — Reflected Process

$\tilde B_s := \begin{cases} B_s & s \leq \tau_a \\ 2a - B_s & s > \tau_a \end{cases}$

강한 Markov + BM time-reversibility → $\tilde B$도 표준 BM.

---

## 🔬 정리와 증명

### 정리 6.1 — 반사원리 (Reflection Principle)

$a > 0$에 대해
$$\mathbb{P}(M_t \geq a) = 2 \mathbb{P}(B_t \geq a) = 2(1 - \Phi(a/\sqrt t)).$$

### 증명

$\{M_t \geq a\} = \{\tau_a \leq t\}$ (M이 $a$ 건드림 ⇔ first hitting $\leq t$).

**Decomposition**:
$\mathbb{P}(M_t \geq a) = \mathbb{P}(\tau_a \leq t, B_t \geq a) + \mathbb{P}(\tau_a \leq t, B_t < a)$.

**첫 항**: $B_t \geq a$이면 자동으로 $M_t \geq a$. $\mathbb{P}(B_t \geq a)$.

**두 번째 항 (반사 trick)**:
Consider reflected BM $\tilde B_s$. $\tilde B$도 표준 BM (독립적). $\{\tau_a \leq t, B_t < a\}$를 $\tilde B$ 관점에서:

$\tilde B_t = 2a - B_t$ on $\{\tau_a \leq t\}$. $B_t < a \iff \tilde B_t > a$.

$\tilde B$가 표준 BM이므로 (강한 Markov property로 정당화):
$\mathbb{P}(\tau_a \leq t, B_t < a) = \mathbb{P}(\tau_a \leq t, \tilde B_t > a) = \mathbb{P}(\tau_a^{\tilde B} \leq t, \tilde B_t > a) = \mathbb{P}(\tilde B_t > a) = \mathbb{P}(B_t > a)$.

(여기서 $\tilde B_t > a$이면 $\tilde B$가 $a$ 건드림 — $\tau_a^{\tilde B} \leq t$.)

합:
$\mathbb{P}(M_t \geq a) = \mathbb{P}(B_t \geq a) + \mathbb{P}(B_t > a) = 2 \mathbb{P}(B_t \geq a)$.

($\mathbb{P}(B_t = a) = 0$). $\square$

### 정리 6.2 — 첫 도달시각의 분포

$\tau_a$ ($a > 0$)의 CDF:
$\mathbb{P}(\tau_a \leq t) = 2(1 - \Phi(a/\sqrt t))$.

밀도 (Lévy 분포):
$f_{\tau_a}(t) = \frac{a}{\sqrt{2\pi}} t^{-3/2} e^{-a^2/(2t)}, \quad t > 0$.

*증명*. CDF는 정리 6.1에서. 밀도는 chain rule:
$\frac{d}{dt}[2(1 - \Phi(a/\sqrt t))] = 2\phi(a/\sqrt t) \cdot \frac{a}{2t^{3/2}} = \frac{a}{\sqrt{2\pi t^3}} e^{-a^2/(2t)}$. $\square$

**특성**:
- Heavy-tailed: $f(t) \sim t^{-3/2}$ as $t \to \infty$ → $\mathbb{E}[\tau_a] = \infty$.
- Unimodal: mode at $t^* = a^2/3$.
- Scaling: $\tau_a \stackrel{d}{=} a^2 \tau_1$.

### 정리 6.3 — Joint Distribution $(M_t, B_t)$

For $a \geq b$ with $a \geq 0$:
$$\mathbb{P}(M_t \geq a, B_t \leq b) = \mathbb{P}(B_t \geq 2a - b) = 1 - \Phi\left(\frac{2a - b}{\sqrt t}\right).$$

*증명*. 반사: $\{M_t \geq a, B_t \leq b\}$ ⇔ $\{$reflected $\tilde B$가 $\tilde B_t \geq 2a - b\}$. 독립 BM 분포: $\mathbb{P}(B_t \geq 2a - b)$. $\square$

**응용**: Barrier option — "현재 가격은 $b$이지만 최대값이 $a$ 이상이었던" 경로의 확률.

### 정리 6.4 — Reflection principle for symmetric Brownian bridge

**Brownian bridge** $B^0_t := B_t - t B_1$ (conditioning on $B_1 = 0$). Reflection으로 $\max_{t \in [0, 1]} B^0_t$ 분포:
$$\mathbb{P}(\max_{t} B^0_t \geq a) = e^{-2a^2}, \quad a > 0.$$

(응용: Kolmogorov-Smirnov test critical values.)

---

## 💻 NumPy 구현 검증

### 실험 1 — Reflection principle 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

rng = np.random.default_rng(0)

# Simulate n_sim BM paths
n_sim = 100000
N = 1000
T = 1.0
dt = T / N

dB = rng.standard_normal((n_sim, N)) * np.sqrt(dt)
B = np.concatenate([np.zeros((n_sim, 1)), np.cumsum(dB, axis=1)], axis=1)
M = np.maximum.accumulate(B, axis=1)   # running maximum

# P(M_T >= a) vs 2 * P(B_T >= a)
a_vals = np.linspace(0.5, 3.0, 20)
empirical = [(M[:, -1] >= a).mean() for a in a_vals]
theoretical = [2 * (1 - norm.cdf(a / np.sqrt(T))) for a in a_vals]

plt.plot(a_vals, empirical, 'o-', label='실측 P(M_T ≥ a)')
plt.plot(a_vals, theoretical, '--', label=r'$2 P(B_T \geq a)$ (이론)')
plt.xlabel('a'); plt.ylabel('probability')
plt.title('Reflection Principle'); plt.legend(); plt.grid(True, alpha=0.3)
plt.show()
```

### 실험 2 — 첫 도달시각 $\tau_a$의 분포

```python
# τ_a = first time B_t = a
a = 1.0
tau_samples = []
for _ in range(10000):
    t_cur = 0.0
    while t_cur < 10.0:   # cap at T = 10
        dB_step = rng.standard_normal() * np.sqrt(dt)
        # Euler approximation
        t_cur += dt
        # Approximate B at this time (reconstruct)
    # 여기서 각 path을 트래킹해야 하지만 벡터화로
    
# 벡터화
N_long = 50000  # long enough
T_long = 50.0
dt_long = T_long / N_long

dB = rng.standard_normal((10000, N_long)) * np.sqrt(dt_long)
B = np.concatenate([np.zeros((10000, 1)), np.cumsum(dB, axis=1)], axis=1)

# First hitting time
hit = (B >= a)
tau_idx = np.argmax(hit, axis=1)   # 0 if never hit
# 첫 hit 없으면 nan
tau = np.where(hit.any(axis=1), tau_idx * dt_long, np.nan)
tau_finite = tau[~np.isnan(tau)]

print(f'Hit rate: {(~np.isnan(tau)).mean():.4f}')
print(f'E[τ_{a}] 실측: {tau_finite.mean():.4f} (이론: ∞ — heavy-tailed)')
print(f'median τ_{a}: {np.median(tau_finite):.4f}')

# Density comparison
from scipy.stats import levy
t_grid = np.linspace(0.01, 10, 200)
levy_density = a / np.sqrt(2*np.pi*t_grid**3) * np.exp(-a**2/(2*t_grid))

plt.hist(tau_finite, bins=100, density=True, alpha=0.5, label='실측')
plt.plot(t_grid, levy_density, 'r-', label='Lévy 분포')
plt.xlim(0, 10); plt.xlabel(r'$\tau_a$'); plt.ylabel('density')
plt.legend(); plt.grid(True, alpha=0.3)
plt.title(rf'$\tau_a$ distribution for $a={a}$')
plt.show()
```

### 실험 3 — Joint distribution

```python
# P(M_T ≥ a, B_T ≤ b) = P(B_T ≥ 2a - b)
a, b = 1.0, 0.5   # 2a - b = 1.5

p_joint_emp = ((M[:, -1] >= a) & (B[:, -1] <= b)).mean()
p_joint_theory = 1 - norm.cdf(2*a - b)

print(f'P(M_T ≥ {a}, B_T ≤ {b}):')
print(f'  실측: {p_joint_emp:.4f}')
print(f'  이론 P(B_T ≥ {2*a - b}): {p_joint_theory:.4f}')
# 일치
```

---

## 🔗 AI/ML 연결

**Barrier Option Pricing**  
"Up-and-out call" option: $K$에 strike, $H$에 barrier. Payoff = $\max(S_T - K, 0) \cdot \mathbf{1}_{\max S_t < H}$. Reflection principle로 closed-form price 유도 (GBM → log-BM 변환). ML-based pricing이 이 formula를 NN으로 근사/확장 (rough models).

**Quickest Change Detection**  
Log-likelihood ratio random walk의 first hitting이 detection rule. ARL (Average Run Length) 계산에 first hitting 분포. ML-based anomaly detection.

**RL Episode Termination**  
Terminal state가 "absorbing" — agent가 도달 시 에피소드 종료. First hitting time 분포가 episode length distribution 결정. Policy gradient variance 분석에 활용.

**Diffusion Model의 path properties**  
Forward/reverse path의 sup, integral 등 functional의 분포. Generation diversity 이해에 반사원리 원리.

**Extreme Event Prediction**  
Time series의 max를 예측하는 NN. 반사원리가 "baseline distribution" 제공 (BM-like assumption 하). Financial VaR, climate extremes.

---

## ⚖️ 가정과 한계

**가정 — Continuous path**  
반사원리는 경로가 연속해야 성립 (jump over barrier가 없음). Jump processes에서는 수정 필요.

**한계 — BM만**  
일반 SDE $dX = b dt + \sigma dB$에서는 반사원리의 직접 일반화 없음. 특정 SDE에서만 analytical first hitting.

**한계 — 다차원**  
높은 차원에서는 BM의 recurrence가 다름 ($d \geq 3$에서 일시). 반사원리의 일반화가 복잡 (Bessel process, Levy area 등).

---

## 📌 핵심 정리

| 결과 | 수식 |
|---|---|
| Reflection | $\mathbb{P}(M_t \geq a) = 2\mathbb{P}(B_t \geq a)$ |
| $\tau_a$ CDF | $\mathbb{P}(\tau_a \leq t) = 2(1 - \Phi(a/\sqrt t))$ |
| $\tau_a$ 밀도 | $\frac{a}{\sqrt{2\pi t^3}} e^{-a^2/(2t)}$ (Lévy) |
| $\mathbb{E}[\tau_a]$ | $\infty$ (heavy tail) |
| Scaling | $\tau_a \stackrel{d}{=} a^2 \tau_1$ |
| Joint | $\mathbb{P}(M_t \geq a, B_t \leq b) = 1 - \Phi((2a-b)/\sqrt t)$ |
| BB sup | $\mathbb{P}(\sup B^0 \geq a) = e^{-2a^2}$ |

**한 줄 요약**: 반사원리는 "BM이 수준 $a$ 건드림" 확률을 "$B_t \geq a$"의 두 배로 계산. 첫 도달시각 $\tau_a$는 Lévy 분포 (heavy-tailed, $\mathbb{E} = \infty$). Barrier options·extreme events·KS test 등 광범위 응용.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. $\mathbb{P}(\max_{t \in [0, 1]} B_t \geq 2)$를 계산하라.

<details>
<summary>해설</summary>

반사원리: $\mathbb{P}(M_1 \geq 2) = 2 \mathbb{P}(B_1 \geq 2) = 2(1 - \Phi(2))$.

$\Phi(2) \approx 0.9772$. 따라서 $2(1 - 0.9772) = 2 \cdot 0.0228 = 0.0456 \approx 4.56\%$.

**비교**: $\mathbb{P}(B_1 \geq 2) \approx 2.28\%$. Max 확률이 정확히 2배.

</details>

**문제 2 (심화)**. Brownian bridge $B^0_t := B_t - t B_1$에서 $\sup_{t \in [0, 1]} B^0_t \geq a$ 확률이 $e^{-2a^2}$임을 보여라 (reflection for BB).

<details>
<summary>해설</summary>

**접근**: BB를 "constrained BM" 관점에서. 또는 standard BM에서 time reversal.

**하지만 더 직접적** — $B^0$의 결합분포 $(B^0_t)_{t \in [0,1]}$은 $(B_t | B_1 = 0)_{t \in [0,1]}$와 같음.

$\mathbb{P}(\sup B^0 \geq a) = \mathbb{P}(\max B_t \geq a | B_1 = 0)$.

Reflection principle for pinned BM (Lévy or Feller):
$\mathbb{P}(M_1 \geq a, B_1 \in dy) / \mathbb{P}(B_1 \in dy)$. 

LHS numerator: $\mathbb{P}(M_1 \geq a, B_1 = y)$. Reflection: $\mathbb{P}(B_1 = 2a - y) / \mathbb{P}(B_1 = y)$ if $y < a$ (unreflected).

Density ratio: $\phi(2a - y)/\phi(y) = e^{-(2a-y)^2/2 + y^2/2} = e^{-2a^2 + 2ay}$.

$y = 0$: $e^{-2a^2}$. $\square$

**응용 (Kolmogorov-Smirnov)**:
Empirical CDF $F_n(x)$ vs true $F(x)$. $\sqrt n(F_n - F) \to B^0 \circ F$ (Donsker). KS statistic:
$D_n = \sup |F_n - F|$, $\sqrt n D_n \to \sup |B^0|$.

$\mathbb{P}(\sup |B^0| > a) = 2 \sum (-1)^{k+1} e^{-2k^2 a^2}$ (reflection 반복).

Critical value for KS test: $a$ such that this probability = significance level. Famous table values: $a = 1.36$ for $p = 0.05$ (approximately).

**실전**: Scikit-learn의 KS test, Anderson-Darling 등 모두 이 이론에 기반.

</details>

**문제 3 (AI 연결)**. Reinforcement learning에서 "episode termination" (absorbing state) time distribution을 모델링할 때, BM first hitting과의 유사성과 차이점을 논하라.

<details>
<summary>해설</summary>

**RL Episode Termination**:
$\tau = \inf\{t : S_t \in \mathcal{T}\}$, $\mathcal{T}$ = terminal states set.

**BM First Hitting과 유사**:
- 둘 다 random stopping time
- Distribution이 heavy-tailed일 수 있음
- $\mathbb{E}[\tau] = \infty$ 가능 (경로가 무한 fluctuation)

**차이**:
1. **State space 이산 vs 연속**:
   - RL: 주로 이산 (finite MDP) 또는 낮은 차원
   - BM: $\mathbb{R}$ (continuous state)

2. **Transition 구조**:
   - RL: Action-dependent transition ($P(s' | s, a)$)
   - BM: No action, deterministic drift + noise

3. **Termination 정의**:
   - RL: Explicit set $\mathcal{T}$ (goal states, failure states)
   - BM: "First crossing of level $a$"

4. **Variance source**:
   - RL: Policy randomness + environment stochasticity
   - BM: Pure Gaussian noise

**AI 응용**:

**(1) Value function 계산**:
$V^\pi(s) = \mathbb{E}[\sum_{t < \tau} \gamma^t r_t | S_0 = s]$. $\tau$ random — Wald's identity (Ch5-03)의 RL 버전.

**(2) Episode length prediction**:
LSTM/Transformer로 $\tau | s_0, a_0, \ldots$ 예측. Survival analysis + deep learning (DeepHit).

**(3) Hindsight Experience Replay**:
Failure episodes에서 "어떤 state가 도달했나"를 relabel. First hitting time 분석이 data efficiency 개선.

**(4) Adaptive episode length**:
Early termination for "unpromising" trajectories. BM-style "passage time" analytics — policy가 reward threshold를 빨리 넘는 trajectory prioritize.

**이론적 확장**:
- **Absorbing MDP + discount**: 반사원리 없이도 fixed-point theory (contraction mapping) 적용.
- **Monte Carlo tree search termination**: PUCT bound의 first hitting time interpretation.
- **Curriculum learning**: Easy → hard task progression을 first hitting 분포로 모델링.

**결론**: BM first hitting이 "기본 시간 모델"이지만 RL의 복잡성 (action, policy, discrete structure) 때문에 직접 대응 없음. 그러나 heavy-tail, expected time finite/infinite, scaling laws 등 **정성적 교훈은 공유**.

</details>

---

<div align="center">

◀ [05. 이차변분 $\langle B\rangle_t = t$](./05-quadratic-variation.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [Ch7-01. MCMC의 아이디어 — 정상분포 설계](../ch7-mcmc/01-mcmc-idea.md)

</div>
