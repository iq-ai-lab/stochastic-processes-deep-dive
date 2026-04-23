# 03. Random Walk Scaling Limit — Donsker 정리

## 🎯 핵심 질문

- **Donsker의 불변원리**: $\frac{1}{\sqrt n} S_{\lfloor nt\rfloor} \xrightarrow{d} B_t$ in $C[0, 1]$가 왜 성립하는가?
- 이것이 왜 "CLT의 **과정(process) 버전**"이고, 개별 시점 CLT를 어떻게 일반화하는가?
- 증명의 두 축 — **tightness** (컴팩트성)와 **유한차원 분포 수렴**이 어떻게 결합되는가?
- **이산 → 연속 이행**이 왜 DDPM → Score-SDE, ResNet → Neural ODE의 수학적 기반인가?

---

## 🔍 왜 이 정리가 AI에서 중요한가

**DDPM → Score-SDE 이산-연속 연결**: DDPM의 이산 시간 Markov chain이 step 수 $T \to \infty$에서 **VP-SDE로 수렴** — Donsker 원리의 직접 적용. Song et al. 2021의 프레임워크 전체가 이 수렴에 기반.

**ResNet → Neural ODE**: $x_{l+1} = x_l + f(x_l)/L$ ($L$ layers) → $\dot x = f(x)$ as $L \to \infty$. Donsker-like 결과가 이 극한을 정당화.

**SGD의 continuous-time limit**: Stochastic gradient flow $d\theta = -\nabla L dt + \sqrt{\eta} dW_t$가 SGD의 극한. Implicit regularization 이론의 토대.

**Gradient flow 분석**: NTK 이론에서 infinitely-wide NN이 "Gaussian process" 극한으로 수렴. Donsker 원리의 일반화.

---

## 📐 수학적 선행 조건

- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): CLT, 수렴 이론 (weak, in distribution)
- [Ch6-01, Ch6-02](./01-axiomatic-definition.md): BM 공리, 존재성
- 함수해석 기초: $C[0, 1]$, metric space

---

## 📖 직관적 이해

### "CLT의 path 버전"

**일반 CLT**: $\frac{1}{\sqrt n} \sum \xi_k \xrightarrow{d} \mathcal{N}(0, 1)$ — **단일 시점**에서의 분포 수렴.

**Donsker**: 전체 **random walk 경로**를 rescale하면 BM 경로로 수렴:
$$\left\{\frac{1}{\sqrt n} S_{\lfloor nt\rfloor}\right\}_{t \in [0, 1]} \xrightarrow{d} \{B_t\}_{t \in [0, 1]} \quad \text{in } C[0, 1].$$

$C[0, 1]$에서의 **weak convergence** — 함수공간의 분포 수렴.

### "Invariance Principle"

중요한 insight: **increment $\xi_k$의 구체적 분포는 상관없음** (평균 0, 분산 1만 필요). 어느 분포로 출발해도 rescale 극한은 **같은 BM**. 이 "invariance"에서 이름.

> **비유**: 여러 개의 다른 언어(random walk의 various distributions)가 같은 번역(BM)으로 수렴. BM이 "universal" continuous random process.

### 증명의 두 축

**축 1 (fdd 수렴)**: 각 유한 시점 $(t_1, \ldots, t_m)$에서 finite-dimensional 분포 수렴. CLT의 다변량 버전.

**축 2 (tightness)**: 경로의 분포가 $C[0, 1]$에서 **tight** — "unlikely to escape to infinity in path space". Continuity modulus의 통제.

두 축 합 → path 분포 수렴.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Skorokhod Embedding

iid $\xi_k$ with $\mathbb{E}\xi = 0, \text{Var}\xi = 1$. Random walk $S_n = \sum_{k=1}^n \xi_k$. **Linear interpolation**:
$$X^{(n)}(t) := \frac{1}{\sqrt n}\left(S_{\lfloor nt\rfloor} + (nt - \lfloor nt\rfloor)(S_{\lfloor nt\rfloor + 1} - S_{\lfloor nt\rfloor})\right).$$

$X^{(n)} \in C[0, 1]$ (piecewise linear, 연속).

### 정의 3.2 — 수렴 in $C[0, 1]$

$X^{(n)} \xrightarrow{d} X$ in $C[0, 1]$ means: for every continuous bounded functional $F : C[0, 1] \to \mathbb{R}$, $\mathbb{E}[F(X^{(n)})] \to \mathbb{E}[F(X)]$.

Equivalent: Distributions $\mu_n$ of $X^{(n)}$ converge weakly to $\mu$ (distribution of $X$).

### 정의 3.3 — Tight family in $C[0, 1]$

$\{\mu_n\}$가 **tight**: $\forall \epsilon > 0$, $\exists$ compact $K \subset C[0, 1]$ with $\mu_n(K) \geq 1 - \epsilon$ for all $n$.

**Arzelà-Ascoli 특성화**: $C[0, 1]$의 compact set = "uniformly bounded + equicontinuous".

따라서 tightness = "modulus of continuity가 uniformly controlled".

---

## 🔬 정리와 증명

### 정리 3.1 — Donsker의 불변원리

$\xi_1, \xi_2, \ldots$ iid with $\mathbb{E}\xi = 0, \text{Var}\xi = 1$. $X^{(n)}$을 정의 3.1처럼 정의. 그러면
$$X^{(n)} \xrightarrow{d} B \quad \text{in } C[0, 1],$$
$B$는 표준 BM.

### 증명 스케치 (두 축)

**Step 1 — Finite-dimensional 분포 수렴**:

임의 $0 \leq t_1 < \cdots < t_m \leq 1$에 대해 $(X^{(n)}(t_1), \ldots, X^{(n)}(t_m))$이 $(B_{t_1}, \ldots, B_{t_m})$의 분포로 수렴해야.

각 증분 $X^{(n)}(t_{j+1}) - X^{(n)}(t_j) = \frac{1}{\sqrt n} \sum_{k = \lfloor n t_j\rfloor + 1}^{\lfloor n t_{j+1}\rfloor} \xi_k$.

CLT (multivariate): 서로 disjoint index range에서의 합이 **독립** (increments iid), 각각 $\xrightarrow{d} \mathcal{N}(0, t_{j+1} - t_j)$.

따라서 joint fdd convergence to BM fdd.

**Step 2 — Tightness**:

$C[0, 1]$에서 tightness의 **Arzelà-Ascoli + Kolmogorov** 충분조건:
$$\lim_{\delta \to 0} \limsup_n \mathbb{P}\left(\sup_{|s - t| \leq \delta} |X^{(n)}(s) - X^{(n)}(t)| > \epsilon\right) = 0.$$

증명: Maximal inequality (Doob) 또는 직접 moment bound.
- $\mathbb{E}[|X^{(n)}(t) - X^{(n)}(s)|^4] \leq C|t - s|^2$ (4차 moment).
- Kolmogorov continuity theorem 유사 arg로 Hölder 연속성 → compact set 통제.

**Step 3 — fdd + tightness ⇒ weak convergence**:

Prokhorov theorem: fdd 수렴 + tight → weak convergence in $C[0, 1]$. $\square$

### 정리 3.2 — Continuous Mapping Theorem (CMT)

$X^{(n)} \xrightarrow{d} X$ in $C[0, 1]$, $F : C[0, 1] \to \mathbb{R}$ a.s. 연속 (at $X$'s support). 그러면 $F(X^{(n)}) \xrightarrow{d} F(X)$.

**응용**: 
- $\max_t X^{(n)}(t) \xrightarrow{d} \max_t B_t$ — reflection principle (Ch6-06)
- $\int_0^1 X^{(n)}(t) dt \xrightarrow{d} \int_0^1 B_t dt \sim \mathcal{N}(0, 1/3)$
- Hitting time $\tau_a^{(n)} \xrightarrow{d} \tau_a$ (BM hitting time)

### 정리 3.3 — Donsker 응용: Kolmogorov-Smirnov 통계

Empirical CDF $F_n(x) = \frac{1}{n}\#\{X_i \leq x\}$ vs true $F$. Donsker-like 결과:
$$\sqrt n (F_n - F) \xrightarrow{d} B^0 \circ F,$$
$B^0$ = Brownian bridge ($B^0_0 = B^0_1 = 0$).

이로부터 Kolmogorov-Smirnov test 통계의 분포 유도.

---

## 💻 NumPy 구현 검증

### 실험 1 — Random walk scaling → BM

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# 여러 n에 대해 X^(n) 시각화
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, n in zip(axes.flat, [10, 50, 200, 1000, 5000, 50000]):
    t_grid = np.linspace(0, 1, n + 1)
    # iid Rademacher
    xi = rng.choice([-1, 1], size=n)
    S = np.concatenate([[0], np.cumsum(xi)])
    X_n = S / np.sqrt(n)
    ax.plot(t_grid, X_n)
    ax.set_title(f'n = {n}')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-3, 3])
plt.suptitle("Rescaled random walk → BM (Donsker)")
plt.tight_layout(); plt.show()
```

### 실험 2 — fdd 수렴 검증

```python
# 각 t에서의 marginal distribution이 N(0, t)로 수렴
n_sim = 10000
t_vals = [0.3, 0.5, 0.8]

for n in [10, 100, 10000]:
    X_final = []
    for _ in range(n_sim):
        xi = rng.choice([-1, 1], size=n)
        S = np.cumsum(xi)
        X_final.append([S[int(t * n) - 1] / np.sqrt(n) for t in t_vals])
    X_final = np.array(X_final)
    
    print(f'\nn = {n}:')
    for i, t in enumerate(t_vals):
        print(f'  Var(X_n({t})) = {X_final[:, i].var():.4f}, 이론 t = {t}')
    
    # Joint covariance
    cov = np.cov(X_final, rowvar=False)
    print(f'  Cov matrix: \n{cov}')
    print(f'  이론 min: \n{np.minimum.outer(t_vals, t_vals)}')
```

### 실험 3 — Max functional

```python
# CMT: max_t X^(n)_t → max_t B_t
n_sim = 10000
n = 1000

M_samples = []
for _ in range(n_sim):
    xi = rng.choice([-1, 1], size=n)
    S = np.cumsum(xi) / np.sqrt(n)
    M_samples.append(max(0, S.max()))

M_samples = np.array(M_samples)

# 이론: P(max B_t ≥ a) = 2 P(B_1 ≥ a) = 2(1 - Φ(a)) (reflection principle)
from scipy.stats import norm
a_vals = np.linspace(0, 3, 30)
empirical_P = np.array([(M_samples >= a).mean() for a in a_vals])
theoretical_P = 2 * (1 - norm.cdf(a_vals))

plt.plot(a_vals, empirical_P, 'o-', label='실측')
plt.plot(a_vals, theoretical_P, '--', label='이론 (reflection)')
plt.xlabel('a'); plt.ylabel(r'$P(M \geq a)$')
plt.legend(); plt.grid(True, alpha=0.3); plt.title('Donsker + reflection principle')
plt.show()
# 일치 → CMT 확인
```

---

## 🔗 AI/ML 연결

**DDPM → Score-SDE 연결**  
Discrete DDPM: $x_t = \sqrt{1 - \beta_t} x_{t-1} + \sqrt{\beta_t} z_t$. Continuous-time limit (Song 2021):
$$\frac{1}{\sqrt T} (x_{\lfloor Tt\rfloor} - x_0) \to \text{(VP-SDE path)}$$
as $T \to \infty$ with appropriate $\beta(t)$ rescale. Donsker-like argument.

**ResNet → Neural ODE**  
$x_{l+1} = x_l + f(x_l)/L$ with $L$ layers. Rescaled path $x(t) = x_{\lfloor Lt\rfloor}$가 ODE $\dot x = f(x)$ 해로 수렴 (Chen 2018). Donsker의 deterministic version (drift만).

**SGD의 SDE Limit**  
$\theta_{n+1} = \theta_n - \eta \nabla L(\theta_n, \xi_n)$ with small $\eta$. Rescaled $\theta(t) = \theta_{\lfloor t/\eta\rfloor}$가
$$d\theta = -\nabla L dt + \sqrt{\eta \Sigma(\theta)} dW_t$$
(Li, Tai, E 2017). Implicit regularization in SGD dynamics.

**Infinite-width NN as GP**  
Neural Tangent Kernel regime: width $\to \infty$에서 network가 Gaussian process로 수렴 (Jacot 2018). Path-valued 수렴이 "training dynamics가 ODE"임을 정당화.

**Bootstrap in statistics**  
Empirical process theory가 Donsker-type 결과 활용. Permutation tests, bootstrap confidence intervals 이론 기반.

---

## ⚖️ 가정과 한계

**가정 — iid + variance 1**  
정확한 변수 분포 무관하지만 iid 필요. 종속성 있으면 다른 한계 (fractional BM for long-range) or 다른 CLT (martingale CLT).

**가정 — 유한 분산**  
$\mathbb{E}\xi^2 = \infty$ (heavy-tailed, $\alpha$-stable)이면 BM 아닌 **$\alpha$-stable Lévy process**로 수렴. Donsker의 generalization.

**한계 — Rate of convergence**  
정리는 qualitative (convergence in distribution). Rate는 별도 (Edgeworth expansion, Berry-Esseen). AI 응용에서 step 수 결정에 중요.

**한계 — Functional form**  
"Linear interpolation"이 technical artifact. Jump (piecewise constant) 버전에서는 convergence가 $D[0, 1]$ (Skorokhod space)에서.

---

## 📌 핵심 정리

| 결과 | 수식 |
|---|---|
| Donsker | $\frac{1}{\sqrt n} S_{\lfloor nt\rfloor} \xrightarrow{d} B_t$ in $C[0, 1]$ |
| 증명 축 | fdd 수렴 + tightness (Prokhorov) |
| Invariance | $\xi_k$의 specific distribution 무관 |
| CMT | $X^{(n)} \xrightarrow{d} X$, $F$ 연속 → $F(X^{(n)}) \xrightarrow{d} F(X)$ |
| KS statistic | $\sqrt n (F_n - F) \to B^0 \circ F$ |

**한 줄 요약**: Donsker의 불변원리는 "CLT의 과정 버전" — iid random walk의 rescaling이 BM 경로로 수렴. "Invariance" 성질 (분포 무관)이 BM의 universality 확립. 이산→연속 이행(DDPM→Score-SDE, ResNet→Neural ODE)의 수학적 근거.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. iid Bernoulli($p$) random walk를 standardize (mean, variance 조정)하여 BM에 수렴시키는 scaling을 써라.

<details>
<summary>해설</summary>

$\xi_k \sim \text{Bernoulli}(p)$: $\mathbb{E}\xi = p$, $\text{Var}\xi = p(1-p)$.

Centered and normalized: $\tilde \xi_k = (\xi_k - p) / \sqrt{p(1-p)}$. Now $\mathbb{E}\tilde\xi = 0, \text{Var}\tilde\xi = 1$.

Donsker:
$$\frac{1}{\sqrt n} \tilde S_{\lfloor nt\rfloor} = \frac{1}{\sqrt{np(1-p)}}(S_{\lfloor nt\rfloor} - pnt) \xrightarrow{d} B_t.$$

**원 scale로**:
$$S_{\lfloor nt\rfloor} \approx pnt + \sqrt{np(1-p)} B_t.$$

Drift + volatility — **generalized BM** $X_t = \mu t + \sigma B_t$ with $\mu = pn, \sigma = \sqrt{np(1-p)}$.

**응용**: Stock price (Bernoulli up/down) → Brownian motion with drift. Merton / Black-Scholes의 기초 (SDE Ch2-05).

</details>

**문제 2 (심화)**. Donsker 정리의 tightness 조건을 **Kolmogorov continuity theorem**을 이용해 직접 증명하라.

<details>
<summary>해설</summary>

**목표**: $X^{(n)}$의 분포족이 $C[0, 1]$에서 tight.

**Kolmogorov continuity condition 유사**: $\mathbb{E}|X^{(n)}(t) - X^{(n)}(s)|^{2k} \leq C |t - s|^k$ for some $k$.

**Calculation**:
$X^{(n)}(t) - X^{(n)}(s) \approx \frac{1}{\sqrt n}(S_{\lfloor nt\rfloor} - S_{\lfloor ns\rfloor}) = \frac{1}{\sqrt n} \sum_{j=\lfloor ns\rfloor + 1}^{\lfloor nt\rfloor} \xi_j$.

IID sum, 4차 moment:
$$\mathbb{E}\left|\frac{1}{\sqrt n}\sum_{j=1}^m \xi_j\right|^4 = \frac{1}{n^2} \mathbb{E}\left|\sum \xi_j\right|^4.$$

For iid mean-0 variance-1 with $\mathbb{E}\xi^4 = \kappa < \infty$:
$\mathbb{E}|\sum_{j=1}^m \xi_j|^4 = 3m^2 + m(\kappa - 3)$ (expansion).

$= O(m^2)$.

따라서 $\mathbb{E}|X^{(n)}(t) - X^{(n)}(s)|^4 \leq \frac{Cm^2}{n^2} = C\left(\frac{\lfloor nt\rfloor - \lfloor ns\rfloor}{n}\right)^2 \leq C |t - s|^2$.

**Kolmogorov**: $\alpha = 4, \beta = 1$ → Hölder continuous with exponent $< 1/4$ a.s. (Hölder continuity 제공). 구체적으로 $X^{(n)}$의 경로가 $\gamma < 1/4$에 대해 uniform Hölder bound.

**Tightness**: Hölder ball $\{f : \|f\|_\gamma \leq R\}$은 $C[0,1]$에서 compact (Arzelà-Ascoli). $\mathbb{P}(\|X^{(n)}\|_\gamma \leq R) \geq 1 - \epsilon$ for appropriate $R$ (Markov inequality on $\|X\|_\gamma$). Tightness 확인. $\square$

**주의**: $\mathbb{E}\xi^4 < \infty$ 가정 필요 — Donsker의 original statement는 $\mathbb{E}\xi^2 < \infty$만 필요하므로 다른 tightness 기법 필요 (maximal inequality, stopping argument).

</details>

**문제 3 (AI 연결)**. Score-SDE의 VP-SDE가 DDPM의 이산 Markov chain의 Donsker-type 수렴 한계임을 formalize하고, noise schedule $\beta_t$가 어떻게 되어야 하는지 논하라.

<details>
<summary>해설</summary>

**Discrete DDPM** (forward process):
$x_{t+1} = \sqrt{1 - \beta_t} x_t + \sqrt{\beta_t} z_{t+1}$, $z_{t+1} \sim \mathcal{N}(0, I)$, $t = 0, \ldots, T - 1$.

**Continuous limit target** (VP-SDE):
$dX_\tau = -\frac{1}{2} \beta(\tau) X_\tau d\tau + \sqrt{\beta(\tau)} dB_\tau$.

**Rescaling**: discrete step $t$를 continuous $\tau = t/T \in [0, 1]$로 map. $\beta$ scaling: $\beta_t = \beta(\tau) \cdot \Delta\tau = \beta(t/T)/T$.

**Discrete step to continuous**:
$\Delta X_t = x_{t+1} - x_t = (\sqrt{1 - \beta_t} - 1) x_t + \sqrt{\beta_t} z_{t+1}$.

$\sqrt{1 - \beta_t} - 1 \approx -\beta_t/2 = -\beta(\tau) / (2T)$.

$\Delta X \approx -\frac{\beta(\tau)}{2T} x + \sqrt{\beta(\tau)/T} \cdot z$.

**Compare to Euler-Maruyama** of VP-SDE: $\Delta X = -\frac{1}{2}\beta(\tau) x \Delta\tau + \sqrt{\beta(\tau) \Delta\tau} z$, $\Delta\tau = 1/T$.

**정확히 매칭** (up to $O(\beta_t^2)$ term). Donsker-like argument + Euler-Maruyama convergence:

$T \to \infty$, $\beta_t = \beta(\tau)/T$로 scale → discrete forward → continuous VP-SDE 수렴 (in $C[0, 1]$ or stronger).

**Noise schedule 요구**:
1. **$\beta(\tau) \geq 0$**: valid noise
2. **$\beta(\tau)$ continuous (또는 bounded)**: regularity for SDE
3. **Decreasing $\bar\alpha_\tau = e^{-\int_0^\tau \beta(u) du}$**: forward process가 "mixing"되도록

**Choice examples**:
- **Linear**: $\beta(\tau) = \beta_{\min} + (\beta_{\max} - \beta_{\min}) \tau$ (DDPM default)
- **Cosine**: $\bar\alpha_\tau = \cos^2((\tau + s)/(1 + s) \cdot \pi/2)$ (iDDPM, Nichol & Dhariwal)
- **EDM** (Karras 2022): variance-preserving with sophisticated schedule

**이론적 consequences**:
- DDPM의 epsilon-matching loss가 VP-SDE의 score matching과 **동치** (up to scaling)
- DDIM (deterministic sampling) = PF-ODE ($\sigma = 0$ version of reverse SDE)
- 모든 discrete → continuous 결과가 Donsker-like 수렴의 instance

**실전 함의**:
- Large $T$ (1000 step)가 continuous approximation 정확도 high
- Small $T$ (50 step)는 discretization error — fast samplers (DPM-Solver, etc.)가 이를 보정
- Continuous SDE 이론이 discrete DDPM의 **asymptotic behavior**를 이해하는 key

**연결**: SDE Deep Dive Ch5 (EM, strong convergence), Ch6 (Reverse SDE), Ch6-06 (DDPM as VP-SDE EM discretization)에서 완전 유도.

</details>

---

<div align="center">

◀ [02. 존재성 — Lévy의 Haar 기저 구성](./02-levy-construction.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [04. 경로 성질 — 비미분 가능성](./04-non-differentiability.md)

</div>
