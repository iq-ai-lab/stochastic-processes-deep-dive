# 01. MCMC의 아이디어 — 정상분포 설계

## 🎯 핵심 질문

- 복잡한 분포 $\pi$에서 **왜 직접 샘플할 수 없는가**, 그리고 MCMC가 어떻게 해결하는가?
- "**$\pi$를 정상분포로 갖는 Markov chain 설계 → 시뮬레이션 → 샘플**"의 이론적 근거는?
- **에르고딕 정리**(Ch2-06)가 왜 MCMC 기댓값 계산의 근간인가?
- 베이지안 추론·물리 시뮬레이션·ML inference에서 MCMC의 표준 역할?

---

## 🔍 왜 MCMC가 AI에서 중요한가

**Bayesian Deep Learning**: 가중치의 사후분포 $p(\theta | D)$ 샘플링 — posterior expectation, uncertainty quantification. BNN, SG-MCMC (Welling-Teh 2011).

**Probabilistic Programming (Stan, PyMC, NumPyro)**: HMC·NUTS 기반 MCMC가 표준. 복잡한 hierarchical Bayesian models의 posterior 샘플.

**RL의 Thompson Sampling**: Posterior from Bayesian bandit → arm selection. MCMC로 posterior 근사.

**Energy-Based Models (EBM)**: $p(x) \propto e^{-E(x)}$. $Z$ 계산 불가 → MCMC로 샘플 / gradient estimation.

**Generative Modeling 이전 MCMC**: Score matching, Langevin dynamics가 MCMC의 연속시간 극한 — Diffusion model의 수학적 조상.

---

## 📐 수학적 선행 조건

- [Ch2 전체](../ch2-discrete-markov/01-markov-property.md): Markov chain, 정상분포, 수렴, 에르고딕 정리
- [Ch2-05](../ch2-discrete-markov/05-detailed-balance.md): Detailed balance

---

## 📖 직관적 이해

### 근본 문제

"임의의" 분포 $\pi(x) \propto \tilde\pi(x)$에서 샘플 뽑기. **문제**:
1. **정규화 상수** $Z = \int \tilde\pi(x) dx$ 계산 불가 (고차원).
2. **고차원 rejection sampling** 효율 매우 낮음.
3. **CDF inversion** 불가 (multivariate).

**예**:
- Bayesian posterior $p(\theta | D) = p(D | \theta) p(\theta) / p(D)$. Evidence $p(D)$ 계산 어려움.
- Ising model $p(\sigma) = \frac{1}{Z} e^{-\beta H(\sigma)}$. $Z = \sum_\sigma e^{-\beta H}$ 기하급수적 많은 항.

### MCMC의 핵심 아이디어

"$\pi$에서 직접 샘플 못함 → **$\pi$를 정상분포로 갖는 Markov chain 구성** → 오래 시뮬레이션 → 샘플이 $\pi$에서 나옴".

**수학적 안전망**: Ch2-04 (수렴 정리) — 기약·비주기 체인은 정상분포로 수렴. Ch2-06 (에르고딕) — 시간평균 = 공간평균.

### 체인 설계 원칙

1. **Detailed balance** $\pi(x) P(x, y) = \pi(y) P(y, x)$ 설계 → 정상분포 $\pi$ 자동 (Ch2-05).
2. **기약성** 보장 — 모든 state 도달 가능.
3. **비주기성** — 수렴 속도.

Metropolis-Hastings (Ch7-02)가 이 원칙을 체계적으로 적용하는 알고리즘.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — MCMC Problem

Target distribution $\pi$ with density $\tilde\pi / Z$ known up to constant. Goal: generate $X_1, X_2, \ldots$ approximately $\sim \pi$ (in some sense).

### 정의 1.2 — MCMC Estimator

"$\mathbb{E}_\pi[f]$를 추정":
$$\hat f_n = \frac{1}{n} \sum_{k=1}^n f(X_k).$$

**Ergodic theorem** (Ch2-06): 기약·양재귀 체인에서 $\hat f_n \to \mathbb{E}_\pi[f]$ a.s.

### 정의 1.3 — Burn-in

초기 몇 step의 $X_1, \ldots, X_{n_{\text{burn}}}$은 정상분포에서 멀음 → 제외:
$$\hat f_n^{\text{burn}} = \frac{1}{n - n_{\text{burn}}} \sum_{k = n_{\text{burn}}+1}^n f(X_k).$$

---

## 🔬 정리와 증명

### 정리 1.1 — MCMC 정당성

기약·비주기·양재귀 Markov chain with 정상분포 $\pi$. 임의 초기 분포 $\mu_0$에서 시작한 체인 $\{X_n\}$에 대해 $\mathbb{E}_\pi|f| < \infty$이면:
$$\hat f_n \to \mathbb{E}_\pi[f] \quad \text{a.s. and in } L^1.$$

*증명*. Ch2-06 ergodic theorem의 직접 적용. $\square$

### 정리 1.2 — 수렴률

Reversible 체인의 스펙트럴 gap $\gamma = 1 - |\lambda_2|$, mixing time $t_{\text{mix}}(\epsilon) = O(\gamma^{-1} \log(1/\epsilon))$. Asymptotic variance:
$$\sqrt n (\hat f_n - \mathbb{E}_\pi f) \xrightarrow{d} \mathcal{N}(0, \sigma_f^2),$$
$\sigma_f^2 = \text{Var}_\pi(f)(1 + 2\sum \rho_k)$ (autocorrelation 합). 좋은 체인 = 작은 autocorrelation.

### 정리 1.3 — MCMC 설계 원칙 (Framework)

Target $\pi$에 대한 transition kernel $P$가 MCMC에 유효 ⇔:
1. $\pi P = \pi$ (정상분포 조건)
2. 체인이 기약 (모든 support 도달 가능)
3. 체인이 비주기 (수렴 안정성)

Detailed balance (정리 2-05의 1)은 (1)의 **충분** 조건.

---

## 💻 NumPy 구현 검증

### 실험 1 — 간단한 MCMC: Independence Sampler

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# Target: mixture of two Gaussians (bimodal)
def log_pi(x):
    return np.log(0.5 * np.exp(-0.5*(x-2)**2) + 0.5 * np.exp(-0.5*(x+2)**2))

# Independence MH: proposal from N(0, 4), accept with MH ratio
def mh_independence(n_iter, proposal_std=2.0):
    x = 0.0   # init
    samples = []
    accepts = 0
    for _ in range(n_iter):
        y = rng.normal(0, proposal_std)
        # Proposal density: N(0, proposal_std²), symmetric → no prop in ratio
        log_alpha = log_pi(y) - log_pi(x) + \
                    0.5*(y - 0)**2/proposal_std**2 - 0.5*(x - 0)**2/proposal_std**2
        if np.log(rng.random()) < log_alpha:
            x = y
            accepts += 1
        samples.append(x)
    return np.array(samples), accepts/n_iter

samples, acc_rate = mh_independence(n_iter=20000, proposal_std=5.0)
print(f'Acceptance rate: {acc_rate:.3f}')

# Samples histogram vs target
x_grid = np.linspace(-6, 6, 200)
target_pdf = 0.5/np.sqrt(2*np.pi) * (np.exp(-(x_grid-2)**2/2) + np.exp(-(x_grid+2)**2/2))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(samples[:500])
plt.xlabel('iteration'); plt.ylabel('x'); plt.title('Trace plot (first 500)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(samples[1000:], bins=50, density=True, alpha=0.5, label='MCMC samples')
plt.plot(x_grid, target_pdf, 'r-', label='Target π')
plt.legend(); plt.title('Histogram vs target')
plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
```

### 실험 2 — 에르고딕 평균 수렴

```python
# E_π[x²] 추정 — 이론값?
# Mixture: 50% N(2, 1), 50% N(-2, 1)
# E[x] = 0, E[x²] = 0.5 * (4 + 1) + 0.5 * (4 + 1) = 5
true_E_x2 = 5.0

running_avg = np.cumsum(samples**2) / np.arange(1, len(samples) + 1)

plt.plot(running_avg, label=r'$\hat{\mathbb{E}}[x^2]$')
plt.axhline(true_E_x2, color='r', linestyle='--', label='True E[x²]=5')
plt.xscale('log')
plt.xlabel('n'); plt.ylabel(r'$\frac{1}{n}\sum x_k^2$')
plt.legend(); plt.title('Ergodic convergence (MCMC)'); plt.grid(True, alpha=0.3)
plt.show()
```

### 실험 3 — 수렴 전 "burn-in" 효과

```python
# Start from very far point
def mh_independence_init(x_init, n_iter, proposal_std=5.0):
    x = x_init; samples = []
    for _ in range(n_iter):
        y = rng.normal(0, proposal_std)
        log_alpha = log_pi(y) - log_pi(x) + 0.5*y**2/proposal_std**2 - 0.5*x**2/proposal_std**2
        if np.log(rng.random()) < log_alpha:
            x = y
        samples.append(x)
    return np.array(samples)

# 여러 starting points
n_iter = 5000
inits = [-10, -5, 0, 5, 10]
fig, ax = plt.subplots(figsize=(10, 4))
for x0 in inits:
    s = mh_independence_init(x0, n_iter)
    ax.plot(s, alpha=0.5, label=f'init={x0}')
ax.set_xlabel('iter'); ax.set_ylabel('x'); ax.legend(); ax.set_xscale('log')
ax.set_title('Burn-in: 여러 initial point에서 수렴')
ax.grid(True, alpha=0.3)
plt.show()
# 수백 step 이후 모두 정상 분포로 수렴
```

---

## 🔗 AI/ML 연결

**Bayesian Neural Network**  
Weight posterior $p(w | D) \propto p(D | w) p(w)$. MCMC로 $w$ 샘플 → prediction $\mathbb{E}_w[f(w, x)]$. Reliability estimate.

**PyMC / Stan / NumPyro**  
확률적 프로그래밍 언어. 사용자가 model 정의 → backend가 HMC/NUTS로 posterior MCMC 자동.

**Energy-Based Models**  
$p(x) \propto e^{-E_\theta(x)}$ with NN $E_\theta$. Training:
- Contrastive Divergence (CD-k): $k$-step MCMC로 $\mathbb{E}_\theta[\nabla E]$ 근사.
- SGLD based training.

**Score-based Generative Models**  
Langevin MCMC $dX = \nabla \log p dt + \sqrt 2 dB$의 discretization (Ch7-04). Score $\nabla \log p$ NN으로 학습 → MCMC로 샘플.

**Probabilistic Inference in Graphical Models**  
LDA (topic models), Ising, HMM posterior — Gibbs sampler (Ch7-03)가 표준.

---

## ⚖️ 가정과 한계

**한계 — Mixing time**  
Multi-modal, high-dim distributions에서 mixing time 길 수 있음. $\rho \to 1$ in spectral gap → stuck in local modes.

**한계 — Correlation 평가 어려움**  
$\sigma_f^2$ 정확한 추정이 ESS(effective sample size, Ch7-05) 기반. 작은 ESS = high autocorrelation = 실효 샘플 적음.

**한계 — Multi-modal**  
MH, Gibbs가 local modes 간 이동 어려움. Simulated tempering, parallel tempering이 해결.

**한계 — 높은 차원에서 rejection rate**  
Naive proposal이 대부분 reject → inefficient. HMC (gradient 활용)가 획기적 개선.

---

## 📌 핵심 정리

| 개념 | 요약 |
|---|---|
| Goal | Sample from $\pi(x) \propto \tilde\pi(x)$ |
| Approach | $\pi$를 정상분포로 갖는 MC 설계 |
| 근거 | Ergodic theorem — 시간평균 = 공간평균 |
| 설계 충분조건 | Detailed balance + 기약 + 비주기 |
| 수렴률 | Spectral gap → mixing time |
| Estimator | $\hat f_n = \frac{1}{n}\sum f(X_k)$ |
| Burn-in | 초기 transient 제거 |

**한 줄 요약**: MCMC는 "$\pi$에서 직접 샘플 불가 → $\pi$를 정상분포로 갖는 Markov chain 구성·시뮬레이션"의 프레임워크. Ch2의 정상분포 이론 + Ch2-06의 에르고딕 정리가 수학적 근거. Bayesian deep learning·EBM·Score-SDE 등 현대 AI의 기반.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. 1D target $\pi(x) \propto e^{-x^4/4}$에서 MCMC 하기 위해 random-walk proposal $y = x + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$ 사용. 이 체인의 기약성과 비주기성을 보여라.

<details>
<summary>해설</summary>

**기약성**: 임의 $x \to y$ 도달 가능. 한 step에서 $\mathbb{P}(X_{n+1} = y | X_n = x) = \phi(y - x; 0, \sigma^2) \cdot \alpha(x, y) > 0$ for any $y$ ($\phi$ = Gaussian density, everywhere positive). 따라서 $x \to y$ one step with positive probability → 기약.

**비주기**: Self-loop $\mathbb{P}(X_{n+1} = x | X_n = x) > 0$ (reject 시 머무름). Period $d_x = \gcd\{n : P_{xx}^{(n)} > 0\}$에서 $P_{xx}^{(1)} > 0$ → $1 \in$ support → $d_x = 1$ → 비주기.

**결과**: 정리 1.1 조건 충족 → MCMC가 $\pi$로 수렴 보장.

**실전 팁**: $\sigma$ 선택이 효율을 결정 (너무 작으면 slow mixing, 너무 크면 rejection rate 높음). Optimal acceptance rate ~0.44 (1D), 0.234 (high-dim) — Roberts-Gelman-Gilks 1997.

</details>

**문제 2 (심화)**. Rejection sampling과 MCMC의 차이: 왜 high-dim에서 MCMC가 이기는가?

<details>
<summary>해설</summary>

**Rejection sampling**:
Propose $x \sim q$, accept with prob $\pi(x) / (M q(x))$ where $M \geq \sup \pi/q$. 
- **장점**: iid samples (no correlation).
- **문제**: High-dim에서 $M$이 기하급수적으로 커짐.
  - $d$-dim Gaussian target with Gaussian proposal (slightly different variance) → $M \sim \exp(cd)$.
  - Acceptance rate $\sim 1/M \to 0$.

**MCMC**:
Chain 샘플 correlated but **no rejection explosion**.
- 각 step에서 acceptance rate controllable (tuning proposal).
- High-dim: gradient-aware (HMC) with $O(d)$ scaling vs $O(e^d)$ for naive rejection.

**Scaling laws**:
- **Random walk MH**: Optimal scaling $\sigma \propto d^{-1/2}$ → mixing time $O(d)$.
- **HMC**: $\sigma \propto d^{-1/4}$ → mixing time $O(d^{1/4})$ (Beskos 2013).
- **Rejection**: $O(e^d)$ **절대 불리**.

**교훈**: MCMC의 correlation cost < exponential rejection rate. **Exchange sample independence for dimensional efficiency**.

**연결**: ESS = $n / \tau_{\text{autocorr}}$. MCMC의 "effective sample size"가 correlation 반영 — 샘플 수로는 많지만 실효적으로 적을 수 있음 (Ch7-05).

</details>

**문제 3 (AI 연결)**. Bayesian neural network에서 posterior $p(\theta | D)$ 샘플링에 MCMC 사용. 수만/수십만 차원 (parameter count)에서 어떤 방법이 효과적인가?

<details>
<summary>해설</summary>

**Challenges in high-dim ($d \sim 10^6$)**:
- Naive RW-MH: $O(d)$ scaling 이론상 있지만 practical iter 수 막대함.
- Rejection: 불가.
- Variational Inference (VI): Approximation 도입하지만 biased.

**효과적 MCMC approaches**:

**(1) Stochastic Gradient MCMC (SG-MCMC)**:
- **SGLD** (Welling-Teh 2011): $\theta_{t+1} = \theta_t + \frac{\eta}{2}\nabla \log p(\theta | D) + \sqrt\eta \xi$. Mini-batch gradient + Langevin noise.
- Scale: $d \sim 10^8$ parameters.
- **단점**: Biased (discretization + mini-batch noise).

**(2) Hamiltonian Monte Carlo (HMC)**:
- Gradient 활용 → $O(d^{1/4})$ mixing.
- **NUTS** (Hoffman-Gelman 2014): Auto-tuning, leapfrog step 수 자동.
- **한계**: Per-step cost $O(d)$ — millions of params에서 per-iter computation 부담.

**(3) Stochastic Gradient HMC (SGHMC)** (Chen 2014):
- HMC + mini-batch. "Friction" term으로 gradient noise 보정.
- Large-scale BNN 기본 선택.

**(4) Continuous-Time Perspectives**:
- **Underdamped Langevin**: SGHMC의 연속 버전.
- **Anneal / tempered MCMC**: Temperature schedule.

**실전 BNN training**:
- Small BNN ($d < 10^5$): NUTS with Stan/PyMC.
- Medium ($10^5 < d < 10^7$): SGLD variants (pyro-ppl, torch SG-MCMC).
- Large ($d > 10^7$): VI (not MCMC) — mean-field, SVI with scalable optimization.

**Hybrid approaches**:
- **Stochastic Weight Averaging with Gaussian (SWAG)**: Posterior를 Gaussian 근사하여 fast approximate inference.
- **Deep Ensembles**: Multiple point estimates로 uncertainty proxy.

**현대 연구 방향**:
- **Neural network preconditioner** for HMC: NN으로 mass matrix 학습.
- **Diffusion model for posterior**: Score-SDE style → posterior sampling as diffusion reverse.

**연결**: Ch7-04 (HMC)에서 HMC 수학적 구조. SDE Deep Dive Ch4 (Langevin)에서 SG-MCMC의 연속 이론.

</details>

---

<div align="center">

◀ [Ch6-06. 반사원리(Reflection Principle)와 최대값](../ch6-brownian/06-reflection-principle.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [02. Metropolis-Hastings 알고리즘](./02-metropolis-hastings.md)

</div>
