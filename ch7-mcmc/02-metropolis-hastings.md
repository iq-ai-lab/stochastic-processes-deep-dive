# 02. Metropolis-Hastings 알고리즘

## 🎯 핵심 질문

- **MH acceptance ratio** $\alpha = \min(1, \pi(y) q(x|y) / (\pi(x) q(y|x)))$는 어디서 오는가?
- 왜 이 선택이 **detailed balance**를 **자동으로** 만족하는가?
- **정규화 상수 $Z$**가 acceptance에서 **사라지는** 이유는?
- **Proposal distribution** 설계의 핵심 원칙 (acceptance rate tuning)?

---

## 🔍 왜 MH가 AI에서 중요한가

**사실상 표준 MCMC**: 대부분의 MCMC 알고리즘(Gibbs, HMC, Random-walk MH, Langevin MH)이 MH의 특수 경우.

**Bayesian inference의 기본 도구**: PyMC, Stan, NumPyro의 기본 sampler가 MH 기반.

**정규화 불필요한 장점**: EBM, GAN discriminator, 복잡한 posterior 모두 **unnormalized $\tilde\pi$**만 필요 → 현대 AI의 probabilistic model에 직접 적용.

**Simulated Annealing**: Combinatorial optimization에서 MH with cooling schedule → global minimum 탐색.

---

## 📐 수학적 선행 조건

- [Ch2-05](../ch2-discrete-markov/05-detailed-balance.md): Detailed balance
- [Ch7-01](./01-mcmc-idea.md): MCMC idea

---

## 📖 직관적 이해

### MH 알고리즘

Target $\pi(x) \propto \tilde\pi(x)$. Proposal $q(y|x)$ (임의 conditional distribution).

**Step at iteration $n$**:
1. 현재 $x = X_n$. Propose $y \sim q(\cdot | x)$.
2. **Acceptance**:
$$\alpha(x, y) = \min\left(1, \frac{\tilde\pi(y) q(x | y)}{\tilde\pi(x) q(y | x)}\right).$$
3. With prob $\alpha(x, y)$: $X_{n+1} = y$; else $X_{n+1} = x$.

**$Z$ 소거**: $\tilde\pi = Z \pi$. Ratio $\tilde\pi(y)/\tilde\pi(x) = \pi(y)/\pi(x)$. 따라서 **$Z$ 몰라도** 됨.

### 왜 "$\min(1, \cdot)$"인가

**Detailed balance 설계 목적**: Transition kernel
$P(x, y) = q(y|x) \alpha(x, y) + (1 - \alpha(x)) \delta_x(y)$ (accept or stay)
가 $\pi(x) P(x, y) = \pi(y) P(y, x)$를 만족하도록.

$x \neq y$ 가정:
$\pi(x) q(y|x) \alpha(x, y) = \pi(y) q(x|y) \alpha(y, x)$ (detailed balance 요구).

$\alpha(x, y)$의 선택이 이 equality를 **만족시키도록** 설계됨:

**Case 1**: $\pi(x) q(y|x) > \pi(y) q(x|y)$ (forward가 더 likely). 
$\alpha(x, y) = \pi(y)q(x|y)/(\pi(x)q(y|x)) < 1$, $\alpha(y, x) = 1$.
LHS = $\pi(x) q(y|x) \cdot \pi(y) q(x|y) / (\pi(x) q(y|x)) = \pi(y) q(x|y)$.
RHS = $\pi(y) q(x|y) \cdot 1 = \pi(y) q(x|y)$. ✓

**Case 2** (대칭): LHS = RHS 자동.

따라서 **detailed balance 자동 성립** — $\pi$가 정상분포 보장.

### 최소성 / 최적성

$\alpha(x, y) = \min(1, \cdot)$이 detailed balance를 만족시키는 **최대값** — "accept 가능한 최대 확률"이라 MCMC 효율 최대. 다른 $\alpha' \leq \min(1, \cdot)$도 valid이지만 효율 낮음.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Metropolis-Hastings Algorithm

Target $\pi$ with unnormalized density $\tilde\pi$. Proposal kernel $q(y | x)$ (density).

**Iteration** $n$: given $X_n = x$,
1. Sample $y \sim q(\cdot | x)$.
2. Sample $u \sim \text{Uniform}(0, 1)$.
3. If $u \leq \alpha(x, y) := \min\left(1, \frac{\tilde\pi(y) q(x|y)}{\tilde\pi(x) q(y|x)}\right)$, set $X_{n+1} = y$.
4. Else $X_{n+1} = x$.

### 정의 2.2 — Transition Kernel

$$P(x, dy) = q(y | x) \alpha(x, y) dy + r(x) \delta_x(dy),$$
$r(x) = 1 - \int q(y|x) \alpha(x, y) dy$ (reject prob, stays at $x$).

### 정의 2.3 — Special Cases

- **Symmetric proposal** (Metropolis 1953): $q(y|x) = q(x|y)$ → $\alpha = \min(1, \pi(y)/\pi(x))$. Random walk MH가 대표.
- **Independence MH**: $q(y|x) = q(y)$ — $x$ 무관.
- **Random walk MH**: $q(y|x) = g(y - x)$ with $g$ symmetric (예: $\mathcal{N}(0, \sigma^2)$).
- **Langevin MH (MALA)**: $q(y|x) = \mathcal{N}(y; x + \frac{\eta}{2}\nabla \log \pi(x), \eta I)$.

---

## 🔬 정리와 증명

### 정리 2.1 — MH가 Detailed Balance를 만족

Transition kernel $P$ (정의 2.2)가 $\pi$에 대해 detailed balance:
$$\pi(x) P(x, y) = \pi(y) P(y, x) \quad \forall x, y.$$

### 증명

$x = y$: 양쪽 같음 (trivially).

$x \neq y$: $P(x, y) = q(y|x) \alpha(x, y)$. 
$$\pi(x) q(y|x) \alpha(x, y) = \pi(x) q(y|x) \min\left(1, \frac{\pi(y) q(x|y)}{\pi(x) q(y|x)}\right).$$

WLOG $\pi(x) q(y|x) \geq \pi(y) q(x|y)$:
- $\alpha(x, y) = \pi(y) q(x|y) / (\pi(x) q(y|x))$.
- $\alpha(y, x) = 1$.

LHS: $\pi(x) q(y|x) \cdot \pi(y) q(x|y) / (\pi(x) q(y|x)) = \pi(y) q(x|y)$.
RHS: $\pi(y) q(x|y) \cdot 1 = \pi(y) q(x|y)$.

**Equal** ✓. $\square$

### 정리 2.2 — MH가 정상분포 $\pi$를 보존

Detailed balance → $\pi$가 정상분포 (Ch2-05 정리 1).

### 정리 2.3 — MH의 기약성과 비주기성 조건

**기약**: Proposal $q(y|x)$의 support가 $\pi$의 support 전체를 덮고, $q > 0$ where $\pi > 0$이면 기약.

**비주기**: $q(y|x) > 0$ at $y = x$ or $\alpha < 1$ 가능 (reject) → self-loop 양의 확률 → 비주기.

*따라서* 대부분의 실전 MH 설정에서 Ergodic theorem의 조건 충족 → MCMC가 $\pi$로 수렴.

### 정리 2.4 — Optimal Acceptance Rate (Scaling Results)

Random-walk MH in $d$-dim with Gaussian proposal $\mathcal{N}(0, \sigma^2 I)$:
- **Roberts, Gelman, Gilks (1997)**: Optimal $\sigma \propto d^{-1/2}$ → acceptance rate $\approx 0.234$ (high-dim asymptotic).
- 1D: $\approx 0.44$.

**실전 heuristic**: Adaptive MH에서 proposal $\sigma$ 조정 → acceptance rate $\in [0.2, 0.5]$.

---

## 💻 NumPy 구현 검증

### 실험 1 — Random-walk MH for bimodal distribution

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

def log_pi(x):
    """Bimodal: 0.5 N(-2, 1) + 0.5 N(2, 1)"""
    return np.log(0.5 * np.exp(-0.5*(x-2)**2) + 0.5 * np.exp(-0.5*(x+2)**2))

def mh_rw(n_iter, step_size=1.0, x0=0.0):
    x = x0; samples = [x]; accepts = 0
    for _ in range(n_iter):
        y = x + rng.normal(0, step_size)
        log_alpha = log_pi(y) - log_pi(x)   # Symmetric proposal
        if np.log(rng.random()) < log_alpha:
            x = y; accepts += 1
        samples.append(x)
    return np.array(samples), accepts/n_iter

samples, acc = mh_rw(n_iter=20000, step_size=2.0)
print(f'Acceptance rate: {acc:.3f}')

# Histogram vs target
x_grid = np.linspace(-6, 6, 200)
target = 0.5/np.sqrt(2*np.pi) * (np.exp(-(x_grid-2)**2/2) + np.exp(-(x_grid+2)**2/2))

plt.hist(samples[2000:], bins=50, density=True, alpha=0.5)
plt.plot(x_grid, target, 'r-', label='Target π')
plt.legend(); plt.title(f'MH-RW, acc={acc:.2f}')
plt.show()
```

### 실험 2 — Acceptance rate vs step_size tuning

```python
# 다양한 step size
step_sizes = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
for s in step_sizes:
    _, acc = mh_rw(n_iter=5000, step_size=s)
    print(f'step_size={s}: acceptance rate = {acc:.3f}')
# 작은 step: acc 높음 (but slow mixing)
# 큰 step: acc 낮음 (but mix faster if accepted)
# 적정 중간값이 옵티멀
```

### 실험 3 — Independence MH vs Random-walk MH

```python
def mh_independence(n_iter, prop_std=3.0, x0=0.0):
    x = x0; samples = [x]; accepts = 0
    for _ in range(n_iter):
        y = rng.normal(0, prop_std)
        # Asymmetric proposal: q(y|x) = q(y)
        log_alpha = log_pi(y) - log_pi(x) + 0.5*(y/prop_std)**2 - 0.5*(x/prop_std)**2
        if np.log(rng.random()) < log_alpha:
            x = y; accepts += 1
        samples.append(x)
    return np.array(samples), accepts/n_iter

ind_samples, ind_acc = mh_independence(20000)
rw_samples, rw_acc = mh_rw(20000, step_size=2.0)

print(f'Independence MH: acc = {ind_acc:.3f}')
print(f'Random walk MH:  acc = {rw_acc:.3f}')

# Autocorrelation 비교
def autocorr(x, max_lag=50):
    x = x - x.mean()
    v = x.var()
    return np.array([np.mean(x[:-k] * x[k:])/v if k > 0 else 1 
                     for k in range(max_lag)])

plt.plot(autocorr(ind_samples[2000:]), label='Indep MH')
plt.plot(autocorr(rw_samples[2000:]), label='RW MH')
plt.xlabel('lag'); plt.ylabel('autocorr'); plt.legend()
plt.title('Autocorrelation comparison')
plt.grid(True, alpha=0.3); plt.show()
# RW가 보통 더 작은 autocorrelation for well-tuned step
```

---

## 🔗 AI/ML 연결

**PyMC / Stan / NumPyro**  
MH는 baseline — 실전은 HMC/NUTS (더 효율). 단 고차원 discrete space에서는 여전히 MH 변형.

**Simulated Annealing**  
$\pi_T(x) \propto e^{-E(x)/T}$, $T$ 점차 0으로. High $T$에서 explore, low $T$에서 exploit. MH with $\pi_T$ + cooling schedule. Combinatorial optimization.

**Bayesian Optimization Acquisition**  
Posterior over $f$ (GP) → MH로 max of $f$ 샘플 → next query point 결정. Thompson sampling의 implementation.

**Energy-Based Models Training**  
$p_\theta(x) \propto e^{-E_\theta(x)}$. Contrastive divergence: MH $k$ step으로 $\mathbb{E}_\theta[\nabla E]$ 근사 → gradient update.

**Particle MCMC**  
State-space model filtering. Particle filter 내부에서 MH로 smooth particles.

---

## ⚖️ 가정과 한계

**가정 — Tractable proposal $q$**  
$q(y|x)$와 $q(x|y)$ 계산 가능. 일반적으로 쉬움 (Gaussian, etc.)이지만 complex proposals에서는 주의.

**한계 — Mixing in high-dim**  
RW-MH에서 acceptance rate $\sim 0.234$ optimal but mixing still slow $O(d)$. HMC (Ch7-04)이 해결.

**한계 — Multi-modal**  
$\pi$가 well-separated modes 가지면 mode 간 이동 어려움. Parallel tempering, simulated tempering으로 극복.

---

## 📌 핵심 정리

| 개념 | 수식 |
|---|---|
| Acceptance | $\alpha(x, y) = \min(1, \pi(y)q(x\|y)/(\pi(x)q(y\|x)))$ |
| $Z$ 소거 | Ratio에서 정규화 상수 취소 |
| Detailed balance | 자동 성립 |
| Symmetric q | $\alpha = \min(1, \pi(y)/\pi(x))$ (Metropolis) |
| Optimal acceptance | ~0.234 (high-dim RW) |
| Special cases | RW-MH, Independence, MALA, Gibbs, HMC |

**한 줄 요약**: MH의 acceptance $\alpha = \min(1, \pi(y)q(x|y)/(\pi(x)q(y|x)))$이 **정규화 상수 $Z$ 없이** detailed balance 자동 보장. 모든 현대 MCMC의 공통 framework, AI의 posterior inference의 backbone.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. MH에서 $\tilde\pi(y)/\tilde\pi(x) > 1$이면 어떻게 되는가? 무엇이 "accept always"를 의미하는가?

<details>
<summary>해설</summary>

**Forward 이익**: $\tilde\pi(y) > \tilde\pi(x)$ (symmetric q면 단순히 $y$가 higher density).

$\alpha(x, y) = \min(1, \text{ratio}) = 1$ → **항상 accept**.

**해석**: "더 좋은 방향으로는 항상 가라" — gradient ascent-like.

**반대**: $\tilde\pi(y) < \tilde\pi(x)$면 $\alpha < 1$ — 확률적으로 reject할 수도 있다. 이 "downhill moves"가 **local mode escape**의 가능성을 제공.

**Simulated Annealing**과 비교: Low temperature → downhill acceptance rate ↓ → exploitation. High temp → ↑ → exploration.

**의의**: MH의 "probabilistic downhill" 성질이 greedy optimization과 다름 — global exploration 가능.

</details>

**문제 2 (심화)**. Independence MH가 proposal $q(y) = \pi(y)$이면 **iid 샘플**이 됨을 보여라. 왜 "perfect MCMC"가 target distribution에서 직접 샘플링하는 것과 같은가?

<details>
<summary>해설</summary>

$q(y) = \pi(y)$:
$\alpha(x, y) = \min(1, \pi(y) \pi(x) / (\pi(x) \pi(y))) = 1$ — 항상 accept.

Chain: $X_{n+1} = Y_n \sim \pi$ (independent of $X_n$).

**결과**: $\{X_n\}$이 iid $\pi$ — "perfect" MCMC.

**paradox**: 하지만 $\pi$에서 직접 iid 샘플링 가능하면 애초에 MCMC 필요 없음. Target proposal $q = \pi$는 **assumption**이 아니라 goal.

**의의**: "MCMC의 비용 = proposal-target mismatch". $q$가 $\pi$에 가까울수록 효율 높음:
- Perfect: $q = \pi$, acceptance = 1, iid.
- Independence with fixed $q$: acceptance rate 낮을 수 있음.
- RW: $q$가 local (previous state에 의존) → proposals overlap well.

**실전**: Adaptive MCMC가 chain 진행하며 $q$를 학습 → "empirical $\pi$ approximation" → 효율 개선.

</details>

**문제 3 (AI 연결)**. Discriminator $D(x)$가 학습된 후, $D \approx \log(\pi_{\text{data}}/\pi_{\text{model}})$. MH로 model distribution에서 data distribution으로 샘플 이동하는 방법은?

<details>
<summary>해설</summary>

**Setup**:
$D(x) \approx \log(\pi_{\text{data}}(x) / \pi_{\text{model}}(x))$.

즉 $\pi_{\text{data}}(x) \approx e^{D(x)} \pi_{\text{model}}(x)$.

**Target**: Sample from $\pi_{\text{data}}$. MH with $q = \pi_{\text{model}}$ (기존 generator):

$\alpha(x, y) = \min(1, \pi_{\text{data}}(y) q(x) / (\pi_{\text{data}}(x) q(y))) = \min(1, e^{D(y)} q(y) q(x) / (e^{D(x)} q(x) q(y))) = \min(1, e^{D(y) - D(x)})$.

**놀라운 결과**: Acceptance가 단순히 discriminator output의 차이 → $e^{D(y) - D(x)}$.

**응용 — MH-GAN (Turner 2019)**:
- Trained GAN의 generator output을 MH 필터로 개선.
- $q$ = generator samples, $D$ = trained discriminator.
- Accept/reject로 "real-like" samples만 남김.

**결과**:
- Standard GAN sample보다 FID 개선.
- 학습된 모델을 "정제"하는 post-hoc 방법.

**이론적 정당성**:
- Discriminator loss optimum에서 $D = \log(p_{\text{real}}/p_{\text{gen}})$ (GAN의 classical result).
- 이 $D$가 MH acceptance 제공.

**확장**:
- **Discriminator Rejection Sampling** (Azadi 2019): $D$로 rejection sampling.
- **SMC-GAN**: Sequential MC with discriminator weights.

**한계**:
- $D$의 training accuracy에 의존.
- Support mismatch 시 $D$ extreme values → unstable.

**연결**: GAN + MCMC가 "generative model refinement"의 hybrid paradigm. MH의 강력한 generalizability 예시.

</details>

---

<div align="center">

◀ [01. MCMC의 아이디어 — 정상분포 설계](./01-mcmc-idea.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [03. Gibbs Sampler](./03-gibbs-sampler.md)

</div>
