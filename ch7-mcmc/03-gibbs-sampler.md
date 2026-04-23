# 03. Gibbs Sampler

## 🎯 핵심 질문

- **Gibbs sampler**: 조건부 분포 $p(x_i | x_{-i})$에서 순차적으로 샘플링하는 아이디어는 어떻게 MCMC인가?
- Gibbs가 **MH의 특수 경우** (acceptance $\alpha \equiv 1$)임을 어떻게 증명하는가?
- **Systematic scan** vs **random scan**의 차이는?
- Gibbs의 **한계**: High correlation 변수들에서 왜 mixing이 느려지는가?

---

## 🔍 왜 Gibbs가 AI에서 중요한가

**Hierarchical Bayesian Models**: Conditional conjugacy 있으면 closed-form posterior → Gibbs 효율적. Hierarchical linear models, topic models (LDA).

**Boltzmann Machine / RBM**: Visible/hidden variables 교대 업데이트. Contrastive divergence 기반.

**Latent Dirichlet Allocation (LDA, Blei 2003)**: Document topic inference — Collapsed Gibbs의 classical use.

**Image Segmentation (Markov Random Field)**: Pixel-wise Gibbs updates → segmentation posterior.

**Missing Data Imputation**: 각 missing value를 conditional posterior에서 샘플 → multiple imputation.

---

## 📐 수학적 선행 조건

- [Ch7-01, Ch7-02](./01-mcmc-idea.md): MCMC, MH
- [Ch2-05](../ch2-discrete-markov/05-detailed-balance.md): Detailed balance

---

## 📖 직관적 이해

### Coordinate-wise Sampling

$\pi(x) = \pi(x_1, x_2, \ldots, x_d)$ — 고차원 joint. Joint 전체 샘플링 어렵지만 **conditional** $\pi(x_i | x_{-i})$이 tractable인 경우 多.

**Gibbs idea**: 한 번에 한 좌표씩 update, 다른 좌표 fixed + 조건부에서 샘플.

**Step**: $x_i \leftarrow \sim \pi(x_i | x_{-i})$.

모든 좌표 cycle → 한 Gibbs iteration.

### 왜 작동하는가

각 step이 **조건부 분포에서 정확한 샘플** → local MH with acceptance 1. 전체 체인이 joint $\pi$ 보존.

> **비유**: 각 좌표 "상호 강화". 한 변수를 다른 변수들에 맞춰 조정 반복 → 전체 joint distribution 수렴.

### Systematic vs Random Scan

**Systematic**: $x_1 \to x_2 \to \cdots \to x_d \to x_1 \to \cdots$. 일정 cycle.

**Random**: 매 step 랜덤 좌표 $i$ 선택.

**Systematic scan 주의**: 각 step 자체는 detailed balance지만 "전체 sweep이 detailed balance는 아님" — 시간 방향성 있음. 그러나 정상분포 보존됨 (composition of "$\pi$-preserving" operators).

**Random scan**은 detailed balance 만족 (reversible).

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Gibbs Sampler (Systematic)

$x = (x_1, \ldots, x_d)$. Iteration $n$: given $x^{(n)}$,
1. $x_1^{(n+1)} \sim \pi(x_1 | x_2^{(n)}, \ldots, x_d^{(n)})$
2. $x_2^{(n+1)} \sim \pi(x_2 | x_1^{(n+1)}, x_3^{(n)}, \ldots, x_d^{(n)})$
3. ...
4. $x_d^{(n+1)} \sim \pi(x_d | x_1^{(n+1)}, \ldots, x_{d-1}^{(n+1)})$

**Block Gibbs**: 여러 좌표를 **block**으로 update — 각 block의 conditional distribution에서.

### 정의 3.2 — Random Scan Gibbs

매 step 랜덤 $i \sim \text{Uniform}(\{1, \ldots, d\})$. Update $x_i \sim \pi(x_i | x_{-i})$.

### 정의 3.3 — Collapsed Gibbs

일부 nuisance parameters를 **analytically integrate out** 후 나머지에 Gibbs. LDA의 collapsed Gibbs가 효율 큰 이유.

---

## 🔬 정리와 증명

### 정리 3.1 — Gibbs가 정상분포 $\pi$를 보존

Gibbs sampler (systematic or random scan)의 transition kernel이 $\pi$를 invariant하게 유지.

*증명*.

**Single coordinate update** $x_i \to x_i'$:
$$\pi(x) K_i(x, x') = \pi(x) \pi(x_i' | x_{-i}) \mathbf{1}_{\{x_{-i}' = x_{-i}\}}.$$

Integrate over $x_i$ to verify invariance:
$$\int \pi(x) K_i(x, x') dx = \pi(x_i' | x_{-i}) \int \pi(x_i | x_{-i}) \pi(x_{-i}) dx_i = \pi(x_i' | x_{-i}) \pi(x_{-i}) = \pi(x_{-i}, x_i') = \pi(x').$$

따라서 $K_i$가 $\pi$ 보존. Composition $K_1 K_2 \cdots K_d$ (systematic) 또는 mixture (random) 모두 $\pi$ 보존. $\square$

### 정리 3.2 — Gibbs가 MH의 특수 경우 (Acceptance $\alpha \equiv 1$)

각 single-coordinate update를 MH로 볼 때, proposal $q(x'|x) = \pi(x_i' | x_{-i}) \mathbf{1}_{\{x_{-i}' = x_{-i}\}}$이면 MH acceptance = 1.

*증명*.
MH ratio:
$$\alpha(x, x') = \min\left(1, \frac{\pi(x') q(x|x')}{\pi(x) q(x'|x)}\right).$$

$\pi(x') = \pi(x_i' | x_{-i}) \pi(x_{-i})$, $\pi(x) = \pi(x_i | x_{-i}) \pi(x_{-i})$.

$q(x|x') = \pi(x_i | x_{-i}')$, $q(x'|x) = \pi(x_i' | x_{-i})$. $x_{-i}' = x_{-i}$이므로 $q(x|x') = \pi(x_i | x_{-i})$.

Ratio:
$$\frac{\pi(x_i' | x_{-i}) \pi(x_{-i}) \cdot \pi(x_i | x_{-i})}{\pi(x_i | x_{-i}) \pi(x_{-i}) \cdot \pi(x_i' | x_{-i})} = 1.$$

**$\alpha = 1$** — 항상 accept. $\square$

### 정리 3.3 — Random Scan Gibbs는 Reversible

Random scan Gibbs의 transition kernel이 detailed balance 만족.

*증명*. Single-coordinate $K_i$에 대해 detailed balance:
$\pi(x) K_i(x, x') = \pi(x) \pi(x_i'|x_{-i}) \mathbf{1}_{\{x_{-i} = x_{-i}'\}}$.  
$\pi(x') K_i(x', x) = \pi(x') \pi(x_i|x_{-i}) \mathbf{1}_{\{x_{-i}' = x_{-i}\}}$.

$\pi(x_i' | x_{-i}) \pi(x_{-i}) \pi(x_i|x_{-i}) = \pi(x_i|x_{-i}) \pi(x_{-i}) \pi(x_i'|x_{-i})$ — **equal**. DB 성립.

Random mixture $\frac{1}{d}\sum K_i$도 DB (linear combination of DB kernels).

**Systematic scan은 DB 깨짐**: $K_1 K_2$의 DB 확인 — $\pi K_1 K_2(x, y) \neq \pi K_2 K_1(y, x)$ 일반적. 하지만 여전히 $\pi$ 보존. $\square$

### 정리 3.4 — 기약성

각 $\pi(x_i | x_{-i})$의 support가 $\pi(x_i | x_{-i})$의 marginal support와 같으면 Gibbs chain이 기약.

*증명 스케치*. 한 사이클 $K_1 K_2 \cdots K_d$로 임의 $(x_1', \ldots, x_d')$ 도달 가능 — 각 step에서 해당 conditional support 전체로 이동. $\square$

---

## 💻 NumPy 구현 검증

### 실험 1 — 2D Gaussian Gibbs

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# Target: N(0, Σ) with Σ = [[1, ρ], [ρ, 1]]
rho = 0.8

# Conditionals: X1 | X2=x2 ~ N(ρ x2, 1 - ρ²)
#               X2 | X1=x1 ~ N(ρ x1, 1 - ρ²)

def gibbs_2d_gaussian(n_iter, x_init=(0., 0.)):
    x = np.array(x_init, dtype=float)
    samples = [x.copy()]
    for _ in range(n_iter):
        x[0] = rng.normal(rho * x[1], np.sqrt(1 - rho**2))
        x[1] = rng.normal(rho * x[0], np.sqrt(1 - rho**2))
        samples.append(x.copy())
    return np.array(samples)

samples = gibbs_2d_gaussian(10000)

# Compare with truth
from scipy.stats import multivariate_normal
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(samples[100:, 0], samples[100:, 1], s=1, alpha=0.3)
plt.title(f'2D Gaussian Gibbs (ρ={rho})')
plt.xlabel('x1'); plt.ylabel('x2')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(samples[:200, 0], samples[:200, 1], '-o', markersize=3, alpha=0.5)
plt.title('Gibbs trajectory (first 200 steps)')
plt.xlabel('x1'); plt.ylabel('x2')
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# Marginal checks
print(f'μ estimates: ({samples[:, 0].mean():.4f}, {samples[:, 1].mean():.4f})')
print(f'Σ estimate:\n{np.cov(samples, rowvar=False)}')
# ρ = 0.8 confirmed
```

### 실험 2 — 높은 correlation에서 mixing 느려짐

```python
# ρ = 0.0 vs 0.8 vs 0.99
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, rho_val in zip(axes, [0.0, 0.8, 0.99]):
    global rho
    rho = rho_val
    s = gibbs_2d_gaussian(500)
    ax.plot(s[:, 0], s[:, 1], '-', linewidth=0.5, alpha=0.5)
    ax.scatter(s[:, 0], s[:, 1], s=3)
    ax.set_title(f'ρ = {rho_val}')
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.grid(True, alpha=0.3)
plt.suptitle('Gibbs mixing: high correlation → slow exploration')
plt.tight_layout(); plt.show()

# ρ = 0.99: 거의 대각선으로만 움직임 — extremely slow
```

### 실험 3 — Gibbs for Ising model (이산)

```python
# 2D Ising: s_{ij} ∈ {±1}, H = -J sum_{<i,j>} s_i s_j
L = 20   # grid size
J = 1.0
beta = 0.5   # inverse temperature

def init_lattice(L):
    return rng.choice([-1, 1], size=(L, L))

def gibbs_ising_step(lattice, beta, J):
    L = lattice.shape[0]
    for i in range(L):
        for j in range(L):
            # Neighboring sum
            nb = (lattice[(i+1)%L, j] + lattice[(i-1)%L, j] + 
                  lattice[i, (j+1)%L] + lattice[i, (j-1)%L])
            # Conditional: P(s_ij = +1 | rest) = sigmoid(2 β J nb)
            p_plus = 1 / (1 + np.exp(-2 * beta * J * nb))
            lattice[i, j] = 1 if rng.random() < p_plus else -1
    return lattice

lattice = init_lattice(L)
for _ in range(100):   # burn-in
    lattice = gibbs_ising_step(lattice, beta, J)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, t in zip(axes, [0, 20, 50, 100]):
    for _ in range(t):
        lattice = gibbs_ising_step(lattice, beta, J)
    ax.imshow(lattice, cmap='gray')
    ax.set_title(f'After {100 + t} steps, β = {beta}')
    ax.axis('off')
plt.suptitle('Gibbs sampling for 2D Ising model')
plt.show()
```

---

## 🔗 AI/ML 연결

**Latent Dirichlet Allocation (LDA)**  
Topic model: word $\to$ topic assignment $z$. Collapsed Gibbs integrates out $\theta, \phi$:
$p(z_i | z_{-i}, w) \propto (n^{d}_{z_i, -i} + \alpha)(n^{w}_{z_i, w_i, -i} + \beta)/(n^{t}_{z_i, -i} + W\beta)$.

30년간 LDA inference의 표준.

**Restricted Boltzmann Machine (RBM)**  
Bipartite: visible $v$, hidden $h$. Conditional $p(h | v), p(v | h)$ factorize — efficient.
Contrastive Divergence (CD-k): $k$-step Gibbs starting from data → gradient estimate.

**Markov Random Field (MRF) Inference**  
Image segmentation, denoising. Pixel-level Gibbs update. Swendsen-Wang block updates for speed.

**Bayesian Linear Regression**  
$\beta | \sigma^2, y, X \sim \mathcal{N}(\ldots)$, $\sigma^2 | \beta, y, X \sim \text{InvGamma}$. Each conditional closed-form → Gibbs가 efficient.

**Missing Data Imputation**  
Each missing value → conditional posterior sample. Multiple Imputation by Chained Equations (MICE)의 core.

---

## ⚖️ 가정과 한계

**한계 — Conditional tractability**  
$\pi(x_i | x_{-i})$가 closed-form이어야 직접 sampling 가능. Complex posteriors에서 unavailable → Metropolis-within-Gibbs.

**한계 — Highly correlated variables**  
$\rho \to 1$에서 coordinate-wise moves가 diagonal 이동만 → slow mixing. **Block updates** or **reparameterization** 해결.

**한계 — High-dim with many dependencies**  
$d \gg 1$ + strong dependence → per-iteration cost $O(d) \cdot$ slow mixing = 매우 비효율. HMC (Ch7-04)가 우수.

---

## 📌 핵심 정리

| 개념 | 수식 |
|---|---|
| Gibbs step | $x_i \leftarrow \pi(x_i \| x_{-i})$ |
| Systematic | 순차 cycle |
| Random scan | 랜덤 coordinate |
| MH 특수 경우 | $\alpha \equiv 1$ |
| Invariance | 각 step $\pi$ 보존 |
| Reversibility | Random scan reversible, systematic not |
| Collapsed | Integrate out nuisance params first |

**한 줄 요약**: Gibbs sampler는 coordinate-wise conditional sampling — MH의 특수 경우 ($\alpha = 1$). Tractable conditionals 있는 경우 강력 (LDA, RBM, MRF), high correlation에서 mixing slow — block updates/reparameterization 필요.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. Gibbs sampler for $\pi(x_1, x_2, x_3) \propto \exp(-\frac{1}{2}(x_1^2 + x_2^2 + x_3^2 + x_1 x_2 + x_2 x_3))$ 설계하라.

<details>
<summary>해설</summary>

**Joint**: $\log \pi = -\frac{1}{2}(x_1^2 + x_2^2 + x_3^2 + x_1 x_2 + x_2 x_3) + \text{const}$.

Matrix form: $\pi \propto \exp(-\frac{1}{2} x^T A x)$ with $A = \begin{pmatrix} 1 & 1/2 & 0 \\ 1/2 & 1 & 1/2 \\ 0 & 1/2 & 1 \end{pmatrix}$.

**Conditionals**:
- $\pi(x_1 | x_2, x_3)$: $\log \pi = -\frac{1}{2}x_1^2 - \frac{1}{2} x_1 x_2 + \text{const}(x_2, x_3)$. $= -\frac{1}{2}(x_1 + x_2/2)^2 + \text{const}$. → $\mathcal{N}(-x_2/2, 1)$.
- $\pi(x_2 | x_1, x_3)$: $-\frac{1}{2}x_2^2 - \frac{1}{2}(x_1 + x_3) x_2 = -\frac{1}{2}(x_2 + (x_1 + x_3)/2)^2 + \text{const}$. → $\mathcal{N}(-(x_1 + x_3)/2, 1)$.
- $\pi(x_3 | x_1, x_2)$: 대칭, $\mathcal{N}(-x_2/2, 1)$.

**Gibbs algorithm**:
```
x2 known → x1 = N(-x2/2, 1)
x1, x3 known → x2 = N(-(x1+x3)/2, 1)
x2 known → x3 = N(-x2/2, 1)
```

Cycle 후 joint $\pi$에서 샘플.

**검증**: $A^{-1}$ 계산 → analytical covariance. Gibbs 실측과 비교.

</details>

**문제 2 (심화)**. 2D Gaussian Gibbs의 수렴률을 spectral gap으로 계산하라. $\rho$가 어떻게 영향을 미치는가?

<details>
<summary>해설</summary>

**2D Gaussian Gibbs** (정리 이전 실험 1):
- Full cycle $K = K_1 K_2$: $x_1 \sim \pi(\cdot | x_2)$ then $x_2 \sim \pi(\cdot | x_1^{\text{new}})$.

**Update in matrix form**:
$(x_1^{n+1}, x_2^{n+1})^T = \text{linear transform of } (x_1^n, x_2^n)^T + \text{noise}$.

$x_1^{n+1} = \rho x_2^n + \sqrt{1 - \rho^2} \epsilon_1$  
$x_2^{n+1} = \rho x_1^{n+1} + \sqrt{1 - \rho^2} \epsilon_2 = \rho^2 x_2^n + \rho\sqrt{1-\rho^2}\epsilon_1 + \sqrt{1-\rho^2}\epsilon_2$.

**AR matrix**: $\begin{pmatrix} x_1^{n+1} \\ x_2^{n+1} \end{pmatrix} = A \begin{pmatrix} x_1^n \\ x_2^n \end{pmatrix} + \text{noise}$ with $A = \begin{pmatrix} 0 & \rho \\ 0 & \rho^2 \end{pmatrix}$ (lower triangular after $K_1$, then $K_2$).

**Spectral radius of $A$**: $|\lambda_{\max}(A)| = \rho^2$. 이것이 "autocorrelation at lag 1" = mixing rate.

**Spectral gap**: $1 - \rho^2$.

**결과**:
- $\rho = 0$: gap = 1, 1 step에서 mix (independent).
- $\rho = 0.5$: gap = 0.75, moderate.
- $\rho = 0.9$: gap = 0.19, slow.
- $\rho = 0.99$: gap = 0.0199, very slow.

**Mixing time**: $t_{\text{mix}}(\epsilon) \approx \frac{1}{\rho^2} \log(1/\epsilon)$ 근사.

**교훈**: Highly correlated Gaussian에서 Gibbs mixing $\sim 1/(1 - \rho^2)$. **Reparameterization** (rotate to principal axes)으로 decorrelate → gap 회복.

**AI 응용**: Deep Bayesian model에서 weight layers 간 correlation이 Gibbs mixing을 죽임 → **centered vs non-centered parameterization** 선택이 효율에 수십 배 영향.

</details>

**문제 3 (AI 연결)**. LDA(Latent Dirichlet Allocation)의 collapsed Gibbs sampler와 variational inference (VB) 중 어느 것이 scalability 우수한가?

<details>
<summary>해설</summary>

**Collapsed Gibbs (Griffiths-Steyvers 2004)**:
Integrate out $\theta_d, \phi_k$ (topic distributions). Only $z_{d, n}$ (topic assignment) sample.

Update: $p(z_{d,n} = k | z_{-(d,n)}, w) \propto (n^{d}_{k, -} + \alpha)(n^{w_{d,n}}_{k, -} + \beta) / (n^{t}_{k, -} + W\beta)$.

**복잡도**: Per iteration $O(N K)$ where $N$ = total tokens, $K$ = topics. Memory $O(D K + K W)$.

**Variational Bayes (VB, Blei 2003)**:
Mean-field: $q(\theta, \phi, z) = q(\theta) q(\phi) \prod q(z_{d,n})$. Iterative optimization of ELBO.

**복잡도**: Similar per-iter, but **parallel updates** possible (each $q(z_{d,n})$ independent).

**Online VB (Hoffman 2010)**:
Stochastic VI — mini-batch of documents. Scales to **billions of documents**.

**Scalability 비교**:
- **Collapsed Gibbs**: Hard to parallelize (sequential conditional). Memory-bound.
- **VB mean-field**: Parallelizable (SGD).
- **Online VB**: Streaming — infinite data.

**Quality trade-off**:
- Gibbs: Asymptotically exact, but slow convergence.
- VB: Fast, but biased (mean-field approximation).
- Online VB: Scales but biased.

**Modern alternatives**:
- **Neural Topic Models (NTM)**: VAE-style with reparameterization. Faster training.
- **Stochastic Gradient MCMC**: SG-HMC for LDA (Gan 2015).

**실전 choice**:
- Small data, correctness critical: Gibbs.
- Large data, scale priority: Online VB or Neural.
- 연구용, ground truth: Full Gibbs + long chain.

**연결**: Ch7-04 (HMC)는 scalability + exactness 둘 다 추구. Deep learning-era에서 VI가 대세이지만 Gibbs는 여전히 principled baseline.

</details>

---

<div align="center">

◀ [02. Metropolis-Hastings 알고리즘](./02-metropolis-hastings.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [04. Hamiltonian Monte Carlo (HMC)](./04-hamiltonian-mc.md)

</div>
