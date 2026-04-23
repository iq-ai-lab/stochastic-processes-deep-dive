# 04. Hamiltonian Monte Carlo (HMC)

## 🎯 핵심 질문

- **Hamiltonian** $H(x, p) = U(x) + \frac{1}{2} p^T M^{-1} p$ — 왜 "potential + kinetic" 역학이 MCMC로 이어지는가?
- **Leapfrog integrator**의 심플렉틱(symplectic) + 시간가역(time-reversible) 성질이 MH acceptance 단순화에 어떻게 기여?
- **Gradient 사용**이 왜 RW-MH 대비 $O(d^{1/4})$ mixing 개선? (vs $O(d)$)
- NUTS (No-U-Turn Sampler)의 path length auto-tuning은 어떤 원리?

---

## 🔍 왜 HMC가 AI에서 중요한가

**현대 Bayesian Inference의 표준**: Stan, PyMC3, NumPyro — 모두 HMC/NUTS 기본. 직전 20년 Bayesian computing의 혁명.

**Deep Learning Bayesian NN**: SG-HMC (Chen 2014), 변형들 — gradient 활용 + MCMC 샘플링.

**물리 기반 AI**: Lattice QCD, molecular dynamics simulation의 core — HMC가 주 sampler.

**Simulated annealing 확장**: Hamiltonian dynamics + temperature → generalized Hamiltonian MC.

---

## 📐 수학적 선행 조건

- [Ch7-01, Ch7-02](./01-mcmc-idea.md): MCMC, MH
- [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive): Hamiltonian mechanics, gradient

---

## 📖 직관적 이해

### 물리적 직관

**Potential energy** $U(x) = -\log \pi(x)$. Target distribution = "Gibbs 분포 at unit temperature".

**Momentum** $p$: Auxiliary 변수. "공을 굴리는 속도".

**Hamiltonian**: $H(x, p) = U(x) + K(p)$ where $K = \frac{1}{2} p^T M^{-1} p$ (kinetic energy, $M$ = mass matrix).

**Joint distribution**: $\pi(x, p) \propto e^{-H(x, p)} = e^{-U(x)} e^{-K(p)} \propto \pi(x) \mathcal{N}(p; 0, M)$.

즉 $x \sim \pi$ ($p$ 독립 Gaussian 보조).

### Hamiltonian 동역학

Hamilton's equations:
$\dot x = \partial H / \partial p = M^{-1} p$
$\dot p = -\partial H / \partial x = -\nabla U(x)$

**Key properties**:
1. **Energy conservation**: $H(x(t), p(t)) = $ const.
2. **Symplectic**: Phase space volume 보존.
3. **Time-reversibility**: Flip momentum $p \to -p$ → 시간 역전.

→ "Constant $H$ 유지하는 deterministic path". MH with this proposal: acceptance = 1 (energy 같음). 문제는 **exact dynamics 풀이 불가**.

### Leapfrog Integrator

실전: **Leapfrog** — explicit numerical integrator.
$$p_{n+1/2} = p_n - \frac{\eta}{2} \nabla U(x_n)$$
$$x_{n+1} = x_n + \eta M^{-1} p_{n+1/2}$$
$$p_{n+1} = p_{n+1/2} - \frac{\eta}{2} \nabla U(x_{n+1})$$

**Properties**:
- **Symplectic**: 정확히 volume-preserving.
- **Time-reversible**: Flip $p$ → 역전.
- **$O(\eta^2)$** local error in $H$.

**MH acceptance**: Exact dynamics가 아니므로 $H$가 약간 변동 — MH 필요. Acceptance:
$$\alpha = \min(1, e^{-H(x', p') + H(x, p)}).$$

Energy 차이가 작으면 (leapfrog 정확) → acceptance 높음.

### Why faster than RW-MH

**RW-MH**: 각 step $\sqrt{dt}$ 움직임 ($\sqrt{d}$ dim-dependent scaling). High-dim $\sigma \sim d^{-1/2}$ for 0.234 acceptance → 전체 path length $\sim 1$ per iter → $O(d)$ iter for mixing.

**HMC**: Gradient $\nabla U$ 활용하여 "correct direction"으로 $L$ leapfrog steps. Effective path length $L \cdot \eta$ with $\eta \sim d^{-1/4}$ (Beskos 2013) → $O(d^{1/4})$ iter.

**Dramatic improvement** in high dim.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Hamiltonian

$H(x, p) = U(x) + \frac{1}{2} p^T M^{-1} p$

- $U(x) = -\log \tilde\pi(x)$ (potential)
- $M$: positive definite mass matrix (usually identity)
- $K(p) = \frac{1}{2} p^T M^{-1} p$ (kinetic)

### 정의 4.2 — HMC Algorithm

Iteration $n$: given $x^{(n)}$,
1. **Momentum refresh**: $p \sim \mathcal{N}(0, M)$.
2. **Leapfrog** $L$ steps with step size $\eta$: $(x, p) \to (x', p')$.
3. **MH acceptance**:
$\alpha = \min(1, e^{H(x, p) - H(x', p')})$.
4. If accept: $x^{(n+1)} = x'$. Else: $x^{(n+1)} = x^{(n)}$.
5. Discard $p'$ (새로운 iteration에서 re-sample).

### 정의 4.3 — Symplectic Map

$\Phi : \mathbb{R}^{2d} \to \mathbb{R}^{2d}$ is **symplectic**: $\Phi_*\omega = \omega$ for the canonical symplectic form $\omega = \sum dp_i \wedge dx_i$.

동등: Jacobian $D\Phi$가 $J$에 대해 "symplectic transformation" (det = 1 등). **Volume-preserving**.

---

## 🔬 정리와 증명

### 정리 4.1 — HMC가 $\pi(x)$를 invariant하게 보존

HMC chain의 transition kernel이 $x$-marginal $\pi$를 보존.

### 증명 스케치

Joint $\pi(x, p) = \pi(x) \mathcal{N}(p; 0, M)$. 증명 요소:
1. **Momentum refresh**: $p$만 update → $\pi(x, p)$의 marginal $p$가 $\mathcal{N}(0, M)$로 유지 ($x$ 고정).
2. **Leapfrog + MH**: $(x, p) \to (x', p')$ update. Symplectic + time-reversible → MH acceptance로 detailed balance satisfied for joint $\pi(x, p)$.

**Joint $\pi(x, p)$가 invariant** → marginal $\pi(x)$ invariant.

**Detailed balance with MH**:
$\pi(x, p) P((x, p), (x', p')) = \pi(x) e^{-K(p)} \cdot \min(1, e^{H(x,p) - H(x', p')}) / Z$.

대칭으로 $\pi(x', p') P((x', p'), (x, p))$ — 같음 (time-reversibility 필요). $\square$

### 정리 4.2 — Leapfrog의 심플렉틱·시간가역 성질

Leapfrog integrator:
1. **Symplectic**: Jacobian determinant = 1 (volume-preserving).
2. **Time-reversible**: Applying with $(-p)$ then negating $p$ inverts the map.

*증명 스케치*.

**Symplectic**:
각 leapfrog step을 3 elementary symplectic maps로 분해:
- $S_1$: $p \to p - \frac{\eta}{2} \nabla U(x)$ (update $p$ only, Jacobian block triangular with identity on $x$)
- $S_2$: $x \to x + \eta M^{-1} p$ (update $x$ only)
- $S_3$: $p \to p - \frac{\eta}{2} \nabla U(x)$ (another half-step on $p$)

각 elementary map의 Jacobian = 1 → composition Jacobian = 1. Full symplectic structure preserved.

**Time-reversibility**: Equations 역방향 대칭 — $(x, p) \to (x', p')$의 역 = $(x', -p') \to (x, -p)$. $\square$

### 정리 4.3 — Energy Conservation (asymptotic)

Leapfrog integrator의 per-step Hamiltonian error:
$|H(x_{n+1}, p_{n+1}) - H(x_n, p_n)| = O(\eta^2)$ locally, $O(\eta^2)$ over long times (not accumulating) — "symplectic integrator의 장점".

(Non-symplectic integrator, 예: Euler, $O(\eta)$ accumulating drift.)

### 정리 4.4 — HMC Optimal Scaling (Beskos 2013)

Target $\pi(x) = \prod \tilde\pi_i(x_i)$ (product form). Optimal leapfrog step size:
$\eta \propto d^{-1/4}$, path length $L\eta = O(1)$, acceptance rate $\approx 0.65$.

Effective mixing time $O(d^{1/4})$ per iteration → dimension-adaptive.

---

## 💻 NumPy 구현 검증

### 실험 1 — HMC for 2D Gaussian

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# Target: N(0, Σ), Σ = [[1, 0.8], [0.8, 1]]
Sigma = np.array([[1.0, 0.8], [0.8, 1.0]])
Sigma_inv = np.linalg.inv(Sigma)

def U(x):
    return 0.5 * x @ Sigma_inv @ x

def grad_U(x):
    return Sigma_inv @ x

def leapfrog(x, p, eta, L, grad_U):
    p = p - eta/2 * grad_U(x)
    for _ in range(L - 1):
        x = x + eta * p
        p = p - eta * grad_U(x)
    x = x + eta * p
    p = p - eta/2 * grad_U(x)
    return x, p

def hmc_step(x, eta=0.1, L=20):
    p = rng.standard_normal(len(x))
    H_old = U(x) + 0.5 * p @ p
    x_new, p_new = leapfrog(x, p, eta, L, grad_U)
    H_new = U(x_new) + 0.5 * p_new @ p_new
    if np.log(rng.random()) < H_old - H_new:
        return x_new, True
    return x, False

# Run
x = np.zeros(2)
samples = []
accepts = 0
n_iter = 5000
for _ in range(n_iter):
    x, acc = hmc_step(x)
    samples.append(x.copy())
    accepts += acc
samples = np.array(samples)
print(f'Acceptance rate: {accepts/n_iter:.3f}')

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.3)
plt.title('HMC samples')
plt.grid(True, alpha=0.3); plt.xlabel('x1'); plt.ylabel('x2')

plt.subplot(1, 2, 2)
plt.plot(samples[:500, 0], samples[:500, 1], '-o', markersize=3, alpha=0.5)
plt.title('HMC trajectory (500 steps)')
plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# Covariance 추정
print(f'\nMean: {samples.mean(axis=0)}')
print(f'Cov:\n{np.cov(samples, rowvar=False)}')
```

### 실험 2 — HMC vs Random-walk MH comparison

```python
# Same target, compare mixing speed

# RW-MH
def mh_step(x, step=1.0):
    y = x + rng.normal(0, step, len(x))
    log_alpha = U(x) - U(y)
    if np.log(rng.random()) < log_alpha:
        return y, True
    return x, False

# Run both
x_rw = np.zeros(2); rw_samples = []
for _ in range(n_iter):
    x_rw, _ = mh_step(x_rw, step=1.0)
    rw_samples.append(x_rw.copy())
rw_samples = np.array(rw_samples)

# Autocorrelation comparison
def autocorr_1d(x, max_lag=50):
    x = x - x.mean()
    return np.array([np.mean(x[:-k]*x[k:])/x.var() if k>0 else 1 
                     for k in range(max_lag)])

plt.plot(autocorr_1d(samples[:, 0]), label='HMC')
plt.plot(autocorr_1d(rw_samples[:, 0]), label='RW-MH')
plt.xlabel('lag'); plt.ylabel('autocorr')
plt.title('Autocorrelation: HMC vs RW-MH')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
# HMC autocorr decays much faster
```

### 실험 3 — Ill-conditioned 분포

```python
# Ill-conditioned Σ: highly elongated
Sigma_ic = np.array([[1.0, 0.99], [0.99, 1.0]])
Sigma_inv_ic = np.linalg.inv(Sigma_ic)

def U_ic(x): return 0.5 * x @ Sigma_inv_ic @ x
def grad_U_ic(x): return Sigma_inv_ic @ x

# HMC 실행
x = np.zeros(2); samples = []
for _ in range(3000):
    p = rng.standard_normal(2)
    x_new = x.copy(); p_new = p.copy()
    # leapfrog L=50 steps
    p_new -= 0.1/2 * grad_U_ic(x_new)
    for _ in range(49):
        x_new += 0.1 * p_new
        p_new -= 0.1 * grad_U_ic(x_new)
    x_new += 0.1 * p_new
    p_new -= 0.1/2 * grad_U_ic(x_new)
    
    H_old = U_ic(x) + 0.5 * p @ p
    H_new = U_ic(x_new) + 0.5 * p_new @ p_new
    if np.log(rng.random()) < H_old - H_new:
        x = x_new
    samples.append(x.copy())
samples = np.array(samples)

plt.scatter(samples[:, 0], samples[:, 1], s=1)
plt.title('HMC on ill-conditioned Gaussian (ρ=0.99)')
plt.grid(True, alpha=0.3); plt.show()
# HMC는 elongated 구조도 잘 탐색 (gradient 정보 활용)
```

---

## 🔗 AI/ML 연결

**PyMC / Stan / NumPyro**  
HMC/NUTS backend. Probabilistic programming의 standard inference. GP, hierarchical models, state-space models 모두 HMC로.

**NUTS (No-U-Turn Sampler, Hoffman-Gelman 2014)**  
HMC path length $L$을 자동 tuning. "U-turn" 감지 시 stop. Stan's default.

**SG-HMC (Chen 2014)**  
HMC + mini-batch gradient + friction term. Scalable BNN inference. Langevin dynamics의 일반화.

**Relativistic MC (Livingstone 2017)**  
Kinetic energy를 heavy-tailed로 설정 → momentum이 jump 가능. Multi-modal posteriors에 유리.

**Riemannian HMC (Girolami-Calderhead 2011)**  
Mass matrix $M$을 local geometry $M(x) = G(x)$ (Fisher information 등)로 → elongated posteriors에 adapt.

---

## ⚖️ 가정과 한계

**가정 — Differentiable target**  
$\nabla U = -\nabla \log \pi$ 필요. Discrete parameters에서는 직접 적용 불가 → Reversible Jump MCMC.

**한계 — Step size tuning 민감**  
$\eta$ 너무 작으면 slow, 너무 크면 instability. NUTS가 자동화하지만 부담.

**한계 — Multi-modal**  
HMC가 smooth 탐색 → well-separated modes 간 이동 어려움. Replica exchange / parallel tempering 조합.

**한계 — Path length $L$**  
고정 $L$은 특정 regime에 맞지 않을 수 있음. NUTS가 dynamic adjustment.

---

## 📌 핵심 정리

| 개념 | 수식 / 설명 |
|---|---|
| Hamiltonian | $H = U + \frac{1}{2} p^T M^{-1} p$ |
| Target | $\pi(x) \propto e^{-U(x)}$ |
| Joint | $\pi(x, p) \propto e^{-H(x, p)}$ |
| Leapfrog | Symplectic + time-reversible |
| MH acceptance | $\min(1, e^{H_{\text{old}} - H_{\text{new}}})$ |
| Scaling | $\eta \propto d^{-1/4}$, mixing $O(d^{1/4})$ |
| NUTS | Auto path length |
| Gradient use | RW보다 dramatically 효율 |

**한 줄 요약**: HMC는 "gradient 활용 + Hamiltonian 역학 + leapfrog 이산화"로 RW-MH의 $O(d)$ → $O(d^{1/4})$ scaling 개선. Symplectic integrator의 volume-preservation + reversibility가 MH acceptance 단순화. 현대 Bayesian inference의 표준.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. HMC의 "Momentum resampling"이 왜 $p \sim \mathcal{N}(0, M)$인가? 왜 매 iteration reset?

<details>
<summary>해설</summary>

**왜 Gaussian momentum**:
Joint $\pi(x, p) \propto e^{-H(x, p)} = e^{-U(x)} \cdot e^{-\frac{1}{2} p^T M^{-1} p}$.

Factorizes. $p | x \sim \mathcal{N}(0, M)$ (independent).

**Target marginal $\pi(x)$ 보존**:
$(x, p) \sim \pi$일 때, momentum refresh $p \leftarrow \mathcal{N}(0, M)$는 $\pi$의 conditional을 다시 sampling — $x$는 변화 없지만 $p$ 재분배.

**왜 reset 필요**:
Leapfrog + MH만 계속하면 (energy conservation + deterministic path) chain이 **symmetric 주위 orbit**. 단일 energy level 위에서만 순회 — $\pi$ 전체 탐색 못 함.

Momentum reset = "energy level 변경" → 다른 $(x, p)$ 영역 탐색 → 전체 $\pi$ 수렴.

**직관**: 포탄 쏘기 비유.
- Leapfrog: 한 번 쏘고 deterministic path.
- MH accept/reject: destination 결정.
- Momentum refresh: "새로운 각도로 다시 쏘기".

**변형 — Langevin Monte Carlo (MALA)**:
$L = 1$ HMC + refresh every step = Langevin dynamics. Continuous limit = Langevin SDE (SDE Deep Dive Ch4).

</details>

**문제 2 (심화)**. Leapfrog의 symplectic 성질을 명시적 Jacobian 계산으로 증명하라.

<details>
<summary>해설</summary>

**Leapfrog map**: $(x, p) \to (x', p')$ with
$p_h = p - \frac{\eta}{2} \nabla U(x)$  
$x' = x + \eta M^{-1} p_h$  
$p' = p_h - \frac{\eta}{2} \nabla U(x')$

**Jacobian** of full map $(x, p) \to (x', p')$:

Step 1 $(x, p) \to (x, p_h)$: $p_h$ depends on $p$ (identity), and $x$ via $\nabla U$.
$J_1 = \begin{pmatrix} I & 0 \\ -\frac{\eta}{2} \nabla^2 U(x) & I \end{pmatrix}$.

$\det J_1 = 1$ (block triangular with identity blocks).

Step 2 $(x, p_h) \to (x', p_h)$: $x' = x + \eta M^{-1} p_h$.
$J_2 = \begin{pmatrix} I & \eta M^{-1} \\ 0 & I \end{pmatrix}$.

$\det J_2 = 1$.

Step 3 $(x', p_h) \to (x', p')$: $p' = p_h - \frac{\eta}{2} \nabla U(x')$.
$J_3 = \begin{pmatrix} I & 0 \\ -\frac{\eta}{2} \nabla^2 U(x') & I \end{pmatrix}$.

$\det J_3 = 1$.

**Full Jacobian**: $J = J_3 J_2 J_1$. Determinant:
$\det J = \det J_3 \cdot \det J_2 \cdot \det J_1 = 1 \cdot 1 \cdot 1 = 1$.

**Volume-preserving** ✓ — 이것이 symplectic의 핵심 결과.

**의의**: MH acceptance에서 Jacobian term $|\det J|$이 1 → acceptance = $\min(1, e^{H_{\text{old}} - H_{\text{new}}})$ 단순화. 비-심플렉틱 integrator에서는 Jacobian correction 필요.

**Non-symplectic Euler**:
$(x, p) \to (x + \eta M^{-1} p, p - \eta \nabla U(x))$.
$J = \begin{pmatrix} I & \eta M^{-1} \\ -\eta \nabla^2 U & I \end{pmatrix}$.
$\det J = 1 + O(\eta^2)$ — approximately 1 but not exact. Long times에서 error accumulates → Hamiltonian drift → incorrect target.

**Leapfrog의 Exact symplecticity**가 HMC의 long-term accuracy의 수학적 근거.

</details>

**문제 3 (AI 연결)**. BNN training with HMC: parameter $\theta \in \mathbb{R}^{10^6}$. Naive HMC의 per-iteration cost vs SG-HMC의 복잡도 비교.

<details>
<summary>해설</summary>

**Naive HMC for BNN**:
$U(\theta) = -\log p(\theta) - \sum_{i=1}^N \log p(y_i | x_i, \theta)$.
$\nabla U$ requires **full dataset** + backprop.

**Per-iteration cost**:
- Gradient: $O(N \cdot \text{forward/backward})$ where $N$ = dataset size.
- Leapfrog steps $L$: $O(L \cdot N \cdot \text{FLOPs per forward-backward})$.
- 한 HMC iter: $O(LN F)$.

$N = 10^6$, $L = 100$ → $10^8 F$ per iter. 수백 iter면 $10^{10}$ FLOPs.

**Stochastic Gradient HMC (SG-HMC, Chen 2014)**:
$\nabla U \approx -\log p(\theta) - \frac{N}{|B|} \sum_{i \in B} \log p(y_i | x_i, \theta)$ (mini-batch).

$|B| = 64$ → per-iter cost $O(L \cdot |B| \cdot F)$ — dataset size에 독립.

**Friction term**:
$d\theta = M^{-1} p dt$, $dp = -\nabla U dt - CM^{-1} p dt + \sqrt{2 C} dW$ (Langevin with friction).

Mini-batch noise를 "natural friction"으로 간주 → correct target distribution.

**Step size + $L$ tuning**:
- 작은 $\eta$, 작은 $L$: mini-batch noise 처리 가능.
- Running averages of gradient (preconditioning) 도움.

**실전 변형**:
- **pSGLD** (preconditioned SGLD): Adaptive step size (RMSProp-like).
- **SGNHT** (Stochastic Gradient Nose-Hoover Thermostat): Friction auto-adapt.

**복잡도 비교**:
- Naive HMC: $O(NL)$ per iter → intractable.
- SG-HMC: $O(|B| L)$ per iter → scalable.
- VI (reference): $O(|B|)$ per iter — fastest but biased.

**Quality trade-off**:
- Naive HMC: Exact, but slow.
- SG-HMC: Biased (mini-batch + discretization), but scalable. 대부분 practitioners 사용.
- Hybrid: SGMCMC + few exact HMC finalization steps.

**현대 BNN 연구 방향**:
- **Normalizing flow posterior**: VI with expressive q.
- **Deep Ensembles**: Multiple point estimates (uncertainty proxy).
- **SWA-G** (Stochastic Weight Averaging-Gaussian): Gaussian approximation from SGD trajectory.

**연결**: 
- HMC는 classical "exact" method, scalability 도전.
- SG-MCMC는 "approximate but scalable" — SDE Deep Dive Ch4 (Langevin) 이론 활용.
- 현대 BNN은 HMC + NN 기법의 hybrid로 진화.

</details>

---

<div align="center">

◀ [03. Gibbs Sampler](./03-gibbs-sampler.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [05. MCMC 수렴 진단과 혼합 시간](./05-mixing-diagnostics.md)

</div>
