# 03. 연속시간 정상분포와 Detailed Balance

## 🎯 핵심 질문

- 연속시간 MC의 정상분포 조건이 왜 $\pi Q = 0$ (Forward equation의 고정점)인가?
- 연속시간 **detailed balance** $\pi_i q_{ij} = \pi_j q_{ji}$는 이산 버전과 어떻게 다르고 같은가?
- Reversible CTMC의 generator가 $\ell^2(\pi)$에서 **self-adjoint**이 되는 조건은?
- 이산 MC의 스펙트럴 결과(Ch2-04)가 연속 버전으로 어떻게 이어지는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**Langevin MCMC의 연속 버전**: $dX_t = -\nabla U dt + \sqrt{2} dB$는 reversible w.r.t. $\pi \propto e^{-U}$ (SDE Deep Dive Ch4). Forward eq = Fokker-Planck. 이산 MC의 detailed balance의 연속 generalization.

**Diffusion Model의 Forward Process**: OU process는 연속시간 reversible MC — $\pi$ = Gaussian. Score-SDE가 이를 활용.

**Continuous-time VAE**: Neural SDE latent dynamics의 equilibrium distribution 학습에 detailed balance 제약.

**Physics-inspired Neural Networks**: Gibbs 분포 → Boltzmann machine → RBM의 연속 시간 버전. Hamiltonian MCMC (Ch7-04)가 이 원리 확장.

---

## 📐 수학적 선행 조건

- [Ch4-01 ~ Ch4-02](./01-generator-q-matrix.md): Q-matrix, Forward/Backward
- [Ch2-03, Ch2-05](../ch2-discrete-markov/03-stationary-distribution.md): 이산 정상분포, detailed balance
- 함수해석: Self-adjoint operator

---

## 📖 직관적 이해

### $\pi Q = 0$의 의미

정상 상태 $X_t \sim \pi$ for all $t$ ⇔ $\dot \mu_t = 0 = \mu_t Q$ (Forward equation). 즉 **$\pi Q = 0$** (행벡터 $\pi$는 $Q$의 left-null vector).

이산의 $\pi(P - I) = 0$과 동등: $Q = P - I$ (embedding)에서 $\pi Q = \pi P - \pi$.

### Global balance vs Detailed balance

**Global**: $\pi Q = 0$ — 각 상태에서 "들어오는 rate = 나가는 rate":
$$\sum_{j \neq i} \pi_j q_{ji} = \pi_i q_i.$$

**Detailed**: 더 강한 조건, 각 쌍 $(i, j)$에서 balance:
$$\pi_i q_{ij} = \pi_j q_{ji}.$$

모든 pair balance이면 sum도 balance → global ← detailed (but not conversely).

### Reversible CTMC와 self-adjoint

$(Q, \pi)$ reversible ⇔ $Q$가 $\ell^2(\pi)$에서 self-adjoint:
$$\langle Qf, g\rangle_\pi = \langle f, Qg\rangle_\pi.$$

결과: 실고유값, 직교 고유함수 (Ch2-05와 병행).

---

## ✏️ 엄밀한 정의

### 정의 3.1 — 정상분포

$\pi$가 **CTMC $(X_t)$의 정상분포** ⇔ $X_0 \sim \pi \Rightarrow X_t \sim \pi$ for all $t$ ⇔ $\pi P(t) = \pi$ for all $t$ ⇔ $\pi Q = 0$.

### 정의 3.2 — 연속시간 Detailed Balance

$(\pi, Q)$가 **detailed balance**를 만족한다 ⇔
$$\pi_i q_{ij} = \pi_j q_{ji} \quad \forall i \neq j.$$

(대각은 자동: $\pi_i q_{ii} = \pi_i q_{ii}$)

---

## 🔬 정리와 증명

### 정리 3.1 — Detailed Balance ⇒ $\pi Q = 0$

$(\pi, Q)$ detailed balance이면 $\pi$는 정상분포.

*증명*.
$$(\pi Q)_j = \sum_i \pi_i q_{ij} = \pi_j q_{jj} + \sum_{i \neq j} \pi_i q_{ij} \stackrel{DB}{=} \pi_j q_{jj} + \sum_{i \neq j} \pi_j q_{ji} = \pi_j \sum_i q_{ji} = 0.$$
$\square$

### 정리 3.2 — Jump chain의 reversibility와의 관계

CTMC가 reversible (detailed balance) ⇔ jump chain $\tilde P_{ij} = q_{ij}/q_i$가 $\tilde \pi_i \propto \pi_i q_i$에 대해 reversible.

*증명*.
$\tilde \pi_i \tilde P_{ij} = \pi_i q_i \cdot q_{ij}/q_i = \pi_i q_{ij} \stackrel{DB}{=} \pi_j q_{ji} = \tilde \pi_j q_{ji}/q_j \cdot q_j = \tilde \pi_j \tilde P_{ji}$. $\square$

### 정리 3.3 — Reversible ⇔ $Q$ self-adjoint on $\ell^2(\pi)$

$(Q, \pi)$ reversible ⇔ $\langle Qf, g\rangle_\pi = \langle f, Qg\rangle_\pi$ for all $f, g \in \ell^2(\pi)$.

*증명*.
$$\langle Qf, g\rangle_\pi = \sum_i \pi_i (Qf)(i) g(i) = \sum_{i,j} \pi_i q_{ij} f(j) g(i).$$
Reversible (detailed balance)로 $\pi_i q_{ij} = \pi_j q_{ji}$: 위 합 = $\sum_{i,j} \pi_j q_{ji} f(j) g(i) = \sum_j \pi_j f(j) (Qg)(j) = \langle f, Qg\rangle_\pi$. $\square$

### 정리 3.4 — Reversible 체인의 스펙트럴 성질

Self-adjoint $Q$의 고유값이 모두 **실수**, 고유함수가 $\ell^2(\pi)$-직교.

정상분포 고유값은 $0$ (with 고유함수 $\mathbf{1}$), 다른 모두 $\leq 0$ (기약 하 strict $<0$).

### 정리 3.5 — 수렴률 with spectral gap

Reversible + 기약 CTMC:
$$\|\mu_t - \pi\|_{TV} \leq C e^{-\gamma t},$$
where $\gamma = -\max\{\text{Re}(\lambda) : \lambda \neq 0, \text{eigenvalue of } Q\}$ = **spectral gap**.

**이산 대응**: Ch2-04의 $|\lambda_2|^n$과 유사. 이산 $e^{-\gamma n} \approx |\lambda_2|^n$에서 $\gamma = -\log|\lambda_2|$.

### 정리 3.6 — Variational characterization of spectral gap

Reversible $Q$에서 spectral gap
$$\gamma = \min_{f : \mathbb{E}_\pi f = 0} \frac{\mathcal{E}(f, f)}{\text{Var}_\pi(f)},$$
$\mathcal{E}(f, f) = -\langle f, Qf\rangle_\pi = \frac{1}{2}\sum_{i, j} \pi_i q_{ij}(f(j) - f(i))^2$ (Dirichlet form).

이 variational form이 **Cheeger 부등식**, **functional inequalities**(Log-Sobolev)의 출발점.

---

## 💻 NumPy 구현 검증

### 실험 1 — $\pi Q = 0$ 풀이

```python
import numpy as np

Q = np.array([
    [-2.0,  1.5,  0.5],
    [ 1.0, -3.0,  2.0],
    [ 0.5,  1.0, -1.5],
])

# π Q = 0: Q^T π^T = 0의 null space
eigvals, eigvecs = np.linalg.eig(Q.T)
idx = np.argmin(np.abs(eigvals))   # eigenvalue 0 해당
pi = np.real(eigvecs[:, idx])
pi = pi / pi.sum()
print(f'정상분포 π = {pi}')
print(f'검증 πQ = {pi @ Q}')   # ≈ 0
```

### 실험 2 — Detailed balance 검증

```python
# 이 예제는 reversible인가?
def check_db(Q, pi):
    for i in range(len(pi)):
        for j in range(len(pi)):
            if i != j:
                if not np.isclose(pi[i] * Q[i, j], pi[j] * Q[j, i]):
                    return False
    return True

print(f'Detailed balance? {check_db(Q, pi)}')
# → False (non-reversible 예)

# Reversible 예: Q를 symmetric하게 설계
pi_target = np.array([0.5, 0.3, 0.2])
q_raw = np.array([
    [0, 0.6, 0.4],
    [1.0, 0, 0.5],
    [0.5, 0.8, 0],
])
# DB: π_i q_ij = π_j q_ji ⇒ q_ji = π_i q_ij / π_j
Q_rev = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        if i != j:
            rate = min(q_raw[i, j], pi_target[j] * q_raw[j, i] / pi_target[i])
            Q_rev[i, j] = rate
    Q_rev[i, i] = -Q_rev[i, :].sum()

# Make symmetric (Metropolis-style)
for i in range(3):
    for j in range(i+1, 3):
        shared = min(pi_target[i] * Q_rev[i, j], pi_target[j] * Q_rev[j, i]) 
        Q_rev[i, j] = shared / pi_target[i]
        Q_rev[j, i] = shared / pi_target[j]
    Q_rev[i, i] = -(Q_rev[i, :i].sum() + Q_rev[i, i+1:].sum())

print(f'수정된 Q:\n{Q_rev}')
print(f'DB 체크: {check_db(Q_rev, pi_target)}')

# 실고유값?
print(f'고유값: {np.linalg.eigvals(Q_rev)}')
```

### 실험 3 — 수렴률 측정

```python
from scipy.linalg import expm

mu_0 = np.array([1.0, 0.0, 0.0])
t_grid = np.linspace(0, 5, 100)
tv_dist = [0.5 * np.abs(mu_0 @ expm(t * Q) - pi).sum() for t in t_grid]

# Spectral gap
eigvals_Q = np.linalg.eigvals(Q)
gamma = -max(np.real(ev) for ev in eigvals_Q if not np.isclose(np.real(ev), 0))
print(f'Spectral gap γ = {gamma:.4f}')

import matplotlib.pyplot as plt
plt.semilogy(t_grid, tv_dist, label='실측 TV')
plt.semilogy(t_grid, np.exp(-gamma * t_grid), '--', label=f'e^(-γt), γ={gamma:.3f}')
plt.xlabel('t'); plt.ylabel('TV dist (log)')
plt.legend(); plt.grid(True, which='both', alpha=0.3); plt.show()
```

---

## 🔗 AI/ML 연결

**Langevin SDE의 Gibbs 정상분포**  
$dX = -\nabla U dt + \sqrt{2\beta^{-1}} dB$의 generator $\mathcal{L} = -\nabla U \cdot \nabla + \beta^{-1} \Delta$. $\pi \propto e^{-\beta U}$가 정상분포 (Fokker-Planck $\partial_t p = \nabla \cdot(\nabla U p) + \beta^{-1} \Delta p = 0$). Reversible w.r.t. $\pi$ → $\mathcal{L}$ self-adjoint on $L^2(\pi)$.

**OU Process (Ornstein-Uhlenbeck)**  
$dX = -\theta X dt + \sigma dB$, stationary $\pi = \mathcal{N}(0, \sigma^2/2\theta)$. SDE Deep Dive Ch3-03에서 해석해.

**Score SDE Forward Process**  
VP-SDE $dX_t = -\frac{1}{2}\beta(t) X dt + \sqrt{\beta(t)} dB$는 시간의존 OU → stationary 수렴. Non-reversible (time-varying $\beta$)이지만 **각 시각에서** 정상 분포 수렴 거동.

**Detailed Balance in Physics-informed AI**  
물리적 시스템(enzyme kinetics, chemical reactions)의 "equilibrium constraint"를 loss에 반영. NN이 detailed balance 깨지 않도록 정규화.

---

## ⚖️ 가정과 한계

**가정 — 기약성**  
여러 closed class이면 정상분포 유일하지 않음 (이산과 동일).

**한계 — 연속상태**  
연속상태 CTMC는 generator가 PDE 연산자. Detailed balance는 **reversed drift** 조건: $\pi \tilde{b} = \pi b$ with $\tilde{b}$ = time-reversed drift. SDE Deep Dive Ch4 참조.

**한계 — 계산**  
$\pi Q = 0$을 큰 상태공간에서 풀기 — iterative methods (GMRES 등). 고차원 Langevin 샘플링에서는 MCMC 자체 (Ch7).

---

## 📌 핵심 정리

| 결과 | 수식 |
|---|---|
| Stationary | $\pi Q = 0$ |
| Detailed balance | $\pi_i q_{ij} = \pi_j q_{ji}$ |
| DB ⇒ Stationary | 자동 |
| Reversible | ⇔ $Q$ self-adjoint on $\ell^2(\pi)$ |
| 수렴률 | $\|\mu_t - \pi\|_{TV} \leq C e^{-\gamma t}$, $\gamma$ = spectral gap |
| Dirichlet form | $\mathcal{E}(f,f) = -\langle f, Qf\rangle_\pi$ |

**한 줄 요약**: 연속시간 MC의 정상분포는 $\pi Q = 0$ (Forward equation의 고정점), detailed balance는 이의 강한 충분조건. Reversible일 때 $Q$가 $\ell^2(\pi)$에서 self-adjoint → 스펙트럴 분석 단순.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. 2-state CTMC $Q = \begin{pmatrix} -\alpha & \alpha \\ \beta & -\beta \end{pmatrix}$. 정상분포와 detailed balance 성립 여부는?

<details>
<summary>해설</summary>

**정상분포**: $\pi Q = 0$. $-\alpha \pi_0 + \beta \pi_1 = 0 \Rightarrow \pi_1 = \alpha \pi_0 / \beta$. 정규화: $\pi = (\beta/(\alpha+\beta), \alpha/(\alpha+\beta))$.

**Detailed balance**: $\pi_0 q_{01} = \pi_1 q_{10}$?  
LHS = $\beta/(\alpha+\beta) \cdot \alpha = \alpha\beta/(\alpha+\beta)$.  
RHS = $\alpha/(\alpha+\beta) \cdot \beta = \alpha\beta/(\alpha+\beta)$.  
**Equal** ✓ → detailed balance 성립.

**일반 원리**: 2-state CTMC는 **항상** reversible (automatically, unique pair (i,j)). 3-state 이상에서 cycle 있으면 DB가 깨질 수 있음.

</details>

**문제 2 (심화)**. 3-state cyclic CTMC $0 \to 1 \to 2 \to 0$ with rates $\lambda, \mu, \nu$. 정상분포를 구하고 DB 검증.

<details>
<summary>해설</summary>

$Q = \begin{pmatrix} -\lambda & \lambda & 0 \\ 0 & -\mu & \mu \\ \nu & 0 & -\nu \end{pmatrix}$.

**정상분포** $\pi Q = 0$:
$-\lambda \pi_0 + \nu \pi_2 = 0$  
$\lambda \pi_0 - \mu \pi_1 = 0$  
$\mu \pi_1 - \nu \pi_2 = 0$

$\pi_1 = \lambda \pi_0 / \mu$, $\pi_2 = \mu \pi_1 / \nu = \lambda \pi_0 / \nu$. 정규화: $\pi_0 (1 + \lambda/\mu + \lambda/\nu) = 1$.

**DB 체크**: $\pi_0 q_{01} = \lambda \pi_0$, but $\pi_1 q_{10} = 0$ (no direct transition 1 → 0). 일치 안 함 → **DB 실패**.

**결론**: Cyclic 구조는 globally balanced이지만 locally not — non-reversible.

**특성**: 복소 고유값 존재. 진동 성질. "probability flow around cycle"이 0이 아님 (time-reversal asymmetry의 signature).

</details>

**문제 3 (AI 연결)**. Score-based diffusion model의 forward process가 reversible인가? 그 의미는?

<details>
<summary>해설</summary>

**정확한 답**: VP-SDE forward $dX_t = -\frac{1}{2}\beta(t) X dt + \sqrt{\beta(t)} dB$는 **비동질**(non-homogeneous) 때문에 표준 reversibility 개념이 직접 적용 안 됨.

그러나 **infinitesimal 각 순간**: 만약 $\beta(t)$ 상수 $\beta$로 고정하면 → OU SDE → stationary $\mathcal{N}(0, I)$ w.r.t. reversible.

**의미**:
1. **Reverse SDE 존재**: Anderson(1982) — forward SDE의 reverse는 또 다른 SDE. Anderson 공식:
$$d\bar X_\tau = (-b + \sigma^2 \nabla \log p_\tau) d\tau + \sigma d\bar B_\tau.$$
이는 **"time reversal" of OU가 OU + score term**임을 보이는 것.

2. **Score term의 기원**: Reversibility가 완벽하지 않기 때문에 $\nabla \log p_t$가 reverse drift에 추가 → score matching의 필요성.

3. **정상분포**: Forward가 $t \to T$에서 $\mathcal{N}(0, I)$로 수렴 (large noise regime). 이것이 "Gaussian prior"의 자연스러운 선택 근거.

**DDPM의 이산 버전**:
$q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t} x_{t-1}, \beta_t I)$. 각 step은 1-state transition — detailed balance 의미 제한적. Reverse Markov chain $p(x_{t-1} | x_t)$ 학습이 **reverse SDE 이산 근사**.

**Continuous normalizing flow와 비교**: Deterministic flow는 "strict reversibility" (diffeomorphism). Diffusion은 stochastic → score로 reverse, 정보 부족을 NN으로 보완.

**연결**: SDE Deep Dive Ch6 (Reverse SDE, Anderson formula), Ch7 (Score matching, DSM)가 이 주제 파헤침. Detailed balance의 연속 generalization이 "time reversal symmetry"로 확장.

</details>

---

<div align="center">

◀ [02. Kolmogorov Forward/Backward 방정식](./02-kolmogorov-equations.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [04. Birth-Death 과정](./04-birth-death.md)

</div>
