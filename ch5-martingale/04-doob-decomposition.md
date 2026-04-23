# 04. Doob 분해와 이차변분

## 🎯 핵심 질문

- 임의의 adapted 과정을 "마팅게일 + predictable 증가 과정"으로 분해하는 **Doob 분해**는 어떻게 유도되는가?
- 마팅게일의 **이차변분** $\langle M\rangle_n$은 무엇이고, 왜 $M_n^2 - \langle M\rangle_n$이 마팅게일인가?
- 이 결과가 **이토 적분**과 **확률해석**의 기초를 놓는 이유는?
- 이산 버전이 연속 BM (Ch6-05)의 이차변분 $\langle B\rangle_t = t$로 어떻게 이어지는가?

---

## 🔍 왜 이 결과가 AI에서 중요한가

**SDE Deep Dive로의 교량**: Ch6의 BM 이차변분 $\langle B\rangle_t = t$이 이 이산 버전의 연속 한계. $(dB)^2 = dt$의 원천 — 이토 적분·이토 공식의 기반.

**Variance analysis of SGD**: $\theta_n^2$의 이차변분이 gradient noise variance를 누적 → convergence rate 분석의 핵심.

**Estimation of diffusion coefficient**: Realized variance $\sum_{i}(X_{t_{i+1}} - X_{t_i})^2$이 $\int \sigma^2 ds$ 추정 — financial econometrics의 핵심 기법.

**Online learning stability**: Regret bound의 variance term이 이차변분으로 표현. Azuma vs Freedman inequality의 차이.

---

## 📐 수학적 선행 조건

- [Ch5-01 ~ Ch5-03](./01-martingale-definition.md): 마팅게일, 수렴, OST
- [Ch1-04](../ch1-foundations/04-filtration.md): Predictable process

---

## 📖 직관적 이해

### Doob 분해의 아이디어

임의 adapted integrable 과정 $X_n$을:
$$X_n = X_0 + M_n + A_n$$
형태로 분해하자, $M$ martingale, $A$ predictable increasing (submartingale이면).

**$A$ 의 정의**: Predictable part $A_n = \sum_{k=1}^n \mathbb{E}[X_k - X_{k-1} | \mathcal{F}_{k-1}]$ — "알 수 있는 drift".

**$M$ 의 정의**: $M_n = X_n - X_0 - A_n$ — "uncertain fluctuation".

이 분해는 **유일** (predictable $A$, $M$ martingale이라는 조건 하).

### 이차변분의 직관

$M_n$ martingale → $M_n^2$이 submartingale (Jensen). Doob 분해:
$$M_n^2 = M_0^2 + \text{(martingale)} + \langle M\rangle_n.$$

$\langle M\rangle_n$ = "$M_n^2$의 predictable 성장 부분" = **quadratic variation (이차변분)**.

**해석**: $M$의 "variance가 시간에 따라 누적된" 값.

### 왜 이것이 이토 적분의 기반인가

연속 BM에서:
$$\sum_{i}(B_{t_{i+1}} - B_{t_i})^2 \to t \quad (n \to \infty, \Delta t \to 0).$$

이는 $\langle B\rangle_t = t$ — BM의 이차변분이 **결정론적** $t$. 이것이 $(dB)^2 = dt$의 수학적 내용 (SDE Ch2-02에서 자세히).

이산 버전: $\langle M\rangle_n = \sum_k \mathbb{E}[(M_k - M_{k-1})^2 | \mathcal{F}_{k-1}]$ — "conditional variance의 누적".

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Doob 분해

$X_n$ adapted integrable. **Doob 분해**: $X_n = X_0 + M_n + A_n$, with
- $M$ martingale, $M_0 = 0$
- $A$ predictable ($A_n \in \mathcal{F}_{n-1}$), $A_0 = 0$
- $A$ increasing if $X$ submartingale

### 정의 4.2 — 이차변분 (Quadratic Variation)

$M_n$ square-integrable martingale. **이차변분**:
$$\langle M\rangle_n = \sum_{k=1}^n \mathbb{E}[(M_k - M_{k-1})^2 | \mathcal{F}_{k-1}].$$

$\langle M\rangle$는 predictable, increasing, $\langle M\rangle_0 = 0$.

### 정의 4.3 — 공분산 변분

두 square-integrable martingales $M, N$. **공분산 변분** (covariation):
$$\langle M, N\rangle_n = \sum_{k=1}^n \mathbb{E}[(M_k - M_{k-1})(N_k - N_{k-1}) | \mathcal{F}_{k-1}].$$

$\langle M, M\rangle = \langle M\rangle$ (이차변분). Polarization:
$$\langle M, N\rangle = \frac{1}{2}[\langle M + N\rangle - \langle M\rangle - \langle N\rangle].$$

---

## 🔬 정리와 증명

### 정리 4.1 — Doob 분해의 존재·유일성

임의 adapted integrable $X$에 대해 Doob 분해 $X_n = X_0 + M_n + A_n$ (with $M$ martingale, $A$ predictable)이 **유일하게** 존재.

*증명*.

**존재**: $A_n = \sum_{k=1}^n \mathbb{E}[X_k - X_{k-1} | \mathcal{F}_{k-1}]$로 정의 (predictable). $M_n = X_n - X_0 - A_n$으로 정의.

$M$ adapted, integrable. Martingale property:
$$\mathbb{E}[M_{n+1} - M_n | \mathcal{F}_n] = \mathbb{E}[X_{n+1} - X_n - (A_{n+1} - A_n) | \mathcal{F}_n] = \mathbb{E}[X_{n+1} - X_n | \mathcal{F}_n] - \mathbb{E}[X_{n+1} - X_n | \mathcal{F}_n] = 0.$$
$\square$

**유일성**: 두 분해 $X = X_0 + M + A = X_0 + M' + A'$. $M - M' = A' - A$. 왼쪽 martingale, 오른쪽 predictable. Predictable martingale must be constant (마팅게일 + predictable ⇒ $\mathbb{E}[M_{n+1} - M_n | \mathcal{F}_n] = M_{n+1} - M_n$, 하지만 martingale property로 이것이 0). 따라서 $M - M' = $ constant $= 0$ (both start at 0). $\square$

### 정리 4.2 — Submartingale의 Doob 분해

$X$ submartingale이면 Doob 분해의 $A$가 **increasing** (비감소).

*증명*.
$A_{n+1} - A_n = \mathbb{E}[X_{n+1} - X_n | \mathcal{F}_n] \geq 0$ (submartingale property). $\square$

### 정리 4.3 — 이차변분의 존재

$M$ square-integrable martingale ($\sup_n \mathbb{E}[M_n^2] < \infty$). $M^2$이 submartingale (Jensen). Doob 분해 $M_n^2 = M_0^2 + N_n + \langle M\rangle_n$ with $N$ martingale, $\langle M\rangle$ predictable increasing.

$\langle M\rangle$이 정의 4.2의 형태임을 보여보자.

*증명*.
Doob 분해에서 $\langle M\rangle_n = \sum_{k=1}^n \mathbb{E}[M_k^2 - M_{k-1}^2 | \mathcal{F}_{k-1}]$.

$\mathbb{E}[M_k^2 - M_{k-1}^2 | \mathcal{F}_{k-1}] = \mathbb{E}[(M_k - M_{k-1})^2 + 2 M_{k-1}(M_k - M_{k-1}) | \mathcal{F}_{k-1}]$. Martingale 성질로 두번째 항 = 0. 결과:
$$\langle M\rangle_n = \sum_{k=1}^n \mathbb{E}[(M_k - M_{k-1})^2 | \mathcal{F}_{k-1}].$$
정의 4.2. $\square$

### 정리 4.4 — $M_n^2 - \langle M\rangle_n$ is martingale

Square-integrable martingale $M$에 대해 $M_n^2 - \langle M\rangle_n$은 martingale.

*증명*.
Doob 분해: $M_n^2 - M_0^2 = N_n + \langle M\rangle_n$, $N$ martingale. 따라서 $M_n^2 - \langle M\rangle_n = M_0^2 + N_n$이 martingale. $\square$

### 정리 4.5 — $L^2$ 수렴과 이차변분

Square-integrable martingale $M$이 **$L^2$-bounded** ($\sup_n \mathbb{E}[M_n^2] < \infty$)이면:
$$\mathbb{E}[\langle M\rangle_\infty] = \lim_n \mathbb{E}[\langle M\rangle_n] = \sup_n \mathbb{E}[M_n^2] - M_0^2.$$

특히 $\langle M\rangle_\infty < \infty$ a.s.

*증명*.
$\mathbb{E}[M_n^2] = M_0^2 + \mathbb{E}[\langle M\rangle_n]$ (martingale $N$의 expectation은 상수).

$\sup$ 취함. MCT로 limit. $\square$

### 정리 4.6 — 이차변분의 대안 정의

$M$ square-integrable martingale. 분할 $\pi = \{0 = t_0 < t_1 < \cdots < t_n\}$ on $[0, T]$ (이산에선 $t_i = i$)의 **quadratic variation**:
$$[M]_n := \sum_{k=1}^n (M_k - M_{k-1})^2.$$

이는 predictable이 아니지만, $\langle M\rangle$과 관계:
$$[M]_n - \langle M\rangle_n \text{ is martingale}.$$

즉 $[M]$이 $\langle M\rangle$의 "noisy version" — 기댓값이 같음.

*증명*.
$[M]_n - \langle M\rangle_n = \sum_k (M_k - M_{k-1})^2 - \sum_k \mathbb{E}[(M_k - M_{k-1})^2 | \mathcal{F}_{k-1}]$.

각 항 $(M_k - M_{k-1})^2 - \mathbb{E}[(M_k - M_{k-1})^2 | \mathcal{F}_{k-1}]$이 martingale difference. Sum은 martingale. $\square$

> **연속시간 일반화**: $[M]_t \to \langle M\rangle_t$ in probability as $|\pi| \to 0$. BM에서 $[B]_t = \langle B\rangle_t = t$ (정확히 결정론적, 이 신비가 $(dB)^2 = dt$).

---

## 💻 NumPy 구현 검증

### 실험 1 — Simple Random Walk의 Doob 분해

```python
import numpy as np
rng = np.random.default_rng(0)

# X_n = S_n² (submartingale)
# Doob 분해: X_n = M_n + A_n, A_n = sum of conditional increments
# E[S_{n+1}² - S_n² | F_n] = E[(S_n + ξ)² - S_n² | F_n] = E[2 S_n ξ + ξ²] = 0 + 1 = 1
# 따라서 A_n = n (deterministic increasing) — 이는 simple RW의 경우
# M_n = S_n² - n

N = 1000
xi = rng.choice([-1, 1], size=N)
S = np.cumsum(xi)
S = np.concatenate([[0], S])

# X_n = S_n²
X = S**2
# A_n = n (이론)
A = np.arange(len(X))
# M_n = X_n - A_n
M = X - A

# M은 martingale여야 → E[M] = 0
print(f'E[M_n] (평균 over 시간, single path):')
# 다수 paths로 E 추정
n_paths = 10000
xi_multi = rng.choice([-1, 1], size=(n_paths, N))
S_multi = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(xi_multi, axis=1)], axis=1)
M_multi = S_multi**2 - np.arange(N+1)
for n in [100, 500, 1000]:
    print(f'  E[M_{n}] = {M_multi[:, n].mean():.4f} (이론: 0)')
# 모두 0에 가까움 → M_n = S_n² - n 이 martingale
```

### 실험 2 — 이차변분 계산

```python
# SRW의 이차변분 ⟨S⟩_n = sum_k E[(S_k - S_{k-1})² | F_{k-1}]
# (ΔS)² = ξ² = 1 (항상), E[ξ²|F_{k-1}] = 1
# ⟨S⟩_n = n

# 경험적 variation [S]_n = sum (ΔS)² = sum 1 = n (deterministic in SRW)
N = 100
dS = xi_multi[0, :N]
bracket_S = np.cumsum(dS**2)
print(f'경험 [S]_N = {bracket_S[-1]}, 이론 ⟨S⟩_N = {N}')
# 정확히 같음 — SRW는 (ΔS)² = 1 항상

# 차이 [S] - ⟨S⟩는 SRW에서 0 (trivial case)
# 하지만 ξ가 N(0,1)이면 (ΔS)² 랜덤, 평균 1
# → [S] ≠ ⟨S⟩ (개별 sample에서), 기댓값은 같음

# Gaussian increments case
dS_gauss = rng.standard_normal(N)
bracket_Sg = np.cumsum(dS_gauss**2)
print(f'\nGaussian: [S]_N = {bracket_Sg[-1]:.4f}, ⟨S⟩_N = {N} (이론 E[ξ²]=1)')
# [S] 랜덤, 기댓값 = N
```

### 실험 3 — 이차변분 $\to t$ (연속 한계 preview)

```python
import matplotlib.pyplot as plt

# BM 이산화: (ΔB)² ≈ dt → 누적 [B]_n ≈ n·dt = t
T = 1.0
for N_steps in [10, 100, 1000, 10000]:
    dt = T / N_steps
    dB = rng.standard_normal(N_steps) * np.sqrt(dt)
    QV = np.cumsum(dB**2)
    print(f'N={N_steps}: [B]_T = {QV[-1]:.4f}, 이론 T = {T}')
    # N 커질수록 QV → T (결정론적, SDE Ch6-05)
```

---

## 🔗 AI/ML 연결

**SGD의 Variance 추적**  
$\theta_n^2$의 이차변분이 "gradient noise variance의 누적". Learning rate schedule $\alpha_n$이 이 누적 variance를 조절.

**Freedman's inequality**  
Azuma의 강화: bound가 이차변분 $\langle M\rangle_n$에 의존. Variance-adaptive concentration → tighter regret bounds in online learning (Ch5-05).

**Realized Volatility in Finance**  
$[X]_T = \sum(X_{t_i} - X_{t_{i-1}})^2$를 high-frequency data로 계산 → volatility $\sigma^2$ estimate. Algorithmic trading, risk management.

**SDE Discretization Error Analysis**  
Euler-Maruyama의 강수렴 차수 0.5가 $\mathbb{E}[(\text{discretization error})^2]$의 bound에 기반. 이차변분 이론이 직접 응용.

**Neural SDE 학습**  
Chen et al. 2018의 Neural SDE에서 diffusion coefficient $\sigma_\theta(x, t)$ 추정에 경험적 이차변분 사용 — "observed path $X$의 $[X]_T$로부터 $\int \sigma^2 dt$ 추정".

---

## ⚖️ 가정과 한계

**한계 — $L^2$ 필요**  
이차변분 정의에 square-integrability 필요. Heavy-tailed martingales (infinite variance)는 일반화 필요 (p-variation 등).

**한계 — Predictable vs Optional**  
Doob 분해의 "predictable $A$" 조건이 중요. Optional (adapted but not predictable) 버전은 **Meyer 분해** (연속시간 semimartingale).

**한계 — 이산에서 연속으로의 이행**  
연속시간에서는 이차변분이 "limit of sums over finer partitions"로 정의되며, existence·well-definedness에 비자명한 작업 필요 (BM continuous path + martingale 성질).

---

## 📌 핵심 정리

| 결과 | 수식 |
|---|---|
| Doob 분해 | $X = X_0 + M + A$, $M$ martingale, $A$ predictable |
| 유일성 | Predictable martingale = constant |
| Submartingale | $A$ increasing |
| 이차변분 | $\langle M\rangle_n = \sum \mathbb{E}[(\Delta M)^2\|\mathcal{F}]$ |
| $M^2 - \langle M\rangle$ | martingale |
| $[M]$ vs $\langle M\rangle$ | $[M] - \langle M\rangle$ martingale |
| $L^2$-bounded | $\mathbb{E}[\langle M\rangle_\infty] < \infty$ |

**한 줄 요약**: Doob 분해는 임의 adapted 과정을 "martingale + predictable" 분해. 이차변분 $\langle M\rangle$은 martingale의 "variance 누적"으로 $M^2 - \langle M\rangle$이 martingale — 이산 $(dM)^2 = d\langle M\rangle$의 정확한 형태이며, 연속 BM의 $(dB)^2 = dt$의 선구자.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. iid $\xi_k$ with mean 0, variance $\sigma^2$. $S_n = \sum \xi_k$. $\langle S\rangle_n$과 $[S]_n$을 계산하라.

<details>
<summary>해설</summary>

**$\langle S\rangle_n$** (predictable):
$\langle S\rangle_n = \sum_{k=1}^n \mathbb{E}[(S_k - S_{k-1})^2 | \mathcal{F}_{k-1}] = \sum_{k=1}^n \mathbb{E}[\xi_k^2] = n \sigma^2$ (iid).

**$[S]_n$** (empirical):
$[S]_n = \sum_{k=1}^n (S_k - S_{k-1})^2 = \sum_{k=1}^n \xi_k^2$.

**비교**:
- $\langle S\rangle_n = n\sigma^2$ (deterministic)
- $[S]_n = \sum \xi_k^2$ (random, mean $n\sigma^2$, variance $n \text{Var}(\xi^2)$)
- $[S]_n - \langle S\rangle_n = \sum(\xi_k^2 - \sigma^2)$: iid mean 0 → martingale.

$[S]_n / n \to \sigma^2$ a.s. (SLLN) — realized variance의 consistency.

**응용**: Financial time series의 $\sigma$ 추정 $\hat\sigma^2 = [X]_T/T$.

</details>

**문제 2 (심화)**. Exponential martingale $M_n = e^{\lambda S_n - n\psi(\lambda)}$ for $S_n = \sum \xi_k$ iid. $\langle M\rangle_n$의 대략적 형태를 구하라.

<details>
<summary>해설</summary>

$\Delta M_k = M_k - M_{k-1} = M_{k-1}(e^{\lambda \xi_k - \psi(\lambda)} - 1)$.

$\mathbb{E}[(\Delta M_k)^2 | \mathcal{F}_{k-1}] = M_{k-1}^2 \mathbb{E}[(e^{\lambda \xi - \psi(\lambda)} - 1)^2]$.

$\mathbb{E}[(e^{\lambda \xi - \psi} - 1)^2] = \mathbb{E}[e^{2\lambda \xi - 2\psi}] - 2 \mathbb{E}[e^{\lambda \xi - \psi}] + 1$.
$\mathbb{E}[e^{\lambda \xi}] = e^{\psi(\lambda)}$이므로 $\mathbb{E}[e^{\lambda \xi - \psi}] = 1$.
$\mathbb{E}[e^{2\lambda \xi - 2\psi}] = e^{\psi(2\lambda) - 2\psi(\lambda)}$ (cgf 활용).

따라서 $\mathbb{E}[(\Delta M_k)^2 | \mathcal{F}_{k-1}] = M_{k-1}^2 (e^{\psi(2\lambda) - 2\psi(\lambda)} - 1)$.

$\langle M\rangle_n = (e^{\psi(2\lambda) - 2\psi(\lambda)} - 1) \sum_{k=1}^n M_{k-1}^2$ — **stochastic** integral of $M^2$ against $dt$ (이산).

**연속 버전 (SDE)**: $dM = \lambda M dB$ (GBM), $\langle M\rangle_t = \lambda^2 \int_0^t M_s^2 ds$. 이산 결과의 continuous 대응.

**응용**: Risk-neutral pricing에서 exponential martingale 사용, $\langle M\rangle$이 "effective variance".

</details>

**문제 3 (AI 연결)**. Freedman's inequality: $|M_{k+1} - M_k| \leq b$ and $\langle M\rangle_n \leq v$ a.s., 그러면 $\mathbb{P}(M_n \geq t) \leq \exp(-t^2/(2v + 2bt/3))$. Azuma ($\langle M\rangle$ 대신 $\sum c_k^2$) 대비 장점을 논하라.

<details>
<summary>해설</summary>

**Azuma**: $|M_k - M_{k-1}| \leq c_k$ (worst-case bound) → $\mathbb{P}(M_n \geq t) \leq \exp(-t^2/(2\sum c_k^2))$.

**Freedman**: $|M_k - M_{k-1}| \leq b$ AND $\langle M\rangle_n \leq v$ → tighter bound $\exp(-t^2/(2v + 2bt/3))$.

**Freedman의 장점**:
1. **Adaptive to actual variance**: 각 step의 conditional variance가 작으면 $v$ 작음 → tighter bound.
2. **Sharp in low-variance regime**: $v \ll \sum c_k^2$일 때 훨씬 강한 concentration.

**예시**: Online learning에서 $|M_k - M_{k-1}| \leq 1$ (bounded loss), 하지만 대부분 time에는 $|M_k - M_{k-1}| \approx 0.01$. Azuma는 $\sum 1^2 = n$ 사용 → loose. Freedman은 $\langle M\rangle_n \approx 0.0001 n$ → 100배 tighter.

**실전 응용**:
- **Multi-armed bandit**: UCB bound를 Freedman으로 유도 → optimal rate ($\sqrt{KT \log T}$).
- **Online convex optimization**: Adaptive learning rate (AdaGrad)의 regret analysis.
- **RL**: Policy evaluation의 variance-aware bound.

**Azuma 대안으로서 유용성**: Worst-case bound $c_k$가 loose할 때 (e.g., heavy-tailed but low conditional variance). Modern concentration inequality 연구(Bernstein, Bennett, Freedman)가 이 아이디어 체계화.

**이차변분의 철학적 역할**: "Realized variability"가 "worst-case variability"보다 현실적 → adaptive learning, meta-learning의 이론 기반.

</details>

---

<div align="center">

◀ [03. Optional Stopping Theorem](./03-optional-stopping.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [05. 마팅게일과 ML — Online Learning](./05-martingale-ml.md)

</div>
