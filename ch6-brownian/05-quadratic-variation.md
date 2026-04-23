# 05. 이차변분 $\langle B\rangle_t = t$

## 🎯 핵심 질문

- $\sum_i (B_{t_{i+1}} - B_{t_i})^2 \to t$ in $L^2$ — 이차변분이 **결정론적** $t$로 수렴하는 이유는?
- 이 결과가 왜 **$(dB)^2 = dt$의 수학적 의미**이고, 어떻게 이토 공식의 기반이 되는가?
- 경로별 이차변분도 a.s. $t$와 같다는 더 강한 사실?
- BM의 기계적 특성: 1차변분 $\infty$, 2차변분 $t$ — 이 대조의 의미?

---

## 🔍 왜 이 결과가 SDE Deep Dive로 직결되는가

**이토 적분의 기반**: $\int H dB$의 이토 등장성 $\mathbb{E}[(\int H dB)^2] = \mathbb{E}[\int H^2 ds]$이 이차변분에서 나옴 (SDE Ch1-02).

**이토 공식**: $df(B_t) = f'(B_t) dB + \frac{1}{2}f''(B_t) dt$의 **둘째 항**이 Taylor 전개에서 $(\Delta B)^2 \to dt$ 치환의 직접 결과 (SDE Ch2-02).

**DDPM과 Score-SDE**: Reverse SDE의 유도에서 이차변분 성질이 "정확한 noise scaling" 제공.

**Realized volatility estimation**: $[X]_T = \sum (\Delta X)^2$이 $\int \sigma^2 ds$의 unbiased estimator. Financial AI의 주요 도구.

---

## 📐 수학적 선행 조건

- [Ch6-01 ~ Ch6-04](./01-axiomatic-definition.md): BM 공리·존재성·non-smoothness
- [Ch5-04](../ch5-martingale/04-doob-decomposition.md): 이차변분의 이산 버전
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): $L^2$-convergence, Chi-square 분포

---

## 📖 직관적 이해

### 수치적 관찰

Partition $\pi = \{0 = t_0 < t_1 < \cdots < t_n = t\}$. **이차변분 sum**:
$$Q_\pi := \sum_i (B_{t_{i+1}} - B_{t_i})^2.$$

**기댓값**: 각 $(B_{t_{i+1}} - B_{t_i})^2 \sim (t_{i+1} - t_i) \chi_1^2$ (1자유도 카이 제곱 스케일). 평균 $(t_{i+1} - t_i)$. 합:
$$\mathbb{E}[Q_\pi] = \sum_i (t_{i+1} - t_i) = t.$$

**분산**: $\text{Var}((B_{t_{i+1}} - B_{t_i})^2) = 2(t_{i+1} - t_i)^2$ (chi-square variance $= 2$). 독립증분으로
$$\text{Var}(Q_\pi) = 2\sum_i (t_{i+1} - t_i)^2 \to 0 \quad \text{as mesh } \to 0.$$

따라서 $Q_\pi \to t$ **in $L^2$** (평균 $t$, 분산 0으로 수렴).

### 왜 이것이 특별한가

**Smooth function** $f \in C^1$에서: $\sum (f(t_{i+1}) - f(t_i))^2 \leq \max|\Delta f| \cdot \sum|\Delta f| = O(\text{mesh}) \cdot V(f) \to 0$. 즉 smooth function의 이차변분 = 0.

**BM에서 이차변분 = $t > 0$**: 극단적 대조. 이것이 BM의 "non-smoothness의 양적 증거".

### $(dB)^2 = dt$의 진짜 의미

이산화 단계: $(B_{t+dt} - B_t)^2 \approx dt$. 
- **분포적으로**: $(B_{t+dt} - B_t)^2 \sim dt \cdot \chi_1^2$, 평균 $dt$.
- **합으로는**: $\sum (\Delta B)^2 \to dt$ (deterministic) — variance가 사라짐.

즉 "infinitesimally, $(dB)^2$가 $dt$와 같게 행동" — 이토 공식에서 이 substitution이 정당.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Partition and Mesh

$\pi = \{0 = t_0 < t_1 < \cdots < t_n = t\}$. **Mesh** $|\pi| := \max_i (t_{i+1} - t_i)$.

### 정의 5.2 — 경로별 이차변분 (Pathwise Quadratic Variation)

$$[B]_t^\pi := \sum_i (B_{t_{i+1}} - B_{t_i})^2.$$

Sequence of partitions $\pi_n$ with $|\pi_n| \to 0$. 

### 정의 5.3 — 이차변분 $\langle B\rangle_t$

$\langle B\rangle_t = t$ — BM의 예측가능 이차변분(predictable QV, Doob 분해로부터). 즉 $B_t^2 - t$가 martingale.

---

## 🔬 정리와 증명

### 정리 5.1 — $L^2$ 수렴

$|\pi_n| \to 0$일 때
$$[B]_t^{\pi_n} = \sum_i (B_{t_{i+1}^n} - B_{t_i^n})^2 \to t \quad \text{in } L^2.$$

### 증명

**평균 계산**:
$(B_{t_{i+1}} - B_{t_i})^2$의 평균 $= \text{Var}(B_{t_{i+1}} - B_{t_i}) = t_{i+1} - t_i$. 합:
$$\mathbb{E}[[B]_t^\pi] = \sum_i (t_{i+1} - t_i) = t.$$

**분산 계산**:
독립증분 → $(B_{t_{i+1}} - B_{t_i})^2$들이 서로 독립. 분산 합:
$$\text{Var}([B]_t^\pi) = \sum_i \text{Var}((B_{t_{i+1}} - B_{t_i})^2).$$

$Z \sim \mathcal{N}(0, \sigma^2)$에 대해 $Z^2 \sim \sigma^2 \chi_1^2$, $\text{Var}(Z^2) = 2\sigma^4$. 따라서
$$\text{Var}((B_{t_{i+1}} - B_{t_i})^2) = 2(t_{i+1} - t_i)^2.$$

합:
$$\text{Var}([B]_t^\pi) = 2\sum_i (t_{i+1} - t_i)^2 \leq 2 |\pi| \cdot \sum_i (t_{i+1} - t_i) = 2|\pi| t \to 0.$$

**$L^2$ 수렴**:
$$\mathbb{E}[([B]_t^\pi - t)^2] = \text{Var}([B]_t^\pi) + (\mathbb{E}[[B]_t^\pi] - t)^2 = 2|\pi|t + 0 \to 0.$$

$\square$

### 정리 5.2 — 경로별 a.s. 수렴 (Dyadic Refinement)

Dyadic partitions $\pi_n = \{k t/2^n : k = 0, \ldots, 2^n\}$에 대해
$$[B]_t^{\pi_n} \to t \quad \text{a.s.}$$

*증명 스케치*.
정리 5.1에서 $L^2$ 수렴 얻음. $\mathbb{E}[([B]_t^{\pi_n} - t)^2] = 2t \cdot t/2^n = O(2^{-n})$.

Borel-Cantelli: $\mathbb{P}(|[B]_t^{\pi_n} - t| > \epsilon) \leq \mathbb{E}[\ldots]/\epsilon^2 = O(2^{-n})$. $\sum < \infty$ → Borel-Cantelli → a.s. 수렴. $\square$

**주의**: Non-dyadic refinement로는 a.s. 수렴이 실패할 수 있음 — 즉 BM의 이차변분은 partition에 의존할 수 있다. 정확한 결과: **"refining partitions" 수열에서 a.s. 수렴**.

### 정리 5.3 — $B_t^2 - t$ is Martingale

$M_t := B_t^2 - t$가 $\mathcal{F}_t^B$-martingale.

*증명*.
$$\mathbb{E}[B_t^2 | \mathcal{F}_s^B] = \mathbb{E}[(B_s + (B_t - B_s))^2 | \mathcal{F}_s^B]$$
$$= B_s^2 + 2 B_s \mathbb{E}[B_t - B_s | \mathcal{F}_s^B] + \mathbb{E}[(B_t - B_s)^2 | \mathcal{F}_s^B]$$
$$= B_s^2 + 0 + (t - s).$$

따라서 $\mathbb{E}[B_t^2 - t | \mathcal{F}_s^B] = B_s^2 + (t - s) - t = B_s^2 - s$. Martingale. $\square$

**Doob 분해**: $B_t^2 = (B_t^2 - t) + t$. Martingale part: $B^2 - t$. Predictable increasing: $t$. 이는 Ch5-04의 이차변분 $\langle B\rangle_t = t$.

### 정리 5.4 — Dual with First Variation

1차 변분 $V_n = \sum |B_{t_{i+1}} - B_{t_i}| \to \infty$ a.s. (Ch6-04 정리 4.3).
2차 변분 $[B]_n \to t < \infty$ a.s.

**대조**: 1차는 무한, 2차는 결정론적 $t$. 이 **정확한 scaling 질서**가 BM의 본질.

### 정리 5.5 — Extension to SDE

SDE solution $dX_t = b(t, X_t) dt + \sigma(t, X_t) dB_t$의 이차변분:
$$[X]_t = \int_0^t \sigma^2(s, X_s) ds, \quad d[X]_t = \sigma^2(t, X_t) dt.$$

(SDE Ch2-03, Ch4-01에서 유도.)

**함의**: 이토 공식에서 $(dX)^2 = d[X] = \sigma^2 dt$ — "모든 SDE의 이차변분이 deterministic growth rate $\sigma^2$".

---

## 💻 NumPy 구현 검증

### 실험 1 — $[B]_t^\pi \to t$ 수렴 확인

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
T = 1.0

# Sample one BM path at high resolution
N_fine = 10**6
dt_fine = T / N_fine
dB = rng.standard_normal(N_fine) * np.sqrt(dt_fine)
B = np.concatenate([[0], np.cumsum(dB)])

# 다양한 partition mesh에 대해 QV 계산
N_vals = [10, 100, 1000, 10000, 100000, 1000000]
QV_values = []
for N in N_vals:
    step = N_fine // N
    B_sub = B[::step]
    QV = np.sum(np.diff(B_sub)**2)
    QV_values.append(QV)

print(f'{"N":>8} {"|π|":>10} {"[B]_T":>10}')
for N, qv in zip(N_vals, QV_values):
    print(f'{N:>8} {1/N:>10.6f} {qv:>10.4f}')

# N 증가 → QV → 1 (= T)
plt.loglog(N_vals, np.abs(np.array(QV_values) - T), 'o-')
plt.xlabel('N (partition size)'); plt.ylabel('|[B]_T - T|')
plt.title(r'$[B]_T^\pi \to T$ as $|\pi| \to 0$')
plt.grid(True, which='both', alpha=0.3); plt.show()
```

### 실험 2 — $[B]_t$ 분산이 $|\pi|$에 비례

```python
# 여러 paths에 대해 fixed N에서 [B]_t의 분산 측정
n_paths = 10000
T = 1.0

for N in [10, 100, 1000]:
    dt = T / N
    QVs = []
    for _ in range(n_paths):
        dB = rng.standard_normal(N) * np.sqrt(dt)
        QVs.append((dB**2).sum())
    
    print(f'N = {N}: 평균 {np.mean(QVs):.4f}, 분산 {np.var(QVs):.6f}, 이론 분산 2T|π| = {2*T/N:.6f}')
# 분산이 2T/N에 맞춤 — 정리 5.1
```

### 실험 3 — 1차변분 vs 2차변분 대조

```python
N_vals = np.logspace(1, 5, 10, dtype=int)
total_var_1 = []  # Σ |ΔB|
total_var_2 = []  # Σ (ΔB)²

for N in N_vals:
    dt = T / N
    dB = rng.standard_normal(N) * np.sqrt(dt)
    total_var_1.append(np.abs(dB).sum())
    total_var_2.append((dB**2).sum())

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.loglog(N_vals, total_var_1, 'o-')
plt.loglog(N_vals, np.sqrt(N_vals), '--', label=r'$\sqrt{N}$ ref')
plt.xlabel('N'); plt.ylabel(r'$\sum |\Delta B|$')
plt.title('1차변분 → ∞'); plt.legend(); plt.grid(True, which='both', alpha=0.3)

plt.subplot(1, 2, 2)
plt.semilogx(N_vals, total_var_2, 'o-')
plt.axhline(T, color='r', linestyle='--', label='T = 1')
plt.xlabel('N'); plt.ylabel(r'$\sum (\Delta B)^2$')
plt.title('2차변분 → T'); plt.legend(); plt.grid(True, which='both', alpha=0.3)
plt.tight_layout(); plt.show()
```

---

## 🔗 AI/ML 연결

**이토 공식 (SDE Ch2-01)**  
$df(X_t) = f' dX + \frac{1}{2} f'' d[X]_t = f' (b dt + \sigma dB) + \frac{1}{2}f'' \sigma^2 dt$. **$\frac{1}{2}f''$ term의 근원**이 이차변분. 직접적 응용:
- Geometric BM $dS = \mu S dt + \sigma S dB$의 $\log S$ 유도.
- Black-Scholes PDE 유도.
- Diffusion model의 정확한 drift 공식.

**Realized Volatility Estimation**  
$[X]_t = \int \sigma^2 ds$ 추정:
$\hat \sigma^2_T = \frac{1}{T}\sum_i (X_{t_{i+1}} - X_{t_i})^2$.
Consistent estimator (정리 5.1). High-frequency trading, volatility arbitrage의 core.

**Neural SDE의 diffusion coefficient 학습**  
$\sigma_\theta(x, t)$를 NN으로 parameterize. Training data의 경험적 이차변분 $\sum (\Delta X)^2$이 $\int \sigma_\theta^2 dt$와 matching.

**DDPM의 정확한 noise scaling**  
VP-SDE에서 $dX_t = f dt + g(t) dB_t$. $[X]_t = \int g^2 ds$. Noise schedule $\beta_t \leftrightarrow g^2 = \beta(t)$가 "discrete noise level"과 "continuous diffusion" 연결.

**Information Bottleneck in SGD**  
SGD의 implicit regularization이 "effective diffusion" rate. Gradient noise의 이차변분이 regularization strength 결정 (Smith 2018, flat minima theory).

---

## ⚖️ 가정과 한계

**가정 — Refining sequence of partitions**  
Dyadic refinement에서 a.s. 수렴 (정리 5.2). General partitions에서는 조심 — 수렴이 **stochastic integral의 정의에 의존** (L²로만).

**한계 — 경로별 이차변분과 stochastic**  
$[B]_t$가 deterministic $t$이지만, 이는 "평균적으로 $t$" (분산 → 0). 개별 path의 정확한 변동 없음 — BM의 smoothness의 **양적** 결여의 표시.

**일반 SDE로 확장**  
$dX_t = b dt + \sigma dB_t$에서 $[X]_t = \int \sigma^2 ds$. SDE Deep Dive Ch2-02에서 엄밀 증명.

---

## 📌 핵심 정리

| 결과 | 수식 |
|---|---|
| $L^2$ 수렴 | $[B]_t^\pi \to t$ in $L^2$ as $|\pi| \to 0$ |
| 평균 | $\mathbb{E}[[B]_t^\pi] = t$ |
| 분산 | $\text{Var}([B]_t^\pi) = 2\sum(\Delta t_i)^2 \to 0$ |
| Pathwise (dyadic) | $[B]_t^{\pi_n} \to t$ a.s. |
| Martingale | $B_t^2 - t$ |
| $d\langle B\rangle_t$ | $= dt$ |
| $(dB)^2$ 해석 | $= dt$ (이차변분 성질) |
| 1차 vs 2차 | 1차 $= \infty$, 2차 $= t$ |

**한 줄 요약**: 이차변분 $[B]_t^\pi \to t$ **결정론적**으로 $L^2$ 수렴 — 분산이 partition mesh에 선형 비례로 0으로. 이것이 **$(dB)^2 = dt$의 정확한 수학적 의미**이며, 이토 공식·SDE 이론 전체의 근간 — **이 레포와 SDE Deep Dive를 잇는 중심 결과**.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. BM의 **삼차변분** $\sum (\Delta B)^3 \to 0$임을 보여라.

<details>
<summary>해설</summary>

$\mathbb{E}[(\Delta B)^3] = 0$ (Gaussian의 3차 모멘트 = 0). 따라서
$$\mathbb{E}\left[\sum_i (B_{t_{i+1}} - B_{t_i})^3\right] = 0.$$

**분산**:
$\text{Var}((\Delta B)^3) = \mathbb{E}[(\Delta B)^6] - 0^2 = 15(\Delta t)^3$ (Gaussian 6차 moment).

합의 분산: $\text{Var}(\sum (\Delta B)^3) = 15 \sum (\Delta t)^3 \leq 15 |\pi|^2 t \to 0$.

$L^2$ 수렴: $\sum (\Delta B)^3 \to 0$.

**일반화**: $k$차변분 $\sum (\Delta B)^k$:
- $k = 1$: $\sum \sqrt{\Delta t} \chi$ — magnitude $\sqrt{N \cdot \Delta t} = \sqrt T$, **유계** for each $N$... 하지만 절댓값이면 $\sqrt{2/\pi} \sqrt{N \Delta t} \cdot \sqrt N \to \infty$. (1차변분은 발산.)
- $k = 2$: $\to t$ (유한 deterministic).
- $k \geq 3$: $\to 0$ (고차 합이 사라짐).

**Zero for $k > 2$의 의미**: 이토 공식에서 $f(B_t) - f(B_0) = \sum f'(\Delta B) + \frac{1}{2}f''(\Delta B)^2 + \text{higher}$. 고차 항이 $\to 0$ → Taylor 전개가 **2차에서 멈춤** — 이것이 이토 공식의 깔끔함.

</details>

**문제 2 (심화)**. Non-dyadic refinement에서 $[B]_t^{\pi_n}$이 a.s.로 $t$에 수렴 **안 할 수도 있음**을 논하라.

<details>
<summary>해설</summary>

**Example** (Dudley): "Chaotic" partition sequence 만들기. Each $\pi_n$의 mesh는 $\to 0$이지만 $\pi_n$이 서로 "incompatible" — subset 관계 없음.

**Dudley 예**: 
$\pi_n$이 $n$번째 시행에서 BM sample path에 의존하여 **adaptively** 선택. 이런 non-adapted partitions에서는 a.s. 수렴 실패 가능.

**Protter's book (2005)** 예: BM path dependent한 $\pi_n$으로 $[B]_t^{\pi_n}$이 arbitrary value로 수렴 가능 (pathological).

**정확한 정리**: **"Refining" 또는 "adapted" partitions**에서 a.s. 수렴. General partition에서는 **probability** (또는 $L^2$) 수렴만 보장.

**연결**: Stochastic integration의 "predictable" 조건 (Ch1-04) 이 이와 관련. 이토 적분의 정의가 partition의 "predictable" 선택을 요구 — Stratonovich 적분과의 대조.

**실전 함의**:
- Numerical SDE에서 **fixed partition** (dyadic 또는 uniform)을 사용해야 stable.
- Adaptive time-stepping (error-based)은 이론적으로 주의 필요.

</details>

**문제 3 (AI 연결)**. Realized volatility에서 "jumps" (discrete jump process 혼합)이 있으면 $\sum (\Delta X)^2$이 $\int \sigma^2 dt + \sum (\text{jump})^2$로 분리된다. NN이 이 jump 성분을 어떻게 식별·처리할 수 있는가?

<details>
<summary>해설</summary>

**Jump-Diffusion Setup**:
$dX_t = b dt + \sigma dB_t + dJ_t$, $J$ = compound Poisson (Ch3-03).

**이차변분 분해**:
$[X]_t = \int_0^t \sigma^2 ds + \sum_{s \leq t} (\Delta J_s)^2$ where $\Delta J_s$ = jump size.

**문제**: 관찰된 $\sum (X_{t_{i+1}} - X_{t_i})^2$이 **continuous part + jumps 합**. Jumps 식별 없이는 $\sigma$ 추정 biased.

**NN 기반 방법**:

**(1) Threshold-based truncation (Barndorff-Nielsen-Shephard)**:
$|\Delta X| > c \sqrt{\Delta t}$이면 jump candidate. **Bi-Power Variation**:
$BPV_t := \sum |X_{t_{i+1}} - X_{t_i}| \cdot |X_{t_{i-1}} - X_{t_i-2}|$
— cross-product이므로 jump의 영향 minimal.

**(2) CNN으로 jump detection**:
Input = 시계열 window, output = "jump 있는가 여부 확률". Time-frequency features (wavelet) 활용.

**(3) Neural Network Structured Deep Learning**:
- **Dense** branch: continuous volatility 추정 ($\sigma$)
- **Jump** branch: jump rate + jump size distribution
- Joint loss: log-likelihood of observed sequence

**(4) Deep Filtering**:
Particle filter / Kalman filter의 NN amortization. 관찰 sequence에서 latent $\sigma_t$와 jump times를 jointly infer.

**실전 model**:
- **Deep Realized Kernel** (Liu et al. 2015): NN이 microstructure noise와 jumps 동시 처리.
- **CNN for tick data**: high-frequency pattern recognition.
- **Transformer for volatility forecasting**: self-attention으로 regime change 감지.

**이론적 측면**:
- Pure continuous BM: $[X]_t$만. NN이 잘 estimates.
- **Heavy-tailed jumps** (Levy process beyond compound Poisson): 경계 불명확, 새로운 이론 필요.
- **Realized power variation**: $\sum |\Delta X|^p$ for $p < 2$가 continuous part만 select (jumps vanish).

**Financial AI의 실전 과제**:
- Tick-level data의 노이즈 분리 (market microstructure)
- Regime change detection (volatility clustering)
- Forecasting — Transformer/LSTM이 state-of-art, 이론과 engineering 결합.

**연결**: SDE Deep Dive Ch5 (numerical methods)에서 jump diffusion 처리. 현대 quantitative finance가 pure BM을 넘어 jumps + rough path + neural parameterization으로 발전.

</details>

---

<div align="center">

◀ [04. 경로 성질 — 비미분 가능성](./04-non-differentiability.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [06. 반사원리(Reflection Principle)와 최대값](./06-reflection-principle.md)

</div>
