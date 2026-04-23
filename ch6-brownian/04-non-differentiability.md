# 04. 경로 성질 — 비미분 가능성

## 🎯 핵심 질문

- BM이 **거의 확실히 어디에서도 미분불가능**한 이유는? (**Paley-Wiener-Zygmund 정리**)
- **Hölder 연속성** $|B_t - B_s| \leq C|t - s|^{1/2 - \epsilon}$이 왜 정확히 $1/2$에서 깨지는가?
- BM 경로의 **Hausdorff 차원** $\dim_H(\text{graph}) = 3/2$의 의미는?
- 이 "paradoxical" 성질이 어떻게 **이토 적분의 필연성**과 **$(dB)^2 = dt$의 신비**로 이어지는가?

---

## 🔍 왜 이 성질이 AI에서 중요한가

**이토 적분의 근거**: BM이 유한변동이 아니면(non-differentiable의 결과) 경로별 Riemann-Stieltjes 적분 불가 → $L^2$ 한계로 이토 적분 정의 (SDE Ch1-01).

**이산 근사의 한계**: 아무리 fine grid로 BM을 이산화해도 "실제 BM 경로"를 잡을 수 없음. Numerical SDE methods의 본질적 제약.

**Diffusion model의 복잡성**: DDPM의 reverse process가 "노이즈 제거"라는 직관 이면에, BM의 non-smooth 성질 때문에 학습이 "simple regression"이 아니게 되는 이유.

**Turbulence / rough volatility**: BM이 너무 rough → financial data는 실제로 더 rough (H < 1/2, rough path theory; fractional volatility).

---

## 📐 수학적 선행 조건

- [Ch6-01 ~ Ch6-03](./01-axiomatic-definition.md): BM 공리, 존재성, Donsker
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Borel-Cantelli, Kolmogorov continuity
- 실해석: Hölder continuity, bounded variation

---

## 📖 직관적 이해

### "모든 점에서 미분불가"

매 $t_0 \in [0, 1]$에서 $\frac{B_t - B_{t_0}}{t - t_0}$가 $t \to t_0$에서 **발산**. 구체적으로 한계 limsup $= +\infty$, liminf $= -\infty$ — 기울기가 정의 안 됨.

**직관**: BM 증분 $B_t - B_s \sim \mathcal{N}(0, t-s)$. 표준편차 $\sqrt{t-s}$. 기울기 $\frac{B_t - B_s}{t - s} \sim \mathcal{N}(0, 1/(t-s)) \to \mathcal{N}(0, \infty)$ as $t \to s$. **분산 발산** → 한계 없음.

### Hölder 지수 1/2

$|B_t - B_s|$이 $|t-s|^{1/2}$ scale에서 변동. 정확히 **$1/2 - \epsilon$ Hölder** (a.s.), 하지만 **not $1/2$ Hölder** — "임계값 $1/2$에서 깨짐".

### Hausdorff 차원 계산

BM 경로 $\{(t, B_t) : t \in [0, 1]\}$의 Hausdorff 차원 = $3/2$. 1차원 domain에 $1/2$-Hölder 경로 → 차원 $1 + (1 - 1/2) \cdot 1 = 3/2$.

**비교**: smooth curve ($C^1$)의 차원 = 1. Space-filling curve = 2. BM의 $3/2$는 그 사이 — "fractal between curve and plane".

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Hölder 연속

$f : [0, 1] \to \mathbb{R}$가 **$\gamma$-Hölder 연속**이다:
$$\exists C > 0, \forall s, t : |f(t) - f(s)| \leq C|t - s|^\gamma.$$

### 정의 4.2 — Hausdorff 차원

집합 $A \subseteq \mathbb{R}^d$의 **$s$-Hausdorff 측도** $\mathcal{H}^s(A) := \lim_{\delta \to 0}\inf\{\sum r_i^s : A \subseteq \bigcup B(x_i, r_i), r_i < \delta\}$.

**Hausdorff 차원** $\dim_H(A) := \inf\{s : \mathcal{H}^s(A) = 0\} = \sup\{s : \mathcal{H}^s(A) = \infty\}$.

### 정의 4.3 — Bounded Variation

$f : [0, 1] \to \mathbb{R}$의 **전변동**:
$$V(f) := \sup_\pi \sum |f(t_{i+1}) - f(t_i)|,$$
partition $\pi : 0 = t_0 < \cdots < t_n = 1$.

$V(f) < \infty$이면 $f$ **bounded variation**.

---

## 🔬 정리와 증명

### 정리 4.1 — Paley-Wiener-Zygmund (1933)

Standard BM $\{B_t\}_{t \geq 0}$. 거의 확실히 $t \mapsto B_t$는 **어떤 점에서도 미분 불가**.

### 증명 스케치

**전략**: 모든 $t_0 \in [0, 1]$에 대해 $\limsup_{t \to t_0} |B_t - B_{t_0}|/|t - t_0|^\gamma = \infty$ for $\gamma > 1/2$.

**Step 1 — Fixed $t_0$**:
$\frac{B_{t_0 + h} - B_{t_0}}{h} \sim \mathcal{N}(0, 1/h)$. $h \to 0$: $|B_{t_0 + h} - B_{t_0}|/h$의 tail이 ever larger. Borel-Cantelli on fine dyadic grids:
$$\mathbb{P}\left(\frac{|B_{t_0 + h_k} - B_{t_0}|}{h_k} > k\right) = \mathbb{P}(|\mathcal{N}(0, 1/h_k)| > k) = 2(1 - \Phi(k\sqrt{h_k})).$$

$h_k = 1/k^3$ 잡으면 $k\sqrt{h_k} = k^{-1/2} \to 0$ → tail $\to 1$. Borel-Cantelli infinite → 무한히 자주 큰 기울기 → $t_0$에서 미분불가능.

**Step 2 — All $t_0$ simultaneously**:
$t_0$가 uncountable이므로 위 argument를 직접 union에 쓸 수 없음. Dyadic approximation + modulus of continuity:

$\sup_{|s - t| \leq \delta} |B_s - B_t| \leq C\sqrt{\delta \log(1/\delta)}$ a.s. (Lévy의 modulus theorem). 이것이 $\sqrt{\delta}$-ish가 한계.

만약 $t_0$에서 미분가능하다면 $|B_t - B_{t_0}| = O(|t - t_0|)$ — $\sqrt{|t - t_0|}$보다 훨씬 작아야. Lévy modulus와 모순.

따라서 **모든 $t_0$에서 미분불가능** a.s. $\square$

(완전한 증명은 Mörters-Peres or Karatzas-Shreve 참조.)

### 정리 4.2 — BM의 Hölder 연속성

$\gamma < 1/2$에 대해 BM 경로가 거의 확실히 **$\gamma$-Hölder 연속** on compact intervals. $\gamma \geq 1/2$에서는 실패.

*증명 스케치*.

**$\gamma < 1/2$ 성립**: Kolmogorov continuity theorem (Ch1-02 정리 2.3) 적용. BM:
$$\mathbb{E}|B_t - B_s|^{2n} = C_n |t - s|^n \quad (\text{Gaussian moments}).$$

$\alpha = 2n, \beta = n - 1$로 Hölder 지수 $< \beta/\alpha = 1/2 - 1/(2n)$. $n$ 임의 크게 하면 $\gamma < 1/2$ 모두에 대해 Hölder.

**$\gamma \geq 1/2$ 실패**: Law of iterated logarithm:
$$\limsup_{t \to 0^+} \frac{B_t}{\sqrt{2t \log\log(1/t)}} = 1 \quad \text{a.s.}$$

이는 $B_t$가 $\sqrt{t \log\log(1/t)}$ scale에서 fluctuate → $1/2$-Hölder bound $C\sqrt t$를 **log factor로 초과** → $1/2$-Hölder 실패. $\square$

### 정리 4.3 — BM은 **Unbounded Variation**

거의 확실히 $V(B) = \infty$ on any interval $[s, t]$ with $s < t$.

### 증명

Upper bound on variation from Hölder:
$V = \sup \sum |B_{t_{i+1}} - B_{t_i}|$. 만약 $V < \infty$였다면
$$\sum (B_{t_{i+1}} - B_{t_i})^2 \leq V \cdot \max |B_{t_{i+1}} - B_{t_i}| \to 0$$
as $|\pi| \to 0$ (continuity). 그러나 **Ch6-05에서** $\sum (\Delta B)^2 \to t > 0$ in $L^2$ (이차변분 = t).

모순 → $V = \infty$. $\square$

**Profound consequence**: BM이 유한변동이 아님 → **Riemann-Stieltjes 적분 $\int H dB$를 경로별로 정의 불가능** (SDE Ch1-01). 이토 적분 필연.

### 정리 4.4 — Hausdorff 차원

Graph of BM $\{(t, B_t) : t \in [0, 1]\}$의 Hausdorff 차원 = **3/2** a.s.

*증명 스케치*. "Rough curve in 2D": horizontal dimension 1, vertical fluctuation $\sqrt{t}$ scale → $\dim_H = 1 + (1 - 1/2) = 3/2$.

(공식 증명은 McKean 1955.)

### 정리 4.5 — Level sets and Hitting Sets

Level set $\{t : B_t = a\}$의 Hausdorff 차원 = $1/2$ a.s. (fractal).
Zero set $\{t : B_t = 0\}$은 Cantor-like closed uncountable set.

---

## 💻 NumPy 구현 검증

### 실험 1 — 비미분 가능성 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# BM path at increasingly fine resolution
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
n_levels = [10, 100, 1000, 10000, 100000, 1000000]
base_noise = rng.standard_normal(n_levels[-1])  # 사용하지 않지만 일관성

for ax, n in zip(axes.flat, n_levels):
    dt = 1.0 / n
    dB = rng.standard_normal(n) * np.sqrt(dt)
    B = np.concatenate([[0], np.cumsum(dB)])
    t_grid = np.linspace(0, 1, n + 1)
    ax.plot(t_grid, B, linewidth=0.5)
    ax.set_title(f'N = {n}')
    ax.grid(True, alpha=0.3)
plt.suptitle("BM path at various resolutions — 확대해도 rough")
plt.tight_layout(); plt.show()
# → 아무리 zoom해도 jagged (self-similar roughness)
```

### 실험 2 — Hölder exponent 측정

```python
# 경험적 Hölder: max_{|t-s|<=δ} |B_t - B_s| vs δ^γ
N = 100000
dt = 1 / N
dB = rng.standard_normal(N) * np.sqrt(dt)
B = np.concatenate([[0], np.cumsum(dB)])

# Sliding window max
def max_increment(B, window_size):
    diffs = B[window_size:] - B[:-window_size]
    return np.abs(diffs).max()

windows = [10, 100, 1000, 10000]
deltas = np.array(windows) * dt
max_incs = [max_increment(B, w) for w in windows]

# Log-log slope = Hölder exponent
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(np.log(deltas), np.log(max_incs))
print(f'Measured Hölder exponent: {slope:.4f}')
print(f'Theory: < 0.5 (e.g., 0.49)')

plt.loglog(deltas, max_incs, 'o-', label=f'Measured, slope ≈ {slope:.3f}')
plt.loglog(deltas, np.sqrt(deltas) * 3, '--', label=r'$\sqrt{\delta}$ reference')
plt.xlabel(r'$\delta$ (window size)'); plt.ylabel(r'$\max|B_t - B_s|$ over $|t-s| \leq \delta$')
plt.legend(); plt.grid(True, which='both', alpha=0.3); plt.show()
# → slope 거의 0.5 (BM의 $1/2 - \epsilon$ Hölder)
```

### 실험 3 — Total variation의 발산

```python
# V_n = sum |B(t_{k+1}) - B(t_k)| as n → ∞
N_vals = [100, 1000, 10000, 100000]
print(f'{"N":>8} {"Σ|ΔB|":>15} {"Σ(ΔB)²":>15}')
for N in N_vals:
    dt = 1/N
    dB = rng.standard_normal(N) * np.sqrt(dt)
    tv = np.abs(dB).sum()
    qv = (dB**2).sum()
    print(f'{N:>8} {tv:>15.4f} {qv:>15.4f}')
# TV는 √N scale로 발산, QV는 1로 수렴 (이차변분)
```

---

## 🔗 AI/ML 연결

**Ito Integral의 필연성** (SDE Ch1-01)  
BM non-smoothness → RS 적분 경로별 불가능 → $L^2$ 극한 정의 필요. "Score matching" 등의 noise-related 계산이 이 위에서.

**Rough Path Theory in Financial ML**  
BM의 $1/2$-Hölder가 "너무 smooth". Rough volatility (Gatheral 2018): $H < 1/2$ — 시장 데이터가 $1/2$보다 rough. Deep learning이 이 $H$를 추정.

**Neural SDE의 numerical error**  
BM 경로의 roughness가 SDE solver의 intrinsic error source. Euler-Maruyama의 strong convergence 0.5차가 이 fluctuation scale.

**Fractal dimension in time series**  
BM-like time series의 scaling exponent 추정 (DFA, R/S analysis). 금융·EEG·network traffic의 self-similarity.

**Diffusion Model의 implicit regularity**  
DDPM이 noisy input에서 작동 = "rough function" 근사. Score network $\epsilon_\theta$가 smoother 모델이어야 BM의 non-smoothness를 적절히 다룸.

---

## ⚖️ 가정과 한계

**한계 — BM의 구체 현실성**  
실세계 현상(주가, 온도)은 BM과 정확히 일치하지 않음. BM이 "극한 유일성"을 가지나, 유한 $n$에서는 approximation.

**한계 — Hausdorff dim 직접 측정 어려움**  
실측에서 $\dim_H$ 추정은 노이즈와 finite sample 영향 큼. Box-counting, correlation dimension 등 proxy.

**대안 process**:
- **Fractional BM** (H ≠ 1/2): Hölder $H$, $H > 1/2$는 smooth, $H < 1/2$는 rough
- **Multifractal**: local $H$가 space-varying (Mandelbrot 1989)

---

## 📌 핵심 정리

| 결과 | 요약 |
|---|---|
| Paley-Wiener-Zygmund | 거의 확실히 어디에서도 미분불가능 |
| Hölder | $\gamma < 1/2$ 성립, $\gamma \geq 1/2$ 실패 |
| Unbounded variation | $V(B) = \infty$ a.s. |
| Hausdorff dim | $\dim_H(\text{graph}) = 3/2$ |
| Level set dim | $1/2$ |
| Law of iterated logarithm | $B_t \sim \sqrt{2t\log\log(1/t)}$ |

**한 줄 요약**: BM 경로는 **연속이지만 어디서도 미분불가능** — 이는 $1/2$-Hölder regularity의 한계치와 unbounded variation의 결과. 이 "pathological" 성질이 이토 적분 구성의 근본 동기이며, SDE 이론 전체의 출발점.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. BM의 Hölder 지수가 $1/2$에서 깨지는 예시를 구체적으로 보여라.

<details>
<summary>해설</summary>

**Law of iterated logarithm**: $\limsup_{t \to 0^+} B_t / \sqrt{2t\log\log(1/t)} = 1$ a.s.

**함의**: 거의 확실히 sequence $t_n \to 0$이 있어
$|B_{t_n} - B_0| \approx \sqrt{2 t_n \log\log(1/t_n)}$.

$1/2$-Hölder 가정: $|B_{t_n}| \leq C \sqrt{t_n}$. 비교:
$$\sqrt{2 t_n \log\log(1/t_n)} \leq C \sqrt{t_n} \Rightarrow \sqrt{\log\log(1/t_n)} \leq C/\sqrt 2.$$

$t_n \to 0$에서 $\log\log(1/t_n) \to \infty$ → **모순**.

따라서 $1/2$-Hölder 실패 (a.s., 원점 근처).

**해석**: BM은 "$1/2$-Hölder에 거의 가까운" regularity이지만 **log 보정 인자** 때문에 정확히 $1/2$에서 깨짐. 이 log factor가 rough path theory의 미세 구조 분석에서 중요.

</details>

**문제 2 (심화)**. BM이 unbounded variation임을 이차변분 $\sum(\Delta B)^2 \to t$ 사용 없이 **직접 증명**하라.

<details>
<summary>해설</summary>

**직접 증명**: Dyadic partition $\pi_n : t_k = k/2^n$, $k = 0, \ldots, 2^n$. 

$V_n = \sum_{k=0}^{2^n - 1} |B_{t_{k+1}} - B_{t_k}| = \sum |Z_k|$, $Z_k = B_{t_{k+1}} - B_{t_k} \sim \mathcal{N}(0, 2^{-n})$ iid.

$\mathbb{E}|Z_k| = \sqrt{2/\pi} \cdot 2^{-n/2}$ (|$\mathcal{N}(0, \sigma^2)|$의 평균 $= \sigma\sqrt{2/\pi}$).

$\mathbb{E}[V_n] = 2^n \cdot \sqrt{2/\pi} \cdot 2^{-n/2} = \sqrt{2/\pi} \cdot 2^{n/2} \to \infty$.

**Variance**: $\text{Var}(V_n) = 2^n \text{Var}(|Z_k|)$. $\text{Var}(|Z_k|) = (1 - 2/\pi) \cdot 2^{-n}$. So $\text{Var}(V_n) = (1 - 2/\pi)$ bounded.

**CLT-style**: $V_n / \mathbb{E}[V_n] \to 1$ in probability. $\mathbb{E}[V_n] \to \infty$, so $V_n \to \infty$ in probability.

A.s. convergence with Borel-Cantelli on deviation $|V_n - \mathbb{E}[V_n]| > \mathbb{E}[V_n]/2$: Chebyshev → $\mathbb{P}(...) \leq 4 \text{Var}/{\mathbb{E}^2} \to 0$. 

Therefore $V_n \to \infty$ a.s. → $V = \sup V_n = \infty$ a.s. $\square$

**Insight**: 이차변분 $\sum |Z_k|^2 \to t$ (deterministic)과 대조적으로, 1차변분 $\sum |Z_k| \to \infty$. 2차는 잘 행동, 1차는 발산 — BM의 정확한 scaling 질서.

</details>

**문제 3 (AI 연결)**. Rough volatility model에서 log-volatility가 $H < 1/2$의 fractional BM을 따른다. 이것이 option pricing과 implied volatility surface에 어떤 영향을 미치는가?

<details>
<summary>해설</summary>

**Rough volatility (Gatheral-Jaisson-Rosenbaum 2018)**:
$d(\log V_t) = \eta dW_t^H$, $W^H$ = fractional BM with Hurst $H \approx 0.1$ (매우 rough).

**구체 특성**:
- $H = 1/2$: standard BM, integrated volatility $\sqrt t$ scale
- $H < 1/2$: **rougher than BM**, Hölder $H - \epsilon$
- Empirical: $H \approx 0.05$ to $0.15$ in equity markets (Bayer 2016)

**Option pricing 영향**:
1. **ATM skew**: At-the-money implied vol skew $\sim t^{H - 1/2}$. $H = 0.5$ (standard)이면 zero skew at $t \to 0$ (Black-Scholes). $H < 1/2$면 **skew explodes** as $t \to 0$ — 실제 관찰된 "short-dated skew anomaly" 재현.

2. **Volatility clustering**: Rough process가 short-term autocorrelation 보여줌 — ARCH/GARCH보다 더 현실적.

3. **Variance term structure**: $\mathbb{E}[\sigma_t^2]$의 scaling이 $H$에 의존.

**Deep Learning 적용**:
- Estimation of $H$ from high-frequency data using NN (Bennedsen, Lunde, Pakkanen 2022).
- **Deep rough pricing**: fractional Brownian motion-based pricer가 closed-form 없이 PDE/Monte Carlo. NN이 fast approximator.
- **Signatures / rough path** features for NN input — fractional BM의 non-Markovian structure를 효율적으로 encode.

**한계**:
- Fractional BM은 non-Markovian — SDE의 standard 이론(이토 공식) 직접 적용 불가.
- Rough path theory (Lyons 1998), regularity structures (Hairer 2014)가 이론적 framework.

**실전 implementation**:
- QuantLib의 RoughHeston 모델
- Python에서 `fbm` package로 fBM 샘플링

**연결**: BM의 $1/2$ scaling의 universality (정리 4.2) vs financial data의 rough reality. Empirical finance가 mathematical BM을 넘어서 더 general process 요구.

</details>

---

<div align="center">

◀ [03. Random Walk Scaling Limit — Donsker 정리](./03-donsker-theorem.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [05. 이차변분 $\langle B\rangle_t = t$](./05-quadratic-variation.md)

</div>
