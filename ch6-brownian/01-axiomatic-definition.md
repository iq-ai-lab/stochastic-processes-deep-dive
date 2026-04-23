# 01. 브라운 운동의 공리적 정의

## 🎯 핵심 질문

- 브라운 운동(BM) $\{B_t\}_{t \geq 0}$을 정의하는 **4가지 공리** — (i) $B_0 = 0$, (ii) 독립증분, (iii) 정규 증분, (iv) 연속 경로 — 는 왜 이 조합인가?
- **자기유사성** $B_{ct} \stackrel{d}{=} \sqrt{c} B_t$는 공리에서 어떻게 유도되는가?
- 공분산 $\mathbb{E}[B_s B_t] = \min(s, t)$이 이 정의의 직접적 결과인 이유는?
- 네 공리가 서로 **독립적**인가, 아니면 중복인가?

---

## 🔍 왜 BM이 AI에서 중요한가

**Diffusion Model의 core**: Forward process $dX_t = f dt + g dB_t$가 BM 기반. DDPM·Score-SDE의 수학적 토대.

**Gaussian Process의 연속시간 원형**: BM은 가장 기본적인 Gaussian process, kernel $k(s, t) = \min(s, t)$.

**Langevin dynamics**: MCMC의 연속 한계에 BM이 "noise source". Bayesian deep learning의 core.

**Reinforcement learning의 exploration**: Continuous-time control (HJB equation)에 BM이 무작위성 도입.

**수학적 무결성**: BM이 "연속이지만 미분불가능", "이차변분 $\langle B\rangle_t = t$" 같은 **paradoxical** 성질을 가지며, 이것이 이토 적분(SDE Deep Dive Ch1)의 필연성.

---

## 📐 수학적 선행 조건

- [Ch1-01 ~ Ch1-04](../ch1-foundations/01-rigorous-definition.md): 확률과정, fdd, 필트레이션
- [Ch1-03](../ch1-foundations/03-stationarity.md): Gaussian process
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): 다변량 정규분포, 특성함수

---

## 📖 직관적 이해

### "가장 fundamental한 연속 random process"

BM은 **최소한의** 공리로 정의:
1. **시작점**: $B_0 = 0$ (origin)
2. **독립 과거·미래**: 겹치지 않는 구간의 증분 독립
3. **시간에 비례하는 분산**: $\text{Var}(B_t - B_s) = t - s$, 정규분포
4. **연속 경로**: 점프 없음

이 네 공리의 유일한 해가 BM (자기유사성, 스케일링 법칙 포함).

### 왜 정규분포인가

**CLT 관점** (Ch6-03 Donsker): 이산 random walk의 rescaled 극한이 BM. CLT로 증분은 정규분포.

**독립증분 + 정상성** → CLT 재귀적 적용 → 정규분포.

### 연속 경로의 의미

공리 (iv)는 **비자명**. 독립증분 + 정규 증분만 있는 과정이 반드시 연속 경로를 갖는 것은 아님. 명시적 요구 (또는 Kolmogorov continuity theorem, Ch1-02로 유도).

---

## ✏️ 엄밀한 정의

### 정의 1.1 — 브라운 운동 (Brownian Motion / Wiener Process)

확률공간 $(\Omega, \mathcal{F}, \mathbb{P})$ 위의 과정 $\{B_t\}_{t \geq 0}$이 **표준 브라운 운동**이다:
(B1) $B_0 = 0$
(B2) **독립증분**: 임의 $0 = t_0 < t_1 < \cdots < t_n$에 대해 $B_{t_1} - B_{t_0}, \ldots, B_{t_n} - B_{t_{n-1}}$가 독립
(B3) $B_t - B_s \sim \mathcal{N}(0, t - s)$ for $s < t$
(B4) 경로 $t \mapsto B_t(\omega)$가 거의 확실히 연속

### 정의 1.2 — 일반 브라운 운동

$\mu \in \mathbb{R}, \sigma > 0$. $X_t = \mu t + \sigma B_t$는 **drift $\mu$, 변동성 $\sigma$의 브라운 운동**. 표준 BM은 $\mu = 0, \sigma = 1$.

### 정의 1.3 — 다차원 브라운 운동

$\mathbf{B}_t = (B^1_t, \ldots, B^d_t)$가 **$d$차원 표준 BM**: 각 $B^i$가 표준 BM이고, 서로 독립.

---

## 🔬 정리와 증명

### 정리 1.1 — BM은 Gaussian Process

BM $\{B_t\}$는 평균 0, 공분산 $\mathbb{E}[B_s B_t] = \min(s, t)$의 Gaussian process.

*증명*.
(B2) + (B3) → 증분 independent Gaussian. 임의 $t_1, \ldots, t_n$에 대해 $(B_{t_1}, \ldots, B_{t_n})$이 iterated linear combination of Gaussians — 다변량 Gaussian.

공분산 ($s \leq t$): $B_t = B_s + (B_t - B_s)$. $B_s \perp (B_t - B_s)$ (independent increments).
$$\mathbb{E}[B_s B_t] = \mathbb{E}[B_s (B_s + (B_t - B_s))] = \mathbb{E}[B_s^2] + \mathbb{E}[B_s] \mathbb{E}[B_t - B_s] = s + 0 = s.$$

대칭성으로 $\min(s, t)$. $\square$

### 정리 1.2 — 자기유사성 (Scaling Invariance)

$c > 0$에 대해 $\{c^{-1/2} B_{ct}\}_{t \geq 0}$도 표준 BM.

*증명*.
$X_t := c^{-1/2} B_{ct}$. 네 공리 확인:
(B1): $X_0 = c^{-1/2} B_0 = 0$ ✓
(B2): $X_{t_1} - X_{t_0} = c^{-1/2}(B_{ct_1} - B_{ct_0})$ 등이 독립 (시간 $ct$ 간격에서 BM 독립증분)
(B3): $X_t - X_s = c^{-1/2}(B_{ct} - B_{cs}) \sim c^{-1/2} \mathcal{N}(0, c(t - s)) = \mathcal{N}(0, t - s)$ ✓
(B4): $B$ 연속 → $X$ 연속 ✓ $\square$

**의미**: BM은 **self-similar** with scaling exponent 1/2. 이는 Hausdorff 차원 분석, fractal 성질 (Ch6-04)의 기초.

### 정리 1.3 — 시간 역전

$X_t := B_1 - B_{1-t}$ on $[0, 1]$이 표준 BM.

*증명*.
$X_0 = 0$. $X_t - X_s = B_{1-s} - B_{1-t} \sim \mathcal{N}(0, t-s)$ (BM 증분의 시간 대칭). 독립증분 상속. 연속. $\square$

**의미**: BM의 "시간 방향은 본질적으로 대칭" — 이 reversibility가 SDE Ch6(reverse SDE)의 기초.

### 정리 1.4 — BM은 마팅게일

자연 필트레이션 $\{\mathcal{F}_t^B\}$에 대해 $B_t$는 martingale.

*증명*.
$\mathbb{E}[B_t | \mathcal{F}_s^B] = \mathbb{E}[B_s + (B_t - B_s) | \mathcal{F}_s^B] = B_s + 0 = B_s$ (independent increments + mean 0). $\square$

또한 $B_t^2 - t$, $\exp(\lambda B_t - \lambda^2 t/2)$도 martingale (Ch5의 문제 + SDE Ch2-05).

### 정리 1.5 — BM의 존재성

위 네 공리를 만족하는 확률과정이 **존재**한다.

*증명 스케치*. 두 방법:
1. **Lévy 구성** (Ch6-02): Haar 기저 + iid Gaussian 계수로 랜덤 급수.
2. **Kolmogorov 확장** (Ch1-02) + **Kolmogorov continuity**: fdd 가족이 Gaussian with covariance $\min(s,t)$. Kolmogorov 확장으로 (i)-(iii) 만족 과정. Kolmogorov continuity (Hölder 조건 $\mathbb{E}|B_t - B_s|^4 = 3(t-s)^2$)으로 (iv) 연속 modification. $\square$

### 정리 1.6 — BM의 전체 자유도 = Gaussian process

"독립증분 + 정상증분 + 연속 경로 + $B_0 = 0$" → **유일하게** 표준 BM (up to modification).

*증명*. 공분산이 $\min(s, t)$로 일관되게 결정 → fdd도 결정 → Kolmogorov로 유일. $\square$

---

## 💻 NumPy 구현 검증

### 실험 1 — 4가지 공리 검증

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

T = 10.0
N = 10000
dt = T / N

# BM 시뮬: independent N(0, dt) increments
dB = rng.standard_normal((N,)) * np.sqrt(dt)
B = np.concatenate([[0], np.cumsum(dB)])
t_grid = np.linspace(0, T, N + 1)

# 공리 (1) B_0 = 0
print(f'B_0 = {B[0]}')

# 공리 (3) B_t - B_s ~ N(0, t-s)
n_sim = 10000
n_samples = rng.standard_normal((n_sim, N)) * np.sqrt(dt)
B_sim = np.cumsum(n_samples, axis=1)

t_test, s_test = 5.0, 2.0
t_idx, s_idx = int(t_test/dt), int(s_test/dt)
increments = B_sim[:, t_idx-1] - B_sim[:, s_idx-1]
print(f'\nB_{t_test} - B_{s_test}:')
print(f'  평균: {increments.mean():.4f} (이론: 0)')
print(f'  분산: {increments.var():.4f} (이론: {t_test - s_test})')

# 공리 (2) 독립증분 확인 (Correlation)
inc1 = B_sim[:, 1000] - B_sim[:, 500]   # [5, 10] dt
inc2 = B_sim[:, 2000] - B_sim[:, 1500]  # [15, 20] dt
print(f'\nCorr(inc1, inc2) = {np.corrcoef(inc1, inc2)[0,1]:.4f} (이론: 0)')

# 공리 (4) 경로 연속 (visual)
plt.plot(t_grid, B)
plt.xlabel('t'); plt.ylabel(r'$B_t$')
plt.title('Brownian motion sample path')
plt.grid(True, alpha=0.3); plt.show()
```

### 실험 2 — 공분산 $\min(s, t)$ 검증

```python
# n_sim 경로 생성, 여러 시점 pair의 covariance
pairs = [(0.2, 0.5), (0.5, 0.5), (0.3, 0.8), (0.5, 1.2)]
for s_rel, t_rel in pairs:
    s, t = s_rel * T, t_rel * T
    Bs = B_sim[:, int(s/dt)-1]
    Bt = B_sim[:, int(t/dt)-1]
    cov_emp = np.cov(Bs, Bt)[0, 1]
    print(f'Cov(B_{s}, B_{t}) = {cov_emp:.4f}, 이론 min(s,t) = {min(s,t)}')
```

### 실험 3 — 자기유사성 $B_{ct} \stackrel{d}{=} \sqrt{c} B_t$

```python
c = 4.0   # scale factor
# B_{c·t} 분포
t_fixed = 1.0
B_scaled = B_sim[:, int(c*t_fixed/dt) - 1]

# sqrt(c) * B_t 분포
B_original = B_sim[:, int(t_fixed/dt) - 1]
B_rescaled = np.sqrt(c) * B_original

print(f'c = {c}, t = {t_fixed}')
print(f'B_{c*t_fixed} 분산: {B_scaled.var():.4f}')
print(f'√c · B_{t_fixed} 분산: {B_rescaled.var():.4f}')
print(f'이론 c·t = {c*t_fixed}')
# 일치 → 자기유사성 확인

# 두 분포의 KS test도 가능
from scipy.stats import ks_2samp
stat, pval = ks_2samp(B_scaled, B_rescaled)
print(f'KS test p-value: {pval:.4f} (>> 0.05면 같은 분포 기각 불가)')
```

---

## 🔗 AI/ML 연결

**DDPM Forward Process**  
VP-SDE: $dX_t = -\frac{1}{2}\beta(t) X dt + \sqrt{\beta(t)} dB_t$. Noise term $dB$가 표준 BM. 각 step에서 "Gaussian noise 추가"가 (B3) 공리의 infinitesimal 형태.

**Gaussian Process Regression**  
BM과 같은 covariance $k(s, t) = \min(s, t)$는 **integrated Wiener kernel**이라 불림. 보다 일반적 RBF, Matérn 등의 kernel이 BM 변형을 base로.

**Reparameterization Trick**  
VAE의 $z = \mu + \sigma \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$는 "시각 1에서의 BM" 샘플. 더 일반적으로 $z \sim p_\theta$를 "standard Gaussian의 transform"으로 재구성.

**Score Function과 BM의 Generator**  
Infinitesimal generator $\mathcal{L}_{\text{BM}} = \frac{1}{2}\Delta$ (Laplacian). $\nabla \log p_t$가 score, Langevin: $dX = \nabla \log p dt + \sqrt{2} dB$.

**Physics-informed NN**  
Heat equation $\partial_t u = \frac{1}{2} \Delta u$의 solution은 BM의 transition density ($u(t, x) = \mathbb{E}_x[f(B_t)]$). Feynman-Kac의 직접 응용.

---

## ⚖️ 가정과 한계

**가정 — Gaussian 증분**  
(B3)이 정규분포 가정. 다른 증분 분포 → **Lévy process** (jump) or **$\alpha$-stable process** (heavy-tailed). BM은 "continuous Lévy"의 유일한 예.

**한계 — 비현실적 "경로의 roughness"**  
BM 경로는 **Hölder $1/2 - \epsilon$** — "매우 들쭉날쭉". 실제 물리 현상은 더 smooth (평균화) 또는 더 rough (turbulence). **Fractional BM**으로 Hölder 지수 조절.

**한계 — Markov 속성**  
BM이 Markov. 현실 경제·physics 중 일부는 non-Markov (long-range dependence) → fractional BM (Hurst parameter ≠ 1/2).

---

## 📌 핵심 정리

| 공리 | 수식 |
|---|---|
| B1 | $B_0 = 0$ |
| B2 | Independent increments |
| B3 | $B_t - B_s \sim \mathcal{N}(0, t-s)$ |
| B4 | a.s. continuous paths |
| Covariance | $\mathbb{E}[B_s B_t] = \min(s, t)$ |
| Self-similarity | $B_{ct} \stackrel{d}{=} \sqrt c B_t$ |
| Martingale | $B_t$, $B_t^2 - t$, $e^{\lambda B_t - \lambda^2 t/2}$ |

**한 줄 요약**: 브라운 운동은 4가지 공리 (초기 0, 독립증분, Gaussian 증분, 연속 경로)로 정의되는 **Gaussian process**. $\min(s, t)$ 공분산 + $\sqrt{c}$ self-similarity가 모든 성질의 근원이며, Diffusion model·SDE·확률해석의 기본 building block.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. $\text{Cov}(B_{0.3}, B_{0.5})$와 $\text{Var}(B_{0.5} - B_{0.3})$을 BM 성질로 계산하라.

<details>
<summary>해설</summary>

**공분산**: $\text{Cov}(B_{0.3}, B_{0.5}) = \min(0.3, 0.5) = 0.3$ (정리 1.1).

**증분 분산**: $B_{0.5} - B_{0.3} \sim \mathcal{N}(0, 0.5 - 0.3) = \mathcal{N}(0, 0.2)$ (B3). $\text{Var} = 0.2$.

**검증**: $\text{Var}(B_{0.5} - B_{0.3}) = \text{Var}(B_{0.5}) - 2\text{Cov}(B_{0.3}, B_{0.5}) + \text{Var}(B_{0.3}) = 0.5 - 0.6 + 0.3 = 0.2$ ✓.

</details>

**문제 2 (심화)**. $X_t := t B_{1/t}$ (with $X_0 = 0$)이 표준 BM임을 보여라 (time inversion).

<details>
<summary>해설</summary>

**공리 체크**:

(B1) $X_0 = 0$ ✓ (정의).

(B3) Gaussian marginals: 각 $X_t$가 $t \cdot \mathcal{N}(0, 1/t) = \mathcal{N}(0, t)$ ✓.

**Joint distribution (Gaussian 이므로 평균·공분산으로 완전 결정)**:
$\text{Cov}(X_s, X_t) = \text{Cov}(sB_{1/s}, tB_{1/t}) = st \min(1/s, 1/t) = st/\max(s, t) = \min(s, t)$ ✓.

공분산이 min(s, t) → BM과 같은 Gaussian process → **distribution 동일**.

(B2) Independent increments: 공분산 구조만으로 증분 독립 유도 가능 (Gaussian orthogonality).

(B4) $X$ 연속: 경로 $t \mapsto t B_{1/t}$가 $t = 0$에서 연속인지가 issue. Law of iterated logarithm으로 a.s. $\lim_{t \to 0^+} t B_{1/t} = 0$ → 연속.

**의미**:
- BM은 **"time inversion" symmetric**.
- $B_t$ for $t \in (0, 1]$와 $X_t = t B_{1/t}$ for $t \in [1, \infty)$ 연결 → "0 근처 거동 = ∞ 근처 거동"의 dual.
- 이 대칭이 reflection principle (Ch6-06), iterated logarithm 법칙의 기반.

</details>

**문제 3 (AI 연결)**. VP-SDE forward process $dX_t = -\frac{1}{2}\beta(t) X_t dt + \sqrt{\beta(t)} dB_t$의 해 $X_t$가 Gaussian random variable임을 보이고, marginal $p_t(x)$의 공식을 구하라.

<details>
<summary>해설</summary>

**Linear SDE with multiplicative noise는 Gaussian 해** — 초기 $X_0$가 Gaussian이면.

**해 공식** (SDE Ch3-05 참조): VP-SDE의 경우 이토 공식 or 적분인자 방법으로
$$X_t = e^{-\frac{1}{2}\int_0^t \beta(s) ds} X_0 + \int_0^t e^{-\frac{1}{2}\int_s^t \beta(u) du} \sqrt{\beta(s)} dB_s.$$

$\alpha_t := e^{-\frac{1}{2}\int_0^t \beta(s) ds}$로 정의. 그러면
$$X_t = \alpha_t X_0 + \sqrt{1 - \alpha_t^2} \cdot Z, \quad Z \sim \mathcal{N}(0, I) \text{ indep of } X_0.$$

(Ito isometry로 noise 부분의 분산 계산: $\int_0^t e^{-\int_s^t \beta} \beta(s) ds = 1 - \alpha_t^2$.)

**Marginal**: $X_0 \sim p_{\text{data}}$이면
$$p_t(x) = \int p_0(x_0) \mathcal{N}(x; \alpha_t x_0, (1 - \alpha_t^2) I) dx_0.$$

**핵심 특성**:
- $t \to 0$: $\alpha_t \to 1$ → $X_t \to X_0$ (data distribution).
- $t \to T$ (large): $\alpha_t \to 0$ → $X_t \to \mathcal{N}(0, I)$ (standard Gaussian).

**BM과의 관계**:
- Noise source $dB_t$가 표준 BM.
- Drift term $-\frac{1}{2}\beta(t) X$가 "data → origin" mean-reversion.
- 전체: **time-varying Ornstein-Uhlenbeck**.

**Training의 의미**:
- 각 $t$에서 $X_t$ 샘플 = $\alpha_t X_0 + \sqrt{1 - \alpha_t^2} Z$로 쉽게 생성.
- $\epsilon_\theta$가 $Z$ 예측하도록 훈련 → marginal distribution의 score $\nabla \log p_t$ 학습.

**이것이 DDPM의 수학적 정확한 기반**:
- 이산 DDPM: $\beta_t$가 discrete → $\alpha_t = \prod(1 - \beta_s)$.
- 연속 VP-SDE: $\alpha_t = e^{-\frac{1}{2}\int \beta}$.
- 둘의 동치성이 Score-SDE의 핵심 결과.

**연결**: SDE Deep Dive Ch3 (SDE 해), Ch6 (Score-SDE), Ch4 (Fokker-Planck for $p_t$)에서 자세히.

</details>

---

<div align="center">

◀ [Ch5-05. 마팅게일과 ML — Online Learning](../ch5-martingale/05-martingale-ml.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [02. 존재성 — Lévy의 Haar 기저 구성](./02-levy-construction.md)

</div>
