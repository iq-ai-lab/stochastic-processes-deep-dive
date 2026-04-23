# 03. 정상성(Stationarity) — 강·약

## 🎯 핵심 질문

- **엄격 정상(strict stationarity)**과 **약 정상(weak / covariance stationarity)**은 어떻게 다른가?
- 엄격 ⇒ 약이지만 역은 성립하지 않는 이유는 무엇이고, **가우시안 과정**에서는 왜 두 개념이 일치하는가?
- 약 정상은 왜 2차 모멘트만 보는가 — 스펙트럴 표현정리(Wiener-Khinchin)로 주파수 영역 해석이 가능해지는가?
- 시계열 모델의 **정상화(differencing, detrending)**은 왜 필요하고, 어떤 정상 가정 위에서 정당화되는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**시계열 예측**(Transformer, LSTM, Mamba)의 통계적 분석은 "입력이 정상 과정"이라는 가정 하에 자주 수행된다. 정상성이 깨지면 iid 샘플 가정이 무너지고 **학습-테스트 분포 시프트**가 발생한다. **금융 시계열**(주가, 변동성)은 로그 수익률 수준에서만 약 정상이고, 이 사실이 Black-Scholes·GARCH 모델의 정당성의 핵심이다(SDE Deep Dive Ch3).

**MCMC 이론**(Ch7)에서 **정상분포**는 한 시점 분포가 아닌 "모든 시점에서 같은 marginal" — 즉 엄격 정상의 시간 시프트 불변성과 직결된다. 에르고딕 정리(Ch2-06)는 정상 과정에서 시간평균 = 공간평균을 보장하는데, 이것이 MCMC 샘플링의 이론적 근거다.

**Self-supervised 시계열 학습**(예: masked autoencoder, contrastive learning)에서 **"시간 시프트가 의미적으로 같다"**는 가정은 사실 엄격 정상성. 이 가정이 깨지는 상황(regime change, trend)에서는 모델 성능이 급락.

---

## 📐 수학적 선행 조건

- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): 결합분포의 동일성, 특성함수, 2차 모멘트
- 푸리에 해석 기초: 실수 위의 Fourier transform, 정의역 해석
- [Ch1-01](./01-rigorous-definition.md): fdd, Ch1-02 Kolmogorov 확장

---

## 📖 직관적 이해

### 두 가지 정상성

시간 시프트 $\theta_h : T \to T$, $\theta_h(t) = t + h$를 고려하자. 정상성은 "시간을 $h$만큼 옮겨도 분포가 같은가"에 대한 답이며, **얼마나 강한 의미로 같은지**에 따라 두 가지.

**엄격 정상(strict stationarity)**  
모든 $n$, 모든 $t_1, \ldots, t_n$, 모든 $h$에 대해
$$(X_{t_1}, \ldots, X_{t_n}) \stackrel{d}{=} (X_{t_1 + h}, \ldots, X_{t_n + h}).$$
**fdd 전체가 시프트 불변**. 가장 강한 의미.

**약 정상(weak / wide-sense / covariance stationarity)**  
$\mathbb{E}[X_t] = \mu$ (시각 무관)이고, 공분산이 시차에만 의존:
$$\text{Cov}(X_s, X_t) = C(t - s).$$
**1차, 2차 모멘트만** 시프트 불변.

> **비유**: 엄격 정상은 "영화 전체가 loop"인 것. 약 정상은 "영화의 밝기·대비만 시간 무관"인 것. 전자가 훨씬 강한 조건.

### 왜 약 정상이 유용한가

엄격 정상은 확인하기 어렵다 — 모든 fdd가 시프트 불변임을 검증해야 하니. 반면 약 정상은 평균과 공분산만 추정하면 되고, **스펙트럴 밀도**를 통해 주파수 영역 해석이 가능(Wiener-Khinchin). Box-Jenkins ARMA·ARIMA의 모든 이론이 약 정상 위에서 펼쳐진다.

### 가우시안 과정에서의 특수성

가우시안 과정의 fdd는 평균과 공분산만으로 **완전히 결정**된다. 따라서 평균·공분산이 시프트 불변이면(약 정상) fdd도 자동 시프트 불변(엄격 정상). **가우시안에서는 두 개념이 일치**.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — 엄격 정상 (Strict Stationarity)

확률과정 $\{X_t\}_{t \in T}$ ($T = \mathbb{Z}$ 또는 $\mathbb{R}$)가 **엄격 정상**이라는 것은, 모든 $n \geq 1$, 모든 $t_1, \ldots, t_n \in T$, 모든 $h \in T$에 대해
$$(X_{t_1 + h}, X_{t_2 + h}, \ldots, X_{t_n + h}) \stackrel{d}{=} (X_{t_1}, X_{t_2}, \ldots, X_{t_n}).$$

### 정의 3.2 — 약 정상 (Weak / Covariance Stationarity)

$\mathbb{E}[X_t^2] < \infty$인 확률과정 $\{X_t\}$가 **약 정상**이라는 것은:
1. $\mathbb{E}[X_t] = \mu$ (상수, $t$ 무관)
2. $\text{Cov}(X_s, X_t) = C(t - s)$ (시차의 함수).

함수 $C : T \to \mathbb{R}$을 **autocovariance function**이라 한다. $\rho(h) = C(h) / C(0)$이 autocorrelation.

### 정의 3.3 — 스펙트럴 밀도 (Spectral Density)

$T = \mathbb{Z}$의 약 정상 과정에서 $C(h)$가 절대가합이면
$$f(\omega) = \frac{1}{2\pi} \sum_{h \in \mathbb{Z}} C(h) e^{-i h \omega}, \quad \omega \in [-\pi, \pi]$$
를 **스펙트럴 밀도**라 하며, 역변환 $C(h) = \int_{-\pi}^{\pi} e^{i h \omega} f(\omega) d\omega$.

### 정의 3.4 — Autocovariance function의 양정치성

$C : T \to \mathbb{R}$이 **양정치(positive semi-definite)**라는 것은 모든 유한 $t_1, \ldots, t_n$, 모든 $c_1, \ldots, c_n \in \mathbb{C}$에 대해
$$\sum_{i, j} c_i \bar{c_j} C(t_i - t_j) \geq 0.$$

---

## 🔬 정리와 증명

### 정리 3.1 (엄격 정상 ⇒ 약 정상 if 2차 모멘트 존재)

$\{X_t\}$가 엄격 정상이고 $\mathbb{E}[X_t^2] < \infty$이면 약 정상이다.

*증명.* 엄격 정상으로 $(X_t) \stackrel{d}{=} (X_{t+h})$이므로 $\mathbb{E}[X_t] = \mathbb{E}[X_{t+h}]$ — 평균은 $t$ 무관. 또한 $(X_s, X_t) \stackrel{d}{=} (X_{s-s}, X_{t-s}) = (X_0, X_{t-s})$이므로
$$\text{Cov}(X_s, X_t) = \text{Cov}(X_0, X_{t-s}) =: C(t-s).$$
$\square$

### 정리 3.2 (약 정상 ⇏ 엄격 정상 — 반례)

약 정상이지만 엄격 정상이 아닌 과정이 존재.

*반례.* $X_t$를 iid 시퀀스로, 각 $X_t$가 다음 분포를 갖도록 설정: 확률 $1/2$로 $\mathcal{N}(0, 1)$에서, 확률 $1/2$로 Laplace(평균 0, 분산 1)에서 샘플. 평균 0, 분산 1, 독립이므로 $C(h) = \delta_{h, 0}$ (약 정상). 그러나 $X_t$의 주변 분포가 "가우시안과 라플라스의 혼합"으로 **비가우시안**이고, 예를 들어 3차 모멘트 $\mathbb{E}[X_t^3]$가 **0**이지만 4차 모멘트는 시점마다 동일한 분포이긴 해도, 이 예에서는 시프트 불변이 성립해버림.

**더 확실한 반례**: $X_0, X_1, X_2, \ldots$를 다음과 같이 구성 — $X_0 \sim \mathcal{N}(0, 1)$, $X_t = X_0 \cdot \epsilon_t$ ($t \geq 1$, $\epsilon_t \in \{\pm 1\}$ iid, $X_0$와 독립). 그러면
- $\mathbb{E}[X_t] = 0$
- $\mathbb{E}[X_t^2] = 1$
- $\text{Cov}(X_s, X_t) = \mathbb{E}[X_0^2] \mathbb{E}[\epsilon_s \epsilon_t] = \delta_{s,t}$ ($s, t \geq 1$), $\text{Cov}(X_0, X_t) = 0$ ($t \geq 1$).

하지만 $(X_1, X_2)$는 구조적으로 $X_0 \epsilon_1, X_0 \epsilon_2$로 의존성을 가지며, $(X_1, X_2) \not\stackrel{d}{=} (X_{1+h}, X_{2+h})$ (예: $h = -1$이면 $(X_0, X_1)$의 분포는 $X_0$와 $X_0 \epsilon_1$ — 다른 joint 분포). $\square$

### 정리 3.3 (가우시안 과정: 엄격 ⇔ 약)

$\{X_t\}$가 **가우시안 과정**이면 엄격 정상과 약 정상이 동치.

*증명.* 약 정상에서 시작해 엄격 정상을 유도. 임의 $t_1, \ldots, t_n$과 $h$에 대해 $(X_{t_1+h}, \ldots, X_{t_n+h})$는 다변량 가우시안이며 그 분포는 평균 벡터와 공분산 행렬로 완전 결정. 약 정상으로
$$\mathbb{E}[X_{t_i + h}] = \mu = \mathbb{E}[X_{t_i}], \quad \text{Cov}(X_{t_i+h}, X_{t_j+h}) = C(t_i - t_j) = \text{Cov}(X_{t_i}, X_{t_j}).$$
평균과 공분산이 일치하므로 같은 다변량 가우시안 — fdd가 같음. 반대 방향은 정리 3.1. $\square$

### 정리 3.4 (Bochner / Herglotz 정리 — autocovariance의 특성화)

$T = \mathbb{Z}$의 약 정상 과정 $\{X_t\}$의 autocovariance $C(h)$는 **양정치 함수**이며, 역으로 양정치 함수 $C$는 어떤 약 정상 가우시안 과정의 autocovariance로 실현된다. 또한 절대가합 $C$는 비음 스펙트럴 밀도 $f \geq 0$의 푸리에 계수:
$$C(h) = \int_{-\pi}^{\pi} e^{ih\omega} f(\omega) d\omega.$$

*증명 스케치.* 양정치성: $\sum_{i,j} c_i \bar{c_j} C(t_i - t_j) = \text{Var}(\sum c_i X_{t_i}) \geq 0$. 역방향은 Kolmogorov 확장(Ch1-02)으로 $C$를 공분산으로 하는 가우시안 과정 구성. 푸리에 변환은 Bochner 정리에 의해 비음 측도로 표현되고, 밀도 $f$ 존재 시 위 식. $\square$

### 정리 3.5 (Wiener-Khinchin 정리 / 스펙트럴 표현)

약 정상 과정 $\{X_t\}_{t \in \mathbb{R}}$의 autocovariance $C(h)$가 연속이고 절대가적이면, 스펙트럴 밀도 $f \geq 0$가 존재해
$$C(h) = \int_{-\infty}^{\infty} e^{ih\omega} f(\omega) d\omega.$$

스펙트럴 밀도 $f$는 과정의 "주파수별 에너지 분포"를 기술하며, ARMA 모델·칼만 필터의 주파수 영역 해석의 출발점.

---

## 💻 NumPy 구현 검증

### 실험 1 — AR(1) 과정의 약 정상성 확인

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
phi = 0.7    # AR(1): X_t = φ X_{t-1} + ε_t, ε_t ~ N(0, 1)
N = 100_000
X = np.zeros(N)
# 정상분포에서 시작: X_0 ~ N(0, 1/(1-φ²))
X[0] = rng.standard_normal() * np.sqrt(1 / (1 - phi**2))
for t in range(1, N):
    X[t] = phi * X[t-1] + rng.standard_normal()

# 이론 autocovariance: C(h) = φ^|h| / (1 - φ²)
def autocov_empirical(x, max_lag):
    x = x - x.mean()
    return np.array([np.mean(x[:-h] * x[h:]) if h > 0 else np.var(x) 
                     for h in range(max_lag + 1)])

max_lag = 20
C_hat = autocov_empirical(X, max_lag)
C_true = phi**np.arange(max_lag + 1) / (1 - phi**2)

print(f'{"h":>3} {"실측 C(h)":>12} {"이론 C(h)":>12}')
for h in range(0, 11):
    print(f'{h:>3} {C_hat[h]:>12.4f} {C_true[h]:>12.4f}')

# 실측과 이론이 일치 → AR(1)은 약 정상, C(h) = φ^|h|/(1-φ²)
```

### 실험 2 — 가우시안 vs 비가우시안에서 엄격 정상성 차이

```python
# AR(1) with 가우시안 noise vs 비가우시안 noise
# 둘 다 약 정상이지만 가우시안만 엄격 정상
# 분위수(quantile) 일치로 대체 검증

N = 50_000
phi = 0.7

# 가우시안 noise
X_g = np.zeros(N)
X_g[0] = rng.standard_normal() * np.sqrt(1 / (1 - phi**2))
for t in range(1, N):
    X_g[t] = phi * X_g[t-1] + rng.standard_normal()

# 비가우시안 (chi²-1 noise) — 비대칭
eps_ng = rng.chisquare(1, size=N) - 1   # 평균 0, 분산 2
X_ng = np.zeros(N)
X_ng[0] = eps_ng[0] / np.sqrt(1 - phi**2)
for t in range(1, N):
    X_ng[t] = phi * X_ng[t-1] + eps_ng[t]

# "기울이 일치"를 3차 모멘트(skewness)로 비교
# 엄격 정상이면 skewness가 시점 무관이지만 비가우시안 AR(1)도 그렇긴 함
# 그러나 joint 3차 교차 모멘트는 다름
print(f'가우시안 AR(1):   X_t³ skew = {((X_g**3).mean() / (X_g**2).mean()**1.5):.4f}')
print(f'비가우시안 AR(1): X_t³ skew = {((X_ng**3).mean() / (X_ng**2).mean()**1.5):.4f}')
# 비가우시안에서는 skew ≠ 0 — 약 정상은 만족하지만 경우에 따라 joint 관점 차이
```

### 실험 3 — Wiener-Khinchin: 경험 스펙트럴 밀도와 autocovariance의 푸리에

```python
# AR(1)의 이론 스펙트럴 밀도 f(ω) = 1 / (2π |1 - φ e^{-iω}|²)
N = 2**14
X = np.zeros(N)
X[0] = rng.standard_normal() * np.sqrt(1 / (1 - phi**2))
for t in range(1, N):
    X[t] = phi * X[t-1] + rng.standard_normal()

# Periodogram (경험 스펙트럴 밀도)
X_fft = np.fft.rfft(X - X.mean())
periodogram = (np.abs(X_fft) ** 2) / N

# 평활화(smoothing) — Bartlett window
window = 32
sm = np.convolve(periodogram, np.ones(window)/window, mode='same')

omega = np.linspace(0, np.pi, len(sm))
f_true = 1 / (2 * np.pi * np.abs(1 - phi * np.exp(-1j * omega))**2)

plt.loglog(omega[1:], sm[1:], alpha=0.5, label='경험 스펙트럼(smoothed)')
plt.loglog(omega[1:], f_true[1:], 'k--', label='이론 f(ω)')
plt.xlabel('ω (rad)'); plt.ylabel('spectral density')
plt.legend(); plt.title('Wiener-Khinchin — AR(1) 스펙트럴 밀도'); plt.grid(True)
plt.show()
```

---

## 🔗 AI/ML 연결

**시계열 예측**  
Transformer/LSTM 학습에서 입력을 "정상"으로 만들기 위해 **differencing**($\Delta X_t = X_t - X_{t-1}$), **detrending**(추세 제거), **deseasonalization**을 적용. 이는 원 과정이 엄격 정상이 아니라 **"차분 후 정상"**(ARIMA의 Integrated 부분)이라는 가정.

**Gaussian Process 회귀의 stationary kernel**  
RBF·Matérn 커널은 $k(x, x') = k(\|x - x'\|)$ 형태로 거리에만 의존 — **공간적 엄격 정상성**(space-homogeneous). 입력 space에서 translation invariance를 가정하는 것과 같다. Neural Tangent Kernel은 일반적으로 이 불변성이 없어 학습의 불균형 영역이 생긴다.

**Self-supervised pretraining**  
Masked Autoencoder(MAE)·SimCLR 같은 기법은 "augmentation이 의미를 바꾸지 않는다"는 불변성을 요구 — 이산 group action에 대한 "분포 불변성"의 한 형태. 시계열에서 "시간 시프트에 대한 불변성"은 엄격 정상성과 동치.

**RL에서의 stationary policy**  
$\pi(a|s)$가 시간에 의존하지 않는 정책(stationary policy) 하에서 state process가 정상 마르코프 체인이 되고, 정상분포 $d^\pi(s)$가 정의. policy gradient 이론(PPO, TRPO)이 이 정상성 위에서 편미분을 취한다.

**Diffusion Model의 non-stationarity**  
Forward process $X_t$는 시간 $t$에 따라 분산이 바뀌므로(VP-SDE) **정상이 아님**. 대신 reverse process를 학습할 때 시간 embedding $t$를 조건부로 받는다 — 이는 "non-stationary process"를 명시적으로 처리하는 표준 기법.

---

## ⚖️ 가정과 한계

**가정 — 2차 모멘트 존재**  
약 정상은 $\mathbb{E}[X_t^2] < \infty$ 없이는 정의되지 않음. 금융에서 **$\alpha$-stable 과정**(0 < α < 2)은 무한 분산이므로 약 정상 개념이 적용 안 됨 — 대신 엄격 정상만 쓸 수 있다.

**한계 — 정상성 검정**  
현실 시계열에서 정상성은 "가정"이지 증명하기 어려움. ADF 검정, KPSS 검정 등은 특정 대안(unit root 등)에 대한 테스트일 뿐. **Regime change**가 있으면 어느 검정도 정상성을 지지하지 못함.

**한계 — locally stationary process**  
"전체적으로는 비정상, 짧은 구간에서는 정상"인 과정(Dahlhaus의 locally stationary)은 실전에서 더 현실적이지만, 약 정상 이론의 직접 적용이 어렵다.

---

## 📌 핵심 정리

| 개념 | 요약 |
|------|------|
| 엄격 정상 | 모든 fdd가 시간 시프트 불변 |
| 약 정상 | 평균 상수 + 공분산이 시차 함수 |
| 관계 | 엄격+2차 모멘트 ⇒ 약; 역 일반적으로 거짓 |
| 가우시안 과정 | 두 정상성이 **동치** |
| autocovariance | $C(h)$는 양정치 함수, Bochner 정리 |
| Wiener-Khinchin | $C(h) = \int e^{ih\omega} f(\omega) d\omega$ (스펙트럴 밀도) |

**한 줄 요약**: 엄격 정상은 fdd 시프트 불변, 약 정상은 2차 모멘트만 불변. 가우시안 과정에서는 같고, 일반에서는 다르다.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. AR(1) 과정 $X_t = \phi X_{t-1} + \epsilon_t$ ($\epsilon_t \sim $ iid zero-mean, 분산 $\sigma^2$)이 약 정상이려면 $|\phi|$에 어떤 조건이 필요한가? 이때 $C(h)$는?

<details>
<summary>해설</summary>

$|\phi| < 1$가 필요 (그렇지 않으면 분산이 발산).  
정상분포 존재 시 $\text{Var}(X_t) = \phi^2 \text{Var}(X_{t-1}) + \sigma^2$로 $\text{Var}(X_t) = \sigma^2 / (1 - \phi^2)$.  
$C(h) = \mathbb{E}[X_t X_{t+h}] = \phi |h| \sigma^2 / (1 - \phi^2)$ ($h \geq 0$일 때 반복 적용; 일반적으로 $\phi^{|h|} \sigma^2/(1-\phi^2)$).

**주의**: 초기값을 정상분포에서 뽑지 않으면 초반에는 비정상, 하지만 기하급수적으로 정상화. $|\phi| = 1$이면 random walk — 분산이 $t\sigma^2$로 발산, **비정상**.

</details>

**문제 2 (심화)**. $\{X_t\}$가 엄격 정상이고 $Y_t = g(X_t, X_{t+1}, \ldots, X_{t+k})$ (유한 시차의 함수)라 하자. $\{Y_t\}$도 엄격 정상임을 보여라. 함수의 출력이 확률변수 벡터일 때도 성립하는가?

<details>
<summary>해설</summary>

임의 $n, t_1, \ldots, t_n, h$에 대해
$$(Y_{t_1 + h}, \ldots, Y_{t_n + h}) = (g(X_{t_1+h:t_1+h+k}), \ldots, g(X_{t_n+h:t_n+h+k})).$$
$(X_{t_1+h:t_1+h+k}, \ldots, X_{t_n+h:t_n+h+k}) \stackrel{d}{=} (X_{t_1:t_1+k}, \ldots, X_{t_n:t_n+k})$이 엄격 정상 가정에서 성립. $g$는 deterministic 측정가능 함수이므로 공통으로 적용하면 분포 같음. 벡터 출력도 마찬가지 — $g : \mathbb{R}^{k+1} \to \mathbb{R}^m$도 같은 논리.

**교훈**: 엄격 정상 과정의 **Markov-sliding** 변환도 엄격 정상. 이는 NN feature extraction이 정상성을 보존함을 뜻한다 — self-supervised 학습의 이론적 안전망.

</details>

**문제 3 (AI 연결)**. Diffusion forward process $X_t = \sqrt{\bar\alpha_t} X_0 + \sqrt{1 - \bar\alpha_t} \epsilon$ ($\bar\alpha_t$가 시간 감소)는 약 정상이 아니다. 왜인가? 이 비정상성을 학습이 처리하는 메커니즘은 무엇인가?

<details>
<summary>해설</summary>

**비정상성**: $\text{Var}(X_t) = \bar\alpha_t \text{Var}(X_0) + (1 - \bar\alpha_t)$ — 이는 $t$의 함수로 변화. 구체적으로 VP-SDE에서 $\bar\alpha_t$는 단조 감소하여 $t = 0$에서 데이터 분산, $t = T$에서 1(순수 노이즈 분산)로. 따라서 marginal 분포 $p_t$ 자체가 시간에 의존 — **약 정상 아님**(평균은 0으로 같지만 분산이 $t$에 의존).

**학습 처리**: Score network $s_\theta(x, t)$는 **시간 $t$를 명시적 입력**으로 받는다 — 이는 "non-stationary process를 조건부 모델링으로 처리"하는 것. 즉 각 $t$마다 서로 다른 노이즈 수준을 알고 있으므로 비정상성이 문제가 되지 않음.

**대조**: 만약 score network가 $t$를 모르면 학습이 붕괴 — 모든 noise level을 "평균적으로" 처리하려다 성능 급락. 이는 **"정상화를 통한 학습"과 "조건부 학습을 통한 비정상 처리"** 두 접근의 대조적 예시.

**연결**: SDE Deep Dive Ch4(Fokker-Planck)에서 $p_t$의 시간 진화를 PDE로 기술 — 비정상 marginal을 정량적으로 추적하는 방법.

</details>

---

<div align="center">

◀ [02. 유한차원 분포와 Kolmogorov 확장정리](./02-kolmogorov-extension.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [04. 필트레이션(Filtration)과 정보 흐름](./04-filtration.md)

</div>
