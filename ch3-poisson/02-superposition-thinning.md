# 02. 결합·분할과 비균질 Poisson

## 🎯 핵심 질문

- 독립 Poisson 과정의 **합(superposition)**은 왜 또 Poisson이고, rate가 더해지는가?
- 각 이벤트를 독립 확률 $p$로 keep하는 **thinning**이 왜 $p\lambda$ rate의 Poisson을 주는가?
- **비균질 Poisson**(rate $\lambda(t)$)은 어떻게 정의되고, 균질 Poisson으로의 시간재척도(time change)가 어떻게 작동하는가?
- 공간 Poisson point process로 확장될 때 "rate measure" $\Lambda$가 왜 모든 것을 결정하는가?

---

## 🔍 왜 이 개념들이 AI에서 중요한가

**Trading/이벤트 스트림 집계**: 여러 sensor/source의 이벤트를 하나의 stream으로 합침 (superposition). 각 source가 Poisson이면 합도 Poisson → 통합 분석 가능.

**Sampling / Subsampling**: 대용량 로그에서 무작위 서브샘플링 (thinning) → 축소된 데이터도 Poisson. Dataset distillation의 기초 이론.

**비균질 rate 학습**: 시간 의존 arrival rate $\lambda(t)$를 NN으로 추정 (Neural TPP) — 광고 click-through rate의 시간대 변화, 뉴스 유행 곡선 등.

**Spatial Point Process**: 의료 영상 (tumor locations), 천문학 (별 위치), geolocation 분석. 2D/3D Poisson 과정이 기본 모델, 이를 깨는 패턴 (cluster, repulsion)이 연구 대상.

---

## 📐 수학적 선행 조건

- [Ch3-01](./01-three-equivalent-definitions.md): Poisson 과정의 3가지 정의
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): 독립·결합분포, 조건부, 특성함수

---

## 📖 직관적 이해

### Superposition

두 이벤트 stream (예: 한국어 트윗, 영어 트윗)을 합치면 전체 rate = 각 rate 합.

**직관**: 지하철에서 1호선(rate $\lambda_1$)과 2호선(rate $\lambda_2$)의 열차 도착. 전체 도착 rate = $\lambda_1 + \lambda_2$ if 독립.

### Thinning

"이벤트 발생 시 동전을 던져 heads이면 keep". 전체 Poisson rate $\lambda$, keep 확률 $p$ → keep된 이벤트는 rate $p\lambda$.

**직관**: 모든 방문객 중 랜덤 10%만 survey 대상 → survey 대상의 도착 rate가 원본의 10%.

### Non-homogeneous rate $\lambda(t)$

rate가 시간에 따라 변함. Cumulative intensity $\Lambda(t) = \int_0^t \lambda(s) ds$가 모든 정보를 요약. 비균질 Poisson 과정을 **시간 변환**으로 균질로 환원:
$$N_t = \tilde N_{\Lambda(t)},$$
여기서 $\tilde N$는 rate 1의 균질 Poisson.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Superposition

$\{N^{(1)}_t\}, \{N^{(2)}_t\}$가 독립 Poisson 과정, rate $\lambda_1, \lambda_2$. **합쳐진 과정**:
$$N_t := N^{(1)}_t + N^{(2)}_t.$$

### 정의 2.2 — Thinning

$\{N_t\}$가 rate $\lambda$ Poisson, 각 이벤트에 독립적으로 $\xi_i \sim \text{Bern}(p)$ 할당. 
$$N^p_t := \sum_{i : T_i \leq t} \xi_i$$
($T_i$는 $i$번째 이벤트 시각.) — **thinned 과정**.

### 정의 2.3 — 비균질 Poisson (inhomogeneous)

$\lambda : [0, \infty) \to [0, \infty)$ locally integrable. $\Lambda(t) := \int_0^t \lambda(s) ds$.

$\{N_t\}$가 **rate $\lambda(\cdot)$의 비균질 Poisson 과정**이다 ⇔
1. $N_0 = 0$
2. 독립증분
3. $N_t - N_s \sim \text{Poisson}(\Lambda(t) - \Lambda(s))$ for $s < t$

### 정의 2.4 — 공간 Poisson Point Process

$\Lambda$가 $\mathbb{R}^d$(또는 Polish) 위의 $\sigma$-유한 측도. **Poisson point process with intensity $\Lambda$**: random countable subset $\Pi \subset \mathbb{R}^d$로서
1. 각 Borel $A$에 대해 $\#(\Pi \cap A) \sim \text{Poisson}(\Lambda(A))$
2. 서로소 $A_1, \ldots, A_n$에 대해 $\#(\Pi \cap A_i)$들이 독립

---

## 🔬 정리와 증명

### 정리 2.1 — Superposition

독립 Poisson 과정의 합은 rate 합의 Poisson:
$$N_t = N^{(1)}_t + N^{(2)}_t \sim \text{Poisson}((\lambda_1 + \lambda_2) t).$$

*증명.* 독립 Poisson 합의 특성함수:
$$\phi_{N_t}(u) = \phi_{N^{(1)}_t}(u) \phi_{N^{(2)}_t}(u) = e^{\lambda_1 t (e^{iu} - 1)} e^{\lambda_2 t (e^{iu} - 1)} = e^{(\lambda_1 + \lambda_2)t(e^{iu} - 1)}.$$
이는 Poisson$((\lambda_1 + \lambda_2)t)$ 특성함수. 독립증분은 각 과정에서 상속. $\square$

### 정리 2.2 — Thinning

Rate $\lambda$ Poisson에서 독립 Bernoulli($p$) 할당으로 얻은 $N^p_t$는 rate $p\lambda$ Poisson. Complement $N^{1-p}_t = N_t - N^p_t$는 rate $(1-p)\lambda$ Poisson. 두 thinned 과정 **독립**.

*증명.* $N^p_t | N_t = n \sim \text{Bin}(n, p)$ (각 이벤트 독립 keep). 조건부 확률:
$$\mathbb{P}(N^p_t = k) = \sum_{n \geq k} \mathbb{P}(N_t = n) \binom{n}{k} p^k (1-p)^{n-k}$$
$$= \sum_{n \geq k} \frac{(\lambda t)^n e^{-\lambda t}}{n!} \binom{n}{k} p^k (1-p)^{n-k} = \frac{(p\lambda t)^k e^{-p\lambda t}}{k!}.$$
(합 계산: 이항 전개 + Poisson 변환 trick) → Poisson$(p\lambda t)$.

독립증분은 원 과정에서 상속. Complement와의 독립성: $N^p_t, N^{1-p}_t | N_t = n$이 $(k, n-k)$ 분포, 합쳐 계산하면 서로 독립 Poisson. $\square$

### 정리 2.3 — 비균질 Poisson의 시간 변환

$\lambda(\cdot)$의 비균질 Poisson $\{N_t\}$와 rate 1의 균질 Poisson $\{\tilde N_t\}$ 사이
$$N_t \stackrel{d}{=} \tilde N_{\Lambda(t)}.$$

$\Lambda$ 연속·단조증가이면 **강한 변환**: $N_t = \tilde N_{\Lambda(t)}$ 경로별.

*증명 스케치.* $N_t - N_s \sim \text{Poisson}(\Lambda(t) - \Lambda(s))$이고 $\tilde N_{\Lambda(t)} - \tilde N_{\Lambda(s)} \sim \text{Poisson}(\Lambda(t) - \Lambda(s))$. 유한차원 분포 일치. $\square$

**응용**: 비균질 Poisson 시뮬레이션을 "균질 시뮬레이션 + 시간 변환"으로 구현.

### 정리 2.4 — Thinning으로 비균질 Poisson 샘플링

**Lewis-Shedler thinning 알고리즘**: $\lambda(t) \leq \lambda^*$ (upper bound) 알려진 비균질 Poisson 시뮬레이션:
1. Rate $\lambda^*$의 균질 Poisson 시뮬레이션
2. 각 이벤트 시각 $t$에서 accept 확률 $\lambda(t)/\lambda^*$

결과가 올바른 비균질 Poisson.

*증명.* 각 $t$에서 이벤트 유지 확률 $= \lambda(t)/\lambda^*$. Infinitesimal:
$$\mathbb{P}(\text{kept event in } [t, t+h]) = \lambda^* h \cdot \lambda(t)/\lambda^* = \lambda(t) h.$$
이는 정의 C(infinitesimal rate). $\square$

### 정리 2.5 — 공간 Poisson의 독립성

공간 Poisson process $\Pi$와 서로소 $A_1, \ldots, A_n$. 카운트 $\#(\Pi \cap A_i)$가 **독립**.

*증명.* 1차원 Poisson의 독립증분 성질의 직접 확장 — Kolmogorov 확장으로 joint 분포 정의. $\square$

### 정리 2.6 — Mapping (Coloring) 정리

Rate $\Lambda$의 공간 Poisson 각 점에 독립 "색" $c \sim q(\cdot | x)$ 할당 ($x$ = 위치). 각 색 $c$만 추출한 process는 rate $\Lambda_c(dx) = q(c|x) \Lambda(dx)$의 독립 Poisson process.

---

## 💻 NumPy 구현 검증

### 실험 1 — Superposition

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
lam1, lam2, T = 1.5, 1.0, 10.0

def poisson_events(lam, T):
    times = []
    t = 0
    while True:
        t += rng.exponential(1/lam)
        if t > T: break
        times.append(t)
    return np.array(times)

# 독립 두 Poisson 과정
events1 = poisson_events(lam1, T)
events2 = poisson_events(lam2, T)
combined = np.sort(np.concatenate([events1, events2]))

# 합쳐진 간격 분포 → Exp(λ1+λ2)?
intervals = np.diff(np.concatenate([[0], combined]))
print(f'합친 간격 평균: {intervals.mean():.4f}, 이론: {1/(lam1+lam2):.4f}')

# 시각화
fig, ax = plt.subplots(figsize=(10, 3))
ax.vlines(events1, 0, 1, color='C0', label=r'$N^{(1)}$')
ax.vlines(events2, 1, 2, color='C1', label=r'$N^{(2)}$')
ax.vlines(combined, 2, 3, color='k', label='Superposition')
ax.legend(); ax.set_yticks([])
plt.show()
```

### 실험 2 — Thinning

```python
lam, p, T = 2.0, 0.3, 20.0
events = poisson_events(lam, T)

# 각 이벤트 독립 keep with prob p
keep = rng.random(len(events)) < p
kept = events[keep]
discarded = events[~keep]

print(f'원본 rate ≈ {len(events)/T:.2f}, 이론 λ = {lam}')
print(f'Kept rate ≈ {len(kept)/T:.2f}, 이론 pλ = {p*lam}')
print(f'Discarded rate ≈ {len(discarded)/T:.2f}, 이론 (1-p)λ = {(1-p)*lam}')

# kept의 간격이 Exp(pλ)인가
kept_intervals = np.diff(np.concatenate([[0], kept]))
print(f'Kept 간격 평균: {kept_intervals.mean():.4f}, 이론: {1/(p*lam):.4f}')
```

### 실험 3 — 비균질 Poisson (Lewis-Shedler)

```python
# λ(t) = 1 + sin(t) + 1 (양수 유지)
lam_func = lambda t: 1.5 + np.sin(t)
lam_max = 2.5
T = 20.0

# Step 1: rate λ* 균질 Poisson
candidates = poisson_events(lam_max, T)
# Step 2: accept with prob λ(t) / λ*
accept = rng.random(len(candidates)) < lam_func(candidates) / lam_max
events = candidates[accept]

# 이론 평균 카운트 E[N_T] = Λ(T) = ∫ λ(s) ds
from scipy.integrate import quad
Lambda_T, _ = quad(lam_func, 0, T)
print(f'시뮬 카운트: {len(events)}, 이론 Λ(T) = {Lambda_T:.2f}')

# 시각화
t_grid = np.linspace(0, T, 500)
plt.plot(t_grid, lam_func(t_grid), label=r'$\lambda(t)$')
plt.vlines(events, 0, 0.5, color='k', alpha=0.5, label='events')
plt.xlabel('t'); plt.legend(); plt.show()
```

---

## 🔗 AI/ML 연결

**다중 source event aggregation**  
독립 sensor streams의 통합 분석 — Superposition이 "aggregation이 Poisson 유지"를 보장. 예: Cloud IoT gateway가 다수 장치 메시지 합침 → aggregated 부하 예측.

**Dataset subsampling의 이론적 안전**  
대규모 로그 데이터 (10억 row) → 10% subsample (thinning). Subsample이 원본의 rate·distribution 구조를 보존 → downstream 분석의 variance는 커지지만 bias 없음. MCMC thinning과 동일 원리.

**Neural TPP의 intensity 모델링**  
$\lambda(t | \mathcal{H}_t) = f_\theta(\mathcal{H}_t, t)$. Softplus로 양수 보장. Training objective:
$$\log L = \sum_i \log \lambda(t_i | \mathcal{H}_{t_i}) - \int_0^T \lambda(s | \mathcal{H}_s) ds.$$
두번째 적분 항이 "rate 적분"을 근사 (MC sampling). Lewis-Shedler thinning이 생성의 핵심.

**Spatial point process modeling**  
Determinantal Point Process (DPP)의 "diversity sampling" — 공간 Poisson의 확장, 반-repulsion 구조. 추천시스템의 diverse recommendation에 활용.

**Rejection Sampling for 복잡 분포**  
Lewis-Shedler thinning의 일반화 — proposal $p_0$, target $p$에서 accept 확률 $p(x)/M p_0(x)$. 이 원리가 Metropolis-Hastings(Ch7-02)의 조상.

---

## ⚖️ 가정과 한계

**가정 — 독립성 (superposition, thinning)**  
두 source가 서로 자극하면(e.g., 한 뉴스가 다른 뉴스 트리거) 독립 깨짐 → superposition이 Poisson 아님. Hawkes cross-excitation 모델 필요.

**한계 — 비균질 rate 추정의 역문제**  
실측 이벤트 시각에서 $\lambda(t)$ 추정은 inverse problem — kernel density estimation, penalized log-likelihood. Underfitting (너무 smooth)과 overfitting (spike) trade-off.

**한계 — $\lambda(t)$가 random process일 때**  
$\lambda_t$가 stochastic → **Cox process** (doubly stochastic Poisson). Conditional on $\lambda$ path, 여전히 비균질 Poisson이지만 $\lambda$ 자체가 marginal 결정. Deep learning으로 $\lambda$ path를 GP로 모델링 (Mei-Eisner).

---

## 📌 핵심 정리

| 결과 | 수식 |
|---|---|
| Superposition | 독립 Poisson의 합 = rate 합의 Poisson |
| Thinning | Bernoulli($p$) keep → rate $p\lambda$ Poisson |
| Complement thinning | keep과 discard는 독립 |
| 비균질 Poisson | $\Lambda(t) = \int_0^t \lambda(s) ds$, $N_t - N_s \sim \text{Poisson}(\Lambda(t) - \Lambda(s))$ |
| Time change | $N_t \stackrel{d}{=} \tilde N_{\Lambda(t)}$ |
| Lewis-Shedler | Thinning으로 비균질 샘플링 |

**한 줄 요약**: Poisson 과정은 **합과 분할에 닫혀 있고** (superposition, thinning), 시간/공간의 rate measure $\Lambda$가 확장/비균질 과정을 결정. 시간 변환 + thinning으로 비균질 과정을 균질 과정으로 환원 가능.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. 한 콜센터에 전화가 rate $\lambda_1 = 5$/시간(문의), $\lambda_2 = 2$/시간(항의)로 독립 도착. 전체 전화 rate와 "항의 전화가 첫 전화일 확률"을 구하라.

<details>
<summary>해설</summary>

**전체 rate**: $\lambda_1 + \lambda_2 = 7$/시간. 

**첫 전화가 항의**: 두 독립 Poisson 중 "어느 것이 먼저 도착"은 "경쟁 exponentials" 문제. 
$$\mathbb{P}(T_2 < T_1) = \frac{\lambda_2}{\lambda_1 + \lambda_2} = \frac{2}{7} \approx 28.6\%.$$

**일반 원리**: Superposition에서 각 이벤트의 source는 **independent Bernoulli with probabilities $\lambda_i/\sum \lambda_j$**. 이는 thinning의 역 — superposition을 "source labeling된 Poisson"으로 해석.

</details>

**문제 2 (심화)**. 비균질 Poisson $\lambda(t) = e^{-t}$ on $[0, \infty)$. 총 이벤트 수의 분포를 구하고, 첫 이벤트 시각 $T_1$의 밀도를 유도하라.

<details>
<summary>해설</summary>

**총 이벤트 수**: $\Lambda(\infty) = \int_0^\infty e^{-s} ds = 1$. $N_\infty \sim \text{Poisson}(1)$ — **유한** (평균 1).

**첫 이벤트 시각 $T_1$**:
$$\mathbb{P}(T_1 > t) = \mathbb{P}(N_t = 0) = e^{-\Lambda(t)} = e^{-(1 - e^{-t})}.$$
밀도:
$$f_{T_1}(t) = -\frac{d}{dt} e^{-(1 - e^{-t})} = e^{-t} e^{-(1 - e^{-t})}.$$

**한계**: $\mathbb{P}(T_1 = \infty) = \mathbb{P}(N_\infty = 0) = e^{-1} \approx 0.368$ — 약 37%는 이벤트가 아예 발생 안 함. 이는 **rate가 빠르게 감소**하기 때문.

**직관**: 초기에 rate 높고 뒤로 갈수록 감소 → 이벤트 대부분 초반 집중 → 총 이벤트 유한.

</details>

**문제 3 (AI 연결)**. Neural TPP 모델의 log-likelihood 계산에 rate 적분 $\int_0^T \lambda_\theta(s | \mathcal{H}_s) ds$가 들어간다. 이 적분을 (a) Monte Carlo (b) 수치적분 (c) closed-form (특정 $\lambda_\theta$ 선택) 각각으로 계산할 때 trade-off는?

<details>
<summary>해설</summary>

**(a) Monte Carlo**:
$\int_0^T \lambda \approx \frac{T}{M} \sum_{k=1}^M \lambda(u_k | \mathcal{H}_{u_k})$, $u_k \sim \text{Uniform}(0, T)$. 
- **장점**: 임의 $\lambda$ 형태 허용, unbiased
- **단점**: Monte Carlo variance, 많은 샘플 필요; $\mathcal{H}_{u_k}$ 구성이 sequence에서 expensive

**(b) 수치적분** (Simpson, Gauss quadrature):
- **장점**: 낮은 차원에서 MC보다 정확
- **단점**: 1D만 효율, high-dim (multivariate TPP)에서 curse of dimensionality

**(c) Closed-form** (e.g., Hawkes with exponential kernel):
$\lambda(t) = \mu + \sum_{T_i < t} \alpha e^{-\beta(t - T_i)}$, 적분 분석적 계산 가능. 
- **장점**: Exact, fast
- **단점**: 모델 형태 제한 (exponential kernel만 — 표현력 감소)

**Neural TPP의 실전 선택**:
- **TPP-AttNHP** (Mei-Eisner, Transformer Hawkes): Trained via MC with importance sampling on time intervals
- **Fully-Neural TPP** (Omi 2019): Cumulative intensity $\Lambda_\theta$를 직접 NN 출력 → derivatives로 $\lambda$ 얻어 적분 불필요
- **IFL-TPP** (Shchur 2020): Closed-form tractable — piecewise log-intensity로 모델링

**Trade-off 요약**: 표현력(flexibility) ↔ 계산 효율(closed-form/analytical). 최근 연구는 "cumulative intensity NN"으로 양쪽 장점 취하려 함.

</details>

---

<div align="center">

◀ [01. Poisson 과정의 3가지 동치 정의](./01-three-equivalent-definitions.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [03. 복합 Poisson 과정](./03-compound-poisson.md)

</div>
