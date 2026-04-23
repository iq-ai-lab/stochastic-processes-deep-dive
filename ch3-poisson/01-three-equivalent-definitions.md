# 01. Poisson 과정의 3가지 동치 정의

## 🎯 핵심 질문

- Poisson 과정의 3가지 정의 — **독립증분 + Poisson marginal**, **iid Exp 간격시간**, **infinitesimal rate** — 이 왜 **동치**인가?
- **메모리리스 성질** $\mathbb{P}(T > s+t | T > s) = \mathbb{P}(T > t)$가 왜 지수분포의 **유일 특성**이고, 이것이 왜 Poisson과 동치인가?
- 이 동치성이 왜 직관적이지 않은가 — "이산 카운트 분포"와 "연속 대기시간"이 같은 대상을 어떻게 기술하는가?

---

## 🔍 왜 이 과정이 AI에서 중요한가

**이벤트 기반 sequence modeling**: 뉴스 피드, 클릭 스트림, 트랜잭션 로그의 도착 시각은 Poisson(또는 변형)으로 모델링. Temporal Point Process (TPP) 학습(Du et al. 2016, Transformer-Hawkes) 모두 이 기반.

**Neural spike train 분석**: 뉴런 발화의 "시간 점 과정" — 가장 단순한 모델이 Poisson. Deep generative spike model (Pillow et al.)의 baseline.

**Queueing이 LLM inference에 들어옴**: 요청 도착(Poisson 가정) + 처리 시간 → M/M/1, Little의 법칙으로 latency 분석(Ch3-04).

**Reliability / Survival analysis**: 이벤트 발생률 $\lambda$의 cumulative hazard. DeepHit, DeepSurv 등 survival deep learning이 Poisson intensity를 NN으로 파라미터화.

---

## 📐 수학적 선행 조건

- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): 지수분포, Poisson 분포, 특성함수
- [Ch1-01 ~ Ch1-04](../ch1-foundations/01-rigorous-definition.md): 확률과정, 필트레이션
- 측도론 기초: absolute continuity, 확률밀도

---

## 📖 직관적 이해

### 세 관점, 한 대상

**Poisson 과정** $\{N_t\}_{t \geq 0}$는 "이벤트 카운팅 함수" — 시각 $t$까지 발생한 이벤트 수.

- **관점 A (카운트)**: 구간별 카운트가 Poisson 분포, 서로소 구간 독립
- **관점 B (간격)**: 연속 이벤트 간 간격이 iid 지수
- **관점 C (인피니티시멀)**: 짧은 시간 $h$에 한 번 이벤트 확률 ≈ $\lambda h$

이들이 모두 같은 대상을 기술. 증명이 핵심 — 세 정의 중 어느 하나로 시작해도 나머지 유도 가능.

### 지수분포의 메모리리스

$T \sim \text{Exp}(\lambda)$이면
$$\mathbb{P}(T > s + t | T > s) = \mathbb{P}(T > t).$$

"이미 $s$만큼 기다렸다"는 정보가 앞으로 얼마나 더 기다릴지에 **아무 영향 없음**. 지수분포만의 유일한 성질 (Cauchy functional equation).

**직관**: 방사성 원자가 붕괴를 "기억하지 않는다" — 10초 뒤에 붕괴할 확률은 0초 뒤에 붕괴할 확률과 같음 (이미 붕괴 안했다면).

### 왜 메모리리스 → Poisson

간격시간 iid 지수 → 각 시점에서 "다음 이벤트까지의 시간"이 과거 이벤트와 독립 → 서로소 구간의 이벤트 수가 독립. 또한 Poisson 분포 자체가 **무한분할 가능**(infinitely divisible) → 누적 카운트가 Poisson.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — 정의 A (카운트 관점)

$\{N_t\}_{t \geq 0}$가 **rate $\lambda > 0$의 Poisson 과정**이다 (정의 A):
1. $N_0 = 0$
2. **독립증분**: $0 \leq t_0 < t_1 < \cdots < t_n$에 대해 $N_{t_1} - N_{t_0}, \ldots, N_{t_n} - N_{t_{n-1}}$가 독립
3. **Stationary Poisson marginal**: $N_t - N_s \sim \text{Poisson}(\lambda(t-s))$ for $s < t$
4. 경로 $t \mapsto N_t$가 우연속 + 유한 jump

### 정의 1.2 — 정의 B (간격 관점)

$T_1, T_2, T_3, \ldots$를 iid $\text{Exp}(\lambda)$, $S_n = T_1 + \cdots + T_n$. 정의:
$$N_t := \max\{n : S_n \leq t\}.$$
$\{N_t\}$가 rate $\lambda$의 Poisson 과정.

### 정의 1.3 — 정의 C (infinitesimal)

$\{N_t\}$가 (정의 C):
1. $N_0 = 0$
2. 독립증분
3. $\mathbb{P}(N_{t+h} - N_t = 1) = \lambda h + o(h)$
4. $\mathbb{P}(N_{t+h} - N_t \geq 2) = o(h)$

### 정의 1.4 — 지수분포의 메모리리스

$T \sim \text{Exp}(\lambda)$ ⇔ $\mathbb{P}(T > t) = e^{-\lambda t}$ ⇔ 메모리리스.

---

## 🔬 정리와 증명

### 정리 1.1 — 지수분포의 메모리리스 유일성

$T$가 양의 값을 갖는 연속 확률변수. 다음은 동치:
1. $T \sim \text{Exp}(\lambda)$ for some $\lambda > 0$.
2. $\mathbb{P}(T > s + t | T > s) = \mathbb{P}(T > t)$ for all $s, t \geq 0$.

*증명.*  
**(1 ⇒ 2)**: 직접 계산. $\mathbb{P}(T > s+t | T > s) = e^{-\lambda(s+t)}/e^{-\lambda s} = e^{-\lambda t}$.

**(2 ⇒ 1)**: $g(t) := \mathbb{P}(T > t)$. 가정 → $g(s+t) = g(s) g(t)$ (Cauchy 함수방정식). $g$가 우연속·비증가 → $g(t) = e^{-\lambda t}$ for some $\lambda \geq 0$. $g > 0$이려면 $\lambda > 0$. $\square$

### 정리 1.2 — A ⇒ B (카운트 → 간격 iid 지수)

정의 A의 Poisson 과정에서 간격시간 $T_k = S_k - S_{k-1}$이 iid $\text{Exp}(\lambda)$.

*증명.*  
$\{T_1 > t\} = \{N_t = 0\}$. 정의 A: $N_t \sim \text{Poisson}(\lambda t)$이므로 $\mathbb{P}(T_1 > t) = e^{-\lambda t}$ → $T_1 \sim \text{Exp}(\lambda)$.

$T_2$의 분포 w.r.t. $T_1$: $\{T_2 > t | T_1 = s\} = \{N_{s+t} - N_s = 0 | \mathcal{F}_s\}$ — 독립증분 + 정상성으로 $= \mathbb{P}(N_t = 0) = e^{-\lambda t}$. 조건부 분포가 $T_1$과 독립, $\text{Exp}(\lambda)$. 귀납으로 $T_k$ iid $\text{Exp}(\lambda)$. $\square$

### 정리 1.3 — B ⇒ A (간격 → 카운트 Poisson)

정의 B에서 $N_t = \max\{n : S_n \leq t\}$가 $\text{Poisson}(\lambda t)$ + 독립증분.

*증명.*  
$S_n = T_1 + \cdots + T_n \sim \text{Gamma}(n, \lambda)$ (iid 지수 합).
$$\mathbb{P}(N_t = n) = \mathbb{P}(S_n \leq t < S_{n+1}) = \mathbb{P}(S_n \leq t) - \mathbb{P}(S_{n+1} \leq t).$$
직접 계산:
$$= \int_0^t \frac{\lambda^n s^{n-1}}{(n-1)!} e^{-\lambda s} ds - \int_0^t \frac{\lambda^{n+1} s^n}{n!} e^{-\lambda s} ds.$$
부분적분으로 $= \frac{(\lambda t)^n}{n!} e^{-\lambda t}$ → $\text{Poisson}(\lambda t)$.

독립증분: 메모리리스로 $S_n$ 이후 간격은 다시 iid 지수 → $N_{t+s} - N_t$가 $\mathcal{F}_t$ 독립, $\text{Poisson}(\lambda s)$. $\square$

### 정리 1.4 — A ⇒ C 및 C ⇒ A

**A ⇒ C**: $N_{t+h} - N_t \sim \text{Poisson}(\lambda h)$. Taylor 전개:
$$\mathbb{P}(N_{t+h} - N_t = 1) = \lambda h e^{-\lambda h} = \lambda h - \lambda^2 h^2 + O(h^3) = \lambda h + o(h).$$
$$\mathbb{P}(N_{t+h} - N_t \geq 2) = 1 - e^{-\lambda h} - \lambda h e^{-\lambda h} = \frac{(\lambda h)^2}{2} + O(h^3) = o(h).$$

**C ⇒ A**: 독립증분과 infinitesimal rate로 $p_n(t) := \mathbb{P}(N_t = n)$이 ODE
$$p_n'(t) = -\lambda p_n(t) + \lambda p_{n-1}(t), \quad p_0(0) = 1, p_n(0) = 0 (n \geq 1)$$
을 만족함을 보임(짧은 구간 $[t, t+h]$ 분석). 해가 $p_n(t) = (\lambda t)^n e^{-\lambda t}/n!$. $\square$

**종합**: A ⇔ B ⇔ C. 세 정의가 모두 동등.

### 정리 1.5 — Mean, Variance, Covariance

$N_t \sim \text{Poisson}(\lambda t)$:
- $\mathbb{E}[N_t] = \lambda t$
- $\text{Var}(N_t) = \lambda t$
- $\text{Cov}(N_s, N_t) = \lambda \min(s, t)$

*증명.* 평균·분산은 Poisson 분포 성질. 공분산: $s < t$로 $N_t = N_s + (N_t - N_s)$, 독립이므로 $\text{Cov}(N_s, N_t) = \text{Var}(N_s) = \lambda s$. $\square$

> **BM 비교**: BM도 $\text{Cov} = \min(s, t)$ — 구조 유사(독립증분). 카운트 vs 연속 값, Poisson vs 가우시안 marginal.

---

## 💻 NumPy 구현 검증

### 실험 1 — 세 정의 모두로 Poisson 시뮬레이션

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
lam, T = 2.0, 10.0

# 방법 A: 직접 Poisson 카운트 생성 (각 작은 구간별)
n_bins = 1000
dt = T / n_bins
counts_A = rng.poisson(lam * dt, n_bins)
N_A = np.concatenate([[0], np.cumsum(counts_A)])
t_A = np.linspace(0, T, n_bins + 1)

# 방법 B: 간격시간 iid 지수
intervals = rng.exponential(1/lam, 100)
times = np.cumsum(intervals)
times = times[times <= T]
t_B = np.concatenate([[0], times, [T]])
N_B = np.concatenate([[0], np.arange(1, len(times)+1), [len(times)]])

# 방법 C: infinitesimal 균일 샘플링 (작은 h로 Bernoulli)
h = 0.001
events_C = rng.random(int(T/h)) < lam * h
N_C = np.concatenate([[0], np.cumsum(events_C)])
t_C = np.arange(0, T + h, h)[:len(N_C)]

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].step(t_A, N_A, where='post'); axes[0].set_title('A: 직접 카운트')
axes[1].step(t_B, N_B, where='post'); axes[1].set_title('B: 간격 Exp')
axes[2].step(t_C, N_C, where='post'); axes[2].set_title('C: infinitesimal Bernoulli')
for ax in axes:
    ax.set_xlabel('t'); ax.set_ylabel('N_t'); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# N_T 분포가 Poisson(λT)인지 확인
N_samples = [rng.poisson(lam * T) for _ in range(10000)]
print(f'평균: {np.mean(N_samples):.2f}, 이론 λT = {lam*T}')
print(f'분산: {np.var(N_samples):.2f}, 이론 λT = {lam*T}')
```

### 실험 2 — 메모리리스 성질 검증

```python
# T ~ Exp(1), P(T > s+t | T > s) vs P(T > t)
samples = rng.exponential(1, 1_000_000)
s_vals = [0, 0.5, 1, 2, 3]
t = 1.0

print(f'{"s":>3} {"P(T > s+t | T > s)":>22} {"P(T > t)":>12}')
for s in s_vals:
    cond = samples > s
    prob_cond = np.mean(samples[cond] > s + t) if cond.sum() else 0
    prob_marg = np.mean(samples > t)
    print(f'{s:>3} {prob_cond:>22.4f} {prob_marg:>12.4f}')
# → 모두 같은 값: 메모리리스 확인

# 비교: 균등분포는 메모리리스 아님
samples_U = rng.uniform(0, 5, 1_000_000)
s, t = 2, 1
cond = samples_U > s
print(f'Uniform: P(T > {s+t} | T > {s}) = {np.mean(samples_U[cond] > s+t):.4f}')
print(f'Uniform: P(T > {t}) = {np.mean(samples_U > t):.4f}')
# → 다름 (지수만 메모리리스)
```

### 실험 3 — 독립증분 검증

```python
lam, T = 1.0, 100.0
n_sim = 10000

# 구간 [0, 5]와 [5, 10]의 증분
N1 = rng.poisson(lam * 5, n_sim)
N2 = rng.poisson(lam * 5, n_sim)

# 상관 없어야 함
corr = np.corrcoef(N1, N2)[0, 1]
print(f'Corr(N_5, N_{10} - N_5) = {corr:.4f} (이론: 0)')

# 주의: N_5, N_10의 상관은 0 아님 (겹치는 부분 때문)
N_5 = rng.poisson(lam * 5, n_sim)
N_10_inc = rng.poisson(lam * 5, n_sim)
N_10 = N_5 + N_10_inc
print(f'Corr(N_5, N_10) = {np.corrcoef(N_5, N_10)[0,1]:.4f} ' 
      f'(이론 5/√(5·10) = {5/np.sqrt(50):.4f})')
```

---

## 🔗 AI/ML 연결

**Temporal Point Process (TPP)**  
Neural TPP 모델 (Du 2016, Mei-Eisner 2017, Transformer-Hawkes)은 "intensity $\lambda(t | \mathcal{H}_t)$를 RNN/Transformer로 파라미터화". 가장 단순 baseline = homogeneous Poisson = constant $\lambda$. 모델이 Poisson보다 얼마나 잘 맞나로 복잡성 평가.

**Neural Spike Train**  
뉴런 발화 시점 → Poisson 점 과정 (가장 단순). 복잡한 모델 (GLM, Poisson GAN) 모두 Poisson baseline에서 출발. 뇌파 분석의 null hypothesis.

**Self-Attention as exponential similarity**  
Attention score $\exp(q \cdot k / \sqrt{d})$의 지수 꼴은 Boltzmann-like softmax. 간격 모델로서 해석 가능 — "이전 token까지의 시간 간격이 Exp-distributed". (수학적 유사성, 정확한 mapping은 아님.)

**Hawkes process (self-exciting)**  
$\lambda(t) = \mu + \sum_{T_i < t} g(t - T_i)$. Poisson의 확장 — 과거 이벤트가 미래 intensity 증가시킴. 뉴스 확산, financial tick data 모델링.

**Reliability / Survival Neural Networks**  
DeepHit, Survival Transformer 등은 hazard rate $\lambda(t | x)$ 예측. Constant hazard = Exponential, flexible hazard = general Poisson-type. Censoring 처리가 간격 관점에서 자연.

---

## ⚖️ 가정과 한계

**가정 — 독립증분**  
현실 이벤트 (뉴스 확산, 지진 여진)는 자기흥분(self-exciting) → **Hawkes process**가 더 적합. Poisson은 first-order 근사.

**가정 — Homogeneous rate**  
$\lambda$ 상수. 현실은 시간대(피크/밤), 요일 등에 따라 변함 → **비균질 Poisson** (Ch3-02).

**한계 — 연속성 & no-jump-cluster**  
정확히 같은 시각 2개 이벤트? 정의 A·B에서 확률 0. 현실 데이터(타임스탬프 tie)는 정제 필요.

**한계 — Overdispersion**  
$\text{Var}/\text{Mean} = 1$이 Poisson. 현실 데이터는 overdispersed (variance > mean) — **Negative Binomial**, **Compound Poisson** 등으로 일반화.

---

## 📌 핵심 정리

| 정의 | 형태 |
|---|---|
| A (카운트) | 독립증분 + $N_t - N_s \sim \text{Poisson}(\lambda(t-s))$ |
| B (간격) | 간격 iid $\text{Exp}(\lambda)$ |
| C (infinitesimal) | 독립증분 + $\mathbb{P}(\Delta N = 1) = \lambda h + o(h)$ |
| 모두 동치 | A ⇔ B ⇔ C |
| 메모리리스 | Exp의 유일 특성 |
| 모멘트 | $\mathbb{E}[N_t] = \text{Var}(N_t) = \lambda t$ |

**한 줄 요약**: Poisson 과정은 "rate $\lambda$의 이벤트 발생"을 기술하는 세 가지 동등한 방식(카운트/간격/infinitesimal)을 가지며, 지수분포의 **메모리리스** 성질이 세 정의를 연결하는 핵심 구조다.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. 웹사이트에 평균 3분에 한 번 방문자가 도착. 1시간 동안 12명 미만 올 확률은?

<details>
<summary>해설</summary>

$\lambda = 1/3$ (방문자/분). 1시간 = 60분. $N_{60} \sim \text{Poisson}(\lambda \cdot 60) = \text{Poisson}(20)$.

$\mathbb{P}(N_{60} < 12) = \sum_{k=0}^{11} \frac{20^k e^{-20}}{k!}$.

SciPy: `scipy.stats.poisson.cdf(11, 20) ≈ 0.021` → **약 2.1%**.

직관: 평균 20 대비 12 미만은 낮은 확률.

</details>

**문제 2 (심화)**. Poisson 과정 $N_t$에서 첫 $n$ 이벤트 시각 $(S_1, \ldots, S_n) | N_T = n$의 조건부 분포는 무엇인가?

<details>
<summary>해설</summary>

**결과**: $(S_1, \ldots, S_n) | N_T = n \stackrel{d}{=}$ "iid Uniform$(0, T)$ 샘플을 정렬한 order statistics".

**증명**: 결합 밀도
$$f(s_1, \ldots, s_n, N_T = n) = \lambda^n e^{-\lambda s_1} \cdot e^{-\lambda(s_2 - s_1)} \cdots e^{-\lambda(s_n - s_{n-1})} \cdot e^{-\lambda(T - s_n)} = \lambda^n e^{-\lambda T}$$
for $0 < s_1 < \cdots < s_n < T$. 즉 **constant** on simplex.

$\mathbb{P}(N_T = n) = (\lambda T)^n e^{-\lambda T}/n!$. 조건부:
$$f(s_1, \ldots, s_n | N_T = n) = \frac{\lambda^n e^{-\lambda T}}{(\lambda T)^n e^{-\lambda T}/n!} = \frac{n!}{T^n}.$$
이는 $[0, T]$에서 iid Uniform의 order statistics의 joint density.

**직관**: "$T$ 시간 동안 정확히 $n$ 이벤트가 발생했다"는 정보만으로는 각 이벤트가 어느 시각에 일어났는지는 uniform random — Poisson의 "시간 균질성"의 결과.

**응용**: 
- Rejection sampling for inhomogeneous Poisson (Ch3-02)
- 2D Poisson point process의 Monte Carlo 시뮬레이션
- Spatial statistics

</details>

**문제 3 (AI 연결)**. Neural Hawkes 모델 (Mei-Eisner 2017)이 "homogeneous Poisson baseline"보다 log-likelihood가 높을 때, 이는 데이터의 어떤 성질을 시사하는가? 모델 복잡도와 성능의 trade-off는?

<details>
<summary>해설</summary>

**Poisson vs Hawkes log-likelihood**:
- Poisson baseline: $\log p = \sum_i \log \lambda - \lambda T$ — $\lambda$ 상수
- Hawkes: $\log p = \sum_i \log \lambda(t_i | \mathcal{H}_{t_i}) - \int_0^T \lambda(s | \mathcal{H}_s) ds$ — $\lambda$ history-dependent

Hawkes가 baseline 능가 = 데이터에 **시간 의존성**이 있음:
- **Self-excitation**: 이벤트가 다음 이벤트 확률 증가 (e.g., 뉴스 확산, 지진 여진)
- **Non-stationarity**: 장기적 rate 변화
- **Clustering**: 이벤트 밀집 구간과 sparse 구간의 대비

**Trade-off**:
- **복잡성**: Hawkes/Neural TPP는 많은 파라미터, 데이터 부족 시 overfit
- **Interpretability**: Poisson은 단일 $\lambda$ — 해석 쉬움
- **Inference speed**: Neural TPP는 sampling 느림 (history conditioning)
- **Statistical power**: 충분한 데이터가 있으면 Hawkes >> Poisson

**AIC/BIC criterion**: $-2 \log L + k \log n$ (BIC), 복잡도 페널티 고려한 model selection. Hawkes가 BIC에서도 이겨야 "진짜 개선".

**실전 사례**:
- Twitter retweet cascade: Hawkes가 Poisson 대비 log-likelihood 크게 개선 → self-excitation 강함
- Random IoT sensor events: Poisson으로도 충분 → complex 모델 불필요

**연결**: 이는 "**통계적 모델 선택**"의 표준 문제. Poisson은 간결한 baseline이자 null hypothesis. 데이터가 이를 유의미하게 깨면 더 복잡한 모델 (Hawkes, Neural TPP) 정당화.

</details>

---

<div align="center">

◀ [Ch2-06. 에르고딕 정리(Ergodic Theorem)](../ch2-discrete-markov/06-ergodic-theorem.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [02. 결합·분할과 비균질 Poisson](./02-superposition-thinning.md)

</div>
