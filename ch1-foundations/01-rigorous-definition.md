# 01. 확률과정의 엄밀한 정의

## 🎯 핵심 질문

- 확률과정 $\{X_t\}_{t \in T}$는 **정확히 무엇**인가 — 확률변수의 가족인가, 함수값 확률변수인가?
- 왜 "**sample path**" 관점과 "**유한차원 분포**" 관점이 모두 필요한가?
- 이산/연속 시간과 이산/연속 상태는 어떻게 조합되며, 각 조합의 측정가능성(measurability) 요구사항은 무엇인가?
- "확률과정이 측정 가능하다"는 조건이 왜 **자명하지 않은가** — 연속시간에서 $(\omega, t) \mapsto X_t(\omega)$의 joint measurability 문제는 무엇인가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**생성모델**의 수학적 토대는 모두 확률과정이다. DDPM의 forward process $\{X_t\}_{t \in [0,T]}$는 연속시간·연속상태 마르코프 과정이며, 이를 샘플링 가능하게 이산화하려면 먼저 **연속시간 확률과정이 무엇인지**가 엄밀히 정의되어야 한다. **강화학습**에서 MDP의 상태 궤적 $\{S_t\}_{t \in \mathbb{N}}$은 이산시간 과정이고, Bellman 방정식의 수렴 분석은 이 궤적의 정상분포에 의존한다. **시계열 모델**(Transformer, RNN, Mamba 등)의 입력은 $T \to \mathbb{R}^d$ 형태의 sample path로 해석되며, 학습 데이터의 "분포"가 무엇인지는 **유한차원 분포의 가족**으로 기술된다.

엄밀한 정의 없이 진행하면 다음이 흔들린다:
- "분포가 같은 두 확률과정"이 경로 수준에서 다를 수 있다는 사실 (modification vs indistinguishability)
- 연속시간 과정에서 $\sup_t X_t$, $\int_0^T X_t dt$ 같은 **경로 범함수**가 측정 가능한가
- Brownian motion, Poisson process 같은 정칙 경로(regular path) 과정을 구성할 때 무엇이 보장되어야 하는가

---

## 📐 수학적 선행 조건

- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): 확률공간 $(\Omega, \mathcal{F}, \mathbb{P})$, 확률변수(측정가능 함수), 곱측도, 분포 $\mu_X = \mathbb{P} \circ X^{-1}$
- 측도론: $\sigma$-대수, Borel $\sigma$-대수 $\mathcal{B}(\mathbb{R}^d)$, product $\sigma$-algebra $\bigotimes_{t \in T} \mathcal{B}(\mathbb{R})$
- 함수해석 기초: $\mathbb{R}^T$(함수 공간), product topology

---

## 📖 직관적 이해

### 3가지 동등한 관점

확률과정 $\{X_t\}_{t \in T}$는 세 가지 방식으로 바라볼 수 있으며, 모두 같은 대상을 가리킨다:

**관점 A — "각 시각마다 확률변수"**
$$X_t : \Omega \to \mathbb{R}, \quad \text{각 } t \in T \text{에 대해 측정가능}$$

**관점 B — "두 변수 함수 (Joint)"**
$$X : \Omega \times T \to \mathbb{R}, \quad (\omega, t) \mapsto X_t(\omega)$$

**관점 C — "함수값 확률변수"**
$$X : \Omega \to \mathbb{R}^T, \quad \omega \mapsto (t \mapsto X_t(\omega))$$

관점 C에서 $X(\omega)$를 **sample path** 또는 **trajectory**라 부른다 — 즉 $\omega$를 고정한 한 번의 실험 결과가 **$T$ 위의 함수**라는 해석.

> **비유**: 주식 가격 $S_t$를 보자.
> - 관점 A: "내일 오전 10시의 가격 $S_{10}$은 확률변수"
> - 관점 B: "모든 $(\omega, t)$에 대해 이벤트 × 시각 함수"
> - 관점 C: "한 번의 시장 세션 = 하루치 가격 곡선 하나"
>
> 관점 C가 "샘플 경로"의 해석을 준다. 훈련 데이터의 한 시계열은 하나의 sample path.

### 왜 세 관점 모두 필요한가

- **관점 A**는 단일 시점 분포(marginal)를 다룰 때 편하다 — "$X_t$의 분산은?"
- **관점 B**는 $\sup_t X_t(\omega)$, $\int_0^T X_t(\omega) dt$ 같이 **시각과 결과가 섞인 양**의 측정가능성을 따질 때 필수다
- **관점 C**는 "경로 공간 위의 확률측도"로서 $\mathbb{P}_X$를 볼 수 있게 한다 — Wiener measure, Poisson measure 등이 여기서 산다

관점 C가 가장 풍부하지만, **측정가능성 보장이 가장 비자명**하다.

### 4가지 분류 — 시간 × 상태

| | **이산 시간** ($T = \mathbb{N}$) | **연속 시간** ($T = [0, \infty)$) |
|---|---|---|
| **이산 상태** | 마르코프 체인 (Ch2), RL의 tabular MDP | 연속시간 MC (Ch4), Poisson 과정 (Ch3) |
| **연속 상태** | AR 모델, random walk, LSTM 은닉상태 | Brownian motion (Ch6), SDE, Diffusion forward process |

이산-이산: 유한/가산 집합 위의 전이확률 $P_{ij}$로 기술.
이산-연속: 회귀식 $X_{n+1} = f(X_n) + \epsilon_n$.
연속-이산: 이벤트 카운팅 과정, jump process.
연속-연속: 확률해석의 본 영역, **측정가능성이 가장 까다롭다**.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — 확률과정

$(\Omega, \mathcal{F}, \mathbb{P})$를 확률공간, $(E, \mathcal{E})$를 가측공간(**상태공간**, state space), $T$를 **지표집합**(index set)이라 하자.

**확률과정**(stochastic process)은 각 $t \in T$마다 확률변수
$$X_t : (\Omega, \mathcal{F}) \to (E, \mathcal{E})$$
를 지정하는 가족 $X = \{X_t\}_{t \in T}$이다.

- $T = \mathbb{N}$ 또는 $T \subset \mathbb{Z}$: **이산시간 확률과정**
- $T = [0, \infty)$ 또는 $T \subset \mathbb{R}$: **연속시간 확률과정**
- $E$가 가산: **이산상태 확률과정**
- $E = \mathbb{R}^d$, $\mathcal{E} = \mathcal{B}(\mathbb{R}^d)$: **연속상태 확률과정**

### 정의 1.2 — Sample path (Trajectory)

$\omega \in \Omega$를 고정하면, 함수
$$t \mapsto X_t(\omega) : T \to E$$
를 $\omega$의 **sample path**(또는 trajectory, realization)라 한다.

확률과정은 "함수 공간 $E^T$ 위의 확률변수" $X : \Omega \to E^T$로 볼 수 있다.

### 정의 1.3 — 유한차원 분포 (finite-dimensional distribution, fdd)

$n \in \mathbb{N}$, $t_1, \ldots, t_n \in T$에 대해
$$\mu_{t_1, \ldots, t_n}(A_1 \times \cdots \times A_n) = \mathbb{P}(X_{t_1} \in A_1, \ldots, X_{t_n} \in A_n), \quad A_i \in \mathcal{E}$$
로 정의되는 $(E^n, \mathcal{E}^{\otimes n})$ 위의 확률측도를 확률과정 $X$의 **$(t_1, \ldots, t_n)$-유한차원 분포**라 한다. 가족 $\{\mu_{t_1, \ldots, t_n}\}_{n, t_i}$를 확률과정의 **fdd 가족**이라 부른다.

### 정의 1.4 — 측정가능 과정 (Measurable process)

$T \subset \mathbb{R}$에 $\mathcal{B}(T)$를 Borel $\sigma$-대수로 부여하자. 확률과정 $X$가 **측정가능**하다는 것은
$$X : (\Omega \times T, \mathcal{F} \otimes \mathcal{B}(T)) \to (E, \mathcal{E})$$
가 product $\sigma$-대수에 대해 joint measurable이라는 의미다.

> 각 $t$에 대해 $X_t$가 측정가능하다(개별 measurability)는 joint measurability를 **자동으로 함의하지 않는다**. 연속시간에서 이는 별도의 조건이다.

### 정의 1.5 — Modification, Indistinguishability

두 확률과정 $X, Y$가 같은 $(\Omega, \mathcal{F}, \mathbb{P})$ 위에서 정의되어 있을 때:

- $Y$는 $X$의 **modification**: 각 $t \in T$에 대해 $\mathbb{P}(X_t = Y_t) = 1$
- $X, Y$는 **indistinguishable**: $\mathbb{P}(X_t = Y_t \text{ for all } t \in T) = 1$

Indistinguishable ⇒ modification이지만, 역은 성립하지 않는다 (연속시간에서 특히).

---

## 🔬 정리와 증명

### 정리 1.1 — Modification과 Indistinguishability는 동등하지 않다

$T = [0, 1]$, $\Omega = [0, 1]$, $\mathcal{F} = \mathcal{B}([0,1])$, $\mathbb{P} = $ Lebesgue measure. 두 과정을 다음과 같이 정의:
$$X_t(\omega) \equiv 0, \qquad Y_t(\omega) = \mathbf{1}_{\{t = \omega\}}$$

**(i)** 각 $t$에 대해 $\mathbb{P}(X_t \neq Y_t) = \mathbb{P}(\omega = t) = 0$이므로 $Y$는 $X$의 modification.

**(ii)** 하지만 모든 $\omega$에 대해 $Y_\omega(\omega) = 1 \neq 0 = X_\omega(\omega)$이므로
$$\{\omega : X_t(\omega) = Y_t(\omega) \text{ for all } t\} = \emptyset.$$
따라서 두 과정은 indistinguishable이 **아니다**. $\square$

**교훈**: Modification은 각 시각의 **"단면"** 일치만 보장하고, 경로 전체의 일치는 보장하지 않는다. 경로 성질(연속성, 최대값 등)을 논할 때는 반드시 indistinguishable 버전을 잡아야 한다.

### 정리 1.2 — Sample path의 연속성은 보장되지 않는다

$\Omega = \{\omega_0\}$, $\mathcal{F} = \{\emptyset, \Omega\}$, $\mathbb{P}(\Omega) = 1$, $T = [0,1]$, $X_t(\omega_0) = \mathbf{1}_{\{t = 1/2\}}$로 놓자. 각 $t$에 대해 $X_t$는 상수 확률변수이므로 측정가능하다. 유한차원 분포도 잘 정의된다. 그러나 sample path $t \mapsto X_t(\omega_0)$는 $t = 1/2$에서 불연속이다.

> **핵심**: "모든 유한차원 분포가 잘 정의된다"와 "sample path가 연속이다"는 **독립된 성질**. Brownian motion이 연속 경로를 갖는다는 것은 공리에서 요구되는 **추가 조건**이다 (Ch6에서 Lévy 구성으로 존재성을 보장).

### 정리 1.3 — 개별 measurability ≠ Joint measurability

$X_t(\omega) = \mathbf{1}_{\{t\}}(\omega)$, $\Omega = T = [0,1]$, $\mathbb{P} = $ Lebesgue.

- 각 $t$에 대해 $X_t(\omega) = \mathbf{1}_{\{t\}}(\omega)$는 $\omega$의 함수로 측정가능 (단일점 집합 $\{t\}$는 Borel).
- 하지만 $X : [0,1] \times [0,1] \to \{0,1\}$, $X(\omega, t) = \mathbf{1}_{\{\omega = t\}}$의 level set $\{X = 1\} = \Delta$ (대각선)은 $\mathcal{B}([0,1]) \otimes \mathcal{B}([0,1])$-measurable이지만, 이 예에서 특별히 문제는 없어 보인다.

이를 더 극단적으로 만들려면 Vitali 집합과 같은 **비가측 집합**의 indicator로 각 시각 변수를 구성하는 기술이 필요하며 — **연속시간에서 joint measurability는 자동이 아님**을 보이는 표준 예다 (Doob 1953, 증명 상세는 생략).

**결론**: 연속시간 확률과정에서는 "joint measurable modification"의 존재를 별도로 증명해야 한다. Doob의 separability 이론과 progressively measurable process(Ch1-04)가 이를 해결한다.

### 명제 1.4 — 가산 지표집합에서는 문제가 없다

$T$가 가산일 때, $\mathcal{F} \otimes 2^T = \mathcal{F} \otimes 2^T$이고 각 $X_t$가 측정가능하면
$$X(\omega, t) = \sum_{s \in T} X_s(\omega) \mathbf{1}_{\{s\}}(t)$$
는 joint measurable이다. 이산시간에서는 개별 measurability만으로 joint measurability가 자동이다.

*증명.* 각 항 $X_s(\omega) \mathbf{1}_{\{s\}}(t)$는 두 measurable function의 곱이므로 measurable. 가산합도 measurable. $\square$

---

## 💻 NumPy 구현 검증

### 실험 1 — 4가지 분류의 sample path 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1) 이산시간·이산상태: 3-state 마르코프 체인
P = np.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])
N = 50
state = 0
path = [state]
for _ in range(N):
    state = rng.choice(3, p=P[state])
    path.append(state)
axes[0, 0].step(range(len(path)), path, where='post')
axes[0, 0].set_title('이산시간·이산상태 (Markov Chain)')
axes[0, 0].set_xlabel('n'); axes[0, 0].set_ylabel('state')
axes[0, 0].set_yticks([0, 1, 2])

# 2) 이산시간·연속상태: AR(1) — X_{n+1} = 0.8 X_n + ε_n
X = np.zeros(N + 1)
for n in range(N):
    X[n + 1] = 0.8 * X[n] + rng.standard_normal()
axes[0, 1].plot(X, 'o-', markersize=3)
axes[0, 1].set_title('이산시간·연속상태 (AR(1))')
axes[0, 1].set_xlabel('n'); axes[0, 1].set_ylabel(r'$X_n$')

# 3) 연속시간·이산상태: Poisson 과정
lam = 2.0; T = 10
t = 0.0; times = [0.0]; counts = [0]
while t < T:
    t += rng.exponential(1 / lam)
    if t < T:
        times.append(t); counts.append(counts[-1] + 1)
times.append(T); counts.append(counts[-1])
axes[1, 0].step(times, counts, where='post')
axes[1, 0].set_title('연속시간·이산상태 (Poisson Process)')
axes[1, 0].set_xlabel('t'); axes[1, 0].set_ylabel(r'$N_t$')

# 4) 연속시간·연속상태: Brownian motion (Euler 이산화)
dt = 0.01; N_bm = int(T / dt)
B = np.concatenate([[0], np.cumsum(rng.standard_normal(N_bm) * np.sqrt(dt))])
axes[1, 1].plot(np.linspace(0, T, N_bm + 1), B)
axes[1, 1].set_title('연속시간·연속상태 (Brownian Motion)')
axes[1, 1].set_xlabel('t'); axes[1, 1].set_ylabel(r'$B_t$')

for ax in axes.flat:
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 실험 2 — Modification vs Indistinguishability 수치적 관찰

```python
# 정리 1.1의 예: X_t ≡ 0, Y_t(ω) = 1_{t=ω}
# 유한 sampling으로 "거의 모든 t에 대해 같다"를 관찰
rng = np.random.default_rng(0)
n_omega, n_t = 10_000, 100
omega = rng.uniform(0, 1, n_omega)
t_grid = np.linspace(0, 1, n_t)

X = np.zeros((n_omega, n_t))   # X ≡ 0
Y = (np.abs(omega[:, None] - t_grid[None, :]) < 1e-6).astype(float)
# 연속에서 Y_t(ω) = 1_{t=ω}는 유한 이산화에서 거의 0

prob_equal_at_each_t = (X == Y).mean(axis=0)
print(f'각 시각에서 X_t = Y_t 확률 (평균): {prob_equal_at_each_t.mean():.6f}')
# → 거의 1 (modification)

prob_equal_all_t = (X == Y).all(axis=1).mean()
print(f'모든 시각에서 X = Y인 ω의 비율: {prob_equal_all_t:.6f}')
# → 이산 grid에선 거의 1이지만, 연속 T에선 0 (indistinguishable 아님)
# 이산화 효과로 구분이 흐려지는 것이 핵심 교훈
```

### 실험 3 — 유한차원 분포(fdd)의 일관성 관찰

```python
# BM: (B_{0.3}, B_{0.7})의 2차원 marginal을 시뮬레이션
# fdd는 N(0, Σ)이고 Σ = [[0.3, 0.3], [0.3, 0.7]] (정리: Cov(B_s, B_t) = min(s,t))
n_paths = 100_000
t1, t2 = 0.3, 0.7
Z1 = rng.standard_normal(n_paths) * np.sqrt(t1)
dZ = rng.standard_normal(n_paths) * np.sqrt(t2 - t1)
B1, B2 = Z1, Z1 + dZ   # 독립 증분

cov = np.cov(B1, B2)
print('이론 Σ:')
print(np.array([[t1, t1], [t1, t2]]))
print('실측 Σ:')
print(cov)
# → 이론과 일치. fdd가 확률과정 분포의 핵심 정보.
```

---

## 🔗 AI/ML 연결

**Diffusion Model (DDPM/Score-SDE)**  
Forward process $X_t = \alpha_t X_0 + \sigma_t \epsilon$는 연속시간·연속상태 마르코프 과정. "연속시간 확률과정" 정의가 없으면 DDPM을 SDE로 재유도하는 Song et al. (2021)의 프레임워크를 쓸 수 없다. 경로 분포 $\mathbb{P}_X$는 **path measure**이며, **reverse process**는 이 측도의 시간반전으로 구성된다(Ch6, SDE Deep Dive).

**Transformer·시계열 모델**  
입력 시퀀스 $(x_1, \ldots, x_n)$은 (이산시간·연속상태) 확률과정의 **유한차원 분포 샘플 하나**. 위치 인코딩은 시각 $t$에 대한 정보이며, attention은 fdd 전체에 의존하는 **non-Markovian** 처리.

**RL의 MDP**  
상태-행동-보상 궤적 $\{(S_t, A_t, R_t)\}$은 이산시간 확률과정. 정책 $\pi$는 transition kernel을 바꿔 sample path 분포를 바꾸는 측도 변경(change of measure). Off-policy 학습의 importance ratio가 바로 이 측도 비율.

**VAE·잠재변수 모델**  
Latent $z \sim p(z)$는 "시각 = 1"의 trivial 확률과정이지만, **continuous-time normalizing flow**(FFJORD 등)은 $z$를 시간 $t \in [0, 1]$ 위의 ODE 궤적으로 취급한다.

---

## ⚖️ 가정과 한계

**가정 1 — "모든 $X_t$가 같은 $(\Omega, \mathcal{F}, \mathbb{P})$ 위에 정의됨"**  
서로 다른 실험의 결과를 모으려면 **곱공간**을 만들어야 하며, 이는 Kolmogorov 확장정리(Ch1-02)가 해결한다.

**가정 2 — "지표집합 $T$는 totally ordered"**  
공간 확률과정(spatial random field)은 $T$가 $\mathbb{R}^d$로 확장된 경우이며, 마르코프 성질이 "공간 Markov field"(Gibbs 측도)로 일반화된다. 이 레포는 $T \subset \mathbb{R}$로 제한.

**가정 3 — "Polish state space"**  
Kolmogorov 확장정리의 표준 버전은 $E$가 Polish space(완비 분리가능 거리공간)임을 요구. 물리학에서 등장하는 분포값 과정(distribution-valued process)은 이 가정 밖.

**한계**  
연속시간 과정은 fdd만으로 경로 연속성·측정가능성이 결정되지 않는다 — 정리 1.2가 보인 바. Brownian motion을 "연속 경로 버전"으로 구성하려면 Lévy 구성(Ch6-02)과 Kolmogorov continuity theorem이 필요하다.

---

## 📌 핵심 정리

| 개념 | 요약 |
|------|------|
| 확률과정 | $(\Omega, \mathcal{F}, \mathbb{P})$ 위의 확률변수 가족 $\{X_t\}_{t \in T}$ |
| Sample path | $\omega \mapsto (t \mapsto X_t(\omega)) : \Omega \to E^T$ |
| fdd | 각 유한 $t_1, \ldots, t_n$에 대한 $(X_{t_1}, \ldots, X_{t_n})$의 결합분포 |
| Modification | 각 $t$에서 a.s. 일치 |
| Indistinguishable | 모든 $t$에 대해 동시에 a.s. 일치 (더 강한 조건) |
| Measurable process | $X : \Omega \times T \to E$가 joint measurable |
| 4분류 | (이산/연속 시간) × (이산/연속 상태) |

**한 줄 요약**: 확률과정은 "각 시각마다 확률변수"이기도 하고 "sample path라는 함수값 확률변수"이기도 하며, 두 관점은 **유한차원 분포**가 매개한다. 연속시간에서 "경로 성질"(연속성·측정가능성)은 별도로 요구해야 한다.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. $T = \{1, 2, 3\}$, 상태공간 $E = \{0, 1\}$일 때 가능한 확률과정의 fdd는 몇 개의 파라미터로 기술되는가?

<details>
<summary>해설</summary>

$(X_1, X_2, X_3)$의 결합분포는 $\{0,1\}^3 = 8$개 원자의 확률이며, 합이 1이므로 **7개의 자유 파라미터**. 이는 fdd 중 최대 차원($n = 3$)의 결합분포로 모든 $n \leq 3$의 fdd를 marginal로 얻을 수 있음을 뜻한다. 이산·유한에서는 fdd = 결합분포가 완전한 정보.

</details>

**문제 2 (심화)**. $T = [0, 1]$ 위에서 두 확률과정 $X, Y$가 modification 관계라 하자. $Z_t = \sup_{s \leq t} X_s$와 $Z'_t = \sup_{s \leq t} Y_s$는 항상 modification인가?

<details>
<summary>해설</summary>

**일반적으로 아니다**. $\sup$은 **비가산 supremum**이고, "각 $t$에서 a.s. 일치"는 경로 전체의 성질을 보존하지 못한다. 반례: $T = [0,1]$, $\Omega = [0,1]$, $\mathbb{P} = $ Lebesgue, $X_t \equiv 0$, $Y_t(\omega) = \mathbf{1}_{\{t = \omega\}}$. 각 $t$에서 $X_t = Y_t$ a.s.(modification). 그러나 $\sup_{s \leq t} Y_s(\omega) = 1$ for all $\omega \leq t$, 즉 $Z'_t \neq Z_t = 0$가 양의 확률로 발생. Indistinguishable이 아니면 경로 범함수는 같지 않을 수 있다.

이것이 **progressively measurable modification**(Doob)가 필요한 이유 — 경로 범함수의 well-definedness를 위해.

</details>

**문제 3 (AI 연결)**. DDPM의 forward process $X_t = \sqrt{\bar{\alpha}_t} X_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$ ($t \in \{1, \ldots, 1000\}$, $\epsilon \sim \mathcal{N}(0, I)$)을 확률과정으로 해석할 때, $T$, 상태공간 $E$, 경로의 측정가능성 요구는 어떻게 되는가? 만약 이를 연속시간 $t \in [0, 1]$로 확장(Score-SDE)하면 새로 요구되는 조건은?

<details>
<summary>해설</summary>

**이산 버전(DDPM)**: $T = \{1, \ldots, 1000\}$ (이산), $E = \mathbb{R}^d$ ($d$ = 이미지 픽셀 수). 가산 $T$이므로 명제 1.4에 의해 개별 measurability만으로 joint measurability가 자동. 경로 연속성 같은 추가 조건은 의미 없음.

**연속 버전(VP-SDE)**: $T = [0, 1]$, $E = \mathbb{R}^d$. 추가 요구:
1. **Joint measurability** — 경로 범함수(예: $\int_0^1 \|\nabla \log p_t(X_t)\|^2 dt$)의 well-definedness를 위해 필요 (Ch1-04의 progressively measurable)
2. **경로 연속성** — reverse SDE의 해의 존재성(Lipschitz drift 하에)과 이차변분 정의를 위해 필수
3. **Adapted 구조** — filtration $\mathcal{F}_t$ 정의 후 forward noise가 adapted이어야 reverse process의 시간반전이 잘 정의됨

즉 이산 → 연속 확장은 **"추가 경로 조건을 공짜로 얻지 못한다"** — 이 조건들이 Song et al. (2021)의 SDE 포뮬레이션의 숨은 전제이고, 이것이 SDE Deep Dive에서 파헤쳐진다.

</details>

---

<div align="center">

◀ [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [02. 유한차원 분포와 Kolmogorov 확장정리](./02-kolmogorov-extension.md)

</div>
