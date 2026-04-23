# 05. 확률과정의 분류 지도

## 🎯 핵심 질문

- 확률과정의 주요 축 네 가지 — (**시간**: 이산/연속) × (**상태**: 이산/연속) × (**마르코프/비마르코프**) × (**정상/비정상**) — 각 조합의 대표 과정은 무엇인가?
- 각 분류에서 **수학적 도구**는 어떻게 달라지는가 — 이산 MC의 전이행렬, 연속 MC의 생성기 $Q$, BM의 이토 계산 등이 왜 서로 다른 형태를 갖는가?
- AI/ML의 주요 모델이 이 분류에서 **어디에 위치**하는가 — Transformer, DDPM, PPO, GP, HMM 등?
- 한 분류에서 다른 분류로의 **이동**(이산→연속, 비정상→정상 변환)은 어떻게 일어나는가?

---

## 🔍 왜 이 분류가 AI에서 중요한가

**모델 설계의 기초**: 입력이 "이산시간·연속상태·마르코프"인지 "연속시간·연속상태·비마르코프"인지에 따라 적합한 아키텍처(RNN vs Neural ODE vs Transformer)와 학습법이 결정된다. 분류를 잘못 식별하면 잘못된 inductive bias를 주입.

**이론의 호환성**: 한 분야의 정리(예: 에르고딕 정리)를 다른 과정에 쓸 때, 분류가 맞는지 먼저 확인해야 한다. 비정상 과정에 정상 가정의 bound를 적용하면 오류.

**생성 모델의 핵심**: DDPM = 이산시간·연속상태·마르코프·비정상. Score-SDE = 연속시간·연속상태·마르코프·비정상. 이 전이(이산 → 연속)가 **전체 레포의 중요한 주제**이며, 분류를 명확히 이해해야 가능.

**RL의 MDP 분류**: Finite MDP (이산·이산·마르코프·정상 정책 하)에서는 tabular 방법, Linear-Gaussian MDP (이산·연속·마르코프)에서는 LQR, 비정상 환경에서는 meta-RL이 필요.

---

## 📐 수학적 선행 조건

- [Ch1-01](./01-rigorous-definition.md): 확률과정의 엄밀한 정의
- [Ch1-02](./02-kolmogorov-extension.md): Kolmogorov 확장정리 (각 분류에서 존재 보장)
- [Ch1-03](./03-stationarity.md): 엄격/약 정상성
- [Ch1-04](./04-filtration.md): 필트레이션, adapted, 마르코프 성질

---

## 📖 직관적 이해

### 4개의 2진 축

| 축 | 이산 (0) | 연속 (1) |
|---|---|---|
| **시간** $T$ | $\mathbb{N}, \mathbb{Z}$ | $[0, \infty), \mathbb{R}$ |
| **상태** $E$ | 가산 집합 | $\mathbb{R}^d$, 함수공간 |
| **마르코프성** | 비마르코프 | $\mathbb{P}(X_{n+1} | \mathcal{F}_n) = \mathbb{P}(X_{n+1} | X_n)$ 마르코프 |
| **정상성** | 비정상 | 엄격 또는 약 정상 |

총 $2^4 = 16$개의 분면. 대부분의 유명 과정은 일부 분면에 속함.

### 주요 대표 과정

- **(이산, 이산, 마르코프, 정상)**: 정상분포에 있는 마르코프 체인, HMM의 hidden state
- **(이산, 이산, 비마르코프, 정상)**: Longer memory(k-order) 체인, mixture process
- **(이산, 연속, 마르코프, 비정상)**: AR(1) with drift, DDPM forward
- **(이산, 연속, 비마르코프, 정상)**: Gaussian process (정상 커널), ARMA($p, q$)
- **(연속, 이산, 마르코프, 정상)**: M/M/1 큐의 정상분포 상태, Poisson process (counts)
- **(연속, 이산, 비마르코프, 비정상)**: Cox process (intensity가 Markov가 아닌 경우)
- **(연속, 연속, 마르코프, 정상)**: OU process in equilibrium, Langevin in Gibbs
- **(연속, 연속, 마르코프, 비정상)**: BM (starting from 0), SDE solutions, Score-SDE forward
- **(연속, 연속, 비마르코프, 정상)**: Fractional BM(Hurst ≠ 1/2), Long-range dependent process

### 분류가 기술적 도구를 결정

- **이산·이산·마르코프**: **전이행렬** $P$로 모든 분석. 정상분포 $\pi P = \pi$, 수렴률은 $|\lambda_2|$ (Ch2)
- **연속·이산·마르코프**: **Q-matrix(생성기)** $Q$로 분석. Forward/Backward Kolmogorov 방정식 (Ch4)
- **연속·연속·마르코프**: **SDE** $dX_t = b dt + \sigma dB_t$, **Fokker-Planck** $\partial_t p = \mathcal{L}^* p$ (SDE 레포)
- **비마르코프**: 고차 markov로 확장(HMM), 혹은 **커널 방법**(GP), 혹은 **Volterra process** 등

---

## ✏️ 엄밀한 정의 — 분류의 축

### 정의 5.1 — 시간 분류

$T = \mathbb{N}$ (이산), $T = \mathbb{R}_{\geq 0}$ (연속). 경로 성질(연속성·가측성)의 난이도가 극적으로 달라짐.

### 정의 5.2 — 상태 분류

$(E, \mathcal{E})$가 가산(atomic) 또는 Polish (예: $\mathbb{R}^d$). 이산상태에서는 전이확률을 행렬로, 연속상태에서는 kernel $K(x, dy)$로 표현.

### 정의 5.3 — 마르코프 성질

$\{X_t\}$가 $\{\mathcal{F}_t\}$-마르코프라는 것은 모든 $t \geq s$와 Borel $A$에 대해
$$\mathbb{P}(X_t \in A | \mathcal{F}_s) = \mathbb{P}(X_t \in A | X_s).$$
**"미래는 과거와 독립, 현재가 주어지면"**.

### 정의 5.4 — 정상 분류

(Ch1-03) 엄격/약/비정상.

### 정의 5.5 — 추가 축 (자주 사용)

- **Gaussian**: 모든 fdd가 다변량 가우시안
- **Lévy process**: 독립·정상증분 + $X_0 = 0$ + 우연속 경로 (BM, Poisson, Compound Poisson의 통합)
- **Semimartingale**: 이토 적분의 적분자로 쓸 수 있는 가장 일반적인 마르코프 + 마팅게일 합
- **Point process**: 시간 $[0, \infty)$에서 이벤트 시점의 집합 (Poisson, Hawkes 등)

---

## 🔬 정리와 증명 — 분류 간 관계

### 정리 5.1 (Markov + 정상 ⇒ 정상 전이 kernel)

$\{X_t\}$가 시간동질(time-homogeneous) 마르코프 과정이고, 초기분포 $\mu_0 = \pi$가 **불변분포**(invariant)라면 $\{X_t\}$는 엄격 정상.

*증명.* 시간동질 가정으로 $P(X_t \in A | X_0 = x) = P_t(x, A)$ (시각 $t$에만 의존). 정상분포 $\pi$ 조건: $\int P_t(x, \cdot) \pi(dx) = \pi(\cdot)$. 따라서
$$\mathbb{P}(X_{t_1+h} \in A_1, \ldots, X_{t_n+h} \in A_n)$$
를 $X_h$로 조건화하면 $\int \pi(dx_h) P_{t_1}(x_h, dx_1) \cdots = \int \pi(dx_0) P_{t_1}(x_0, dx_1) \cdots$ (마르코프 + $X_h \sim \pi$). 시프트 독립. $\square$

### 정리 5.2 (이산시간 마르코프 과정은 연속시간으로 자연 embed)

$\{X_n\}_{n \in \mathbb{N}}$이 이산시간 MC이면, Poisson clock $N_t$ (rate 1)를 이용해 $\tilde{X}_t = X_{N_t}$로 하면 **연속시간 MC**가 된다 (piecewise-constant 경로). 두 과정은 같은 정상분포를 공유.

*증명 스케치.* $N_t$의 jump time $\tau_k \sim \text{Gamma}(k, 1)$. $\tilde{X}_t$가 구간 $[\tau_k, \tau_{k+1})$에서 $X_k$로 일정. 독립증분성(Poisson)으로 마르코프 성질 유지. 생성기 $Q = P - I$로 나타남. $\square$

> **응용**: 이산 MC의 분석 결과를 연속 MC로 이식하는 표준 기술.

### 정리 5.3 (BM은 연속 Lévy + 마르코프 + 비정상)

BM $B_t$:
- **Lévy**: 독립·정상증분, $B_0 = 0$.
- **마르코프**: $\mathbb{P}(B_t \in A | \mathcal{F}_s^B) = \mathbb{P}(B_t \in A | B_s)$ (증분 독립).
- **비정상**: $\text{Var}(B_t) = t$ (시간 의존).

*증명.* 독립증분 가정으로 마르코프. 분산 $t$는 정의에서. $\square$

> **교훈**: "Lévy + 마르코프"의 예로서 BM. Poisson도 Lévy + 마르코프이지만 이산 상태.

### 정리 5.4 (AR(1)은 이산시간 + 연속상태 + 마르코프)

$X_{n+1} = \phi X_n + \epsilon_{n+1}$, $\epsilon$ iid. 그러면:
- 이산시간·연속상태
- 마르코프 (정의에서)
- $|\phi| < 1$일 때 정상분포 $\mathcal{N}(0, \sigma^2/(1-\phi^2))$ 존재, 정상분포에서 시작하면 엄격 정상.

### 정리 5.5 (Gaussian process는 비마르코프가 일반적)

GP $\{X_t\}$ with covariance $k(s, t)$가 **마르코프**이기 위한 **필요충분조건**은 $k$가 **triangular factorizable**:
$$k(s, t) = u(\min(s, t)) v(\max(s, t))$$
형태 (Brownian motion, OU 등). 일반 RBF 커널 $k(s, t) = \exp(-(s-t)^2/2)$는 **비마르코프**.

*증명 스케치.* 마르코프 GP는 $(X_s, X_t)$ joint given $X_r$ ($r < s < t$)이 Gaussian decomposition $X_t = A(r, t) X_r + \text{indep noise}$ 형태여야 함 — 공분산이 conditioned covariance로 간결화. Triangular factorization 조건이 이를 보장 (자세한 증명은 Karatzas-Shreve). $\square$

### 정리 5.6 (정지과정 ⊂ 에르고딕 과정)

모든 **에르고딕**(ergodic) 과정은 엄격 정상. 역은 거짓. 에르고딕이 마팅게일 극한 정리·MCMC 이론의 핵심.

*증명 아이디어.* 에르고딕의 정의 자체가 "시간 시프트 불변 + shift-invariant 사건의 확률이 0 또는 1"이므로 정상은 자동. 비에르고딕 정상의 예: 두 서로 다른 정상분포의 혼합(각 sample path는 한쪽에만 들어감) — Ch2-06에서 자세히.

---

## 🗺️ 분류 지도 — 전체 요약

### 이산시간 × 이산상태

| 마르코프 | 정상 | 대표 과정 | 핵심 도구 |
|---|---|---|---|
| ✓ | ✓ | MC in stationary distribution | 전이행렬 $P$, 정상분포 $\pi$ |
| ✓ | ✗ | MC before mixing | $P^n$, mixing time |
| ✗ | ✓ | k-order Markov | Hankel matrix, HMM |
| ✗ | ✗ | Non-stationary HMM | 시간의존 전이행렬 |

### 이산시간 × 연속상태

| 마르코프 | 정상 | 대표 | 도구 |
|---|---|---|---|
| ✓ | ✓ | AR(1) stationary | 전이 kernel $K(x, dy)$ |
| ✓ | ✗ | AR(1) non-stationary, DDPM | 시간의존 kernel |
| ✗ | ✓ | ARMA, GP with stationary kernel | 스펙트럴 밀도 |
| ✗ | ✗ | Transformer input sequence | attention mechanism |

### 연속시간 × 이산상태

| 마르코프 | 정상 | 대표 | 도구 |
|---|---|---|---|
| ✓ | ✓ | 정상 CTMC, M/M/1 정상 | Q-matrix, $\pi Q = 0$ |
| ✓ | ✗ | Transient CTMC | Forward/Backward eq |
| ✗ | ✓ | Renewal process | Renewal theorem |
| ✗ | ✗ | Hawkes process | Branching structure |

### 연속시간 × 연속상태

| 마르코프 | 정상 | 대표 | 도구 |
|---|---|---|---|
| ✓ | ✓ | OU in equilibrium, Langevin | Fokker-Planck $\mathcal{L}^* \pi = 0$ |
| ✓ | ✗ | BM, GBM, Score-SDE forward | SDE, Ito calculus |
| ✗ | ✓ | Stationary Gaussian in time | Covariance function $C(t)$ |
| ✗ | ✗ | Fractional BM, Hawkes, rough vol | Volterra / fractional calculus |

---

## 💻 NumPy 구현 검증

### 실험 1 — 4가지 핵심 분면의 sample path 비교 시각화

```python
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# (1) 이산·이산·MC·정상: 2-state 마르코프 체인
P = np.array([[0.8, 0.2], [0.3, 0.7]])
pi = np.linalg.solve(np.vstack([(P.T - np.eye(2))[:-1], np.ones(2)]),
                     np.array([0, 1]))
state = rng.choice(2, p=pi)
path = [state]
for _ in range(100):
    state = rng.choice(2, p=P[state])
    path.append(state)
axes[0, 0].step(range(len(path)), path, where='post')
axes[0, 0].set_title('이산·이산·Markov·정상: 2-state MC\n(정상분포에서 시작)')

# (2) 이산·연속·Markov·비정상: Random walk (simple RW가 아니라 BM 이산화)
N = 100; dt = 0.01
X = np.concatenate([[0], np.cumsum(rng.standard_normal(N) * np.sqrt(dt))])
axes[0, 1].plot(X)
axes[0, 1].set_title('이산·연속·Markov·비정상: Random walk')

# (3) 연속·이산·Markov·비정상: Poisson 과정
lam = 1.5; T = 10; t = 0.0; times = [0]; counts = [0]
while t < T:
    t += rng.exponential(1/lam)
    if t < T:
        times.append(t); counts.append(counts[-1] + 1)
times.append(T); counts.append(counts[-1])
axes[1, 0].step(times, counts, where='post')
axes[1, 0].set_title(r'연속·이산·Markov·비정상: Poisson($\lambda=1.5$)')

# (4) 연속·연속·Markov·정상: OU in equilibrium
theta, sig = 2.0, 1.0
N = 10000; dt = 0.001; T = N * dt
X = np.zeros(N + 1)
X[0] = rng.standard_normal() * sig / np.sqrt(2 * theta)  # 정상 start
for n in range(N):
    X[n+1] = X[n] - theta * X[n] * dt + sig * rng.standard_normal() * np.sqrt(dt)
axes[1, 1].plot(np.linspace(0, T, N+1), X)
axes[1, 1].set_title(r'연속·연속·Markov·정상: OU ($\theta=2$) in equilibrium')

for ax in axes.flat:
    ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
```

### 실험 2 — GP with 정상 커널의 마르코프성 확인

```python
# OU: covariance σ²e^{-θ|t-s|} — 마르코프 (triangular factorizable 형태)
# RBF: covariance exp(-(s-t)²/2) — 비마르코프
# 경험적으로 P(X_2 | X_0, X_1) 과 P(X_2 | X_1)의 차이로 확인

def sample_gp(kernel, t_grid, n_samples=10_000):
    K = kernel(t_grid[:, None], t_grid[None, :])
    L = np.linalg.cholesky(K + 1e-8 * np.eye(len(t_grid)))
    return (rng.standard_normal((n_samples, len(t_grid)))) @ L.T

t_grid = np.array([0.0, 0.5, 1.0])
# OU 커널
K_ou = lambda s, t: np.exp(-2 * np.abs(s - t))
X_ou = sample_gp(K_ou, t_grid)
# RBF 커널
K_rbf = lambda s, t: np.exp(-(s - t)**2 / 0.5)
X_rbf = sample_gp(K_rbf, t_grid)

# 조건부 분산 비교
from numpy.linalg import solve
def cond_var(K, future_idx, past_idx):
    K_pp = K[np.ix_(past_idx, past_idx)]
    K_pf = K[np.ix_(past_idx, [future_idx])]
    K_ff = K[future_idx, future_idx]
    return K_ff - K_pf.T @ solve(K_pp, K_pf)

K_ou_full = np.exp(-2 * np.abs(t_grid[:, None] - t_grid[None, :]))
K_rbf_full = np.exp(-(t_grid[:, None] - t_grid[None, :])**2 / 0.5)

print('OU: Var(X_2 | X_1) =      {:.4f}'.format(cond_var(K_ou_full, 2, [1]).item()))
print('OU: Var(X_2 | X_0, X_1) = {:.4f}'.format(cond_var(K_ou_full, 2, [0, 1]).item()))
# → 두 값이 같음: 마르코프 성질 (X_0 추가 정보 없음)

print('RBF: Var(X_2 | X_1) =      {:.4f}'.format(cond_var(K_rbf_full, 2, [1]).item()))
print('RBF: Var(X_2 | X_0, X_1) = {:.4f}'.format(cond_var(K_rbf_full, 2, [0, 1]).item()))
# → RBF에선 두 값이 다름: 비마르코프 (과거가 추가 정보 제공)
```

---

## 🔗 AI/ML 연결 — 분류 지도 on 주요 모델

| 모델 | 시간 | 상태 | 마르코프? | 정상? |
|---|---|---|---|---|
| **Transformer (encoder)** | 이산 | 연속 (embedding) | ✗ (attention all-to-all) | 정상 가정 (기본) |
| **Transformer (decoder, causal)** | 이산 | 연속 | ✗ (long memory via attention) | 비정상 (position-dependent) |
| **RNN/LSTM** | 이산 | 연속 | ✓ (hidden state) | 비정상 (training) |
| **Mamba (SSM)** | 이산 | 연속 | ✓ (state space) | 비정상 |
| **HMM** | 이산 | 이산 hidden / 연속 observation | ✓ (hidden), ✗ (observation) | 주로 정상 |
| **DDPM** | 이산 | 연속 | ✓ (forward/reverse) | 비정상 |
| **Score-SDE** | 연속 | 연속 | ✓ | 비정상 |
| **Flow Matching** | 연속 | 연속 | ✓ (deterministic flow) | 비정상 |
| **Neural ODE** | 연속 | 연속 | ✓ (ODE) | 가변 |
| **Gaussian Process 회귀** | 이산 (관측) | 연속 | ✗ (일반적) | 주로 정상 kernel |
| **PPO / TRPO** | 이산 | 연속 (policy) | ✓ (MDP) | 정상 policy 가정 |
| **Bandits** | 이산 | 이산/연속 arms | ✓ (iid arms) | 정상 |
| **Hawkes process (예: 뉴스 피드 모델링)** | 연속 | 이산 | ✗ (self-exciting) | 비정상 |

**Key insight**: DDPM의 이산→연속 확장(Score-SDE)은 분류의 "시간" 축을 이산에서 연속으로 이동시킨 것. 이는 **이산 반복문 → SDE 미분방정식**의 이론적 일반화이며, Ch6(SDE 레포)의 Euler-Maruyama가 그 반대 방향 이산화를 제공.

---

## ⚖️ 가정과 한계

**가정 — 분류가 명확한 경우가 드뭄**  
실세계 데이터는 "거의" 정상이거나 "부분적으로" 마르코프인 경우가 많다. 예: 주식 수익률은 약 정상에 가깝지만 변동성 클러스터링(GARCH)에서 장기의존성 발현. 분류는 "모델링 선택"이지 현실의 객관 성질은 아님.

**한계 — 비마르코프의 취급**  
비마르코프 과정은 일반적으로 분석이 어렵다. 주요 전략:
1. **Markov 확장**: $\tilde{X}_n = (X_n, X_{n-1}, \ldots, X_{n-k+1})$로 $k$-order Markov를 1-order로 (HMM 등)
2. **Latent Markov**: hidden state로 Markov 구조 복원 (Kalman, LSTM)
3. **Functional representation**: GP covariance 등으로 직접 modeling

**한계 — 비정상 과정의 정상화 트릭**  
1. **Differencing**: $\Delta X_t = X_t - X_{t-1}$ (ARIMA의 I 부분)
2. **Detrending**: $X_t - (a + bt)$
3. **Log transformation**: 금융 수익률
4. **Scaling**: Score-SDE의 time embedding 조건부 — 비정상을 "인식"하고 처리

**분류의 한계 — 연속 경로와 jump**  
Lévy process는 "연속 + jump"의 혼합 — BM은 연속만, Poisson은 jump만, Compound Poisson은 둘 다. 이 세분류가 분류표에 추가로 필요.

---

## 📌 핵심 정리

| 4개 축 | 의미 | 실전 함의 |
|---|---|---|
| 시간 이산/연속 | $T = \mathbb{N}$ vs $\mathbb{R}_+$ | 수치해석 vs 적분방정식 |
| 상태 이산/연속 | 가산 vs $\mathbb{R}^d$ | 행렬 vs 커널 |
| Markov/non-Markov | 미래 ⊥ 과거 | given 현재 | 단일 상태 vs 전체 기억 |
| 정상/비정상 | 시프트 불변 | 에르고딕 vs 시간 조건부 |

**한 줄 요약**: 확률과정은 4개의 2진 축으로 분류되며, 각 분면마다 전용 수학적 도구(전이행렬, Q-matrix, SDE, 커널)와 AI 응용(MC, DDPM, Score-SDE, GP)이 대응한다. 모델 설계의 첫 단계는 "내 데이터가 이 맵의 어디에 있는가"를 식별하는 것.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. 다음 각 과정이 4개 축에서 어느 분면에 속하는지 분류하라.
(a) $X_n = \cos(\omega n + \Phi)$, $\Phi \sim \text{Uniform}(0, 2\pi)$
(b) GARCH(1,1): $X_t = \sigma_t \epsilon_t$, $\sigma_t^2 = \omega + \alpha X_{t-1}^2 + \beta \sigma_{t-1}^2$
(c) Hawkes process (self-exciting point process)

<details>
<summary>해설</summary>

**(a) Sinusoid with random phase**: 이산시간·연속상태. $(X_n)$만으로 다음을 예측할 수 있는가? $X_n$의 값을 알면 $\cos(\omega n + \Phi)$ 제약에서 $\Phi$ 후보가 몇 개로 줄지만, 다음 시각 값이 deterministic ($X_{n+1} = \cos(\omega(n+1) + \Phi)$). 실제로 $(X_{n-1}, X_n)$로부터 $\Phi$ 완전 복구(2개 조건에서 $\omega$ 주어짐). 따라서 $(X_{n-1}, X_n)$로 확장한 2-order Markov → 1-order 아님. **비마르코프**.  
정상? 평균 $\mathbb{E}[X_n] = \mathbb{E}[\cos(\omega n + \Phi)] = 0$ (균등 $\Phi$에서), 공분산 $\text{Cov}(X_s, X_t) = \frac{1}{2}\cos(\omega(t-s))$ — 시차 의존. **약 정상**. 확률 구조가 $\Phi$ 하나에만 의존하므로 **엄격 정상**도 성립.  
→ **이산·연속·비마르코프·엄격정상**.

**(b) GARCH(1,1)**: 이산시간·연속상태. $\sigma_t$가 $X_{t-1}$, $\sigma_{t-1}$에 의존하므로, $(X_t, \sigma_t)$ joint state로 보면 마르코프. $X_t$만으로는 비마르코프. 정상 조건 $\alpha + \beta < 1$ 하에서 약 정상 (실제로는 엄격 정상이 더 강한 조건).  
→ **이산·연속·마르코프 (extended state)**, **정상** (파라미터 조건 하).

**(c) Hawkes process**: 연속시간·이산상태 (이벤트 카운트). Intensity $\lambda_t = \mu + \sum_{T_i < t} g(t - T_i)$가 과거 이벤트에 의존 → **비마르코프** (unless $g$ exponential, 그때 확장 state로 Markov). 초기 기간에는 평형에 도달하지 않으므로 **비정상**이지만 $t \to \infty$에서 stationary regime 수렴 가능.  
→ **연속·이산·비마르코프·비정상 (일반적)**.

</details>

**문제 2 (심화)**. "비정상" 과정을 "정상"으로 만드는 세 가지 변환(differencing, detrending, 시간 rescaling)을 설명하고, 각각이 원 과정의 어떤 구조적 성질을 가정하는지 논하라.

<details>
<summary>해설</summary>

**(1) Differencing $\Delta X_t = X_t - X_{t-1}$**  
**가정**: 원 과정이 **$I(d)$** (integrated of order $d$) — $d$번 차분해야 정상. 예: Random walk $X_t = X_{t-1} + \epsilon_t$는 $I(1)$, 1번 차분하면 iid. 차분은 **trend와 unit root를 제거**하지만 장기 누적 정보도 지움 → 재구성 시 적분 필요(I in ARIMA).

**(2) Detrending $Y_t = X_t - (a + bt)$**  
**가정**: $X_t = a + bt + Z_t$, $Z$가 정상. 즉 **결정론적 추세(deterministic trend)**. Trend가 stochastic(unit root)이면 detrending이 잘못됨 — 이것이 **"trend-stationary" vs "difference-stationary"**의 구분. 현실 시계열에서 둘 중 어느 것인지는 unit root 검정(ADF, KPSS)으로 확인.

**(3) 시간 rescaling $\tau = \phi(t)$**  
**가정**: 원 과정의 "intrinsic clock"이 $t$가 아니라 $\phi(t)$. 예: Brownian motion $B_{\phi(t)}$를 관찰하면 분산이 $\phi(t)$에 비례하는 비정상. 역변환 $B_t$로 돌리면 "정상적인 BM". 금융의 **stochastic time change**(Clark 1973, trading volume을 시간으로 해석) 등에서 쓰임.

**종합**: 세 기법 모두 "비정상성을 설명하는 구조(trend, random walk, time scale)를 식별 → 제거"하는 패턴. 어느 것이 맞는지는 데이터 생성 기전에 대한 **모델 가정**에 의존.

</details>

**문제 3 (AI 연결)**. Transformer의 self-attention은 "모든 과거 (+ 미래)"를 직접 참조하므로 **비마르코프**인 반면, Mamba (Selective SSM)는 내부 state $h_t$를 통해 **마르코프**적이다. 이 구조적 차이가 학습·추론 복잡도에 어떻게 영향을 미치는가? 분류 관점에서 논하라.

<details>
<summary>해설</summary>

**비마르코프 (Transformer)**:
- 출력 $y_t = \text{Attention}(q_t, \{k_s, v_s\}_{s \leq t})$ — 모든 과거에 직접 의존
- **추론 복잡도**: 시각 $t$에서 $O(t)$ 연산 (all past), 전체 $O(T^2)$
- **학습**: 병렬화 용이 (all tokens simultaneously)
- **문맥 길이 제약**: 메모리 $O(T^2)$로 긴 문맥 어려움
- **정확도**: long-range dependency를 잘 잡음 (attention이 직접 연결)

**마르코프 (Mamba, RNN)**:
- $y_t = f(h_t)$, $h_{t+1} = g(h_t, x_t)$ — hidden state가 전체 "과거 요약"
- **추론 복잡도**: $O(1)$ per step, 전체 $O(T)$
- **학습**: 순차적 (recurrent), 병렬화 어려움 (Mamba는 selective scan으로 병렬화 trick)
- **문맥 길이**: $O(T)$ 메모리로 긴 문맥 처리 가능
- **정확도**: hidden state 차원 한계로 long-range 일부 손실

**분류 관점**:
두 접근은 "비마르코프 full memory" vs "마르코프 bottleneck state"의 trade-off. 이는 **HMM**과 **AR(∞)** 대조와 같은 구조:
- HMM: hidden Markov state → compact representation, bottleneck 제약
- AR(∞): 무한 과거 직접 참조 → expressive하지만 연산 비용

**실전 함의**: 문맥이 매우 길고(>100K tokens) RAM이 제약적이면 Mamba (Markov)가 유리. 단, 복잡한 reasoning이 필요하면 Transformer (non-Markov)가 여전히 우세. 이 분류 인식은 **아키텍처 선택의 원리적 가이드**를 제공.

</details>

---

<div align="center">

◀ [04. 필트레이션(Filtration)과 정보 흐름](./04-filtration.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [Ch2-01. 마르코프 성질과 전이행렬](../ch2-discrete-markov/01-markov-property.md)

</div>
