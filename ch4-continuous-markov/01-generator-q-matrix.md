# 01. 생성기(Generator)와 Q-matrix

## 🎯 핵심 질문

- 연속시간 마르코프 체인(CTMC)에서 **생성기** $Q$는 어떻게 정의되고 — $Q_{ij} = \lim_{h \to 0} \frac{P_{ij}(h) - \delta_{ij}}{h}$가 왜 이 극한으로 정의되는가?
- 왜 **행합 0** ($\sum_j Q_{ij} = 0$)이고, $-q_{ii}$가 상태 $i$의 **탈출률**인가?
- CTMC를 **jump chain + holding time**으로 분해하면 어떻게 되는가 — $i$에 머무는 시간이 $\text{Exp}(q_{ii})$인 이유는?
- 행렬지수 $P(t) = e^{tQ}$의 의미는?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**연속시간 RL / Stochastic Differential Games**: 의사결정 간격이 불규칙 (random) — 이산 MDP의 연속 일반화. 생성기 $Q$로 value function의 ODE: $\mathcal{T} V = \text{optimal Q-matrix}$.

**Temporal Point Process의 state transitions**: 사용자 행동 sequence를 CTMC로 모델링 (활동 → 비활동 → 활동). 생성기로 dwell time 분포 추정.

**Continuous-time VI / VAE extensions**: latent dynamics를 CTMC로 parameterize (discrete latent with continuous clock). Hidden Markov Model의 연속시간 버전.

**Wu-type queueing + LLM inference**: M/M/1의 일반화 M/M/c를 Q-matrix로 분석. GPU serving의 세밀한 latency 모델.

---

## 📐 수학적 선행 조건

- [Ch2-01](../ch2-discrete-markov/01-markov-property.md): Markov 성질, 전이확률
- [Ch3-01](../ch3-poisson/01-three-equivalent-definitions.md): Exp 메모리리스 — CTMC holding time의 근거
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): 행렬지수 $e^{tA}$

---

## 📖 직관적 이해

### CTMC의 두 관점

**관점 A (분포 진화)**: $P_{ij}(t) = \mathbb{P}(X_t = j | X_0 = i)$ 전이확률. 각 $t$마다 행렬. Semigroup 성질 $P(t+s) = P(t) P(s)$.

**관점 B (jump + holding)**: $X_t$가 각 상태 $i$에 $\text{Exp}(q_{ii})$ 시간 머물다 다른 상태 $j \neq i$로 jump. Jump chain 분포 $\mathbb{P}(j | i) = q_{ij}/q_{ii}$.

**두 관점 연결**: $Q$가 infinitesimal generator이고 $P(t) = e^{tQ}$.

### 왜 생성기인가

$P(t)$ 전체를 연속 시간의 함수로 기술하는 것은 무한차원. **미분 정보**(생성기 $Q = P'(0)$)만으로 모든 $P(t)$ 결정 가능 — 행렬 ODE $P'(t) = QP(t)$의 해 $P(t) = e^{tQ}$.

> **비유**: "속도"와 "위치"의 관계. 위치 $x(t)$ 전체를 기록하는 것보다 초기 위치 $x(0)$ + 속도장 $\dot x = f(x)$를 알면 모든 $x(t)$ 결정. $Q$가 확률과정의 "속도장".

### Exp holding time의 필연성

**메모리리스**(Exp)가 CTMC의 core:
- 시각 $s$에서 $X_s = i$라 하자. Markov 성질 → 미래는 현재만 의존 → $s$ 시점부터 상태 $i$에 얼마나 머무는지의 분포가 $\mathcal{F}_s$와 독립.
- 하지만 "이미 얼마나 머물렀는지"도 $\mathcal{F}_s$에 없음 (현재 상태만) → holding time 분포가 **age-free** = 메모리리스 = Exp.

이것이 CTMC holding time이 **Exp(rate)** 여야 하는 이유.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — 연속시간 마르코프 체인

$\{X_t\}_{t \geq 0}$가 가산 상태공간 $E$에서 정의된 CTMC이다:
1. $X_0 \sim $ some distribution
2. Markov 성질: $\mathbb{P}(X_{t+s} = j | \mathcal{F}_t) = \mathbb{P}(X_{t+s} = j | X_t)$
3. 경로가 우연속, 유한 jump (piecewise constant with isolated jump points)

### 정의 1.2 — 전이 semigroup

$P_{ij}(t) := \mathbb{P}(X_{t+s} = j | X_s = i)$ (시간 동질성). 행렬 $P(t) = (P_{ij}(t))$.

**성질**:
- $P(0) = I$
- $P(t + s) = P(t) P(s)$ (Chapman-Kolmogorov 연속판)
- 각 행 확률분포

### 정의 1.3 — 생성기 (Generator)

$Q_{ij} := \lim_{h \to 0^+} \frac{P_{ij}(h) - \delta_{ij}}{h}$ (존재하면).

행렬 $Q = (Q_{ij})$. 

### 정의 1.4 — Q-matrix의 성질

**Q-matrix**는 다음을 만족:
1. $q_{ij} \geq 0$ for $i \neq j$
2. $q_{ii} \leq 0$
3. $\sum_j q_{ij} = 0$ (행합 0)

$q_i := -q_{ii} \geq 0$ = 상태 $i$의 **탈출률**.

### 정의 1.5 — Jump chain and holding time

$X_t$가 상태 $i$를 방문하면 $\text{Exp}(q_i)$ 시간 머문 뒤 상태 $j$로 jump (with $j \neq i$).
$$\mathbb{P}(\text{jump to } j | \text{leave } i) = q_{ij}/q_i \quad (j \neq i).$$

**Jump chain** $Y_n = X_{S_n}$ ($S_n$ = $n$번째 jump 시각)은 이산 Markov chain with transition $\tilde P_{ij} = q_{ij}/q_i$ ($i \neq j$), $\tilde P_{ii} = 0$.

---

## 🔬 정리와 증명

### 정리 1.1 — Q-matrix의 기본 성질

$Q = P'(0)$는 위 세 성질 (1)-(3) 만족.

*증명.*
(1) $P_{ij}(h) \geq 0$ and $\delta_{ij} = 0$ for $i \neq j$, so $Q_{ij} = \lim P_{ij}(h)/h \geq 0$.

(2) $P_{ii}(h) \leq 1 = \delta_{ii}$, so $Q_{ii} \leq 0$.

(3) $\sum_j P_{ij}(h) = 1$ (확률행렬). 미분: $\sum_j Q_{ij} = \frac{d}{dh}\bigg|_{h=0} 1 = 0$. $\square$

### 정리 1.2 — Jump chain 구성과 holding time이 Exp

$X_0 = i$. 첫 jump 시각 $T_1 := \inf\{t > 0 : X_t \neq i\}$. 그러면:
- $T_1 \sim \text{Exp}(q_i)$
- $X_{T_1}$이 $j \neq i$일 확률 = $q_{ij}/q_i$
- $T_1$과 $X_{T_1}$ 독립

*증명 스케치*.
**Holding time Exp**: Markov 성질 + 메모리리스 (Ch3-01 정리 1.1)로 $T_1$은 메모리리스 → Exp. Rate 결정: $\mathbb{P}(T_1 > h) = P_{ii}(h) = 1 + q_{ii} h + o(h) = 1 - q_i h + o(h)$. Exp distribution과 매치 → rate $q_i$.

**Jump 분포**: $\mathbb{P}(X_{T_1} = j, T_1 \leq h) = \mathbb{P}(X_h = j, \text{no double jump}) = P_{ij}(h) + o(h) = q_{ij} h + o(h)$. Normalize.

**독립**: 강한 Markov 성질. $\square$

### 정리 1.3 — Kolmogorov Forward / Backward 방정식 (Preview)

**Forward**: $P'(t) = P(t) Q$  
**Backward**: $P'(t) = Q P(t)$

(Ch4-02에서 유도. 여기서는 결과만.)

### 정리 1.4 — 행렬지수 해

유한 상태 CTMC에서
$$P(t) = e^{tQ} := \sum_{n \geq 0} \frac{(tQ)^n}{n!}.$$

*증명.* $P(t) = e^{tQ}$로 두고 ODE $P'(t) = QP(t)$, $P(0) = I$의 해가 unique (Picard) → $e^{tQ}$이 해. $\square$

### 정리 1.5 — Jump chain과 원 CTMC의 관계

Jump chain $\{Y_n\}$의 정상분포와 CTMC의 정상분포는 관련있지만 다름:
- CTMC 정상 $\pi$: $\pi Q = 0$
- Jump chain 정상 $\tilde \pi$: $\tilde \pi \tilde P = \tilde \pi$
- 관계: $\pi_i = \tilde \pi_i / q_i / (\sum_j \tilde \pi_j/q_j)$

*해석*: 시간 평균 방문 비율 = "jump 방문 비율" × "머무는 시간". Ch4-03 에서 자세히.

---

## 💻 NumPy 구현 검증

### 실험 1 — 3-state CTMC 시뮬레이션 (Jump chain + holding)

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# 3-state Q-matrix
Q = np.array([
    [-2.0,  1.5,  0.5],
    [ 1.0, -3.0,  2.0],
    [ 0.5,  1.0, -1.5],
])
assert np.allclose(Q.sum(axis=1), 0)  # 행합 0

# Jump chain
q = -np.diag(Q)   # 탈출률
P_jump = Q.copy()
for i in range(3):
    P_jump[i, i] = 0
    P_jump[i] /= q[i]

# 시뮬레이션
T = 30.0
t = 0.0; state = 0
times = [t]; states = [state]
while t < T:
    dwell = rng.exponential(1/q[state])
    t += dwell
    state = rng.choice(3, p=P_jump[state])
    times.append(t); states.append(state)

# 경로 plot
plt.step(times, states, where='post')
plt.xlabel('t'); plt.ylabel('state')
plt.title('CTMC sample path'); plt.grid(True, alpha=0.3); plt.show()

print(f'탈출률: q = {q}')
print(f'Jump chain P: \n{P_jump}')
```

### 실험 2 — P(t) = e^{tQ} 검증

```python
from scipy.linalg import expm

t = 2.0
P_t = expm(t * Q)
print(f'P(t=2):\n{P_t}')
print(f'행합: {P_t.sum(axis=1)}')   # 모두 1

# Monte Carlo로 P_{00}(t) 추정
n_sim = 10000
count_00 = 0
for _ in range(n_sim):
    state = 0; t_cur = 0.0
    while t_cur < t:
        dwell = rng.exponential(1/q[state])
        if t_cur + dwell > t:
            break   # 시각 t에서 여전히 state
        t_cur += dwell
        state = rng.choice(3, p=P_jump[state])
    if state == 0:
        count_00 += 1

print(f'MC P_00(2): {count_00/n_sim:.4f}, 이론: {P_t[0,0]:.4f}')
```

### 실험 3 — Holding time이 Exp인지 검증

```python
state = 0
holding_times = []
for _ in range(10000):
    dwell = rng.exponential(1/q[state])
    holding_times.append(dwell)

holding_times = np.array(holding_times)
print(f'실측 평균: {holding_times.mean():.4f}, 이론: {1/q[0]:.4f}')

# Q-Q plot vs Exp
from scipy.stats import probplot
probplot(holding_times, dist='expon', sparams=(0, 1/q[0]), plot=plt)
plt.title('Holding time Q-Q plot vs Exp')
plt.show()
```

---

## 🔗 AI/ML 연결

**Continuous-Time HMM (CT-HMM)**  
Healthcare에서 환자 상태 (정상/경증/중증/치명)이 CTMC. 치료 decisions를 최적화 (MDP on CTMC). Q-matrix가 disease progression rate를 parameterize, 데이터로부터 MLE 추정.

**Continuous-Time RL**  
Continuous decision interval. HJB equation의 이산 버전 = CTMC 위의 Bellman. Policy gradient의 연속 한계.

**Neural Jump Process Models**  
Mei-Eisner의 Neural Hawkes는 continuous-time 신경망으로 intensity $\lambda(t | \mathcal{H}_t)$ 모델링. State transitions가 있을 때 CTMC + history dependence.

**Compartmental Models (SIR, 전염병)**  
Susceptible → Infected → Recovered의 CTMC. Q-matrix의 rate를 NN으로 예측 (COVID AI 연구).

---

## ⚖️ 가정과 한계

**가정 — 이산 state space**  
Continuous state CTMC는 **Feller process**로 일반화 — generator가 미분연산자 (Ch4에서는 이산만).

**가정 — 유한 탈출률 $q_i < \infty$**  
무한 탈출률(instantaneous)은 pathological — explosion 문제 발생 가능 (jump sequence $T_n$이 유한 시간에 무한 수렴).

**한계 — 유한상태에서 $e^{tQ}$ 계산 비용**  
$N \times N$ 행렬 지수 비용 $O(N^3)$. Large state space ($N \sim 10^6$)는 Krylov subspace methods 등 근사 필요.

**한계 — Non-stationary Q(t)**  
시간에 따라 변하는 $Q(t)$ (inhomogeneous CTMC)는 $P(t) = \exp(\int_0^t Q(s) ds)$가 아님 (교환성 실패). Time-ordered product 필요.

---

## 📌 핵심 정리

| 개념 | 정의 / 수식 |
|---|---|
| Q-matrix | $Q_{ij} = \lim_{h \to 0} (P_{ij}(h) - \delta_{ij})/h$ |
| Q 성질 | 비대각 $\geq 0$, 대각 $\leq 0$, 행합 0 |
| 탈출률 | $q_i = -q_{ii}$ |
| Holding time | $\text{Exp}(q_i)$ |
| Jump prob | $q_{ij}/q_i$ ($j \neq i$) |
| 행렬 ODE | $P'(t) = QP(t)$ |
| 행렬지수 | $P(t) = e^{tQ}$ |

**한 줄 요약**: 연속시간 MC의 **infinitesimal dynamics**는 Q-matrix로 완전 encoding되며, holding time의 메모리리스 (Exp) 성질이 Markov 구조의 필연. $P(t) = e^{tQ}$는 ODE의 해.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. 2-state CTMC with $Q = \begin{pmatrix} -\alpha & \alpha \\ \beta & -\beta \end{pmatrix}$. $P(t)$와 정상분포를 구하라.

<details>
<summary>해설</summary>

**행렬지수**: 고유값 0, $-(\alpha + \beta)$. 대응 고유벡터로 diagonalize.

$P(t) = e^{tQ} = \frac{1}{\alpha+\beta}\begin{pmatrix} \beta + \alpha e^{-(\alpha+\beta)t} & \alpha(1 - e^{-(\alpha+\beta)t}) \\ \beta(1 - e^{-(\alpha+\beta)t}) & \alpha + \beta e^{-(\alpha+\beta)t} \end{pmatrix}$.

**정상분포**: $\pi Q = 0$, $\pi_0 + \pi_1 = 1$. $-\alpha \pi_0 + \beta \pi_1 = 0 \Rightarrow \pi_1 = \alpha \pi_0/\beta$. 정규화 $\pi = (\beta/(\alpha+\beta), \alpha/(\alpha+\beta))$.

**해석**: $\alpha$ 크면 (상태 0에서 빨리 떠남) $\pi_1$이 큼 (1에 많이 머묾). $t \to \infty$에서 $P(t) \to \mathbf{1}\pi^T$.

</details>

**문제 2 (심화)**. CTMC에서 첫 jump 이후 두번째 jump까지의 시간 $T_2 - T_1$의 분포는 $T_1$과 독립인가?

<details>
<summary>해설</summary>

**독립**. 강한 Markov 성질 + 메모리리스.

$T_1$에서 jump 후 새 상태 $j$ 도달. 그 시점부터 $j$의 holding time = $\text{Exp}(q_j)$, 이는 $\mathcal{F}_{T_1}$과 독립. 따라서 $T_2 - T_1$의 분포는 $j$에만 의존하고 $T_1$ 값과 독립.

**주의**: $T_2 - T_1$과 $T_1$은 independent distribution이지만, 조건부 $T_2 - T_1 | X_{T_1}$는 의존. 즉 $T_1$과 marginal 독립이지만 full joint로는 state-dependent.

**연결**: 이 구조가 embedded jump chain $Y_n = X_{T_n}$이 이산 MC임을 보장. Holding time은 각 jump 사이에 독립적으로 "삽입".

</details>

**문제 3 (AI 연결)**. 사용자 행동을 CTMC로 모델링: 상태 = {active, idle, offline}. $Q$의 각 entry를 NN으로 context-dependent하게 $Q_{ij}(x)$로 예측하려면 어떤 구조적 제약과 학습 전략이 필요한가?

<details>
<summary>해설</summary>

**구조적 제약**:
1. **행합 0**: $\sum_j Q_{ij}(x) = 0$ — softmax 대신 **diagonal = -sum of off-diagonal**로 parameterize.
2. **비대각 $\geq 0$**: ReLU 또는 softplus로 positive 보장.
3. **Identifiability**: rate scale과 time scale이 함께 바뀌면 equivalent → 시간 단위 정규화 필요.

**구체적 parameterization**:
```python
def Q_network(x):
    rates = softplus(f_theta(x))  # shape (num_states, num_states-1)
    # 대각 자리 건너뛰고 채움, 대각은 -row_sum
    Q = np.zeros((num_states, num_states))
    for i in range(num_states):
        Q[i, :i] = rates[i, :i]
        Q[i, i+1:] = rates[i, i:]
        Q[i, i] = -rates[i].sum()
    return Q
```

**학습 전략**:
1. **Log-likelihood**: 관찰된 trajectory $\{(t_k, s_k)\}$에 대해
$$\log L = \sum_k \log Q_{s_k \to s_{k+1}}(x_{t_k}) - \int_{t_k}^{t_{k+1}} q_{s_k}(x_s) ds.$$
   Forward holding time contribution + jump transition contribution.

2. **Numerical stability**: $q_i = |Q_{ii}|$가 너무 작으면 log-likelihood 발산. Clip rate minimum.

3. **Regularization**: $\ell_2$ on $Q$ entries, 또는 sparsity (대부분의 transition이 rare).

4. **Covariate $x$ 변화율 관리**: $x_t$가 CTMC와 같은 시간 scale이면 **non-autonomous CTMC**. Variational inference 필요.

**실전 challenge**:
- **Multi-timescale dynamics**: 일부 transitions 초, 일부 분/시간.
- **Censored data**: 사용자 session이 도중에 끊김 → 정확히 언제 idle이 되었는지 모름.
- **Sparse events**: 많은 사용자의 대부분 transition은 healthy → model이 majority에 bias됨 → weighted loss.

**연결**: DDPM forward (Ch6 SDE) 구조와 유사 — discrete-state 버전. Continuous-time HMM 확장 (Liu 2020, Rasmussen 2020).

</details>

---

<div align="center">

◀ [Ch3-04. Queueing 이론 맛보기 — M/M/1, Little의 법칙](../ch3-poisson/04-queueing-little.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [02. Kolmogorov Forward/Backward 방정식](./02-kolmogorov-equations.md)

</div>
