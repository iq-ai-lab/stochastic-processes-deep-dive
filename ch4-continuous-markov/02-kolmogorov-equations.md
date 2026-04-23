# 02. Kolmogorov Forward/Backward 방정식

## 🎯 핵심 질문

- **Forward** 방정식 $P'(t) = P(t)Q$와 **Backward** 방정식 $P'(t) = QP(t)$ — 두 방정식이 왜 서로 다른 해석을 갖는가?
- 두 방정식의 해가 같은 $e^{tQ}$인 이유는 — 행렬 $P$와 $Q$가 언제 교환하는가?
- Forward 방정식이 "미래의 distribution evolution", Backward가 "초기 기댓값 evolution"을 기술하는 수학적 근거는?
- 유한상태에서 $P(t) = e^{tQ}$의 **명시적 계산**은 어떻게 하는가 (spectral, Padé, Krylov)?

---

## 🔍 왜 이 방정식들이 AI에서 중요한가

**Continuous-time value function**: RL의 Bellman equation이 CTMC 위에서 $\partial_t V(x, t) = (QV)(x, t) + r(x)$ — Backward 방정식의 직접 응용.

**Transition 예측**: 의료 AI에서 "질환 상태의 $T$시간 후 분포 예측" = Forward $P'(t) = P(t) Q$ 풀이. Deep learning이 $Q$를 예측하면 $P(T)$는 ODE 풀이.

**Fokker-Planck 방정식**의 이산 전조: SDE의 연속상태 버전(SDE Deep Dive Ch4-01)은 Forward 방정식의 generalization.

**Semigroup theory**: 현대 generative model의 score-based SDE가 **Markov semigroup**을 활용. Forward/Backward의 추상 일반화.

---

## 📐 수학적 선행 조건

- [Ch4-01](./01-generator-q-matrix.md): Generator $Q$, $P(t) = e^{tQ}$
- ODE 기초: 행렬 ODE $\dot X = AX$, Picard iteration
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): 행렬지수, diagonalization

---

## 📖 직관적 이해

### 두 방정식의 다른 이야기

**Forward** ($P' = PQ$): "$t$에서 $t + dt$로 한 step 더 진행" — $P(t+dt) \approx P(t)(I + Q dt) = P(t) + P(t) Q dt$. **"미래 방향"** 진화.

**Backward** ($P' = QP$): "$0$에서 $dt$로 초기에 한 step 더 진행" — $P(dt) P(t - dt) \approx (I + Q dt) P(t - dt)$. **"초기 방향"** 진화.

### 왜 둘 다 같은 해?

행렬 $P(t)$와 $Q$가 **교환**(commute) — $PQ = QP$ — 이기 때문. 이는 Markov semigroup 성질 $P(t) P(s) = P(s) P(t)$에서 나옴 (in fact, $P(s) = e^{sQ}$와 $P(t) = e^{tQ}$는 같은 생성기에 대한 semigroup이므로 교환).

유한상태에서는 양쪽 다 $P(t) = e^{tQ}$ 같은 해. 무한상태 / 비동질에서는 차이가 생길 수 있음.

### Distribution evolution

행벡터 $\mu_t$ (시각 $t$의 분포)에 대해 Forward:
$$\mu_t = \mu_0 P(t) \Rightarrow \dot \mu_t = \mu_0 P'(t) = \mu_0 P(t) Q = \mu_t Q.$$

즉 **$\dot \mu_t = \mu_t Q$** — "distribution의 시간 진화".

### Expectation evolution

관찰함수 $f : E \to \mathbb{R}$에 대해 $(P(t) f)(i) = \mathbb{E}_i[f(X_t)]$. Backward로:
$$\frac{d}{dt} \mathbb{E}_i[f(X_t)] = (QP(t) f)(i) = (Q \mathbb{E}_\bullet[f(X_t)])(i).$$

**"기댓값의 시간 진화"** — Feynman-Kac의 시작.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Forward equation

$P(t)$의 원소별 행렬 ODE:
$$P'_{ij}(t) = \sum_k P_{ik}(t) Q_{kj} \iff P'(t) = P(t) Q, \quad P(0) = I.$$

### 정의 2.2 — Backward equation

$$P'_{ij}(t) = \sum_k Q_{ik} P_{kj}(t) \iff P'(t) = Q P(t), \quad P(0) = I.$$

---

## 🔬 정리와 증명

### 정리 2.1 — Forward 방정식 유도

CTMC $P(t + h) = P(t) P(h)$로부터:
$$P(t + h) - P(t) = P(t)(P(h) - I).$$
$h \to 0$:
$$P'(t) = P(t) \lim_{h \to 0} \frac{P(h) - I}{h} = P(t) Q.$$

### 정리 2.2 — Backward 방정식 유도

$P(h + t) = P(h) P(t)$로부터:
$$P(t + h) - P(t) = (P(h) - I) P(t) \Rightarrow P'(t) = Q P(t).$$

*결론*: 두 유도는 같은 semigroup 성질의 **서로 다른 관점**. 유한상태에서는 $Q$와 $e^{tQ}$가 교환 → 같은 해.

### 정리 2.3 — 해의 존재·유일성 (유한상태)

유한상태 $Q$에 대해 Forward/Backward 방정식의 해 $P(t) = e^{tQ}$가 유일하게 존재.

*증명*. $F(t) = e^{tQ}$이 양쪽 ODE의 해 (by chain rule + $Q$와 $e^{tQ}$가 교환). Picard 반복의 contraction mapping → 유일. $\square$

### 정리 2.4 — Expectation evolution (Backward 응용)

$f : E \to \mathbb{R}$, $u(t, i) := \mathbb{E}_i[f(X_t)]$. 그러면
$$\frac{\partial u}{\partial t} = (Qu)(t, i) = \sum_j Q_{ij} u(t, j), \quad u(0, i) = f(i).$$

*증명*. $u(t, \cdot) = P(t) f$이고 $P'(t) = QP(t)$. $\square$

이는 **Feynman-Kac 공식의 이산 버전** — Backward equation을 확률적으로 풀이.

### 정리 2.5 — Distribution evolution (Forward 응용)

초기분포 $\mu_0$, $\mu_t = \mu_0 P(t)$는
$$\dot \mu_t = \mu_t Q, \quad \mu_0 = \text{given}.$$

이는 **Master equation** (물리학) — probability flow의 ODE.

### 정리 2.6 — $e^{tQ}$의 명시적 계산

(a) **Diagonalization**: $Q = V \Lambda V^{-1}$ → $e^{tQ} = V e^{t\Lambda} V^{-1}$ ($e^{t\Lambda}$ = 대각원소 $e^{t\lambda_i}$).

(b) **Padé 근사**: $e^A \approx R(A)$ (rational approximation); SciPy `scipy.linalg.expm`의 default.

(c) **Krylov subspace**: 큰 sparse $Q$에 대해 $e^{tQ} v$만 필요하면 Lanczos/Arnoldi로 approximate.

*실전*: SciPy `expm`이 Padé + scaling 사용, 대부분의 경우 adequate.

---

## 💻 NumPy 구현 검증

### 실험 1 — Forward/Backward 모두 같은 해

```python
import numpy as np
from scipy.linalg import expm

Q = np.array([
    [-2.0,  1.5,  0.5],
    [ 1.0, -3.0,  2.0],
    [ 0.5,  1.0, -1.5],
])

t = 1.5

# 직접 계산
P_direct = expm(t * Q)

# Forward ODE로 풀이
from scipy.integrate import solve_ivp
def fwd(t, y):
    P = y.reshape(3, 3)
    return (P @ Q).flatten()
sol_fwd = solve_ivp(fwd, [0, t], np.eye(3).flatten(), rtol=1e-9)
P_fwd = sol_fwd.y[:, -1].reshape(3, 3)

# Backward ODE
def bwd(t, y):
    P = y.reshape(3, 3)
    return (Q @ P).flatten()
sol_bwd = solve_ivp(bwd, [0, t], np.eye(3).flatten(), rtol=1e-9)
P_bwd = sol_bwd.y[:, -1].reshape(3, 3)

print('Max diff (direct vs forward):', np.abs(P_direct - P_fwd).max())
print('Max diff (direct vs backward):', np.abs(P_direct - P_bwd).max())
# 모두 ~0 → 같은 해
```

### 실험 2 — Expectation ODE $\dot u = Qu$

```python
f = np.array([1.0, 2.0, 3.0])   # f(0)=1, f(1)=2, f(2)=3

# 이론: u(t) = P(t) f
t_grid = np.linspace(0, 5, 50)
u_theory = np.array([expm(t * Q) @ f for t in t_grid])

# ODE로 풀이
def du_dt(t, u):
    return Q @ u
sol = solve_ivp(du_dt, [0, 5], f, t_eval=t_grid)

import matplotlib.pyplot as plt
for i in range(3):
    plt.plot(t_grid, u_theory[:, i], label=f'이론 u({i})')
    plt.plot(t_grid, sol.y[i], '--', label=f'ODE u({i})')
plt.legend(); plt.xlabel('t'); plt.ylabel(r'$\mathbb{E}_i[f(X_t)]$')
plt.title('Backward equation: expectation evolution'); plt.show()
```

### 실험 3 — Distribution evolution

```python
mu_0 = np.array([1.0, 0.0, 0.0])  # Dirac at state 0

def dmu_dt(t, mu):
    return mu @ Q

sol = solve_ivp(dmu_dt, [0, 5], mu_0, t_eval=t_grid)

# 이론 정상분포 (πQ = 0)
eigvals_Q, eigvecs_Q = np.linalg.eig(Q.T)
pi = np.real(eigvecs_Q[:, np.argmin(np.abs(eigvals_Q))])
pi = pi / pi.sum()
print(f'정상분포 π = {pi}')

for i in range(3):
    plt.plot(t_grid, sol.y[i], label=f'μ_t({i})')
    plt.axhline(pi[i], color=f'C{i}', linestyle='--', alpha=0.3)
plt.xlabel('t'); plt.ylabel(r'$\mu_t$')
plt.title('Forward equation: distribution evolution'); plt.legend(); plt.show()
# → μ_t가 π로 수렴
```

---

## 🔗 AI/ML 연결

**Continuous-Time RL Bellman Equation**  
$\dot V(x) + \mathcal{L} V(x) + r(x) = 0$ with terminal $V(x, T) = g(x)$. Here $\mathcal{L}$ is generator. Backward equation에 source term $r$ 추가 — **Hamilton-Jacobi-Bellman**의 CTMC 버전.

**Score-based Generative Model의 PDE 해석**  
Forward SDE $dX_t = f dt + g dB$의 density $p_t$가 Fokker-Planck (SDE Ch4-01) — 이는 Forward equation의 연속상태 generalization. **Generator** in SDE = 미분연산자.

**Compartmental ODE Models in Epidemiology**  
SIR 모델의 continuous deterministic limit이 Forward equation from CTMC. Deep learning이 rate parameters를 estimate → ODE 적분으로 prediction.

**Master Equation Neural Networks**  
Chemical reaction networks에서 $\dot \mu = \mu Q$를 NN으로 surrogate. Large state space 축소 (moment closure + NN) — systems biology의 AI 응용.

---

## ⚖️ 가정과 한계

**가정 — 시간 동질성**  
$Q$가 시간 독립. 비동질 $Q(t)$이면 $P(t) = \text{T-exp}(\int Q(s) ds)$ (time-ordered exponential) — 교환성 없어 행렬지수 단순하지 않음.

**한계 — 유한상태**  
가산 무한상태에서 Forward/Backward 방정식 양쪽이 성립할 조건은 $Q$의 **적절한 성장** ($-q_{ii}$의 upper bound). 생성된 체인에서 **explosion** 발생 가능 (유한 시간에 무한 jump).

**한계 — 연속상태**  
연속상태는 $Q$가 행렬이 아닌 **연산자** (Feller semigroup). Forward / Backward가 **PDE** (Fokker-Planck / Kolmogorov Backward, SDE Ch4에서).

---

## 📌 핵심 정리

| 방정식 | 수식 | 해석 |
|---|---|---|
| Forward | $P'(t) = P(t) Q$ | 미래로 한 step |
| Backward | $P'(t) = Q P(t)$ | 초기로 한 step |
| Distribution | $\dot \mu = \mu Q$ | Master equation |
| Expectation | $\dot u = Q u$ | Expectation evolution |
| 해 (유한상태) | $P(t) = e^{tQ}$ | 행렬지수 |

**한 줄 요약**: Forward equation은 **분포 진화**($\dot \mu = \mu Q$), Backward는 **기댓값 진화**($\dot u = Qu$). 두 방정식이 semigroup 성질의 서로 다른 관점이며, 유한상태에서 해는 $P(t) = e^{tQ}$로 통일된다.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. 2-state CTMC, $Q = \begin{pmatrix} -\alpha & \alpha \\ \beta & -\beta \end{pmatrix}$. Forward equation으로 $P(t)$를 명시적으로 유도하라.

<details>
<summary>해설</summary>

$P'(t) = P(t) Q$. 2×2 경우 ODE system 풀이.

$P_{00}'(t) = -\alpha P_{00}(t) + \beta P_{01}(t)$  
$P_{01}'(t) = \alpha P_{00}(t) - \beta P_{01}(t)$ (with $P_{00} + P_{01} = 1$)

$P_{01} = 1 - P_{00}$로 치환:
$P_{00}' = -\alpha P_{00} + \beta(1 - P_{00}) = \beta - (\alpha + \beta) P_{00}$.

해: $P_{00}(t) = \frac{\beta}{\alpha + \beta} + Ce^{-(\alpha+\beta)t}$. 초기조건 $P_{00}(0) = 1$:
$1 = \frac{\beta}{\alpha + \beta} + C \Rightarrow C = \frac{\alpha}{\alpha + \beta}$.

$P_{00}(t) = \frac{\beta + \alpha e^{-(\alpha+\beta)t}}{\alpha + \beta}$.

유사하게 다른 entry 계산. Ch4-01 문제 1의 $P(t)$와 일치.

</details>

**문제 2 (심화)**. Forward와 Backward equation이 해석적으로 달라 보이는데, 같은 답을 주는 이유를 "$Q$와 $e^{tQ}$의 교환성"으로 설명하라.

<details>
<summary>해설</summary>

**Forward**: $P'(t) = P(t) Q$. 해는 $P(t) = e^{tQ}$ (verify: $\frac{d}{dt} e^{tQ} = e^{tQ} Q = P(t) Q$ ✓).

**Backward**: $P'(t) = Q P(t)$. 해는 역시 $e^{tQ}$ (verify: $\frac{d}{dt} e^{tQ} = Q e^{tQ} = Q P(t)$ ✓).

왜 양쪽이 성립하는가: 행렬지수 $e^{tQ}$는 **자기 자신의 생성기와 교환**:
$$e^{tQ} Q = Q e^{tQ}$$
(series $\sum \frac{(tQ)^n}{n!}$에서 각 항이 $Q$와 교환).

따라서 "왼쪽에서 $Q$ 곱" 와 "오른쪽에서 $Q$ 곱"이 같은 결과 → 두 ODE가 같은 해.

**일반 $P, A$에서는 불가**: $P'(t) = P(t) A$와 $P'(t) = A P(t)$가 일반적으로 다른 해 (if $PA \neq AP$). Markov semigroup의 특수성 — 같은 생성기.

**주의**: **Inhomogeneous CTMC** ($Q = Q(t)$)에서는 $Q(s)$와 $Q(t)$가 일반적으로 교환 안 함 → Forward ≠ Backward. Time-ordered exponential $\mathcal{T}\exp(\int_s^t Q(u) du)$ 필요.

</details>

**문제 3 (AI 연결)**. Continuous-time RL에서 value function ODE $\partial_t V + \mathcal{L} V + r = 0$ (Backward)를 deep network $V_\theta$로 parameterize할 때, 어떤 loss와 training 전략을 쓸 수 있는가?

<details>
<summary>해설</summary>

**Loss 1 — Physics-informed NN (PINN)**:
$$\mathcal{L}_{\text{PDE}} = \mathbb{E}_{(x, t)}\left[\left(\partial_t V_\theta(x, t) + (\mathcal{L} V_\theta)(x, t) + r(x)\right)^2\right] + \lambda \mathbb{E}_x[(V_\theta(x, T) - g(x))^2].$$

- Backward equation을 residual로 minimize
- 특정 $(x, t)$ 샘플링 (uniform, adaptive)
- Autodiff로 $\partial_t, \mathcal{L}$ 계산

**Loss 2 — Monte Carlo with Feynman-Kac**:
$V(x, t) = \mathbb{E}_x[\int_t^T r(X_s) ds + g(X_T) | X_t = x]$ (Ch4-02 정리 2.4 확장).

- Sample trajectories from $(x, t)$
- $V_\theta$를 Monte Carlo estimate에 regression
- Bootstrapping with target network (DQN-style)

**Loss 3 — Martingale-based**:
$M_t = V_\theta(X_t, t) + \int_0^t r(X_s) ds$가 martingale이어야 (Ch5). 이 조건을 martingale residual로:
$$\mathcal{L}_{\text{martingale}} = \mathbb{E}[(M_{t+\Delta t} - M_t - \text{predicted change})^2].$$

**실전 Training 전략**:

1. **Curriculum learning**: 짧은 시간 horizon부터 시작, 점차 extend.

2. **Importance sampling**: 드문 state $(x, t)$에 more weight.

3. **Hamilton-Jacobi-Bellman 구조**: $V$가 특수 구조 (e.g., LQR에서 quadratic in $x$) → 명시적 parameterization.

4. **Benchmark on known solutions**: OU process, GBM 등 closed-form 있는 문제에서 첫 검증.

**기존 연구**:
- **Deep BSDE** (Han, Jentzen, E 2018): Backward SDE를 NN으로 푸는 대표
- **PINN for PDE**: Raissi et al. 2019
- **Continuous Deep Q-learning**: Gu et al. 2016

**연결**: Ch5 (Martingale), Ch6 (BM) 내용이 continuous-time RL/Finance AI의 수학적 토대를 제공. Feynman-Kac이 probability와 PDE의 다리 — 이 레포 뿐 아니라 SDE Deep Dive의 핵심 주제.

</details>

---

<div align="center">

◀ [01. 생성기(Generator)와 Q-matrix](./01-generator-q-matrix.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [03. 연속시간 정상분포와 Detailed Balance](./03-stationary-continuous.md)

</div>
