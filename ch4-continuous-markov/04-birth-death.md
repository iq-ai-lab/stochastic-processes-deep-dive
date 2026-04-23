# 04. Birth-Death 과정

## 🎯 핵심 질문

- **Birth-Death 과정**의 정상분포 공식 $\pi_n = \pi_0 \prod_{k=1}^n \frac{\lambda_{k-1}}{\mu_k}$는 어떻게 detailed balance로부터 유도되는가?
- M/M/1 큐, M/M/c 큐가 왜 birth-death process의 특수 경우인가?
- 언제 정상분포가 **존재**하는가 — $\sum \prod \lambda/\mu$의 수렴 조건은?
- Moran model (population genetics), Ising 1D dynamics 등의 응용은 어떻게 birth-death로 환원되는가?

---

## 🔍 왜 birth-death가 AI에서 중요한가

**Population / Evolutionary RL**: GA(Genetic Algorithm)와 같은 evolutionary computation의 population size dynamics를 birth-death로 모델링. Mutation/selection rates.

**LLM serving 큐잉**: M/M/c는 birth-death — $\lambda_n = \lambda$ (상수), $\mu_n = \min(n, c)\mu$ (state-dependent). Capacity planning의 분석 엔진.

**Chemical Reaction Networks**: Cell biology, drug binding — one-dimensional reaction networks가 birth-death의 확장.

**Neural architecture search / growing nets**: 네트워크 layer/neuron의 추가/삭제를 birth-death로 모델링. Theoretical analysis.

---

## 📐 수학적 선행 조건

- [Ch4-01, Ch4-03](./01-generator-q-matrix.md): Q-matrix, detailed balance
- [Ch3-04](../ch3-poisson/04-queueing-little.md): M/M/1 queuing

---

## 📖 직관적 이해

### 구조

**Birth-Death process**: 상태 공간 $\{0, 1, 2, \ldots\}$ (가산), 각 상태 $n$에서 $n+1$ (birth, rate $\lambda_n$) 또는 $n-1$ (death, rate $\mu_n$)로만 이동. 즉
$$Q_{n, n+1} = \lambda_n, \quad Q_{n, n-1} = \mu_n, \quad Q_{nn} = -(\lambda_n + \mu_n).$$

(Boundary: $\mu_0 = 0$)

### Detailed balance로 해석해

1D 구조 → 항상 reversible (3-state cyclic 같은 loop가 없음).

$\pi_n \lambda_n = \pi_{n+1} \mu_{n+1} \Rightarrow \pi_{n+1} = \pi_n \lambda_n / \mu_{n+1}$.

반복 적용: $\pi_n = \pi_0 \prod_{k=1}^n \frac{\lambda_{k-1}}{\mu_k}$.

### 정상분포 존재 조건

$\pi$가 확률분포이려면 $\sum_n \pi_n < \infty$:
$$1 + \sum_{n \geq 1} \prod_{k=1}^n \frac{\lambda_{k-1}}{\mu_k} < \infty.$$

이 **무한 급수의 수렴**이 정상분포 존재의 충분필요 조건. 가산상태이므로 유한상태와 달리 자명하지 않음.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Birth-Death Process

CTMC $\{X_t\}$ on $\{0, 1, 2, \ldots\}$ with
- Birth rates $\{\lambda_n\}_{n \geq 0}$, $\lambda_n > 0$
- Death rates $\{\mu_n\}_{n \geq 1}$, $\mu_n > 0$, $\mu_0 = 0$

Generator:
$$Q_{nm} = \begin{cases}
\lambda_n & m = n+1 \\
\mu_n & m = n-1 \; (n \geq 1) \\
-(\lambda_n + \mu_n) & m = n \\
0 & \text{else}
\end{cases}$$

---

## 🔬 정리와 증명

### 정리 4.1 — Detailed Balance와 정상분포 공식

Birth-Death process의 정상분포(존재하면):
$$\pi_n = \pi_0 \prod_{k=1}^n \frac{\lambda_{k-1}}{\mu_k}, \quad \pi_0 = \left(1 + \sum_{n \geq 1} \prod_{k=1}^n \frac{\lambda_{k-1}}{\mu_k}\right)^{-1}.$$

정상분포 존재 ⇔ 위 분모 급수 수렴.

*증명 (detailed balance)*.

Pair $(n, n+1)$의 DB: $\pi_n \lambda_n = \pi_{n+1} \mu_{n+1}$. 반복으로 $\pi_n$ 공식. 정규화로 $\pi_0$.

**왜 DB가 이 구조에서 자동인가**: 1D nearest-neighbor (상태 $n$에서 $n \pm 1$로만) → cycle 없음 → 정상분포가 있으면 DB 자동.

**정확한 검증**: 정의한 $\pi_n$이 $\pi Q = 0$을 만족. $(\pi Q)_n = \pi_{n-1} \lambda_{n-1} - \pi_n (\lambda_n + \mu_n) + \pi_{n+1} \mu_{n+1}$. DB로 $\pi_{n-1} \lambda_{n-1} = \pi_n \mu_n$, $\pi_n \lambda_n = \pi_{n+1} \mu_{n+1}$ → $(\pi Q)_n = \pi_n \mu_n - \pi_n \lambda_n - \pi_n \mu_n + \pi_n \lambda_n = 0$. $\square$

### 정리 4.2 — M/M/1의 특수 경우

$\lambda_n = \lambda$ (상수), $\mu_n = \mu$ ($n \geq 1$). 그러면
$$\pi_n = \pi_0 (\lambda/\mu)^n = \pi_0 \rho^n.$$

$\sum \rho^n$ 수렴 ⇔ $\rho < 1$. $\pi_0 = 1 - \rho$ → Ch3-04의 M/M/1 결과.

### 정리 4.3 — M/M/c의 특수 경우

$\lambda_n = \lambda$, $\mu_n = \min(n, c)\mu$:
$$\pi_n = \begin{cases}
\pi_0 \frac{(c\rho)^n}{n!} & 0 \leq n < c \\
\pi_0 \frac{c^c \rho^n}{c!} & n \geq c,
\end{cases}$$
$\rho = \lambda/(c\mu)$. (Ch3-04 문제 2와 일치.)

### 정리 4.4 — 가산상태에서 재귀성 판정

Birth-Death 과정에서 상태 0의 재귀성:
- **재귀** ⇔ $\sum_{n \geq 0} \frac{\mu_1 \mu_2 \cdots \mu_n}{\lambda_1 \lambda_2 \cdots \lambda_n} = \infty$
- **양재귀** ⇔ 위 + $\sum_{n \geq 1} \frac{\lambda_0 \lambda_1 \cdots \lambda_{n-1}}{\mu_1 \mu_2 \cdots \mu_n} < \infty$

(Chung-Fuchs 기반 판정)

*응용*: 1D simple BD ($\lambda_n = \lambda, \mu_n = \mu$)에서 $\rho < 1$이면 양재귀 → 정상분포 존재; $\rho > 1$이면 일시 → 폭발; $\rho = 1$ 영재귀.

### 정리 4.5 — Moran Model (Population genetics)

N개 개체 population, 각각 allele $A$ or $a$. $X_t$ = type $A$ 개체 수.

- Birth $A$: 확률 $X(N-X)/N^2$ — 한 $A$가 복제되어 $a$를 대체
- Death $A$: $\lambda_X = \mu_X = X(N-X)/N^2$

→ symmetric BD, $X_t$가 absorbing states $\{0, N\}$로 수렴. Fixation probability의 분석에 활용.

---

## 💻 NumPy 구현 검증

### 실험 1 — M/M/1 정상분포 공식

```python
import numpy as np
import matplotlib.pyplot as plt

lam, mu = 0.8, 1.0
rho = lam / mu

# 정상분포 공식: π_n = (1-ρ) ρ^n
n_range = np.arange(20)
pi_theory = (1 - rho) * rho**n_range

# 실측: event-driven 시뮬레이션 (Ch3-04 실험 재사용)
rng = np.random.default_rng(0)
T = 5000.0
X = 0; t = 0.0
times = [0.0]; states = [0]
while t < T:
    birth_rate = lam
    death_rate = mu if X > 0 else 0
    total = birth_rate + death_rate
    dwell = rng.exponential(1/total)
    t += dwell
    if rng.random() < birth_rate / total:
        X += 1
    else:
        X -= 1
    times.append(t); states.append(X)

times = np.array(times); states = np.array(states)
durations = np.diff(times)
pi_emp = np.zeros(len(n_range))
for n in n_range:
    pi_emp[n] = durations[states[:-1] == n].sum() / times[-1]

plt.plot(n_range, pi_theory, 'o-', label='이론')
plt.plot(n_range, pi_emp, 's--', label='실측')
plt.yscale('log'); plt.xlabel('n'); plt.ylabel(r'$\pi_n$')
plt.legend(); plt.title(f'M/M/1 정상분포 (ρ={rho})')
plt.grid(True, which='both', alpha=0.3); plt.show()
```

### 실험 2 — M/M/c (multi-server)

```python
from scipy.special import factorial

def mmc_stationary(lam, mu, c, n_max=30):
    rho = lam / (c * mu)
    # π_0
    sum_part1 = sum((c*rho)**n / factorial(n) for n in range(c))
    sum_part2 = (c*rho)**c / (factorial(c) * (1 - rho))
    pi_0 = 1 / (sum_part1 + sum_part2)
    
    pi = np.zeros(n_max)
    pi[0] = pi_0
    for n in range(1, n_max):
        if n < c:
            pi[n] = pi_0 * (c*rho)**n / factorial(n)
        else:
            pi[n] = pi_0 * (c**c * rho**n) / factorial(c)
    return pi

# 3-server example
pi_c3 = mmc_stationary(lam=2.4, mu=1.0, c=3)
print(f'M/M/3 with λ=2.4, μ=1: π_0...π_9 = {pi_c3[:10]}')
print(f'λ/(cμ) = 0.8, 3-서버로 utilization 감소')
```

### 실험 3 — Moran Model simulation

```python
# N = 50 개체, X_0 = 25 (50-50 시작)
N = 50
X = 25
rng = np.random.default_rng(42)
T = 5000

trajectory = [X]
t_path = [0.0]
t = 0.0
while 0 < X < N and t < T:
    rate = X * (N - X) / N**2
    total_rate = 2 * rate   # birth + death
    dwell = rng.exponential(1/total_rate)
    t += dwell
    if rng.random() < 0.5:
        X += 1
    else:
        X -= 1
    trajectory.append(X)
    t_path.append(t)

plt.plot(t_path, trajectory)
plt.xlabel('t'); plt.ylabel('X_t (개체 수 of A)')
plt.title(f'Moran Model: fixation at X={X} (N={N})')
plt.grid(True, alpha=0.3); plt.show()
# → 결국 0 or N에 도달 (fixation)
# Fixation probability: P(X→N | X_0=x) = x/N (symmetric case)
```

---

## 🔗 AI/ML 연결

**Genetic Algorithm의 population dynamics**  
GA step = birth (새 individual 생성) + death (selection에서 제거). Population size를 birth-death로 모델링, 수렴 속도·diversity 분석.

**M/M/c + deep RL**  
서버 pool 동적 관리 (k8s auto-scaling) → state $X_t = $ active servers, birth = scale up, death = scale down. RL agent가 $\lambda, \mu$를 조절하여 SLA + cost 최적화.

**Epidemic Models**  
SIR의 discrete stochastic version: $S \to I$ (infection, rate $\beta SI/N$), $I \to R$ (recovery, rate $\gamma I$). Infection+recovery를 birth-death로 embedding.

**Chemical master equation**  
$A + B \rightleftharpoons C$ 와 같은 1D 가역 반응을 birth-death로. Gillespie algorithm (exact stochastic simulation)이 jump chain + holding time 구조.

---

## ⚖️ 가정과 한계

**한계 — 1D 이동만 허용**  
"2 step jumps" ($n \to n+2$)나 다중 상태 변화는 **jump 과정**으로 확장 필요. Birth-Death는 가장 단순 구조.

**한계 — 가산무한상태에서 정상분포 없을 수 있음**  
$\sum_n \prod \lambda/\mu$ 발산 → 정상분포 없음 → 체인이 drift to infinity. Linear birth-death ($\lambda = \lambda, \mu = \mu$)로 $\rho \geq 1$이면 해당.

**한계 — Detailed balance 단순성이 일반 CTMC로 일반화 어려움**  
2D random walk 등은 BD 아님 — cycle 가능, reversibility가 자명하지 않음.

---

## 📌 핵심 정리

| 결과 | 수식 |
|---|---|
| BD 구조 | $Q_{n, n+1} = \lambda_n, Q_{n, n-1} = \mu_n$ |
| 정상분포 | $\pi_n = \pi_0 \prod \lambda_{k-1}/\mu_k$ |
| 자동 reversible | 1D 구조 → DB 자동 |
| 존재 조건 | $\sum_{n \geq 1} \prod \lambda/\mu < \infty$ |
| M/M/1 | $\pi_n = (1-\rho)\rho^n$ |
| M/M/c | Erlang-C 형태 |

**한 줄 요약**: Birth-Death 과정은 1D nearest-neighbor CTMC로, 항상 reversible(DB 자동)이고 정상분포 공식이 단순 곱 형태. M/M/1, M/M/c, Moran 모델 등 다양한 응용의 통일된 프레임워크.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. M/M/∞ 큐(무한 서버): $\lambda_n = \lambda$, $\mu_n = n\mu$. 정상분포를 구하라.

<details>
<summary>해설</summary>

공식 적용:
$\pi_n = \pi_0 \prod_{k=1}^n \frac{\lambda}{k\mu} = \pi_0 \frac{(\lambda/\mu)^n}{n!}$.

정규화: $\pi_0 \sum \frac{(\lambda/\mu)^n}{n!} = \pi_0 e^{\lambda/\mu} = 1$.

$\pi_n = \frac{(\lambda/\mu)^n}{n!} e^{-\lambda/\mu}$.

**이것은 Poisson($\lambda/\mu$) 분포!**

**해석**: 무한 서버이므로 모든 고객이 즉시 서비스 받음, 체류시간 $\text{Exp}(\mu)$. $\lambda$ rate arrival + Exp duration → Poisson steady-state count (Little: $L = \lambda \cdot 1/\mu$).

**모든 $\lambda, \mu$에 대해 정상분포 존재** ($\sum < \infty$). 이것이 M/M/∞가 항상 안정한 이유 — 무한 capacity.

</details>

**문제 2 (심화)**. Birth-Death의 fixation probability: 상태 $\{0, N\}$이 absorbing일 때, $P_x(X \to N)$을 유도하라.

<details>
<summary>해설</summary>

Let $h(x) = \mathbb{P}_x(X \to N)$. Harmonic function: $(Qh)(x) = 0$ for $0 < x < N$:
$$-(\lambda_x + \mu_x) h(x) + \lambda_x h(x+1) + \mu_x h(x-1) = 0.$$

Boundary: $h(0) = 0, h(N) = 1$.

해: Let $\gamma_k = \mu_k/\lambda_k$. 그러면
$$h(x) = \frac{\sum_{k=0}^{x-1} \prod_{j=1}^k \gamma_j}{\sum_{k=0}^{N-1} \prod_{j=1}^k \gamma_j}.$$

(by $h(x+1) - h(x) = \gamma_x (h(x) - h(x-1))$ and telescoping.)

**특수: symmetric** ($\lambda_x = \mu_x$) — $\gamma \equiv 1$, $h(x) = x/N$ (linear).

**Moran 모델에서**: fixation prob = $x/N$ — 무편향 유전자 drift의 특징.

**응용**: Gambler's ruin (도박자 파산, Ch5 optional stopping), population genetics fixation.

</details>

**문제 3 (AI 연결)**. LLM serving의 M/M/c 큐에서 $c$를 2배로 늘렸을 때 latency의 개선은 2배가 아니고 비선형이다. Erlang-C 공식으로 이를 설명하라.

<details>
<summary>해설</summary>

**M/M/c 평균 대기시간**:
$W_q = \frac{C(c, c\rho)}{c\mu(1 - \rho)}$, $\rho = \lambda/(c\mu)$, $C$ = Erlang-C.

**$c$ 2배 증가 효과**:
1. $\rho$가 절반 ($\lambda$ 고정 시): $\rho \to \rho/2$.
2. Erlang-C $C(c, c\rho)$가 급격히 감소 (모든 서버가 바쁠 확률).

**구체 예**: $\lambda = 3, \mu = 1$:
- $c = 4$: $\rho = 0.75$, $C(4, 3) \approx 0.51$, $W_q \approx 0.51/(4 \cdot 0.25) \approx 0.51$.
- $c = 8$: $\rho = 0.375$, $C(8, 3) \approx 0.012$, $W_q \approx 0.012/(8 \cdot 0.625) \approx 0.0024$.

**약 200배 개선** (2배가 아님!). 이는:
1. Utilization $\rho$ 감소로 system이 훨씬 "여유"
2. $C(c, c\rho)$가 sub-exponentially 감소 — **capacity buffer의 non-linear benefit**

**실전 Implication**:
- **Over-provisioning의 가치**: p99 latency 개선이 cost 증가 대비 매우 큼
- **Auto-scaling threshold**: $\rho > 0.7$에서 scale up — latency 폭발 전
- **Peak vs average**: 피크 기준 capacity 필요, 평균 기준은 부족

**AI serving 적용**:
- GPU pool size 결정
- Distributed inference across nodes
- Batch size 선택 (batch ↔ effective $\mu$ 증가)

**연결**: Ch3-04 (M/M/1 기본) → Ch4-04 (birth-death generalization) → 실전 capacity planning. 이론이 engineering 결정에 직접 영향.

</details>

---

<div align="center">

◀ [03. 연속시간 정상분포와 Detailed Balance](./03-stationary-continuous.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [Ch5-01. 마팅게일의 정의](../ch5-martingale/01-martingale-definition.md)

</div>
