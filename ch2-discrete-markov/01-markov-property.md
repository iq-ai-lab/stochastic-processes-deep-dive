# 01. 마르코프 성질과 전이행렬

## 🎯 핵심 질문

- "**미래는 과거와 독립, 현재가 주어지면**"이라는 직관을 수학적으로 어떻게 formalize하는가?
- **강한 마르코프 성질**(정지시각에서도 마르코프)은 일반 마르코프 성질과 어떻게 다른가?
- 전이행렬 $P$는 왜 **확률행렬**(행합 1, 비음)이고, $n$-step 전이 $P^n$의 구조는 무엇인가?
- **Chapman-Kolmogorov 항등식** $P_{ij}^{(n+m)} = \sum_k P_{ik}^{(n)} P_{kj}^{(m)}$는 어디서 오고 왜 행렬 곱으로 간결해지는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**강화학습의 MDP**는 바로 마르코프 성질의 인스턴스: $P(S_{t+1} | S_t, A_t) = P(S_{t+1} | S_t, A_t, S_{t-1}, \ldots)$. 이 성질이 **Bellman 방정식**을 $V(s) = \max_a [r + \gamma \sum_{s'} P(s'|s, a) V(s')]$로 단순화하는 근거. 마르코프 성질 없이는 Q-learning·Policy gradient 모두 성립하지 않음.

**언어 모델의 n-gram과 Transformer**의 대조: n-gram은 $(n-1)$-order Markov이고, Transformer는 causal mask + positional encoding으로 **non-Markovian attention**을 구현. Mamba/S4 같은 State Space Model은 hidden state를 통해 마르코프성 복원.

**DDPM의 이산 forward**: $q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t} x_{t-1}, \beta_t I)$는 **1-step Markov**. 이 구조 덕분에 variational bound에서 $\log q(x_{1:T} | x_0) = \sum_t \log q(x_t | x_{t-1})$로 합으로 분해 — 학습 가능한 loss의 기반.

**Page Rank / 랜덤 워크 on 그래프**: 웹페이지 간 링크의 확률 행렬 → 정상분포 = PageRank. 이 레포의 Ch2-03에서 Perron-Frobenius로 유일성 증명.

---

## 📐 수학적 선행 조건

- [Ch1-04](../ch1-foundations/04-filtration.md): 필트레이션, 조건부 확률·기댓값
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): $\sigma$-대수, tower property
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): 행렬 곱, 고유값

---

## 📖 직관적 이해

### "기억 없는" 과정

**마르코프 성질**은 "한 걸음 전의 상태만 알면 과거 전체의 정보는 필요 없다"는 뜻. 즉 $X_n$을 아는 순간 $X_{n-1}, X_{n-2}, \ldots$의 값은 **더 이상 $X_{n+1}$ 예측에 유용하지 않다**.

> **비유**: 체스 게임. 현재 보드 상태만 알면 이전 수의 순서는 앞으로의 게임에 영향 없음(체스 규칙이 마르코프적인 예). 반면 포커는 비마르코프 — 상대의 **이전 베팅 패턴**이 미래 판단에 영향.

### 왜 1-step 전이행렬이 모든 것을 결정

1-step 전이확률 $P_{ij} = \mathbb{P}(X_{n+1} = j | X_n = i)$가 주어지면, $n$-step 전이
$$P_{ij}^{(n)} = \mathbb{P}(X_n = j | X_0 = i)$$
는 **행렬 곱** $P^n$의 $(i, j)$ 성분. 이것이 마르코프 성질의 핵심 결과 — 미래 예측이 1-step kernel로 환원.

### 강한 마르코프 성질

**일반 마르코프**: 결정론적 시각 $t$에서 재시작해도 마르코프.
**강한 마르코프**: **정지시각 $\tau$** (예: 첫 방문시각, 도박장 파산시각)에서 재시작해도 마르코프.

강한 마르코프는 자동이 아님 — 이산 MC와 연속 경로 마르코프(BM)에서는 성립하지만, 일반 연속시간 마르코프에서는 별도로 증명 필요.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — 마르코프 성질

이산시간 과정 $\{X_n\}_{n \in \mathbb{N}}$이 필트레이션 $\{\mathcal{F}_n\}$에 대해 **마르코프**라는 것은, 모든 $n \geq 0$과 Borel $A \subseteq E$에 대해
$$\mathbb{P}(X_{n+1} \in A | \mathcal{F}_n) = \mathbb{P}(X_{n+1} \in A | X_n) \quad \text{a.s.}$$

$\{\mathcal{F}_n\} = \{\mathcal{F}_n^X\}$ (자연 필트레이션)이면 동치 조건:
$$\mathbb{P}(X_{n+1} \in A | X_0, \ldots, X_n) = \mathbb{P}(X_{n+1} \in A | X_n).$$

### 정의 1.2 — 시간동질 마르코프 (Time-homogeneous)

$\mathbb{P}(X_{n+1} \in A | X_n = x)$가 $n$에 **의존하지 않으면** 시간동질. 이 경우 전이 kernel
$$K(x, A) = \mathbb{P}(X_1 \in A | X_0 = x)$$
만 주면 과정이 결정된다.

### 정의 1.3 — 이산상태 공간의 전이행렬

$E = \{1, 2, \ldots, N\}$ (유한) 또는 가산이면, 전이행렬 $P$는
$$P_{ij} := \mathbb{P}(X_{n+1} = j | X_n = i), \quad P \in \mathbb{R}^{N \times N}.$$

성질:
1. $P_{ij} \geq 0$ (확률)
2. $\sum_j P_{ij} = 1$ (각 행이 확률분포)

이를 **확률행렬**(stochastic matrix, row-stochastic)이라 한다.

### 정의 1.4 — $n$-step 전이

$P_{ij}^{(n)} := \mathbb{P}(X_n = j | X_0 = i).$

### 정의 1.5 — 초기분포와 joint 분포

초기분포 $\mu_0 = (\mu_0(1), \ldots, \mu_0(N))$ (row vector), 시각 $n$의 분포 $\mu_n = \mu_0 P^n$. 유한차원 결합분포:
$$\mathbb{P}(X_0 = i_0, X_1 = i_1, \ldots, X_n = i_n) = \mu_0(i_0) P_{i_0 i_1} P_{i_1 i_2} \cdots P_{i_{n-1} i_n}.$$

### 정의 1.6 — 정지시각 (Stopping time)

$\tau : \Omega \to \mathbb{N} \cup \{\infty\}$가 **정지시각**이라는 것은 각 $n$에 대해 $\{\tau \leq n\} \in \mathcal{F}_n$.

### 정의 1.7 — 강한 마르코프 성질

$\{X_n\}$이 **강한 마르코프** 성질을 가진다는 것은 임의 정지시각 $\tau$ (with $\tau < \infty$ a.s.)에 대해, $\{X_{\tau + n}\}_{n \geq 0}$의 분포가 $X_\tau$만 given으로 조건부화할 때 $\{\mathcal{F}_\tau\}$ 전체 조건부와 일치한다는 의미.

---

## 🔬 정리와 증명

### 정리 1.1 (Chapman-Kolmogorov 항등식)

시간동질 이산 마르코프 체인에 대해:
$$P_{ij}^{(n+m)} = \sum_k P_{ik}^{(n)} P_{kj}^{(m)}, \quad \forall n, m \geq 0.$$

행렬 형태: $P^{(n+m)} = P^n P^m$ (특히 $P^{(n)} = P^n$).

*증명.* 마르코프 성질 + tower property 적용:
$$P_{ij}^{(n+m)} = \mathbb{P}(X_{n+m} = j | X_0 = i) = \sum_k \mathbb{P}(X_{n+m} = j, X_n = k | X_0 = i)$$
$$= \sum_k \mathbb{P}(X_{n+m} = j | X_n = k, X_0 = i) \mathbb{P}(X_n = k | X_0 = i).$$
마르코프 성질로 $\mathbb{P}(X_{n+m} = j | X_n = k, X_0 = i) = \mathbb{P}(X_{n+m} = j | X_n = k)$. 시간동질로 $= P_{kj}^{(m)}$. 나머지 factor는 $P_{ik}^{(n)}$. 따라서 결과.  
특히 $P^{(1)} = P$이고 귀납으로 $P^{(n)} = P^n$. $\square$

### 정리 1.2 (마르코프 성질의 강화 — 다중 스텝)

$\{X_n\}$이 마르코프이면, 임의 $0 \leq n_1 < \cdots < n_k < m$에 대해
$$\mathbb{P}(X_m \in A | X_{n_1}, \ldots, X_{n_k}) = \mathbb{P}(X_m \in A | X_{n_k}).$$

*증명.* $k$에 대한 귀납과 tower property. $\square$

### 정리 1.3 (초기분포에서의 marginal)

$\mu_n$ = 시각 $n$의 marginal 분포 (row vector). 그러면 $\mu_n = \mu_0 P^n$.

*증명.* $\mu_n(j) = \mathbb{P}(X_n = j) = \sum_i \mathbb{P}(X_0 = i) P_{ij}^{(n)} = (\mu_0 P^n)(j)$. $\square$

### 정리 1.4 (이산 마르코프 체인의 강한 마르코프 성질)

이산시간 이산상태 마르코프 체인은 **자동으로 강한 마르코프 성질**을 가진다.

*증명.* $\tau$가 정지시각, $n$이 fixed일 때
$$\{X_{\tau + n} \in A, \tau = k\} = \{X_{k+n} \in A, \tau = k\}.$$
$\{\tau = k\} \in \mathcal{F}_k$이므로 마르코프 성질에서
$$\mathbb{P}(X_{k+n} \in A | \mathcal{F}_k) = \mathbb{P}(X_{k+n} \in A | X_k).$$
$X_\tau = X_k$ on $\{\tau = k\}$이고, countable decomposition으로 합치면
$$\mathbb{P}(X_{\tau + n} \in A | \mathcal{F}_\tau) = \mathbb{P}(X_{\tau + n} \in A | X_\tau).$$
$\square$

> **중요**: 연속시간에서는 강한 마르코프가 자동이 아니며, 별도 정칙성 조건(우연속 생성기 등) 필요.

### 정리 1.5 (확률행렬의 스펙트럴 기초)

$P$가 확률행렬이면:
1. $\mathbf{1} = (1, \ldots, 1)^T$이 우고유벡터: $P \mathbf{1} = \mathbf{1}$, 따라서 $1$은 고유값.
2. 모든 고유값 $\lambda$는 $|\lambda| \leq 1$.

*증명.*  
(1) $\sum_j P_{ij} = 1$ ⇒ $P \mathbf{1} = \mathbf{1}$.  
(2) $P v = \lambda v$, $v \neq 0$. 최대 성분 $|v_{i^*}| = \max_i |v_i|$를 선택:
$$|\lambda v_{i^*}| = |P v|_{i^*} = |\sum_j P_{i^* j} v_j| \leq \sum_j P_{i^* j} |v_j| \leq |v_{i^*}|.$$
따라서 $|\lambda| \leq 1$. $\square$

### 정리 1.6 (마르코프 체인의 functional of path)

$\{X_n\}$이 마르코프이고 $f, g$가 각 측정가능 함수일 때:
$$\mathbb{E}[f(X_0, \ldots, X_n) g(X_{n+1}, X_{n+2}, \ldots) | \mathcal{F}_n] = f(X_0, \ldots, X_n) \mathbb{E}[g(X_{n+1}, \ldots) | X_n].$$

*증명 스케치.* 첫 항은 $\mathcal{F}_n$-measurable. 두 번째는 $\sigma(X_{n+1}, \ldots)$-measurable, 마르코프 성질로 $X_n$만 조건부화 충분. $\square$

이것이 **Bellman 방정식의 근간** — "과거는 현재에 요약된다".

---

## 💻 NumPy 구현 검증

### 실험 1 — Chapman-Kolmogorov 직접 확인

```python
import numpy as np
rng = np.random.default_rng(0)

# 3-state 전이행렬
P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5],
])

# Chapman-Kolmogorov: P^(5+3) = P^5 @ P^3 = P^8
P_5 = np.linalg.matrix_power(P, 5)
P_3 = np.linalg.matrix_power(P, 3)
P_8_direct = np.linalg.matrix_power(P, 8)
P_8_cg = P_5 @ P_3

print('CG 검증: max |P^8 - P^5 @ P^3| = {:.2e}'.format(
    np.abs(P_8_direct - P_8_cg).max()))
# → 기계 정밀도 내 0

# 확률행렬 성질
print('row sums of P^n: {}'.format(P_8_direct.sum(axis=1)))
# → 모두 1 (확률 보존)
```

### 실험 2 — Monte Carlo로 P^n 실측 vs 이론

```python
# Monte Carlo로 P(X_5 = 2 | X_0 = 0) 추정
n_samples = 500_000
n_steps = 5
end_state = 2
start_state = 0

counts = 0
for _ in range(n_samples):
    state = start_state
    for _ in range(n_steps):
        state = rng.choice(3, p=P[state])
    if state == end_state:
        counts += 1

P5 = np.linalg.matrix_power(P, n_steps)
print(f'MC 추정: P(X_5=2|X_0=0) = {counts/n_samples:.4f}')
print(f'이론:    P^5[0,2]        = {P5[0, 2]:.4f}')
# → 일치
```

### 실험 3 — 마르코프 성질 vs 비마르코프 구별

```python
# 비마르코프: X_n이 (X_{n-1}, X_{n-2}) 둘 다에 의존
# 예: X_n = X_{n-1} + X_{n-2} mod 3 + noise (2-order Markov)

N = 100_000
state = [0, 0]
path = list(state)
for _ in range(N):
    new = (state[-1] + state[-2] + rng.choice(3, p=[0.6, 0.3, 0.1])) % 3
    state = [state[-1], new]
    path.append(new)
path = np.array(path)

# 추정 P(X_n | X_{n-1}) — 만약 1-order Markov면 conditional이 X_{n-2}에 무관
# Conditional on X_{n-1} = 1의 다음 분포를 X_{n-2}별로 분할
for prev_prev in range(3):
    mask = (path[:-2] == prev_prev) & (path[1:-1] == 1)
    if mask.sum() > 100:
        next_dist = np.bincount(path[2:][mask], minlength=3) / mask.sum()
        print(f'P(X_n | X_{{n-1}}=1, X_{{n-2}}={prev_prev}) = {next_dist}')
# → X_{n-2}에 따라 분포가 크게 달라짐 → 비마르코프
```

---

## 🔗 AI/ML 연결

**MDP와 Q-learning**  
Q 함수의 Bellman 방정식
$$Q(s, a) = r(s, a) + \gamma \sum_{s'} P(s' | s, a) \max_{a'} Q(s', a')$$
는 마르코프 성질이 있어야 **단일 step kernel로 미래 보상 요약 가능**. 비마르코프 환경에서는 상태를 확장(history-dependent state)해야 함 — RNN 기반 RL 에이전트의 근거.

**HMM의 hidden state 마르코프성**  
HMM은 hidden $Z_t$가 마르코프, observation $O_t$는 $Z_t$만 조건부 독립. Forward-backward algorithm이 $O(T \cdot |Z|^2)$ 복잡도로 작동하는 것은 이 마르코프 구조 덕. 비마르코프이면 $O(T \cdot |Z|^T)$로 폭발.

**DDPM variational bound 분해**  
$$\log p(x_0) \geq \mathbb{E}_q\left[\log \frac{p(x_{0:T})}{q(x_{1:T}|x_0)}\right] = \sum_t \mathcal{L}_t$$
이 per-step 분해가 가능한 것은 $q$와 $p$ 둘 다 마르코프 구조이기 때문. 비마르코프이면 분해 불가 → 학습 목적함수 정의 불가.

**Transformer vs Mamba 재방문**  
- Transformer: $y_t = f(x_{1:t})$ **비마르코프**, 모든 과거에 의존
- Mamba: $h_{t+1} = Ah_t + Bx_t$, $y_t = Ch_t$ **마르코프** in hidden state
- 이 구조적 차이가 추론 비용 $O(T^2)$ vs $O(T)$의 근원.

---

## ⚖️ 가정과 한계

**가정 — 시간동질**  
대부분의 이론이 시간동질에 의존. 비동질(time-inhomogeneous) MC는 전이행렬 $P_n$이 $n$마다 달라 — 이 경우 정상분포 개념이 약해지고, DDPM forward가 대표적 비동질 예(noise schedule이 $t$-의존).

**한계 — 유한 vs 가산상태**  
가산상태에서는 재귀성이 비자명(Ch2-02에서 다룸). 유한상태는 모든 상태가 재귀(기약이면). 연속상태는 kernel 이론이 필요(Meyn & Tweedie).

**한계 — 마르코프 성질이 자연에 맞지 않는 경우**  
금융 변동성, 뉴스 확산, 인간 행동 패턴 등은 장기 의존성(long-range dependency)을 가짐. 강제로 마르코프로 modeling하면 예측력 상실. 이 경우 확장 state (GARCH, Hawkes) 또는 fractional process 사용.

---

## 📌 핵심 정리

| 개념 | 요약 |
|---|---|
| 마르코프 성질 | $\mathbb{P}(X_{n+1} \in A | \mathcal{F}_n) = \mathbb{P}(X_{n+1} \in A | X_n)$ |
| 전이행렬 | $P_{ij} = \mathbb{P}(X_{n+1} = j | X_n = i)$, 행합 1, 비음 |
| CK 항등식 | $P^{(n+m)} = P^n P^m$ (시간동질) |
| Marginal | $\mu_n = \mu_0 P^n$ |
| 강한 마르코프 | 정지시각에서도 재시작 마르코프 — 이산 MC는 자동 |
| 확률행렬 고유값 | $1$이 고유값, 모두 $|\lambda| \leq 1$ |

**한 줄 요약**: "미래는 현재가 주어지면 과거와 독립" — 이 성질이 미래 예측을 1-step kernel(전이행렬 $P$)로 환원하고, $n$-step은 $P^n$으로 **행렬 곱**으로 계산 가능하게 만든다.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. $P = \begin{pmatrix} 0.6 & 0.4 \\ 0.3 & 0.7 \end{pmatrix}$ 일 때 $P^2$를 계산하고, $\mathbb{P}(X_2 = 1 | X_0 = 0)$을 구하라.

<details>
<summary>해설</summary>

$P^2 = P \cdot P$:
- $(P^2)_{00} = 0.6 \cdot 0.6 + 0.4 \cdot 0.3 = 0.36 + 0.12 = 0.48$
- $(P^2)_{01} = 0.6 \cdot 0.4 + 0.4 \cdot 0.7 = 0.24 + 0.28 = 0.52$
- $(P^2)_{10} = 0.3 \cdot 0.6 + 0.7 \cdot 0.3 = 0.18 + 0.21 = 0.39$
- $(P^2)_{11} = 0.3 \cdot 0.4 + 0.7 \cdot 0.7 = 0.12 + 0.49 = 0.61$

$\mathbb{P}(X_2 = 1 | X_0 = 0) = (P^2)_{01} = 0.52$.

**검증**: 행합 0.48 + 0.52 = 1 ✓

</details>

**문제 2 (심화)**. 마르코프 체인 $\{X_n\}$에 대해 $Y_n = (X_n, X_{n-1})$이 **마르코프**임을 보여라. 또한 이를 이용해 "$k$-order Markov는 1-order Markov로 환원 가능"임을 논하라.

<details>
<summary>해설</summary>

**마르코프성 증명**:
$\mathbb{P}(Y_{n+1} | Y_n, Y_{n-1}, \ldots) = \mathbb{P}((X_{n+1}, X_n) | (X_n, X_{n-1}), \ldots)$. $X_n$은 이미 $Y_n$ 안에 있으므로 deterministic. $X_{n+1}$의 조건부 분포는 $\{X_n\}$의 마르코프 성질로 $X_n$만 조건부화하면 충분:
$$\mathbb{P}(X_{n+1} | X_n, X_{n-1}, \ldots) = \mathbb{P}(X_{n+1} | X_n).$$
이는 $Y_n = (X_n, X_{n-1})$만으로 결정. 따라서 $Y$는 마르코프.

**$k$-order Markov → 1-order**: $k$-order Markov는 $\mathbb{P}(X_{n+1} | X_n, \ldots, X_{n-k+1}) = \mathbb{P}(X_{n+1} | X_n, \ldots, X_{n-k+1})$ (즉 $k$개의 과거에만 의존). 확장 상태 $Y_n = (X_n, X_{n-1}, \ldots, X_{n-k+1})$로 정의하면 $Y$는 1-order Markov.

**함의**: 어떤 과거 의존성도 state를 확장하면 마르코프화 가능. 이것이 RL에서 "state augmentation"(과거 $k$ 스텝을 state에 포함), NLP에서 n-gram → LSTM hidden state로의 전환 근거.

**대가**: state 공간이 $k$배, $|E|^k$로 기하급수적 증가. 따라서 무한 메모리는 현실적으로 latent/hidden state로 압축(HMM, RNN)해야 함.

</details>

**문제 3 (AI 연결)**. DDPM forward process $q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$는 1-step Markov이지만 **시간비동질** ($\beta_t$가 $t$에 의존). 이 비동질성이 (1) CK 항등식, (2) 학습 목적함수 구성, (3) 추론 샘플링에 각각 어떤 영향을 미치는가?

<details>
<summary>해설</summary>

**(1) CK 항등식**: 시간동질이면 $P^{(n+m)} = P^n P^m$이 단일 $P$의 거듭제곱. 비동질이면 $P^{(s,t)} P^{(t,u)} = P^{(s,u)}$로 일반화(transition kernel의 곱셈적 구조), 각 시각 다른 kernel이므로 곱셈 순서 중요. DDPM은 $q(x_t | x_0) = \mathcal{N}(\sqrt{\bar\alpha_t} x_0, (1 - \bar\alpha_t) I)$로 **해석적 해**를 얻는데, 이는 단순 CK 곱셈의 특별 결과 — 각 $q$가 가우시안이고 곱이 가우시안이 되므로.

**(2) 학습 목적함수**:
$$\mathcal{L}_{\text{VLB}} = \mathbb{E}_q\left[\sum_{t=1}^T D_{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1} | x_t))\right].$$
각 $t$마다 $\beta_t$ 다르지만, 1-step Markov 덕에 per-step KL로 분해. 비동질이지만 1-step Markov가 유지되는 한 이 분해 유효. 비동질성은 각 step의 scale이 다를 뿐이고, 공통 network $\epsilon_\theta(x_t, t)$가 $t$를 입력으로 받아 처리.

**(3) 추론 샘플링**:
$$x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z, \quad z \sim \mathcal{N}(0, I).$$
$t$에 따라 $\mu_\theta, \sigma_t$가 다르므로 $t = T, T-1, \ldots, 1$로 순차 샘플링. 비동질 Markov chain의 sequential 샘플링 — 1000 step 걸린다. **이산 → 연속 확장**(Score-SDE)이 이를 ODE/SDE 적분으로 환원하여 더 적은 step으로 가속 가능(DDIM, DPM-Solver).

**종합**: 비동질성은 "각 시각 다른 noise schedule"을 허용하여 표현력을 높이지만(learning용이성 vs sample quality trade-off), CK·KL 분해·Markov 샘플링의 수학적 구조는 온전히 보존.

</details>

---

<div align="center">

◀ [Ch1-05. 확률과정의 분류 지도](../ch1-foundations/05-classification.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [02. 상태의 분류 — 재귀·일시·주기](./02-state-classification.md)

</div>
