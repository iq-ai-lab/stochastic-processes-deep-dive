# 02. 상태의 분류 — 재귀·일시·주기

## 🎯 핵심 질문

- 상태 $i$가 **재귀**(recurrent)인지 **일시**(transient)인지를 어떻게 판별하는가 — $\sum_n P_{ii}^{(n)}$이 왜 결정적 지표인가?
- **양재귀**와 **영재귀**는 어떻게 다르고, 유한상태에서는 왜 구별이 없는가?
- 상태의 **주기** $d_i = \gcd\{n : P_{ii}^{(n)} > 0\}$와 **비주기성**이 수렴에 어떻게 영향을 미치는가?
- **기약성**(irreducibility)과 **상호통신 클래스**(communicating class)가 체인 구조의 본질을 어떻게 드러내는가?

---

## 🔍 왜 이 분류가 AI에서 중요한가

**MCMC 수렴 보장**(Ch7): Metropolis-Hastings·Gibbs sampler의 정상분포 수렴 정리는 체인이 **기약·비주기·양재귀**임을 요구. 하나라도 깨지면 수렴 보장 없음.

**Page Rank의 dumping factor**: 웹 그래프가 strongly connected가 아닐 때(일부 노드가 dead-end) 기약성 깨짐 → 정상분포 없음. 이를 해결하는 **dumping factor**(0.85)는 기약성을 인위적으로 복구.

**강화학습의 탐색 문제**: 정책 $\pi$ 하에서 MDP 체인이 기약이 아니면, 일부 상태가 절대 방문되지 않음 → Q-value 추정 불가능. $\epsilon$-greedy 탐색은 본질적으로 "체인을 기약으로 만드는" 장치.

**DDPM의 주기성 문제 회피**: Denoising step이 주기 $d > 1$을 가지면 특정 step에서만 수렴 → 샘플 품질 저하. Noise schedule 설계 시 주기성 방지가 무언의 요구사항.

---

## 📐 수학적 선행 조건

- [Ch2-01](./01-markov-property.md): 마르코프 성질, 전이행렬, Chapman-Kolmogorov
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Borel-Cantelli, 확률의 연속성

---

## 📖 직관적 이해

### 재귀 vs 일시

$\{X_n\}_{n \geq 0}$가 상태 $i$에서 시작해 **상태 $i$로 돌아오는 횟수**를 $N_i$라 하자:
$$N_i = \sum_{n \geq 1} \mathbf{1}_{\{X_n = i\}}.$$

**재귀**: $\mathbb{P}_i(N_i = \infty) = 1$ — 무한히 자주 돌아온다.  
**일시**: $\mathbb{P}_i(N_i = \infty) = 0$ — 언젠가 안 돌아온다 (유한 번 돌아옴).

> **비유**: 랜덤 워크 on $\mathbb{Z}$(1차원). 원점에서 시작해 매 step $\pm 1$로 이동. 원점 재귀? **YES** (Pólya 1921). 그러나 3차원 이상에서는 일시 — "취객은 집에 돌아오지만, 술 취한 새는 결코 집에 돌아오지 못한다".

### 양재귀 vs 영재귀

재귀를 더 세분:
- **양재귀**: 평균 재방문 시간 $\mathbb{E}_i[T_i] < \infty$ (여기서 $T_i = \inf\{n \geq 1 : X_n = i\}$)
- **영재귀**: $\mathbb{P}_i(T_i < \infty) = 1$이지만 $\mathbb{E}_i[T_i] = \infty$

**유한상태 기약 체인은 자동으로 양재귀** — 상태 공간이 유한하니 평균 재방문 시간도 유한.

1차원 단순 random walk는 **영재귀** — 재방문은 하지만 평균 시간 무한. 2차원도 영재귀. 3차원부터 일시.

### 주기

상태 $i$의 **주기**
$$d_i = \gcd\{n \geq 1 : P_{ii}^{(n)} > 0\}.$$

$d_i = 1$이면 **비주기**(aperiodic). $d_i > 1$이면 주기 $d_i$ — 반드시 $d_i$의 배수 step마다만 $i$로 돌아감.

> **예**: 2-state "flip-flop" 체인 $0 \to 1 \to 0 \to \cdots$ 완전 결정론. 상태 0의 주기 $d_0 = 2$ — 정확히 짝수 step마다만 0에 있음. 이 경우 $P^n$이 한쪽 상태와 다른 쪽 상태로 **진동** — 정상분포로 수렴 안 함.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — 첫 도달시각, 첫 반환시각

상태 $i$에서 시작해 상태 $j$로의 **첫 도달시각**:
$$T_{ij} := \inf\{n \geq 1 : X_n = j\}, \quad \text{w.r.t. } X_0 = i.$$
$T_{ij} = \infty$이면 "결코 도달 못함".

**첫 반환시각** $T_i := T_{ii}$.

### 정의 2.2 — 재귀 / 일시

상태 $i$가 **재귀**이다 ⇔ $\mathbb{P}_i(T_i < \infty) = 1$.  
상태 $i$가 **일시**이다 ⇔ $\mathbb{P}_i(T_i < \infty) < 1$.

### 정의 2.3 — 양재귀 / 영재귀

재귀 상태 $i$가:
- **양재귀**(positive recurrent) ⇔ $\mathbb{E}_i[T_i] < \infty$
- **영재귀**(null recurrent) ⇔ $\mathbb{E}_i[T_i] = \infty$ (재귀는 여전히)

### 정의 2.4 — 상호통신 (Communicating)

$i \to j$: $\exists n \geq 0, P_{ij}^{(n)} > 0$ ("$i$에서 $j$로 도달 가능").  
$i \leftrightarrow j$: $i \to j$이고 $j \to i$ ("상호통신").

$\leftrightarrow$는 동치관계(반사·대칭·추이) — 상태공간을 **상호통신 클래스**로 분할.

### 정의 2.5 — 기약성

마르코프 체인이 **기약**(irreducible)이다 ⇔ 모든 $i, j$ 사이에 $i \leftrightarrow j$. 즉 상호통신 클래스가 단 하나.

### 정의 2.6 — 주기

상태 $i$의 **주기**: $d_i := \gcd\{n \geq 1 : P_{ii}^{(n)} > 0\}$.

$\{n : P_{ii}^{(n)} > 0\}$가 공집합이면 $d_i$는 정의되지 않음(혹은 $d_i = \infty$).

### 정의 2.7 — 비주기

$d_i = 1$이면 $i$는 **비주기**(aperiodic).

---

## 🔬 정리와 증명

### 정리 2.1 (재귀/일시의 $\sum P^{(n)}$ 판별)

상태 $i$에 대해:
- $i$가 **재귀** ⇔ $\sum_{n \geq 0} P_{ii}^{(n)} = \infty$
- $i$가 **일시** ⇔ $\sum_{n \geq 0} P_{ii}^{(n)} < \infty$

또한 일시일 때 $\mathbb{E}_i[N_i] = \frac{P_i(T_i < \infty)}{1 - P_i(T_i < \infty)} < \infty$.

*증명.*  
$N_i = \sum_{n \geq 1} \mathbf{1}_{\{X_n = i\}}$. 기댓값:
$$\mathbb{E}_i[N_i] = \sum_{n \geq 1} \mathbb{P}_i(X_n = i) = \sum_{n \geq 1} P_{ii}^{(n)}.$$

**재귀 가정**: $p := \mathbb{P}_i(T_i < \infty) = 1$. 강한 마르코프 성질로 $N_i$는 기하분포(parameter $1-p$)의 특수 case — 분포는 $\mathbb{P}_i(N_i \geq k) = p^k$. $p = 1$이면 $N_i = \infty$ a.s., $\mathbb{E}_i[N_i] = \infty$. 따라서 $\sum_n P_{ii}^{(n)} = \infty$.

**일시 가정**: $p < 1$. $\mathbb{P}_i(N_i = k) = p^k(1 - p)$이므로
$$\mathbb{E}_i[N_i] = \sum_k k p^k (1-p) = p / (1-p) < \infty.$$
따라서 $\sum_n P_{ii}^{(n)} < \infty$.

역방향도 위 관계에서. $\square$

### 정리 2.2 (재귀성은 클래스 성질 — class property)

$i \leftrightarrow j$이면 $i$가 재귀 ⇔ $j$가 재귀.

*증명.* $i$가 재귀라 가정. $i \leftrightarrow j$에서 $n, m$ 존재하여 $P_{ij}^{(n)} > 0, P_{ji}^{(m)} > 0$. 임의 $k$에 대해
$$P_{jj}^{(m + k + n)} \geq P_{ji}^{(m)} P_{ii}^{(k)} P_{ij}^{(n)}.$$
$k$에 대해 합하면
$$\sum_{l \geq m + n} P_{jj}^{(l)} \geq P_{ji}^{(m)} P_{ij}^{(n)} \sum_k P_{ii}^{(k)} = \infty.$$
따라서 $j$도 재귀. $\square$

### 정리 2.3 (유한상태 기약 체인은 양재귀)

유한상태 기약 체인에서 모든 상태는 양재귀.

*증명.*  
**재귀 먼저**: 반대로 모든 상태가 일시라 가정. 그러면 $\sum_n P_{ij}^{(n)} < \infty$ for all $i, j$ (정리 2.1). 그런데 $\sum_j P_{ij}^{(n)} = 1$이고 $\sum_j \sum_n P_{ij}^{(n)} = \sum_n 1 = \infty$ — 모순. 따라서 적어도 하나는 재귀, class property로 모두 재귀.

**양재귀**: 정리 2.1의 기댓값 $\mathbb{E}_i[T_i]$가 무한이라 가정. 그러면 평균 방문률이 0으로 간다. 유한상태에서는 평균 방문률의 합이 1(정규화)이므로 모순. 따라서 양재귀.

(공식적 증명은 Ch2-06의 에르고딕 정리를 요구; 자세한 전개는 Durrett §6 참조) $\square$

### 정리 2.4 (주기성은 클래스 성질)

$i \leftrightarrow j$이면 $d_i = d_j$.

*증명.* $i \leftrightarrow j$에서 $n, m$ 존재: $P_{ij}^{(n)} > 0, P_{ji}^{(m)} > 0$. 따라서 $P_{ii}^{(n+m)} \geq P_{ij}^{(n)} P_{ji}^{(m)} > 0$ — $d_i | (n+m)$.

만약 $P_{jj}^{(k)} > 0$이면 $P_{ii}^{(n+k+m)} \geq P_{ij}^{(n)} P_{jj}^{(k)} P_{ji}^{(m)} > 0$ — $d_i | (n+k+m)$. 따라서 $d_i | k$. 그러므로 $d_i | d_j$. 대칭으로 $d_j | d_i$ ⇒ $d_i = d_j$. $\square$

### 정리 2.5 (비주기성의 닫힘 성질)

$i$가 비주기 ($d_i = 1$)이면 충분히 큰 모든 $n$에 대해 $P_{ii}^{(n)} > 0$.

*증명 (수론 결과).* $S = \{n : P_{ii}^{(n)} > 0\}$는 닫힘(closed under addition)이고 $\gcd = 1$. 수론 보조정리: $S$가 덧셈에 대해 닫혀 있고 $\gcd = 1$이면 **충분히 큰 모든 정수**가 $S$에 속함(Chicken McNugget 정리의 변형). $\square$

### 정리 2.6 (주기 $d$일 때 상태공간의 주기적 분할)

기약 주기 $d$ 체인은 상태공간을 $d$개의 disjoint 집합 $C_0, C_1, \ldots, C_{d-1}$으로 분할, 각 step마다 $C_k \to C_{k+1 \mod d}$로 이동.

*증명 스케치.* 임의 $i_0$ 고정, $C_k = \{j : P_{i_0 j}^{(kd + r)} > 0$ for some $r \equiv k \mod d\}$로 정의. 주기 일관성으로 상태 $j$가 어느 $C_k$에 속하는지 well-defined. $\square$

> **응용**: 주기성이 깨지면 체인이 $d$개 서브체인을 순환 — 수렴 분석은 $P^d$ (각 서브체인에서 기약·비주기 체인)로 환원.

---

## 💻 NumPy 구현 검증

### 실험 1 — 재귀성 판별: $\sum P_{ii}^{(n)}$

```python
import numpy as np
rng = np.random.default_rng(0)

# 3-state: 0과 1은 상호통신 클래스, 2는 absorbing
P = np.array([
    [0.5, 0.3, 0.2],
    [0.2, 0.5, 0.3],
    [0.0, 0.0, 1.0],   # 2는 absorbing (재귀지만 나머지는 도달 불가)
])

# Σ P_00^(n): 재귀 기대 → 발산
cumsum_00 = 0.0
Pn = np.eye(3)
for n in range(1, 10_000):
    Pn = Pn @ P
    cumsum_00 += Pn[0, 0]
print(f'Σ_{{n=1}}^{{10000}} P_00^(n) = {cumsum_00:.4f}')
# → 작은 값 (0, 1은 일시 — 2로 흡수)
# Σ_n P_00^n 가 수렴하면 일시. 이 예에선 0이 일시.

# 2의 주기 행 (absorbing이므로 자명한 재귀)
# 만약 흡수 상태가 없이 기약·비주기면 Σ = ∞
P_irr = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5],
])
cumsum = 0.0
Pn = np.eye(3)
for n in range(1, 1000):
    Pn = Pn @ P_irr
    cumsum += Pn[0, 0]
print(f'기약 P에서 Σ P_00^(n) (n≤1000) = {cumsum:.4f}')
# → 큰 값 (재귀), 실제로 선형 증가
```

### 실험 2 — 주기성 관찰

```python
# 2-state flip-flop: 주기 2
P_period = np.array([[0, 1], [1, 0]])
for n in range(1, 7):
    Pn = np.linalg.matrix_power(P_period, n)
    print(f'P^{n} = \n{Pn}')
# n이 홀수면 identity 아님, 짝수면 identity → 주기 2
# → 이 체인은 정상분포로 수렴 안 함 (진동)

# 비주기로 만들려면 self-loop 추가
P_aper = np.array([[0.1, 0.9], [0.9, 0.1]])
for n in [1, 5, 20, 100]:
    Pn = np.linalg.matrix_power(P_aper, n)
    print(f'비주기 P^{n} = {Pn[0]}')
# → (0.5, 0.5)로 수렴
```

### 실험 3 — 상호통신 클래스 찾기 (그래프 SCC)

```python
# networkx의 strongly connected components로 communicating classes 찾기
import networkx as nx

P = np.array([
    [0.5, 0.5, 0.0, 0.0],
    [0.3, 0.7, 0.0, 0.0],   # {0, 1} 클래스
    [0.0, 0.0, 0.4, 0.6],
    [0.0, 0.0, 0.5, 0.5],   # {2, 3} 클래스 (서로 독립적)
])
G = nx.DiGraph()
for i in range(4):
    for j in range(4):
        if P[i, j] > 0:
            G.add_edge(i, j)
scc = list(nx.strongly_connected_components(G))
print(f'Communicating classes: {scc}')
# → [{0, 1}, {2, 3}]
# 이 체인은 **기약이 아님** — 두 클래스 간 이동 불가
# 정상분포 유일하지 않음 (각 클래스마다 하나씩)
```

---

## 🔗 AI/ML 연결

**MCMC의 기약성 검증**  
Metropolis-Hastings의 proposal $q(y|x)$가 state space 전체를 "도달 가능"해야 함. 예: 이산 optimization에서 proposal이 이웃 상태만 제안하면, disconnected 구역이 있으면 기약성 깨짐 → 일부 국소 최적에 영원히 갇힘. 실전 MCMC 진단은 trace plot에서 "jumping 패턴"을 확인 → 기약성 체크.

**PageRank 알고리즘**  
웹 그래프 자체는 기약이 아님(dangling node). Google의 해결: **dumping factor** $\alpha = 0.85$로 모든 페이지로 확률 $(1-\alpha)/N$로 jump. 이는 인위적으로 기약·비주기 체인을 구성하여 유일한 정상분포(= PageRank) 보장.

**RL의 exploration-exploitation**  
$\epsilon$-greedy policy: 확률 $\epsilon$로 random action → 모든 state-action 쌍을 기약하게 만듦. $\epsilon = 0$이면 기약성 깨질 수 있음 → Q-learning 수렴 실패.

**DDPM의 주기성 회피**  
각 step의 $\beta_t \neq 0$을 보장 — self-loop($\sqrt{1 - \beta_t} < 1$) 있음 → 비주기성 자연스럽게 성립. 만약 어떤 step에서 $\beta_t = 0$이면 해당 step은 deterministic → 주기성 발생 위험.

**자동생성 체인의 burn-in**  
MCMC "burn-in period"는 사실상 "transient 상태에서 벗어나 양재귀 regime으로 진입하는 시간". 일시 상태가 많은 체인이면 burn-in이 길어짐 → 효율 저하.

---

## ⚖️ 가정과 한계

**가정 — 유한상태의 단순성**  
유한상태에서 "기약 ⇒ 양재귀"는 자동이지만, 가산상태($\mathbb{Z}$, $\mathbb{N}$)에서는 영재귀 가능. 예: 1차원 simple random walk on $\mathbb{Z}$는 기약·재귀지만 **영재귀**(평균 재방문 무한).

**한계 — 연속상태 공간**  
연속상태에서는 $P_{ii} = 0$ for all $i$ (단일 점 확률 0). 재귀/일시 정의가 다르게 필요 — Harris recurrence, $\phi$-irreducibility 등 (Meyn & Tweedie 2009). 이 레포의 범위 밖.

**한계 — 검증의 어려움**  
실전 MCMC에서 "체인이 정말 기약인가"는 확률 1로 증명하기 어렵. 실증적으로는 여러 chain을 서로 다른 초기점에서 시작하여 섞이는지 확인(Gelman-Rubin $\hat{R}$).

---

## 📌 핵심 정리

| 개념 | 판별 |
|---|---|
| 재귀 | $\mathbb{P}_i(T_i < \infty) = 1$ ⇔ $\sum_n P_{ii}^{(n)} = \infty$ |
| 일시 | $\mathbb{P}_i(T_i < \infty) < 1$ ⇔ $\sum_n P_{ii}^{(n)} < \infty$ |
| 양재귀 | 재귀 + $\mathbb{E}_i[T_i] < \infty$ |
| 영재귀 | 재귀 + $\mathbb{E}_i[T_i] = \infty$ |
| 주기 $d_i$ | $\gcd\{n : P_{ii}^{(n)} > 0\}$ |
| 비주기 | $d_i = 1$ |
| 기약 | 모든 상태 쌍이 상호통신 |
| 유한·기약 | 자동으로 양재귀 |
| 재귀성·주기 | 클래스 성질 (상호통신 클래스 내에서 같음) |

**한 줄 요약**: 상태는 **(재귀/일시) × (양/영) × (주기)** 로 분류되며, 기약·비주기·양재귀 체인만이 유일한 정상분포로 수렴(Ch2-03, Ch2-04). 이 분류가 MCMC·PageRank·RL의 이론적 안전망.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. 2차원 simple random walk on $\mathbb{Z}^2$ (각 step $\pm e_1, \pm e_2$ 중 확률 1/4)이 기약·재귀이지만 **영재귀**임을 Pólya 정리를 이용해 설명하라.

<details>
<summary>해설</summary>

**기약**: 모든 $(i, j) \in \mathbb{Z}^2$가 서로 도달 가능(양의 확률로). 따라서 기약.

**재귀**: Pólya 정리 — $d = 1, 2$에서 simple random walk가 재귀, $d \geq 3$에서 일시. 증명 아이디어는 $P_{00}^{(2n)} \sim c/n^{d/2}$ ($n \to \infty$). 2차원에서 $\sum_n 1/n$은 발산 → 재귀.

**영재귀**: $\mathbb{E}_0[T_0]$을 계산하면, 2차원 random walk의 return time은 heavy-tail 분포 — 평균 무한. 이는 $\sum_n P_{00}^{(2n)} \sim \sum_n 1/n$가 log scale로 발산하지만, 평균 return time은 더 발산.

**직관**: "취객은 결국 집에 돌아온다. 하지만 **평균적으로 무한히 오래 걸린다**." 돌아오긴 하지만 그 기다림의 기댓값이 $\infty$.

**실전 함의**: 격자 상의 자연 확산 모델(diffusion on grid)이 2차원에서 영재귀라는 사실은, 평형 도달 시간 추정이 신중해야 함을 시사. PageRank 같은 응용에서는 이를 피하려고 dumping factor를 추가.

</details>

**문제 2 (심화)**. $P = \begin{pmatrix} 0.5 & 0.5 \\ 1.0 & 0.0 \end{pmatrix}$ 이 체인의 주기를 구하고, $P^n$이 큰 $n$에서 어떻게 행동하는지 분석하라.

<details>
<summary>해설</summary>

**주기 계산**: 상태 0에서 돌아오는 $n$:
- $n = 1$: $P_{00}^{(1)} = 0.5 > 0$ ✓
- $n = 2$: $P_{00}^{(2)} = 0.5 \cdot 0.5 + 0.5 \cdot 1.0 = 0.75 > 0$ ✓

$\gcd(1, 2, \ldots) = 1$ → **비주기** ($d = 1$).

**$P^n$의 극한**:
정상분포 $\pi$: $\pi_0 \cdot 0.5 + \pi_1 \cdot 1 = \pi_0$, $\pi_0 + \pi_1 = 1$로 $\pi_1 = 0.5 \pi_0$, 즉 $\pi = (2/3, 1/3)$.

고유값: $\det(P - \lambda I) = 0 \Rightarrow (0.5 - \lambda)(-\lambda) - 0.5 = 0 \Rightarrow \lambda^2 - 0.5\lambda - 0.5 = 0$.  
$\lambda = \frac{0.5 \pm \sqrt{0.25 + 2}}{2} = \frac{0.5 \pm 1.5}{2} \Rightarrow \lambda_1 = 1, \lambda_2 = -0.5$.

$P^n \to \mathbf{1}\pi^T$ with rate $|\lambda_2|^n = (0.5)^n$. 구체적으로:
$P^n = \mathbf{1}\pi^T + \lambda_2^n \cdot v_2 u_2^T$ (Ch2-04).

$n$이 클 때 $P^n_{ij} \to \pi_j$ 지수적으로. 10 step이면 $(0.5)^{10} \approx 0.001$로 거의 정상분포.

**검증 (NumPy)**:
```python
P = np.array([[0.5, 0.5], [1.0, 0.0]])
for n in [1, 3, 10, 20]:
    print(f'P^{n}[0] = {np.linalg.matrix_power(P, n)[0]}')
```
→ $(0.5, 0.5), (0.625, 0.375), \ldots, (0.667, 0.333)$

</details>

**문제 3 (AI 연결)**. PPO 학습 중 policy $\pi_\theta$가 특정 state를 전혀 방문하지 않을 수 있다. 이 현상을 마르코프 체인의 상태 분류로 설명하고, 해결책을 3가지 제시하라.

<details>
<summary>해설</summary>

**현상 설명**: Policy $\pi_\theta$ 하의 state transition chain은 **일시 상태**를 가질 수 있음. 특히 deterministic policy는 특정 trajectory만 따름 → 다른 state는 도달 불가능 ("일시" = 결국 안 방문). 이 경우:
- 해당 state의 value/Q 추정 불가능
- Exploration 부족으로 suboptimal policy에 갇힘
- Advantage 추정이 biased

**해결책 1: Entropy bonus** (PPO의 기본 기법)  
Loss에 $-\beta H(\pi_\theta)$를 추가하여 policy의 stochasticity 유지. 모든 action에 대해 $\pi_\theta(a|s) > 0$를 강제 → 체인이 기약에 가까워짐. 이는 $\epsilon$-greedy의 연속 버전.

**해결책 2: Off-policy exploration (uniform sampling)**  
Behavior policy $\mu$로 exploration (e.g., DQN의 $\epsilon$-greedy), target policy $\pi_\theta$로 학습. $\mu$가 기약이면 모든 state 방문 보장. Importance sampling으로 $\pi_\theta$ gradient 계산.

**해결책 3: Intrinsic motivation / Curiosity**  
Visit count $n(s)$에 반비례하는 intrinsic reward $r_{\text{int}} = 1/\sqrt{n(s)}$ 추가. 드물게 방문한 state에 보상 → 기약성 촉진. RND(Random Network Distillation), ICM 등이 이 아이디어.

**이론적 보장**: 위 3가지 모두 "학습 중 모든 state가 양의 확률로 방문됨" → 체인이 기약·양재귀. 정리 2.3에 의해 양재귀 → 에르고딕 정리(Ch2-06) 적용 → Q-learning 수렴 보장.

</details>

---

<div align="center">

◀ [01. 마르코프 성질과 전이행렬](./01-markov-property.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [03. 정상분포(Stationary Distribution)와 Perron-Frobenius](./03-stationary-distribution.md)

</div>
