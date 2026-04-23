# 03. 정상분포(Stationary Distribution)와 Perron-Frobenius

## 🎯 핵심 질문

- 정상분포 $\pi P = \pi$는 언제 **존재**하는가? 언제 **유일**한가?
- **Perron-Frobenius 정리**가 확률행렬에 대해 어떻게 특수화되어 정상분포의 유일성을 보장하는가?
- **좌고유벡터 $\pi$**(정상분포)와 **우고유벡터 $\mathbf{1}$**은 각각 어떤 해석을 가지는가?
- 기약·비주기 조건이 각각 어떤 결론에 어떻게 기여하는가 — 각각 떼어내면 무엇이 깨지는가?

---

## 🔍 왜 이 정리가 AI에서 중요한가

**MCMC의 모든 것**: Metropolis-Hastings·Gibbs sampler가 샘플하는 분포는 체인의 **정상분포**. Perron-Frobenius가 "이 정상분포가 유일하고 체인이 거기로 수렴한다"를 보장.

**PageRank의 수학**: Google PageRank는 정확히 "dumping 변형된 웹 그래프 전이행렬의 좌고유벡터($\pi P = \pi$ with eigenvalue 1)". Perron-Frobenius가 이 벡터의 존재·유일·비음을 보장.

**RL의 state distribution**: 정책 $\pi$ 하의 state-visitation distribution $d^\pi$가 정상분포 — policy gradient theorem의 기댓값은 이 분포에 대한 것.

**Spectral clustering**: 그래프 Laplacian의 고유벡터 분석에서 정상분포가 등장(Perron 고유값 근처). 이것이 graph ML (GNN의 message passing)의 수학적 기반.

---

## 📐 수학적 선행 조건

- [Ch2-01](./01-markov-property.md): 전이행렬, Chapman-Kolmogorov
- [Ch2-02](./02-state-classification.md): 기약성, 비주기성, 양재귀
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): 고유값, Jordan 분해, 비음 행렬의 스펙트럴 이론

---

## 📖 직관적 이해

### 정상분포의 의미

**정상분포** $\pi$는 "체인이 이 분포에서 시작하면 모든 시각에서 같은 분포"를 만족하는 행벡터:
$$\pi P = \pi.$$

동치: $X_0 \sim \pi$이면 $X_n \sim \pi$ for all $n$. 시간에 따라 **marginal 분포가 불변**.

$\pi$는 **좌고유벡터** with eigenvalue 1. 기본 사실($P \mathbf{1} = \mathbf{1}$)로 고유값 1은 항상 존재 → 좌고유벡터도 항상 존재. 문제는 **유일성**과 **비음성(확률이므로)**.

> **비유**: 방 안의 가스 분자들. 초기에 한쪽에 몰려 있으면 비평형. 충분한 시간 후 공간적으로 균일한 "평형 분포"에 도달. 이 평형 분포가 정상분포 $\pi$. 역학(전이행렬) 자체가 평형을 결정.

### 왜 Perron-Frobenius가 필요한가

일반 행렬은 고유벡터가 음수/복소수일 수 있음. 확률분포는 **비음·합 1**이어야. Perron-Frobenius는 다음을 보장:
- 고유값 1이 **단순**(multiplicity 1) — 정상분포 유일
- 대응 좌고유벡터가 **비음성분 전부** — 확률분포로 정규화 가능
- 다른 모든 고유값은 $|\lambda_k| < 1$ (비주기 하) — 수렴률 지표

**조건**: 전이행렬이 기약 (그래프 관점: strongly connected). 비주기는 수렴률에만 영향.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — 정상분포 (Stationary / Invariant Distribution)

확률벡터 $\pi = (\pi_1, \ldots, \pi_N)$ ($\pi_i \geq 0, \sum_i \pi_i = 1$)이 **정상분포**라는 것은
$$\pi P = \pi \iff \sum_i \pi_i P_{ij} = \pi_j, \forall j.$$

### 정의 3.2 — 불변측도 (Invariant Measure)

확률이 아닌 일반 비음 측도 $\mu$가 $\mu P = \mu$를 만족하면 **불변측도**. 합이 유한하면 정규화해서 정상분포로 변환 가능.

### 정의 3.3 — 기약 확률행렬 (Irreducible Stochastic Matrix)

$P$가 기약이라는 것은 모든 $i, j$에 대해 $n = n(i, j)$가 존재해 $P_{ij}^{(n)} > 0$. (Ch2-02의 상호통신 정의와 동등.)

### 정의 3.4 — 원시 확률행렬 (Primitive / Regular)

$P$가 **원시**(primitive)이다 ⇔ 어떤 $n_0$이 존재해 $P^{n_0} > 0$ (모든 성분 양수). 이는 기약 + 비주기와 동치.

---

## 🔬 정리와 증명

### 정리 3.1 (Perron-Frobenius — 확률행렬 버전)

$P \in \mathbb{R}^{N \times N}$가 **기약 확률행렬**이면:
1. **고유값 1이 단순** (multiplicity 1, both algebraic and geometric).
2. 다른 모든 고유값은 $|\lambda| \leq 1$.
3. 1에 대응하는 **유일한 좌고유벡터 $\pi$가 성분 모두 양수**인 확률분포 (정상분포).
4. 1에 대응하는 우고유벡터는 $\mathbf{1}$ (up to scalar).

### 정리 3.2 (기약·비주기이면 최대 고유값 1만 $|\lambda| = 1$)

$P$가 **기약·비주기**(원시)이면:
- 고유값 1만 $|\lambda| = 1$에 있음.
- 다른 모든 고유값 $|\lambda_k| < 1$.

### 통합 증명 — 유한상태 기약·비주기 확률행렬

**Step 1 — 고유값 1이 존재 (우고유벡터 $\mathbf{1}$)**  
$P \mathbf{1} = \mathbf{1}$ 명백.

**Step 2 — 정상분포 $\pi$ 존재**  
유한상태이므로 $P^T$도 확률행렬이 아니지만 같은 고유값 1. 대응 좌고유벡터 $\pi$ 존재(가능 복소수). 실수화 + 비음 정규화:
- $P$ 기약 ⇒ $(I + P + P^2 + \cdots + P^{n-1})$에서 모든 성분이 어느 $n$에 양(기약성).
- 따라서 고유값 1의 좌고유공간에 비음 벡터가 있음을 Perron 원본 증명(Gelfand, Seneta)으로 보일 수 있음.

(간단 버전: Brouwer 고정점 정리를 $\pi \mapsto \pi P$ on simplex에 적용. $P$가 simplex를 simplex로 보내므로 연속, compact → 고정점 존재.)

**Step 3 — 유일성**  
$\pi_1, \pi_2$ 두 정상분포이면 $\pi = \pi_1 - \pi_2$가 $\pi P = \pi$ 이고 $\sum \pi_i = 0$ (둘 다 확률이라). $\pi$를 양·음 부분으로 분해 $\pi = \pi^+ - \pi^-$. 각각 $\pi^\pm P \geq \pi^\pm$이어야 하는데, 기약성으로 하나만 양이면 모순 ⇒ $\pi = 0$. $\square$

**Step 4 — 비주기 ⇒ 다른 고유값은 $|\lambda| < 1$**  
$Pv = \lambda v$, $|\lambda| = 1$. 최대 성분 $|v_i^*| = \max$ 선택, 정리 1.5 증명처럼 부등식 등식이 되어야 한다:
$$\sum_j P_{i^* j} v_j = \lambda v_{i^*}, \quad |\lambda v_{i^*}| = |v_{i^*}|.$$
등식 조건으로 $v_j = (\lambda / |\lambda|) v_{i^*}$ 상수. 즉 $v$가 상수 벡터. 원시성(비주기) 가정으로 이 상수 벡터는 $\lambda = 1$ only. $\square$

### 정리 3.3 (양재귀 + 기약 ⇒ 정상분포 유일 존재 — 가산 상태)

가산상태 기약 체인이 **양재귀**이면 정상분포 $\pi_i = 1/\mathbb{E}_i[T_i]$가 유일하게 존재.

*증명 스케치.* 임의 상태 $i$ 고정. $\mu_j^{(i)} = \mathbb{E}_i[\text{방문 } j \text{ in one cycle of return to } i]$로 정의. Tower property로 $\mu^{(i)}$가 불변측도. 합 $\sum_j \mu_j^{(i)} = \mathbb{E}_i[T_i] < \infty$ (양재귀). 정규화로 $\pi_i = 1/\mathbb{E}_i[T_i]$의 정상분포. $\square$

이 식 $\pi_i = 1/\mathbb{E}_i[T_i]$은 **정상분포의 재방문 시간 해석** — 매우 유용.

### 정리 3.4 (기약성 필수)

$P$가 기약이 아니면(2개 이상 closed class), 정상분포 **유일하지 않음**.

*반례.* 2개 disjoint closed class $\{C_1, C_2\}$. 각각 기약 부분체인의 정상분포 $\pi^{(1)}, \pi^{(2)}$. 임의 convex 조합 $\lambda \pi^{(1)} + (1-\lambda) \pi^{(2)}$가 전체 체인의 정상분포 — 무한히 많음. $\square$

---

## 💻 NumPy 구현 검증

### 실험 1 — Perron-Frobenius 직접 확인

```python
import numpy as np
rng = np.random.default_rng(0)

P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5],
])

# 고유값과 좌고유벡터
eigvals, eigvecs = np.linalg.eig(P.T)   # P^T의 고유벡터 = P의 좌고유벡터
print(f'고유값: {eigvals}')
print(f'절댓값: {np.abs(eigvals)}')

# 고유값 1에 해당하는 좌고유벡터 (정상분포)
idx_max = np.argmin(np.abs(eigvals - 1))
pi_raw = np.real(eigvecs[:, idx_max])
pi = pi_raw / pi_raw.sum()
print(f'정상분포 π = {pi}')
print(f'검증: π P = {pi @ P}')   # π와 일치해야

# 우고유벡터 (should be 1 up to scalar)
eigvals_r, eigvecs_r = np.linalg.eig(P)
print(f'우고유벡터 (eigval=1): {np.real(eigvecs_r[:, np.argmin(np.abs(eigvals_r - 1))])}')
# → 상수 벡터 (1, 1, 1) 스케일
```

### 실험 2 — 기약 아닐 때 정상분포 무한히 많음

```python
# 2개 closed class: {0, 1}, {2, 3}
P = np.array([
    [0.5, 0.5, 0.0, 0.0],
    [0.3, 0.7, 0.0, 0.0],
    [0.0, 0.0, 0.4, 0.6],
    [0.0, 0.0, 0.5, 0.5],
])

eigvals, eigvecs = np.linalg.eig(P.T)
print(f'고유값: {np.sort(np.abs(eigvals))[::-1]}')
# → 1이 multiplicity 2 (두 클래스 각각 하나씩)

# 두 독립 정상분포
for idx in np.where(np.abs(eigvals - 1) < 1e-8)[0]:
    v = np.real(eigvecs[:, idx])
    v = v / np.abs(v).sum()
    print(f'정상분포 후보: {v}')
# → 두 개 나옴, 서로 disjoint support
```

### 실험 3 — 재방문 시간과 정상분포 관계

```python
# π_i = 1/E_i[T_i] 검증
P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5],
])

# 정상분포
eigvals, eigvecs = np.linalg.eig(P.T)
pi = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1))])
pi = pi / pi.sum()

# Monte Carlo로 E_i[T_i] 추정
def mean_return_time(P, start, n_trials=20_000):
    times = []
    for _ in range(n_trials):
        state = start
        for n in range(1, 10_000):
            state = rng.choice(3, p=P[state])
            if state == start:
                times.append(n)
                break
    return np.mean(times)

print(f'{"state":>5} {"π_i":>10} {"1/E_i[T_i]":>15}')
for i in range(3):
    T_i = mean_return_time(P, i)
    print(f'{i:>5} {pi[i]:>10.4f} {1/T_i:>15.4f}')
# → π_i와 1/E_i[T_i]가 일치
```

---

## 🔗 AI/ML 연결

**PageRank 엔진**  
웹 페이지 그래프 $G$, 각 링크 $(i \to j)$에 확률 $1/\text{outdeg}(i)$. 기약성 보장 위해 **dumping**: $P_{ij} = \alpha \frac{1}{\text{outdeg}(i)} \mathbf{1}_{\{i \to j\}} + (1 - \alpha) / N$. Perron-Frobenius로 정상분포 $\pi$ 유일 존재 → $\pi_j$ = 페이지 $j$의 PageRank. $\alpha = 0.85$ (Brin & Page 원본).

**MCMC 안전망**  
Metropolis-Hastings가 기약 proposal + $\pi$ accept 조건 하에 올바른 정상분포 유지(Ch7-02). Perron-Frobenius 덕분에 "$\pi$가 정상이고 유일"이라는 보장. 기약성 깨지면 MH도 일부 영역에만 갇힘 → sampling bias.

**RL의 state distribution**  
정상 정책 $\pi$ 하 Markov chain of states. 체인이 기약·비주기이면 state-visitation distribution $d^\pi(s)$는 유일. Policy gradient의 objective $\mathbb{E}_{s \sim d^\pi}[\nabla \log \pi(a|s) Q(s, a)]$가 well-defined.

**Graph Convolutional Network의 stationary feature**  
Graph Laplacian $L = I - D^{-1/2} W D^{-1/2}$의 고유벡터 중 "0 고유값" eigenvector가 정상분포와 연결. GCN의 spectral filtering은 이 정상분포 근처의 저주파 정보를 보존.

**Language Model의 stationary token distribution**  
Unigram 모델은 "iid token" — 정상. N-gram은 N-1 order Markov — 정상분포 존재. Transformer는 비마르코프이지만 각 layer의 "key vector distribution"이 정상 상태에 가까워진다는 연구 관찰(Nguyen et al.).

---

## ⚖️ 가정과 한계

**가정 — 유한 vs 가산**  
유한상태 기약 체인은 자동으로 양재귀 → 정상분포 존재. 가산상태는 양재귀 별도 확인 필요(1D simple RW는 영재귀 → 정상분포 **없음**).

**한계 — 정상분포 계산의 scale**  
큰 상태공간 ($N \sim 10^6$ 웹 페이지)에서 고유벡터 계산 비용 ↑. Power iteration $\pi^{(k+1)} = \pi^{(k)} P$를 수렴까지 반복 — 수렴률은 $|\lambda_2|^k$ (Ch2-04).

**한계 — 고차원 MCMC 수렴 느림**  
고차원에서 $|\lambda_2| \to 1$이 흔함 → mixing time 폭발. 이를 측정하는 것이 **spectral gap** $1 - |\lambda_2|$. 작으면 실전 수렴 불가능.

---

## 📌 핵심 정리

| 정리 | 조건 | 결론 |
|---|---|---|
| Perron-Frobenius | 기약·비주기·유한상태 | 정상분포 $\pi > 0$ 유일 존재, $|\lambda_k| < 1$ for $k \neq 1$ |
| 가산상태 버전 | 기약·양재귀 | 정상분포 유일 존재, $\pi_i = 1/\mathbb{E}_i[T_i]$ |
| Fixed point | Brouwer | $\pi \mapsto \pi P$의 고정점 = 정상분포 |
| 기약 아니면 | 여러 closed class | 정상분포 유일하지 않음 |

**한 줄 요약**: 기약 확률행렬의 좌고유벡터(eigenvalue 1) = 정상분포, 유일·양성분. Perron-Frobenius가 수학적 보장을 제공. 이 보장이 MCMC·PageRank·RL 모두의 근간.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. $P = \begin{pmatrix} 0.8 & 0.2 \\ 0.5 & 0.5 \end{pmatrix}$의 정상분포를 손으로 구하라.

<details>
<summary>해설</summary>

$\pi P = \pi$ and $\pi_0 + \pi_1 = 1$.

$\pi_0 (0.8) + \pi_1 (0.5) = \pi_0$  
$\pi_0 + \pi_1 = 1$

첫 식: $0.8 \pi_0 + 0.5 (1 - \pi_0) = \pi_0 \Rightarrow 0.5 + 0.3 \pi_0 = \pi_0 \Rightarrow \pi_0 = 0.5 / 0.7 = 5/7$.  
$\pi_1 = 2/7$.

**확인**: $\pi P = (5/7 \cdot 0.8 + 2/7 \cdot 0.5, 5/7 \cdot 0.2 + 2/7 \cdot 0.5) = (0.5 + 1/7, 1/7 + 1/7) = (5/7, 2/7)$ ✓

</details>

**문제 2 (심화)**. Perron-Frobenius 증명의 Step 3에서 "$\pi^+$와 $\pi^-$ 중 하나만 양이면 모순"을 명확히 하라.

<details>
<summary>해설</summary>

$\pi = \pi^+ - \pi^-$, $\pi^+, \pi^- \geq 0$ with disjoint supports. 가정 $\pi P = \pi$.

$\pi^+ P$과 $\pi^- P$ 는 모두 비음이고, $\pi P = \pi^+ P - \pi^- P = \pi^+ - \pi^-$.

만약 $\pi^+$의 support가 $S^+$ (서로 disjoint $S^-$), $\pi^+ P$는 각 상태에 대해
$$(\pi^+ P)_j = \sum_i \pi^+_i P_{ij} \geq 0.$$

$j \in S^+$: $(\pi^+ P - \pi^- P)_j = \pi_j^+ - 0 = \pi_j^+$. 따라서 $(\pi^- P)_j \geq 0$ (당연) 이고 $(\pi^+ P)_j = \pi_j^+ + (\pi^- P)_j \geq \pi_j^+$.

즉 $\pi^+ P \geq \pi^+$ (component-wise). 그런데 합 $\sum_j (\pi^+ P)_j = \sum_j \pi_j^+$ (확률행렬 합 보존). 따라서 **모든 $j$에서 $(\pi^+ P)_j = \pi_j^+$** — $\pi^+$이 자체로 불변측도.

기약성으로 $\pi^+$이 unique (up to scalar), 마찬가지 $\pi^-$도. 두 측도의 support가 disjoint인데 기약성으로는 불가능 (모든 상태 간 상호통신) ⇒ 하나는 0. 즉 $\pi^+ = 0$ or $\pi^- = 0$. 합쳐서 $\pi = 0$. $\square$

</details>

**문제 3 (AI 연결)**. RL에서 학습이 "suboptimal policy에 갇힌다"는 현상을 정상분포 관점에서 설명하고, 이를 회피하는 알고리즘적 장치를 2가지 이상 제시하라.

<details>
<summary>해설</summary>

**현상**: Suboptimal policy $\pi_\theta$가 특정 trajectory만 생성 → 체인이 "거의 결정론적" (closed class로 수렴). State 방문 분포 $d^{\pi_\theta}$가 optimal policy 하 방문 분포 $d^{\pi^*}$와 크게 달라짐. 그러나 policy gradient는 $\mathbb{E}_{s \sim d^{\pi_\theta}}[\nabla \log \pi_\theta (a|s) Q(s,a)]$로 **현재 방문 분포에서의** gradient만 계산 → 방문 안 하는 상태의 Q 개선 기회 없음 → 정상분포가 suboptimal에 고정.

**수학적 표현**: 체인이 multiple closed class로 분리, 현재 class의 정상분포 $\pi^{(\text{local})}$만 최적화 (Perron-Frobenius 유일성 실패).

**장치 1: Entropy regularization**  
$\mathcal{L} = \mathbb{E}[R] + \beta H(\pi_\theta)$. $H$ 가 $\pi_\theta$의 stochasticity 유지 → 체인이 기약에 가까워져 단일 정상분포로 수렴 → 모든 state 방문.

**장치 2: Off-policy data augmentation**  
Behavior policy $\mu \neq \pi_\theta$로 다양한 trajectory 수집. $\mu$가 exploration을 장려하면 정상분포 $d^\mu$가 넓게 퍼져 — 모든 state 학습 가능. Importance sampling $\pi_\theta/\mu$로 policy gradient bias 보정.

**장치 3: Trust region + diverse initialization**  
TRPO/PPO의 clipping으로 greedy update 억제 + 여러 random seed로 학습하여 "여러 체인 instance의 앙상블" — 하나는 global optimum에 도달할 확률↑.

**장치 4: Intrinsic motivation (curiosity)**  
Rare state에 bonus reward → 방문 확률 증가 → 정상분포 reshape.

**종합**: 모든 장치의 공통 원리 — "체인을 기약하게 유지하여 Perron-Frobenius의 유일 정상분포 보장 → 모든 상태가 방문되도록"로 정리 가능.

</details>

---

<div align="center">

◀ [02. 상태의 분류 — 재귀·일시·주기](./02-state-classification.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [04. 극한정리와 수렴률 — 스펙트럴 접근](./04-convergence-rate.md)

</div>
