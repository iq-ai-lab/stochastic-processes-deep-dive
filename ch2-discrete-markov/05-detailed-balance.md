# 05. Reversibility와 Detailed Balance

## 🎯 핵심 질문

- **Detailed balance** $\pi_i P_{ij} = \pi_j P_{ji}$가 어떻게 정상분포 $\pi P = \pi$를 **자동으로** 보장하는가?
- 이 조건의 물리적 의미 — "시간 역전 대칭성"은 어떻게 해석되는가?
- Reversible 체인에서 $P$가 **self-adjoint operator** on $\ell^2(\pi)$가 되는 스펙트럴 의미는?
- MCMC(Ch7)의 Metropolis-Hastings가 왜 **detailed balance를 만족**하도록 설계되었는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

**Metropolis-Hastings의 정당성**(Ch7-02): MH의 acceptance ratio $\alpha = \min(1, \frac{\pi(y)q(x|y)}{\pi(x)q(y|x)})$가 **정확히 detailed balance를 만족**하도록 설계. 이것이 MH 체인이 $\pi$에 수렴하는 유일한 근거.

**Glauber dynamics in 물리/베이지안 NN**: Ising 모델·Boltzmann machine의 Gibbs sampling이 detailed balance → Boltzmann 분포로 수렴. Hinton의 RBM 학습이 이 원리 기반.

**Reversible SDE와 Langevin dynamics** (SDE Deep Dive Ch4): Overdamped Langevin $dX = -\nabla U dt + \sqrt{2} dB$가 reversible w.r.t. $\pi \propto e^{-U}$. Score-based generative model의 수학적 토대.

**스펙트럴 분석의 단순화**: Reversible 체인의 전이행렬이 self-adjoint → **실고유값·실직교 고유벡터** → 스펙트럴 gap·mixing time의 깔끔한 분석(Ch2-04).

---

## 📐 수학적 선행 조건

- [Ch2-01 ~ Ch2-04](./01-markov-property.md): 마르코프, 정상분포, 수렴률
- 함수해석 기초: inner product, self-adjoint operator
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): 실대칭 행렬의 스펙트럴 정리

---

## 📖 직관적 이해

### 물리적 시간 역전

**Detailed balance**를 직관적으로 — "시각 $n$에 상태 $i$에서 $j$로 가는 **확률 flow**"와 "시각 $n+1$에서 $j$에서 $i$로 가는 flow"가 같다:
$$\underbrace{\pi_i P_{ij}}_{i \to j \text{ flow}} = \underbrace{\pi_j P_{ji}}_{j \to i \text{ flow}}.$$

이는 "평형에서 분자 수준에서 순flow가 0"이라는 statistical mechanics의 **detailed balance**와 동일한 개념.

### 시간 역전된 체인

Reversible 체인 $\{X_n\}$의 시간을 뒤집으면 $\{X_{N-n}\}_{n=0}^N$이 역시 **같은 전이행렬을 갖는 MC**. 즉 체인 자체가 "시간 방향을 구별할 수 없음".

> **비유**: 동영상을 거꾸로 돌려도 자연스러워 보이는 장면(reversible) vs 깨진 유리가 복원되는 장면(irreversible). Detailed balance는 "동영상 앞·뒤 구별 불가"의 통계 버전.

### 왜 detailed balance가 정상을 보장

$\pi P = \pi$ 검증:
$$(\pi P)_j = \sum_i \pi_i P_{ij} = \sum_i \pi_j P_{ji} \text{ (DB)} = \pi_j \sum_i P_{ji} = \pi_j \cdot 1 = \pi_j.$$

**한 줄 증명**. Detailed balance는 **"global balance"**($\pi P = \pi$)의 **강한 충분조건**. 모든 $(i, j)$ 쌍에서 흐름이 맞으니, 총합에서도 맞음.

### Reversible vs Non-reversible

**Non-reversible 체인**도 정상분포를 가질 수 있음(global balance 만족하지만 detailed balance는 깨짐). 예: "원형 3-state 일방향 체인" $0 \to 1 \to 2 \to 0$. 정상분포 $\pi = (1/3, 1/3, 1/3)$이지만 $\pi_0 P_{01} = 1/3 \neq 0 = \pi_1 P_{10}$.

**MH 디자인 선택**: DB를 요구하는 것이 더 제약적이지만 설계가 쉽고 분석이 단순. Non-reversible MH는 효율 높을 수 있지만 이론이 복잡.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Detailed Balance

확률분포 $\pi$와 전이행렬 $P$가 **detailed balance**를 만족한다는 것은
$$\pi_i P_{ij} = \pi_j P_{ji}, \quad \forall i, j.$$

### 정의 5.2 — Reversible Markov Chain

Detailed balance를 만족하는 체인 $(P, \pi)$를 **reversible**이라 한다. 동치 조건: 정상 시작 체인 $\{X_n\}$의 분포가 $\{X_{N-n}\}$의 분포와 같음.

### 정의 5.3 — $\ell^2(\pi)$ Inner Product

함수 $f, g : E \to \mathbb{R}$에 대해
$$\langle f, g \rangle_\pi := \sum_i \pi_i f(i) g(i).$$

### 정의 5.4 — 전이연산자 $P$ on $\ell^2(\pi)$

$(Pf)(i) := \sum_j P_{ij} f(j) = \mathbb{E}_i[f(X_1)]$.

---

## 🔬 정리와 증명

### 정리 5.1 (Detailed Balance ⇒ Stationary)

$(P, \pi)$가 detailed balance를 만족하면 $\pi$는 $P$의 정상분포.

*증명.*
$$(\pi P)_j = \sum_i \pi_i P_{ij} = \sum_i \pi_j P_{ji} = \pi_j \sum_i P_{ji} = \pi_j.$$
$\square$

### 정리 5.2 (정상분포 ⇏ Detailed Balance)

일반적으로 $\pi P = \pi$이어도 DB는 만족하지 않는다.

*반례.* 3-state cyclic $P$:
$$P = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0 \end{pmatrix}, \quad \pi = (1/3, 1/3, 1/3).$$
$\pi P = \pi$ (쉬움) but $\pi_0 P_{01} = 1/3 \neq 0 = \pi_1 P_{10}$. $\square$

### 정리 5.3 (Reversible ⇔ $P$ is Self-adjoint on $\ell^2(\pi)$)

$(P, \pi)$가 reversible ⇔ 전이연산자 $P$가 $\ell^2(\pi)$ 위에서 self-adjoint.

*증명.* Self-adjoint: $\langle Pf, g\rangle_\pi = \langle f, Pg\rangle_\pi$, i.e.,
$$\sum_i \pi_i (Pf)(i) g(i) = \sum_i \pi_i f(i) (Pg)(i).$$

**(⇒)** DB로 좌변:
$$\sum_i \pi_i \sum_j P_{ij} f(j) g(i) = \sum_{i,j} \pi_i P_{ij} f(j) g(i) \stackrel{DB}{=} \sum_{i,j} \pi_j P_{ji} f(j) g(i) = \sum_j \pi_j f(j) \sum_i P_{ji} g(i) = \sum_j \pi_j f(j) (Pg)(j).$$

**(⇐)** Self-adjoint를 $f = \mathbf{1}_j, g = \mathbf{1}_i$ (indicator)에 적용하면 DB 회복. $\square$

### 정리 5.4 (Reversible 체인의 스펙트럴 성질)

$(P, \pi)$가 reversible이면 $P$는 $\ell^2(\pi)$ 위에서 self-adjoint compact operator (유한상태 가정). 따라서:
1. **고유값이 모두 실수**
2. **고유벡터가 $\ell^2(\pi)$ 직교**
3. 스펙트럴 분해 $P = \sum_k \lambda_k u_k u_k^T \text{diag}(\pi)$ 형태

*증명.* Self-adjoint 유한차원 연산자는 실 스펙트럼, 직교 고유공간 (스펙트럴 정리). $\square$

> **의미**: 복소 고유값이 없음 → 진동 모드 없음 → 분석 훨씬 단순.

### 정리 5.5 (Reversible 체인의 수렴률 — variational 표현)

Reversible 체인의 스펙트럴 gap $\gamma = 1 - \lambda_2$는 **Rayleigh quotient** 최소화:
$$\gamma = \min_{f : \mathbb{E}_\pi[f] = 0} \frac{\mathcal{E}(f, f)}{\|f\|_\pi^2},$$
여기서 $\mathcal{E}(f, f) = \frac{1}{2}\sum_{i,j} \pi_i P_{ij} (f(i) - f(j))^2$ — **Dirichlet form**.

*증명.* Self-adjoint 스펙트럴 분해와 Rayleigh-Ritz의 직접 적용. $\square$

이 variational form이 **Cheeger 부등식**, **Poincaré 부등식** 등 현대 MCMC 분석의 출발점.

### 정리 5.6 (Time-reversed chain)

$(P, \pi)$ reversible이고 $X_0 \sim \pi$, 그러면 $(X_0, X_1, \ldots, X_N) \stackrel{d}{=} (X_N, X_{N-1}, \ldots, X_0)$ (시간 뒤집기가 분포 불변).

*증명.*
$$\mathbb{P}(X_0 = i_0, \ldots, X_N = i_N) = \pi_{i_0} P_{i_0 i_1} \cdots P_{i_{N-1} i_N}.$$
DB $\pi_{i_0} P_{i_0 i_1} = \pi_{i_1} P_{i_1 i_0}$ 반복 적용:
$$= P_{i_1 i_0} \pi_{i_1} P_{i_1 i_2} \cdots = \cdots = \pi_{i_N} P_{i_N i_{N-1}} \cdots P_{i_1 i_0}.$$
이것이 $\mathbb{P}(X_0 = i_N, X_1 = i_{N-1}, \ldots, X_N = i_0)$. $\square$

---

## 💻 NumPy 구현 검증

### 실험 1 — Detailed balance로 정상분포 검증

```python
import numpy as np

# 예 1: Random walk on {0,1,2,3,4} with reflecting boundaries, reversible
N = 5
P = np.zeros((N, N))
for i in range(N):
    if i == 0:
        P[i, 0] = 0.5; P[i, 1] = 0.5
    elif i == N-1:
        P[i, N-2] = 0.5; P[i, N-1] = 0.5
    else:
        P[i, i-1] = 0.5; P[i, i+1] = 0.5

# 정상분포 uniform (대칭 random walk)
pi = np.ones(N) / N

# DB 체크
print('Detailed balance 체크:')
for i in range(N):
    for j in range(N):
        if P[i, j] > 0:
            lhs = pi[i] * P[i, j]
            rhs = pi[j] * P[j, i]
            print(f'π_{i} P_{{i,j}}={lhs:.3f}, π_{j} P_{{j,i}}={rhs:.3f}, '
                  f'equal={np.isclose(lhs, rhs)}')
```

### 실험 2 — Non-reversible 예 (cyclic)

```python
# 3-cycle: 0→1→2→0 (일방향)
P_cycle = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
])
pi = np.ones(3) / 3

print('정상성: π P =', pi @ P_cycle)  # π 유지

# DB 실패
for i in range(3):
    for j in range(3):
        lhs = pi[i] * P_cycle[i, j]
        rhs = pi[j] * P_cycle[j, i]
        if lhs != rhs:
            print(f'DB 실패: π_{i} P_{{i,j}}={lhs}, π_{j} P_{{j,i}}={rhs}')
            break
```

### 실험 3 — Reversible 체인의 실고유값

```python
# 3-state reversible (detailed balance)
pi = np.array([0.5, 0.3, 0.2])
P_sym = np.array([    # 먼저 raw rates 설계
    [0.4, 0.4, 0.2],
    [0.6, 0.2, 0.2],  # 아직 DB 아님
    [0.3, 0.1, 0.6],
])

# DB 맞게 수정: Q_{ij} = min(π_i P_{ij}, π_j P_{ji}) / π_i (Metropolis-style)
P = np.zeros_like(P_sym)
for i in range(3):
    for j in range(3):
        if i != j:
            rate = min(P_sym[i, j], pi[j] * P_sym[j, i] / pi[i])
            P[i, j] = rate
    P[i, i] = 1 - P[i, :].sum()

# 고유값 확인 (실수여야)
eigvals = np.linalg.eigvals(P)
print('Reversible P의 고유값:', eigvals)
print('모두 실수?', np.all(np.abs(eigvals.imag) < 1e-10))

# Non-reversible 비교
eigvals_cyc = np.linalg.eigvals(P_cycle)
print('Cyclic P의 고유값:', eigvals_cyc)
print('복소 고유값 존재?', np.any(np.abs(eigvals_cyc.imag) > 1e-6))
# → reversible은 실수, cyclic은 복소 고유값 (1의 단위근)
```

---

## 🔗 AI/ML 연결

**Metropolis-Hastings의 acceptance**  
$\alpha(x, y) = \min(1, \pi(y)q(x|y)/\pi(x)q(y|x))$의 핵심:
- Proposal $q$와 accept $\alpha$를 합친 전이 kernel $P(x, y) = q(y|x) \alpha(x, y)$
- $\pi(x) q(y|x) \alpha(x, y) = \pi(y) q(x|y) \alpha(y, x)$로 **detailed balance 자동 성립**
- 정리 5.1에 의해 $\pi$가 정상 → MH 수렴 (Ch7-02)

**Langevin dynamics의 reversibility**  
$dX = -\nabla U dt + \sqrt{2\beta^{-1}} dB$의 정상분포는 $\pi \propto e^{-\beta U}$. Fokker-Planck 연산자가 $L^2(\pi)$에서 self-adjoint ⇔ reversibility. Euler-Maruyama 이산화 (SGLD)가 이 구조를 근사 보존.

**Gibbs Sampling의 coordinate-wise DB**  
$x \to x' = (x_1, \ldots, x_{i-1}, y_i, x_{i+1}, \ldots)$ — 한 좌표만 조건부 $p(y_i | x_{-i})$에서 샘플. Per-coordinate DB:
$$\pi(x) p(y_i | x_{-i}) = \pi(x') p(x_i | x_{-i})$$
(둘 다 $\pi(x_{-i}) p(x_i | x_{-i}) p(y_i | x_{-i})$로 factorize).

**Reversible Jump MCMC (Green 1995)**  
다른 dimension space 간 sampling에서도 DB 확장 → Jacobian factor 포함. 모델 선택에 활용.

**Non-reversible MCMC (현대 연구)**  
"Lifted chain" 구성 → DB 깨지만 mixing 더 빠른 알고리즘. 예: Piecewise Deterministic MC (Bouncy Particle, Zigzag). Reversible의 약점(확산 기반 slow mixing) 극복 시도.

---

## ⚖️ 가정과 한계

**가정 — DB는 "sufficient, not necessary"**  
Global balance $\pi P = \pi$만 있어도 정상분포. DB는 **더 강한 조건**. 현실 체인 중 DB를 만족하는 것은 일부. 그러나 MH 설계 시 DB를 요구해 **올바름 보장이 쉬움**.

**한계 — Reversibility가 non-trivial to check**  
그래프 위 random walk, reversible physics 시스템 등은 자연스럽게 reversible. 그러나 임의 체인이 reversible인지 알려면 모든 $(i, j)$ 쌍에서 DB 확인 — $O(N^2)$.

**한계 — Non-reversible이 더 효율적일 수 있음**  
Non-reversible 체인이 기약·비주기이면 수렴하지만, spectral analysis가 복소수 → 진동 있을 수 있음. 그러나 diffusive random walk의 "되돌아감 경향"을 극복 → 고차원에서 mixing 가속 가능 (Hwang, Hwang-Ma-Sheu 1993).

---

## 📌 핵심 정리

| 개념 | 수식 / 의미 |
|---|---|
| Detailed balance | $\pi_i P_{ij} = \pi_j P_{ji}$ |
| DB ⇒ Stationary | 자동 (한 줄 증명) |
| Reversible | DB 만족하는 $(P, \pi)$ |
| Self-adjoint on $\ell^2(\pi)$ | $\langle Pf, g\rangle_\pi = \langle f, Pg\rangle_\pi$ ⇔ reversible |
| 스펙트럴 성질 | 실고유값, 직교 고유벡터 |
| Time reversal | Reversible ⇔ $(X_0, \ldots, X_N) \stackrel{d}{=} (X_N, \ldots, X_0)$ |
| Dirichlet form | $\mathcal{E}(f, f) = \frac{1}{2}\sum \pi_i P_{ij}(f(i) - f(j))^2$ |

**한 줄 요약**: Detailed balance $\pi_i P_{ij} = \pi_j P_{ji}$는 정상분포의 **충분 조건**으로, 한 줄 증명이 자동. 이 조건이 **Metropolis-Hastings의 설계 원리**이자, 전이연산자의 **self-adjoint 구조**로 스펙트럴 분석을 단순화한다.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. $N$-state random walk on cycle ($i \to i+1 \mod N$ with prob 1/2, $i-1 \mod N$ with prob 1/2). 이 체인이 reversible인가? $\pi$는?

<details>
<summary>해설</summary>

$P_{i, i+1} = P_{i, i-1} = 1/2$. 대칭.

정상분포: 대칭성으로 $\pi = 1/N$ (uniform).

DB: $\pi_i P_{i, j} = (1/N)(1/2) = 1/(2N) = \pi_j P_{j, i}$ (같은 값). → **DB 성립 → Reversible**.

**직관**: "균일 분포 + 대칭 이동"이면 DB는 자동. 이것이 symmetric proposal (RWMH)의 MH acceptance가 간단해지는 이유.

**주의**: 주기성 체크 — $N$ 짝수면 주기 2, 홀수면 비주기. DB는 주기성과 별개 — reversible + 주기 = 실고유값 포함 $-1$ (마지막 고유값이 $-1$).

</details>

**문제 2 (심화)**. Reversible 체인 $(P, \pi)$가 주어지고, $f : E \to \mathbb{R}$이 고유함수 $Pf = \lambda f$ ($\lambda \neq 1$)일 때 $\mathbb{E}_\pi[f] = 0$임을 보여라.

<details>
<summary>해설</summary>

$\lambda \neq 1$에 대응하는 고유함수 $f$와 상수 함수 $\mathbf{1}$(고유값 1)이 self-adjoint operator의 서로 다른 eigenvalue이므로 $\ell^2(\pi)$에서 **직교**:
$$\langle f, \mathbf{1}\rangle_\pi = 0 \Rightarrow \sum_i \pi_i f(i) = \mathbb{E}_\pi[f] = 0.$$

**함의**: 수렴률 분석에서 $(\mu_0 - \pi) f = \mu_0 f - \mathbb{E}_\pi f$ 등의 표현이 **mean-zero 고유함수들의 합**으로 표현됨 — 스펙트럴 분해의 기반.

**연결**: Ch2-04 수렴률 증명에서 $\mu_0 - \pi = \sum_{k \geq 2} c_k u_k$로 decomposition, 각 $u_k$가 mean-zero 고유함수 → $(\mu_0 - \pi) P^n = \sum_k c_k \lambda_k^n u_k \to 0$ at rate $|\lambda_2|^n$.

</details>

**문제 3 (AI 연결)**. Gibbs sampler for 2D Ising model: spin $s_i \in \{\pm 1\}$, energy $H(s) = -\beta \sum_{\langle i,j\rangle} s_i s_j$. Coordinate-wise update $s_i \to s_i'$ with $P(s_i' | s_{-i}) \propto e^{-\beta s_i' \sum_{j \sim i} s_j}$. 이 Gibbs sampler가 detailed balance with Boltzmann $\pi(s) \propto e^{-\beta H(s)}$를 만족함을 보여라.

<details>
<summary>해설</summary>

**Gibbs transition**: 한 좌표 $i$만 update, $s \to s' = (s_{-i}, s_i')$.
$$P(s \to s') = \frac{e^{-\beta s_i' \sum_{j \sim i} s_j}}{Z_i(s_{-i})}, \quad Z_i(s_{-i}) = e^{\beta \sum_{j \sim i} s_j} + e^{-\beta \sum_{j \sim i} s_j}.$$

**DB 검증** $\pi(s) P(s \to s') = \pi(s') P(s' \to s)$:

LHS = $\frac{e^{-\beta H(s)}}{Z} \cdot \frac{e^{-\beta s_i' \sum_{j \sim i} s_j}}{Z_i(s_{-i})}$.

$H(s) = H(s_{-i}) - s_i \sum_{j \sim i} s_j$ (기여분 분리), 여기서 $H(s_{-i})$는 $s_i$에 의존 안 함. 따라서
$$e^{-\beta H(s)} = e^{-\beta H(s_{-i})} \cdot e^{\beta s_i \sum_{j \sim i} s_j}.$$

LHS = $\frac{e^{-\beta H(s_{-i})}}{Z} \cdot \frac{e^{\beta s_i \sum_{j \sim i} s_j} \cdot e^{-\beta s_i' \sum_{j \sim i} s_j}}{Z_i(s_{-i})} = \frac{e^{-\beta H(s_{-i})}}{Z Z_i(s_{-i})} \cdot e^{\beta (s_i - s_i') \sum_{j \sim i} s_j}$.

RHS 대칭으로: $\frac{e^{-\beta H(s_{-i})}}{Z Z_i(s_{-i})} \cdot e^{\beta (s_i' - s_i) \sum_{j \sim i} s_j} \cdot e^{\beta s_i' \sum_{j \sim i} s_j} \cdot e^{-\beta s_i \sum_{j \sim i} s_j}$

실제로 양변 계산하면: LHS = $\frac{e^{-\beta H(s_{-i})}}{Z Z_i} e^{\beta s_i \sum s_j - \beta s_i' \sum s_j}$, RHS = $\frac{e^{-\beta H(s_{-i})}}{Z Z_i} e^{\beta s_i' \sum s_j - \beta s_i \sum s_j}$

어라, 이거 부호만 뒤집힘. 아니, 다시 계산:

$\pi(s) P(s \to s') = \pi(s_{-i}, s_i) \cdot \frac{e^{-\beta s_i' M}}{Z_i}$ where $M = \sum_{j \sim i} s_j$.

$\pi(s) = \frac{1}{Z} e^{-\beta H(s_{-i})} e^{\beta s_i M}$ (${}H$의 $i$ 기여 $-s_i M$).

LHS = $\frac{1}{Z} e^{-\beta H(s_{-i})} e^{\beta s_i M} \cdot \frac{e^{-\beta s_i' M}}{Z_i(M)}$  
    $= \frac{e^{-\beta H(s_{-i})}}{Z Z_i(M)} e^{\beta (s_i - s_i') M}$.

RHS = $\pi(s') P(s' \to s) = \frac{1}{Z} e^{-\beta H(s_{-i})} e^{\beta s_i' M} \cdot \frac{e^{-\beta s_i M}}{Z_i(M)}$  
    $= \frac{e^{-\beta H(s_{-i})}}{Z Z_i(M)} e^{\beta (s_i' - s_i) M}$.

어, LHS와 RHS의 지수가 반대 부호다. 재검토:

**정정**: $\pi(s) \propto e^{-\beta H(s)}$, $H(s) = -\sum s_i s_j$ (기호 주의 — 음수 에너지 선호). $s_i$ 항 $= -s_i \sum_{j \sim i} s_j = -s_i M$. $e^{-\beta H}$에서 $e^{\beta s_i M}$ 나옴.

Gibbs conditional: $P(s_i' | s_{-i}) \propto e^{-\beta \cdot (-s_i' M)} = e^{\beta s_i' M}$. 정규화로 $P(s_i' | s_{-i}) = e^{\beta s_i' M} / Z_i(M)$.

$P(s \to s') = P(s_i' | s_{-i}) = e^{\beta s_i' M}/Z_i$.

LHS = $\pi(s) P(s \to s') = e^{-\beta H(s_{-i})} e^{\beta s_i M}/Z \cdot e^{\beta s_i' M}/Z_i$  
$= e^{-\beta H(s_{-i})} e^{\beta (s_i + s_i') M} / (Z Z_i)$.

RHS = $\pi(s') P(s' \to s) = e^{-\beta H(s_{-i})} e^{\beta s_i' M}/Z \cdot e^{\beta s_i M}/Z_i$  
$= e^{-\beta H(s_{-i})} e^{\beta (s_i + s_i') M} / (Z Z_i)$.

**LHS = RHS** ✓ → DB 성립.

**핵심 통찰**: Gibbs sampler는 조건부 $\pi(s_i | s_{-i})$에서 샘플하므로 **acceptance = 1인 MH의 특수 경우**(Ch7-03). DB는 이 "조건부 정확 샘플링" 구조에서 자연스럽게 나옴.

**실전**: Boltzmann machine 학습의 CD-k (Contrastive Divergence) 알고리즘이 이 Gibbs DB를 유한 step으로 근사.

</details>

---

<div align="center">

◀ [04. 극한정리와 수렴률 — 스펙트럴 접근](./04-convergence-rate.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [06. 에르고딕 정리(Ergodic Theorem)](./06-ergodic-theorem.md)

</div>
