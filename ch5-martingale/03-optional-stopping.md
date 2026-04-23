# 03. Optional Stopping Theorem

## 🎯 핵심 질문

- **Optional Stopping Theorem**: 정지시각 $\tau$에서 $\mathbb{E}[X_\tau] = \mathbb{E}[X_0]$은 어떤 조건에서 성립하는가?
- 세 가지 **충분조건**(bounded $\tau$, bounded $X$, uniformly integrable)이 각각 어떻게 작동하는가?
- **도박장 파산 문제**(Gambler's ruin) — 대칭 random walk가 $\{0, N\}$에 도달할 확률을 어떻게 유도?
- **Wald's identity**가 이 정리의 대표적 응용인 이유는?

---

## 🔍 왜 이 정리가 AI에서 중요한가

**Gambler's ruin → Explore-Exploit**: Bandit problem의 "언제 멈춰 exploit할지" decision에 optional stopping 이론. Upper confidence bound 알고리즘의 분석 도구.

**MCMC convergence diagnostics**: "burn-in을 얼마나 길게"에 대한 이론적 답이 stopping time 이론에 기반. Geweke test, CUSUM test 등.

**Reinforcement Learning episode termination**: Episode length이 random (정지시각) → total reward의 기댓값 계산에 optional stopping.

**Sequential Hypothesis Testing (Wald)**: A/B test의 optimal stopping rule — Wald의 SPRT가 이 정리의 직접 응용.

**Deep learning early stopping**: Validation loss가 martingale-like → optimal stopping rule 이론적 접근.

---

## 📐 수학적 선행 조건

- [Ch5-01, Ch5-02](./01-martingale-definition.md): 마팅게일, 수렴 정리
- [Ch1-04](../ch1-foundations/04-filtration.md): Stopping time 정의
- Dominated convergence

---

## 📖 직관적 이해

### "정지해도 공정성이 유지되는가"

마팅게일은 "공정한 게임" — $\mathbb{E}[X_n] = \mathbb{E}[X_0]$. **질문**: 특정 시점 $\tau$(예: "이익 $100 도달")에 멈추면 여전히 공정?

**답**: **일반적으로는 아니다**. 예: Simple random walk $S_n$, $\tau = \inf\{n : S_n = 10\}$. 이 $\tau$는 a.s. 유한 ($S_n$ recurrent), 그러나 $\mathbb{E}[S_\tau] = 10 \neq 0 = \mathbb{E}[S_0]$. 

**문제**: $\tau$가 a.s. 유한이지만 $\mathbb{E}[\tau] = \infty$. "무한히 기다리면 어떤 경우든 100 도달" — 하지만 그 기다림 평균이 무한.

**해결**: 3가지 충분조건으로 등식 보장.

### 세 가지 충분조건

1. **Bounded stopping time**: $\tau \leq N$ (deterministic bound). "유한 시간 안에 반드시 멈춤."

2. **Bounded martingale**: $|X_n| \leq C$ for all $n$. "게임이 한계 안에서만 움직임."

3. **Uniformly integrable martingale**: $\{X_{n \wedge \tau}\}$ UI. 가장 일반적 조건.

### 도박장 파산 문제

도박꾼이 $\$x$로 시작, 매 판 $\pm 1$ (확률 1/2). $\$0$ 또는 $\$N$ 도달 시 멈춤. 확률 $P_x(\$N \text{ 먼저})$?

**답**: $P_x = x/N$ (linear).

**Optional stopping 증명**: $\tau = \inf\{n : S_n \in \{0, N\}\}$. $S_n$ martingale, $\tau < \infty$ a.s., $|S_n| \leq N$ (bounded martingale). 조건 (2) 적용:
$$\mathbb{E}[S_\tau] = \mathbb{E}[S_0] = x.$$
또 $\mathbb{E}[S_\tau] = N \cdot P_x(N) + 0 \cdot P_x(0) = N P_x$. 등식: $P_x = x/N$.

**우아**: Martingale + OST가 즉시 결과 제공. 직접 계산(원래 difference equation)보다 훨씬 간결.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — 정지시각 (Stopping Time)

$\tau : \Omega \to \mathbb{N} \cup \{\infty\}$ — 각 $n$에 대해 $\{\tau \leq n\} \in \mathcal{F}_n$.

**직관**: "시각 $n$까지의 정보만으로 $\tau \leq n$인지 결정 가능". 미래를 보지 않음.

### 정의 3.2 — Stopped Process

$X_n^\tau := X_{\min(n, \tau)}$.

성질: adapted, martingale이 자동 (정리 1의 문제 2 참조).

---

## 🔬 정리와 증명

### 정리 3.1 — Optional Stopping Theorem (세 충분조건)

$\{X_n\}$ martingale, $\tau$ stopping time. 다음 중 하나가 성립하면 $\mathbb{E}[X_\tau] = \mathbb{E}[X_0]$:

**(1) Bounded $\tau$**: $\exists N$ s.t. $\tau \leq N$ a.s.

**(2) Bounded $X$**: $\exists C$ s.t. $|X_n| \leq C$ for all $n$ and $\tau < \infty$ a.s.

**(3) UI martingale**: $\{X_{n \wedge \tau}\}$ UI, $\mathbb{E}[\tau] < \infty$, $\mathbb{E}[|X_{n+1} - X_n| \mathbf{1}_{\tau > n}] \leq C$ for bounded $C$.

### 증명 스케치

**(1) Bounded $\tau$**:
$$\mathbb{E}[X_\tau] = \sum_{k=0}^N \mathbb{E}[X_k \mathbf{1}_{\{\tau = k\}}].$$

Martingale 성질로 $\mathbb{E}[X_N \mathbf{1}_{\{\tau = k\}}] = \mathbb{E}[\mathbb{E}[X_N | \mathcal{F}_k] \mathbf{1}_{\{\tau = k\}}] = \mathbb{E}[X_k \mathbf{1}_{\{\tau = k\}}]$. 합:
$$\mathbb{E}[X_\tau] = \sum_k \mathbb{E}[X_N \mathbf{1}_{\{\tau = k\}}] = \mathbb{E}[X_N] = \mathbb{E}[X_0].$$

**(2) Bounded $X$**: Stopped martingale $X_{n \wedge \tau}$ bounded (by $C$) → $L^1$-bounded → UI → DCT 적용으로
$$\mathbb{E}[X_\tau] = \mathbb{E}[\lim_n X_{n \wedge \tau}] = \lim_n \mathbb{E}[X_{n \wedge \tau}] = \mathbb{E}[X_0].$$

**(3) UI**: UI + a.s. 수렴 → $L^1$ 수렴 → 동등. $\square$

### 정리 3.2 — Gambler's Ruin

Simple random walk $S_n$ starting at $x$, $\tau = \inf\{n : S_n = 0 \text{ or } N\}$, $0 \leq x \leq N$. 그러면:
1. $\tau < \infty$ a.s.
2. $P_x(S_\tau = N) = x/N$
3. $\mathbb{E}_x[\tau] = x(N - x)$

*증명*.

**(1) $\tau < \infty$ a.s.**: Recurrent random walk이 $\{0, N\}$ hit.

**(2) Hitting probability**: $S_n$ martingale, $|S_n| \leq N$ on $\{n \leq \tau\}$ (bounded), $\tau < \infty$ a.s. → OST 조건 (2):
$\mathbb{E}[S_\tau] = x$. 그리고 $\mathbb{E}[S_\tau] = N \cdot p + 0 \cdot (1-p)$ where $p = P_x(S_\tau = N)$. → $p = x/N$.

**(3) Expected time**: $M_n = S_n^2 - n$ martingale (정리 2의 문제 2). Bounded on $\{n \leq \tau\}$ (by $N^2 + \tau$ — 조건 3 필요). OST:
$\mathbb{E}[S_\tau^2] - \mathbb{E}[\tau] = \mathbb{E}[M_\tau] = M_0 = x^2$.
$\mathbb{E}[S_\tau^2] = N^2 p + 0 = N^2 (x/N) = Nx$. 따라서 $\mathbb{E}[\tau] = Nx - x^2 = x(N - x)$. $\square$

**예**: $x = 50, N = 100$ → $\mathbb{E}[\tau] = 50 \cdot 50 = 2500$ step. 게임 끝까지 평균 2500 판.

### 정리 3.3 — Wald's Identity

$\{\xi_k\}$ iid with $\mathbb{E}\xi = \mu$. $S_n = \sum_{k=1}^n \xi_k$. $\tau$ stopping time with $\mathbb{E}[\tau] < \infty$, $\mathbb{E}|\xi| < \infty$. 그러면:
$$\mathbb{E}[S_\tau] = \mu \mathbb{E}[\tau].$$

*증명*. $M_n = S_n - n\mu$가 martingale (iid from mean $\mu$). OST (with 조건 3 version)로 $\mathbb{E}[M_\tau] = 0$ → $\mathbb{E}[S_\tau] = \mu \mathbb{E}[\tau]$. $\square$

**응용**: 예상 게임 길이 × per-round 기댓값 = 총 보상 기댓값. Insurance, risk management, RL episode length 분석.

### 정리 3.4 — Wald's Second Identity (분산 버전)

$\{\xi_k\}$ iid mean 0, variance $\sigma^2$. $\tau$ stopping with $\mathbb{E}[\tau] < \infty$. 그러면:
$$\mathbb{E}[S_\tau^2] = \sigma^2 \mathbb{E}[\tau].$$

*증명*. $S_n^2 - n\sigma^2$ martingale. OST → $\mathbb{E}[S_\tau^2] - \sigma^2 \mathbb{E}[\tau] = 0$. $\square$

---

## 💻 NumPy 구현 검증

### 실험 1 — Gambler's Ruin

```python
import numpy as np
rng = np.random.default_rng(0)

x_start, N = 30, 100
n_sim = 10000

hit_N_count = 0
times = []
for _ in range(n_sim):
    S = x_start
    n = 0
    while 0 < S < N:
        S += rng.choice([-1, 1])
        n += 1
    if S == N:
        hit_N_count += 1
    times.append(n)

p_hat = hit_N_count / n_sim
p_theory = x_start / N
E_tau_hat = np.mean(times)
E_tau_theory = x_start * (N - x_start)

print(f'P(hit N) 실측: {p_hat:.4f}, 이론: {p_theory:.4f}')
print(f'E[τ] 실측:      {E_tau_hat:.1f}, 이론: {E_tau_theory}')
# 일치 확인
```

### 실험 2 — Biased random walk

```python
# P(+1) = p, P(-1) = q, p + q = 1, p ≠ 1/2
# S_n 자체는 martingale 아님 (bias)
# 하지만 (q/p)^{S_n}이 martingale!
# E[(q/p)^{S_n+1} | F_n] = (q/p)^{S_n} (p(q/p) + q(p/q)) = (q/p)^{S_n} (q + p) = (q/p)^{S_n}

p, q = 0.55, 0.45
x_start, N = 20, 50
r = q / p   # < 1

n_sim = 10000
hit_N = 0
for _ in range(n_sim):
    S = x_start
    while 0 < S < N:
        S += rng.choice([-1, 1], p=[q, p])
    if S == N:
        hit_N += 1

p_hit_N = hit_N / n_sim
# 이론: E[r^{S_τ}] = r^{x_start}
# r^N P + r^0 (1-P) = r^{x_start}
# P = (r^{x_start} - 1) / (r^N - 1) = (1 - r^{x_start}) / (1 - r^N) for r < 1
P_theory = (1 - r**x_start) / (1 - r**N)
print(f'Biased RW, p={p}:')
print(f'  P(hit N) 실측: {p_hit_N:.4f}, 이론: {P_theory:.4f}')
# p > 0.5면 → P가 선형 x/N보다 훨씬 큼 (편향 유리)
```

### 실험 3 — Wald's Identity 검증

```python
# ξ_k ~ Bernoulli(0.3) - 0.3 (mean 0, var = 0.21)
# τ = inf{n : S_n ≤ -5 or S_n ≥ 5}
mu_xi, var_xi = 0, 0.21
sigma = np.sqrt(var_xi)

n_sim = 10000
tau_samples = []
S_tau_samples = []

for _ in range(n_sim):
    S = 0; n = 0
    while -5 < S < 5:
        S += rng.choice([-0.3, 0.7], p=[0.7, 0.3])
        n += 1
        if n > 10000: break
    tau_samples.append(n)
    S_tau_samples.append(S)

E_tau = np.mean(tau_samples)
E_S_tau_sq = np.mean(np.array(S_tau_samples)**2)

print(f'E[τ] = {E_tau:.2f}')
print(f'E[S_τ²] = {E_S_tau_sq:.4f}')
print(f'E[S_τ²] / σ² = {E_S_tau_sq / var_xi:.2f} (Wald 2nd: E[τ])')
# 일치 → Wald's 2nd identity 검증
```

---

## 🔗 AI/ML 연결

**Bandit Problems — UCB**  
Upper Confidence Bound의 "stopping rule" — 특정 arm이 충분히 explored되면 stop exploring. Regret bound 증명에 OST + Azuma (Ch5-05) 사용.

**Sequential Probability Ratio Test (SPRT, Wald)**  
두 가설 $H_0$ vs $H_1$의 sequential test. Log-likelihood ratio가 random walk, "boundary hit" 시 결정. Expected sample size가 Wald's identity로 유도.

**Early Stopping in NN Training**  
Validation loss가 최소값 upcrossing 시 stop. Optional stopping 이론이 "이 stopping rule이 biased인가"의 분석 제공.

**Reinforcement Learning — Episode termination**  
Episode $\tau = \inf\{t : \text{terminal}\}$. Total return $\sum_{t < \tau} r_t$의 기댓값 계산에 Wald's identity 확장(state-dependent rewards).

**Quickest Change Detection**  
CUSUM 알고리즘 — martingale이 threshold 넘으면 "change detected". OST가 false alarm rate과 detection delay의 trade-off 분석.

---

## ⚖️ 가정과 한계

**한계 — 조건 실패 시 OST 실패**  
Random walk 예에서 $\tau = \inf\{n : S_n = 10\}$ 만 고려하면 $\mathbb{E}[S_\tau] = 10 \neq 0$. "상한 없음"이 OST 조건을 깸. 실전에서 "정지 조건" 설계 시 주의.

**한계 — 연속시간 확장**  
CTMC, BM에서도 OST 비슷하지만 추가 정칙성 필요 (우연속 경로, 적절한 filtration).

**한계 — Estimation bias**  
Early stopping으로 얻은 estimator는 biased. "Unbiased estimator가 아님"을 인지해야 (debiasing techniques 필요).

---

## 📌 핵심 정리

| 정리 | 조건 | 결론 |
|---|---|---|
| OST (1) | $\tau \leq N$ | $\mathbb{E}[X_\tau] = \mathbb{E}[X_0]$ |
| OST (2) | $\|X_n\| \leq C$, $\tau < \infty$ | 같음 |
| OST (3) | UI + $\mathbb{E}[\tau] < \infty$ + bounded increments | 같음 |
| Wald 1st | iid, $\mathbb{E}\tau < \infty$ | $\mathbb{E}[S_\tau] = \mu \mathbb{E}[\tau]$ |
| Wald 2nd | iid mean 0, $\mathbb{E}\tau < \infty$ | $\mathbb{E}[S_\tau^2] = \sigma^2 \mathbb{E}[\tau]$ |
| Gambler's ruin | RW, $\{0, N\}$ | $P_x = x/N$, $\mathbb{E}\tau = x(N-x)$ |

**한 줄 요약**: Optional Stopping은 "정지시각까지 마팅게일 성질 보존"의 정리. 세 가지 충분조건(bounded time, bounded process, UI) 중 하나 필요. Gambler's ruin·Wald identity가 대표 응용, 현대 ML의 SPRT·UCB·early stopping에 활용.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. 공정한 동전 던지기에서 "앞면 3번 연속"이 나올 때까지의 평균 던지기 수를 martingale로 구하라.

<details>
<summary>해설</summary>

**카지노 해석 (Dubin 방법)**: "$\$1씩 connecting bets" 기법.

각 시각 $n$에 새 도박꾼이 \$1 가지고 와서 "다음 $k$번 앞면 = \$2^k"에 베팅 ($k = 1, 2, 3$). 연속 앞면 나올 때마다 자산 곱. 실패 시 0.

총 자산 $M_n$ = martingale (공정 베팅의 합).

$\tau$ = "앞면 3연속" 첫 발생 시각. $\tau$에서:
- 가장 최근 3명 도박꾼이 연속 성공 → $\$8 + \$4 + \$2 = \$14$
- 그전 도박꾼들은 실패 → \$0
$M_\tau = 14$.

$\mathbb{E}[M_\tau] = \mathbb{E}[M_0] = 0$? 사실 각 도박꾼이 $\$1$로 들어와서 $\sum = \tau$ (시각 $\tau$까지 $\tau$명).

정확한 setup: $M_n = $ (총 투입 - 현재 자산). Martingale. $\mathbb{E}[M_\tau] = 0$ ⇒ 투입 = 자산 ⇒ $\mathbb{E}[\tau] \cdot 1 = \mathbb{E}[14] = 14$.

$\mathbb{E}[\tau] = 14$.

**검증**: Direct 계산 (재귀 방정식)으로도 $2 + 2^2 + 2^3 = 14$.

**일반화**: $k$번 연속 앞면까지의 평균 = $2 + 2^2 + \cdots + 2^k = 2^{k+1} - 2$.

</details>

**문제 2 (심화)**. Simple random walk $S_n$, $\tau_a = \inf\{n : S_n = a\}$ ($a > 0$). $\mathbb{E}[\tau_a]$는 무엇이고, 왜 OST가 직접 적용 안 되는가?

<details>
<summary>해설</summary>

**답**: $\mathbb{E}[\tau_a] = \infty$.

**왜**:
$S_n$ martingale. 만약 OST 적용 가능하면 $\mathbb{E}[S_{\tau_a}] = 0$. 그러나 $S_{\tau_a} = a$ (도달했으니) → $a = 0$, 모순 (if $a \neq 0$).

결론: OST **조건 실패**. $\tau_a$가 a.s. 유한하긴 하지만 (recurrent), $\mathbb{E}[\tau_a] = \infty$. OST condition (1) bounded $\tau$ 실패, (2) bounded $X$ 실패, (3) UI 실패.

**구체적으로**: $\mathbb{E}[\tau_a] = \infty$ 증명 — Wald 1st: 만약 $\mathbb{E}[\tau_a] < \infty$이면 $\mathbb{E}[S_{\tau_a}] = 0 \cdot \mathbb{E}[\tau_a] = 0$. 그러나 $S_{\tau_a} = a$. 모순 → $\mathbb{E}[\tau_a] = \infty$.

**해석**: "Random walk는 결국 $a$에 도달하지만, 평균 도달 시간이 무한" — $\tau_a$의 분포가 heavy tail (sub-exponential).

**양방향 정지** ($\tau = \tau_a \wedge \tau_{-b}$): bounded martingale → OST 적용 가능, $\mathbb{E}[\tau] = ab$ (Gambler's ruin 변형).

**교훈**: **양방향 bounded stopping**이 OST 조건 만족. 한 방향 hitting은 문제 소지.

</details>

**문제 3 (AI 연결)**. Sequential A/B testing에서 Wald's SPRT: 두 variant 중 우수한 것을 빠르게 결정. "Early stopping bias"를 martingale 관점에서 설명하라.

<details>
<summary>해설</summary>

**SPRT Setup**: 각 샘플 $X_i$ 관찰. $H_0$: rate $p_0$ vs $H_1$: rate $p_1$. Log-likelihood ratio $L_n = \sum_i \log\frac{p_1(X_i)}{p_0(X_i)}$.

Decision:
- $L_n \geq A$: accept $H_1$
- $L_n \leq -B$: accept $H_0$
- 그 외: 계속

**Martingale**: $H_0$ 하에서 $e^{L_n}$이 martingale ($\mathbb{E}[e^{L}] = \int p_1 = 1$).

**Early stopping bias**: Naive estimator $\hat p = \bar X_\tau = \frac{1}{\tau}\sum X_i$에서 $\mathbb{E}[\hat p] \neq p$ (true rate).

**왜**: $\tau$는 stopping time (stochastic). $\bar X_n$이 martingale-like이지만 $\tau$가 data에 의존 → $\hat p = \bar X_\tau$가 unbiased 아님. 구체적으로 "advantageous stopping" — 운 좋은 early data에 의해 stop → $\hat p$가 과대평가.

**수학적**:
iid $X$ with mean $\mu$. $\tau$ stopping time. $\mathbb{E}[\bar X_\tau] = \mu$ (iid이면 Wald ratio $\mathbb{E}[S_\tau / \tau]$ 형태).

그러나 $\tau$가 $X$에 의존하면 $\bar X_\tau$와 iid sample의 분포가 다름. Sequential procedure는 **correlated** — 나중 관측이 "cross threshold" 조건에 의존.

**Unbiased estimation**:
- **Inverse probability weighting**: 관측 시 stopping 기여도 감안.
- **Likelihood-based**: Conditional MLE given $\tau$.
- **Bootstrap/resampling**: Simulation으로 bias estimate.

**Deep learning 적용**:
- **Early stopping on val loss**: stopped model의 "true" generalization이 실제로 early stopping point보다 worse (pessimistic estimate).
- **Hyperparameter tuning with successive halving**: Top performing configurations가 over-optimistic — bias correction needed.

**연결**: OST는 정확한 equality 주는 반면, **sequential decision-making**에서는 조건 깨져 bias 발생. 이를 인지하고 correction 필요. 현대 adaptive experimental design (Thompson sampling, Bayesian bandits)이 이 bias를 체계적으로 처리.

</details>

---

<div align="center">

◀ [02. 마팅게일 수렴 정리](./02-convergence-theorem.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [04. Doob 분해와 이차변분](./04-doob-decomposition.md)

</div>
