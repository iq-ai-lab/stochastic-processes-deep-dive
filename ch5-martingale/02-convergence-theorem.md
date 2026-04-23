# 02. 마팅게일 수렴 정리

## 🎯 핵심 질문

- **Doob의 $L^1$ bounded martingale convergence**: $\sup_n \mathbb{E}|X_n| < \infty$이면 $X_n \to X_\infty$ **a.s.**? 왜?
- **Upcrossing inequality** $(b-a)\mathbb{E}[U_n(a, b)] \leq \mathbb{E}[(X_n - a)^+]$가 수렴 증명의 핵심인 이유는?
- 음이 아닌 **supermartingale**이 a.s. 수렴하는 이유 — 가장 단순한 수렴 결과?
- $L^1$ 수렴과 a.s. 수렴의 차이 — uniformly integrable (UI) 조건이 왜 중요한가?

---

## 🔍 왜 이 정리가 AI에서 중요한가

**Stochastic Gradient Descent 수렴**: Robbins-Monro의 핵심 도구가 마팅게일 수렴. Learning rate schedule이 supermartingale을 형성하도록 설계.

**Stochastic Approximation 일반**: Q-learning, Actor-Critic, EM의 iterative 과정의 수렴 분석이 모두 마팅게일 수렴 정리 호출.

**Martingale Central Limit**: 수렴하는 martingale의 fluctuation 분석 — SGD의 asymptotic normality 증명에 활용.

**Bayesian inference consistency**: 사후분포의 a.s. 수렴 (posterior consistency) 이론이 Doob's 마팅게일 수렴 활용.

**Deep learning training dynamics**: Over-parameterized NN의 loss curve가 supermartingale로 decrease → 수렴 보장 연구 (NTK 이론).

---

## 📐 수학적 선행 조건

- [Ch5-01](./01-martingale-definition.md): 마팅게일 정의
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Monotone/Dominated convergence, $L^p$-spaces, uniform integrability
- 실해석: $\liminf, \limsup$

---

## 📖 직관적 이해

### 왜 Doob 정리가 성립하는가 — Upcrossing의 아이디어

$\{X_n\}$이 $a$와 $b$ ($a < b$) 사이를 **무한히 오르내리면** 수렴하지 않음. Doob의 아이디어:

**Upcrossing**: $X$가 $a$ 이하에서 $b$ 이상으로 올라간 횟수 $U_n(a, b)$.

**Upcrossing inequality**: $(b - a)\mathbb{E}[U_n(a, b)] \leq \mathbb{E}[(X_n - a)^+]$.

$\mathbb{E}[(X_n - a)^+]$ 유계면 $\mathbb{E}[U_n(a, b)]$ 유계 → $\mathbb{E}[U_\infty(a, b)] < \infty$ → $U_\infty(a, b) < \infty$ a.s. → $X$가 $[a, b]$에서 유한 번 upcrossing.

모든 rationals $a < b$에 대해 성립 → $X_n$의 $\liminf = \limsup$ a.s. → 수렴.

### 음이 아닌 supermartingale의 수렴

**핵심**: $X_n \geq 0$, supermartingale → $\mathbb{E}[X_n] \leq \mathbb{E}[X_0]$ (감소) → $\mathbb{E}|X_n| \leq \mathbb{E}[X_0] < \infty$ → $L^1$-bounded → Doob 수렴.

가장 "쉬운" 마팅게일 수렴 결과 — reinforcement learning, Bayesian inference에서 자주 활용.

### a.s. 수렴 vs $L^1$ 수렴

**a.s.**: $X_n \to X_\infty$ 확률 1로.
**$L^1$**: $\mathbb{E}|X_n - X_\infty| \to 0$.

일반적으로 **서로 implication 없음**. Doob 수렴은 a.s.만 보장; $L^1$ 수렴을 얻으려면 **uniformly integrable** (UI) 조건 추가 필요:
$$\lim_{K \to \infty} \sup_n \mathbb{E}[|X_n| \mathbf{1}_{|X_n| > K}] = 0.$$

UI martingale $\iff$ $L^1$-UI + bounded $\iff$ closed by $X_\infty$ (i.e., $X_n = \mathbb{E}[X_\infty | \mathcal{F}_n]$).

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Upcrossings

$a < b$. 과정 $X_0, X_1, \ldots, X_n$의 $[a, b]$ **upcrossing 횟수** $U_n(a, b)$: $X$가 $a$ 이하에 있다가 $b$ 이상으로 올라간 횟수.

공식:
$\tau_1 = \inf\{k : X_k \leq a\}, \tau_2 = \inf\{k > \tau_1 : X_k \geq b\}, \tau_3, \ldots$ 교대.
$U_n(a, b) = \max\{j : \tau_{2j} \leq n\}$.

### 정의 2.2 — Uniformly Integrable (UI)

가족 $\{X_\alpha\}$가 **uniformly integrable**이다:
$$\lim_{K \to \infty} \sup_\alpha \mathbb{E}[|X_\alpha| \mathbf{1}_{|X_\alpha| > K}] = 0.$$

**충분조건**: $\sup_\alpha \mathbb{E}|X_\alpha|^p < \infty$ for some $p > 1$ (즉 $L^p$-bounded for $p > 1$).

---

## 🔬 정리와 증명

### 정리 2.1 — Doob's Upcrossing Inequality

$\{X_n\}$ submartingale, $a < b$:
$$(b - a) \mathbb{E}[U_n(a, b)] \leq \mathbb{E}[(X_n - a)^+] - \mathbb{E}[(X_0 - a)^+].$$

*증명 스케치*. Martingale transform (정리 1.4). Predictable strategy $H_k = \mathbf{1}_{\{k \in \text{upcrossing interval}\}}$ (매 upcrossing 동안 1, 아닌 동안 0). 구체적으로: $\tau_{2j-1} < k \leq \tau_{2j}$이면 $H_k = 1$.

$H$는 predictable (stopping time-based), 유계 $[0, 1]$. Martingale transform:
$$(H \cdot X)_n = \sum_{k=1}^n H_k (X_k - X_{k-1}) \geq (b - a) U_n(a, b) - (X_n - a)^-.$$

(각 완성된 upcrossing은 $\geq b - a$ 기여, 진행 중인 부분 upcrossing은 음수 기여 최대 $(X_n - a)^-$.)

Submartingale transform의 기댓값 (Jensen 유사): $\mathbb{E}[(H \cdot X)_n] \geq 0$ (or $\mathbb{E}[(X_n - a)^+]$로 완성).

정리된 형태로: $(b-a)\mathbb{E}[U_n(a,b)] \leq \mathbb{E}[(X_n - a)^+]$ (시작점 기여 $(X_0 - a)^+$로 미세 조정).
$\square$

### 정리 2.2 — Doob의 $L^1$ Bounded Martingale Convergence

$\{X_n\}$ submartingale with $\sup_n \mathbb{E}[X_n^+] < \infty$ (또는 마팅게일 with $\sup_n \mathbb{E}|X_n| < \infty$). 그러면 $X_n \to X_\infty$ **a.s.** and $\mathbb{E}|X_\infty| < \infty$.

*증명*. Upcrossing inequality로 모든 $a < b$ ($a, b$ rational)에 대해
$$\mathbb{E}[U_\infty(a, b)] \leq \frac{|a| + \sup_n \mathbb{E}[X_n^+]}{b - a} < \infty.$$
따라서 $U_\infty(a, b) < \infty$ a.s.

$\{X_n$이 수렴하지 않는 $\omega\} = \{\liminf X_n < \limsup X_n\} = \bigcup_{a < b \text{ rational}} \{U_\infty(a, b) = \infty\}$.

가산합으로 $\mathbb{P}(X_n$이 수렴하지 않음$) = 0$. 따라서 $X_n \to X_\infty$ a.s.

$\mathbb{E}|X_\infty|$ 유한: Fatou + sup bound. $\square$

### 정리 2.3 — 음이 아닌 Supermartingale의 a.s. 수렴

$X_n \geq 0$ supermartingale이면 $X_n \to X_\infty$ a.s. with $\mathbb{E}[X_\infty] \leq \mathbb{E}[X_0]$.

*증명*. $\mathbb{E}[X_n] \leq \mathbb{E}[X_0]$ (supermartingale). $X_n \geq 0$이므로 $\mathbb{E}|X_n| = \mathbb{E}[X_n] \leq \mathbb{E}[X_0]$ — $L^1$-bounded. 정리 2.2. $\mathbb{E}[X_\infty] \leq \mathbb{E}[X_0]$ by Fatou. $\square$

### 정리 2.4 — UI Martingale convergence: $L^1$ & a.s.

$\{X_n\}$ UI martingale. $X_n \to X_\infty$ **both a.s. and in $L^1$**, 그리고 $X_n = \mathbb{E}[X_\infty | \mathcal{F}_n]$ (closed by $X_\infty$).

*증명 스케치*. UI → $L^1$-bounded → a.s. 수렴 (정리 2.2). UI 성질로 $L^1$ 수렴도 (Vitali's theorem). Closure: tower property. $\square$

> **해석**: UI가 Doob martingale $Z_n = \mathbb{E}[X | \mathcal{F}_n]$의 특징 — 그리고 이런 martingale은 **$\mathcal{F}_\infty$에서 terminal value로 "닫힌다"**.

### 정리 2.5 — $L^2$ Martingale의 수렴

$\{X_n\}$ martingale, $\sup_n \mathbb{E}[X_n^2] < \infty$. 그러면 $X_n \to X_\infty$ a.s. and in $L^2$.

*증명*. $L^2 \Rightarrow L^1$-bounded → a.s. 수렴. $L^2$ orthogonality of martingale increments으로 $\|X_n - X_m\|_2^2 = \sum_{k=m+1}^n \|X_k - X_{k-1}\|_2^2$ → Cauchy in $L^2$. $\square$

---

## 💻 NumPy 구현 검증

### 실험 1 — 양의 supermartingale 수렴

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# X_{n+1} = X_n * Y_n, Y_n ~ i.i.d. with E[Y]=1, P(Y>0)=1
# E[X_{n+1}|F_n] = X_n → martingale (사실 곱 martingale)
# 하지만 log(Y)의 Jensen으로 log X는 supermartingale (decreasing in expectation)

N = 1000
n_paths = 500
Y = rng.lognormal(mean=-0.5, sigma=1.0, size=(n_paths, N))   # E[Y]=1
# Verify: E[Y] ≈ 1
print(f'E[Y] ≈ {Y.mean():.4f}')

X = np.zeros((n_paths, N + 1))
X[:, 0] = 1.0
for n in range(N):
    X[:, n+1] = X[:, n] * Y[:, n]

# E[X_n] = 1 (martingale) vs individual paths
plt.figure(figsize=(10, 4))
for i in range(20):
    plt.plot(X[i], alpha=0.5)
plt.plot(X.mean(axis=0), 'k-', lw=2, label=r'$\mathbb{E}[X_n]$ (≈ 1)')
plt.yscale('log')
plt.xlabel('n'); plt.ylabel(r'$X_n$ (log scale)')
plt.title('Positive martingale (log-normal) — 수렴하지만 대부분 0으로')
plt.legend(); plt.grid(True, which='both', alpha=0.3); plt.show()

# Most paths converge to 0 (a.s.), but E[X_n] = 1 (martingale)
print(f'X_N 분포: min={X[:, -1].min():.2e}, median={np.median(X[:, -1]):.4f}, max={X[:, -1].max():.2f}')
# → median은 0에 매우 가까움, max는 큼 → heavy-tailed
```

### 실험 2 — Doob martingale $Z_n = \mathbb{E}[X | \mathcal{F}_n]$

```python
# X: target random variable, Z_n: sequence of partial information estimates
# 예: X = sum of iid Bernoulli(0.5), F_n = first n samples

n_sim = 5000
T = 100

X_samples = rng.binomial(1, 0.5, size=(n_sim, T))
X_total = X_samples.sum(axis=1)

# Z_n = E[X | F_n] = partial sum + expected remaining
Z = np.zeros((n_sim, T + 1))
Z[:, 0] = T * 0.5   # 초기 예상
for n in range(1, T + 1):
    Z[:, n] = X_samples[:, :n].sum(axis=1) + (T - n) * 0.5

# Martingale 성질: E[Z_n] 모두 같음
for n in [0, 50, 100]:
    print(f'E[Z_{n}] = {Z[:, n].mean():.4f} (이론: {T*0.5})')

# 개별 path는 X_total로 수렴
print(f'\nPath 수렴 (몇 예):')
for i in range(5):
    print(f'  Z_{T} = {Z[i, -1]:.0f}, X_total = {X_total[i]}')
# 정확히 일치 (UI martingale → closed by X_total)
```

### 실험 3 — $L^1$-unbounded martingale → non-convergence 예

```python
# X_{n+1} = 2 X_n (with prob 1/2), 0 (with prob 1/2)
# E[X_{n+1}|F_n] = X_n (martingale), but E|X_n| → ∞
N = 50
X = np.ones(N + 1)
rng = np.random.default_rng(1)
for n in range(N):
    if rng.random() < 0.5:
        X[n+1] = 2 * X[n]
    else:
        X[n+1] = 0

print(f'Path: {X}')
# → 한번 0이 되면 영원히 0, 안 되면 기하급수적 증가
# 수렴은 하지만 (a.s. 0 or +∞), L^1 수렴 안 함

# 그러나 이는 non-negative martingale이므로 a.s. 수렴 보장됨 (정리 2.3)
# → 한계값 X_∞ ∈ {0, ∞}
```

---

## 🔗 AI/ML 연결

**Robbins-Monro Stochastic Approximation**  
$\theta_{n+1} = \theta_n - \alpha_n g_n$, $g_n$ = noisy gradient. $V_n = \|\theta_n - \theta^*\|^2$가 supermartingale (주어진 조건 하) → $V_n \to V_\infty$ a.s. $V_\infty = 0$임을 증명하면 convergence.

**Q-learning 수렴**  
Watkins-Dayan: Q-learning iteration이 contraction mapping → norm이 supermartingale → 수렴. Doob 수렴 정리가 근간.

**EM Algorithm의 monotone 개선**  
EM step은 likelihood를 단조 증가: $\log p(X_{\text{obs}} | \theta_{n+1}) \geq \log p(X_{\text{obs}} | \theta_n)$. Bounded from above → 수렴. Martingale 대신 단조 bounded seq의 수렴이지만 유사 원리.

**Neural Tangent Kernel (NTK) regime 분석**  
Over-parameterized NN에서 training loss가 supermartingale (approximately) → 수렴. 최근 deep learning theory 연구.

**Bayesian Inference Consistency**  
Posterior distribution의 consistency (Schwartz 1965): $\pi_n(A) = \mathbb{P}(\theta \in A | D_n)$가 Doob martingale → posterior converge to truth with a.s. as $n \to \infty$.

---

## ⚖️ 가정과 한계

**가정 — $L^1$-boundedness**  
없으면 정리 실패 (위 실험 3 참조). 실전에서 variance 발산하는 SGD는 수렴 안 함 — learning rate schedule이 이를 방지.

**한계 — a.s. vs $L^1$**  
a.s. 수렴만 얻는 경우가 많음. $L^1$ 수렴이 필요하면 UI 추가 확인 필요 (e.g., deep learning convergence analysis).

**한계 — Rate of convergence**  
정리는 qualitative (수렴 한다는 것만). Rate는 별도 분석 — 이산 MC의 mixing time (Ch2-04), CLT + Azuma (Ch5-05) 등.

---

## 📌 핵심 정리

| 정리 | 조건 | 결론 |
|---|---|---|
| Doob $L^1$-bdd | $\sup_n \mathbb{E}\|X_n\| < \infty$ | $X_n \to X_\infty$ a.s. |
| $\geq 0$ supermartingale | 자동 $L^1$-bdd | $X_n \to X_\infty$ a.s. |
| UI martingale | UI | a.s. & $L^1$, closed by $X_\infty$ |
| $L^2$ martingale | $\sup \mathbb{E}X_n^2 < \infty$ | a.s. & $L^2$ |
| Upcrossing ineq | submartingale | $(b-a)\mathbb{E}U_n(a,b) \leq \mathbb{E}(X_n-a)^+$ |

**한 줄 요약**: $L^1$-bounded 마팅게일은 반드시 a.s. 수렴 (Doob). Upcrossing inequality가 증명 핵심. UI 조건이 추가되면 $L^1$ 수렴까지 — Deep Learning의 SGD·Q-learning 수렴 보장의 기반.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. Simple random walk $S_n = \sum \xi_k$ ($\xi$ iid $\pm 1$)은 마팅게일이지만 **수렴하지 않는다** (a.s.). 이를 Doob 정리와 어떻게 조화시키는가?

<details>
<summary>해설</summary>

**Simple random walk의 성질**:
$\mathbb{E}[S_n^2] = n \to \infty$, $\mathbb{E}|S_n| = \Theta(\sqrt n) \to \infty$.

따라서 $\sup_n \mathbb{E}|S_n| = \infty$ → **$L^1$-bounded 가 아님** → Doob 정리 조건 **실패**. 정리가 적용 안 됨, 수렴 안 하는 것과 모순 없음.

실제로 $\limsup S_n = +\infty, \liminf S_n = -\infty$ a.s. (recurrent but unbounded).

**교훈**: Doob 정리는 $L^1$-bounded 조건이 **필수**. 무한히 fluctuate 가능한 martingale (unbounded variance)은 수렴 안 할 수 있다.

**실전 연결**: SGD에서 learning rate schedule이 variance를 제어 → $L^1$-bounded 보장 → 수렴 보장. 상수 learning rate는 이 조건 실패 → SGD oscillates around optimum.

</details>

**문제 2 (심화)**. $L^2$-bounded martingale $X_n$에 대해 $\sum_k \mathbb{E}[(X_k - X_{k-1})^2] < \infty$임을 보이고, 이로부터 $L^2$ 수렴을 유도하라.

<details>
<summary>해설</summary>

**Increments orthogonal in $L^2$**: $k \neq j$에 대해
$$\mathbb{E}[(X_k - X_{k-1})(X_j - X_{j-1})] = 0.$$
(Tower: $\mathbb{E}[(X_k - X_{k-1}) \mathbb{E}[X_j - X_{j-1} | \mathcal{F}_{k-1}]] = 0$ by martingale property for $k < j$.)

**Pythagoras**: $\mathbb{E}[X_n^2] = \mathbb{E}[X_0^2] + \sum_{k=1}^n \mathbb{E}[(X_k - X_{k-1})^2]$.

$L^2$-bounded → $\sum_k \mathbb{E}[(X_k - X_{k-1})^2] \leq \sup_n \mathbb{E}[X_n^2] - \mathbb{E}[X_0^2] < \infty$.

**$L^2$ Cauchy**: $\|X_n - X_m\|_2^2 = \sum_{k=m+1}^n \|X_k - X_{k-1}\|_2^2$. 꼬리합 $\to 0$ → Cauchy in $L^2$.

$L^2$ complete → $X_n \to X_\infty$ in $L^2$.

**추가 (a.s.)**: $L^2$-bounded → $L^1$-bounded → a.s. 수렴 (정리 2.2). 두 한계 일치.

**AI 응용**: Kalman filter의 state estimate, TD($\lambda$) value estimates가 $L^2$-bounded martingale → $L^2$ 수렴 → convergence rate 분석 가능.

</details>

**문제 3 (AI 연결)**. SGD $\theta_{n+1} = \theta_n - \alpha_n (\nabla L(\theta_n) + \epsilon_n)$, $\epsilon_n$ mean-zero noise. $V_n = \|\theta_n - \theta^*\|^2$이 Robbins-Monro 조건 하에 supermartingale이 되는 조건을 찾아라.

<details>
<summary>해설</summary>

**$V_n$의 one-step 진화**:
$V_{n+1} = \|\theta_{n+1} - \theta^*\|^2 = V_n - 2\alpha_n \langle \theta_n - \theta^*, \nabla L + \epsilon_n\rangle + \alpha_n^2 \|\nabla L + \epsilon_n\|^2$.

$\mathbb{E}[\cdot | \mathcal{F}_n]$: $\mathbb{E}[\epsilon_n | \mathcal{F}_n] = 0$로
$\mathbb{E}[V_{n+1} | \mathcal{F}_n] = V_n - 2\alpha_n \langle \theta_n - \theta^*, \nabla L(\theta_n)\rangle + \alpha_n^2(\|\nabla L(\theta_n)\|^2 + \mathbb{E}\|\epsilon_n\|^2)$.

**Strong convexity of $L$**: $\langle \theta - \theta^*, \nabla L(\theta)\rangle \geq \mu \|\theta - \theta^*\|^2 = \mu V_n$.

$\mathbb{E}[V_{n+1} | \mathcal{F}_n] \leq (1 - 2\alpha_n \mu) V_n + \alpha_n^2 (G^2 + \sigma^2)$ where $G = \sup \|\nabla L\|, \sigma^2 = \sup \mathbb{E}\|\epsilon\|^2$.

**$V_n$이 Supermartingale이 되는 조건**:
$\mathbb{E}[V_{n+1} | \mathcal{F}_n] \leq V_n$ ⇔ $(1 - 2\alpha_n \mu) V_n + \alpha_n^2 C \leq V_n$ ⇔ $\alpha_n^2 C \leq 2\alpha_n \mu V_n$ ⇔ $\alpha_n \leq 2\mu V_n / C$.

즉 **$V_n$이 크면** 큰 $\alpha_n$ 가능, **$V_n$ 작아지면** $\alpha_n$도 작아져야 함. 고정 schedule $\alpha_n = c/n$가 Robbins-Monro 충분조건 ($\sum \alpha = \infty, \sum \alpha^2 < \infty$).

**정확한 수렴 증명**:
$V_n$이 directly supermartingale 아니어도 $V_n + \text{correction term}$이 supermartingale. 구체적으로
$$M_n = V_n + \sum_{k \geq n} \alpha_k^2 C$$
이 supermartingale (under Robbins-Monro).

정리 2.3 (양의 supermartingale 수렴) → $M_n$ 수렴 → $V_n$ 수렴 (tail sum → 0).

$V_\infty = 0$ 증명 (with probability 1): $\mathbb{E}[V_\infty] \leq \mathbb{E}[V_0] - 2\mu \sum \alpha_n \mathbb{E}V_n + \sum \alpha_n^2 C$. 발산 $\sum \alpha_n = \infty$와 $\mathbb{E}V_n$ 유계로 $\mathbb{E}V_n \to 0$.

**실전 함의**:
- **Learning rate schedule의 필요**: 상수 $\alpha$는 $\sum \alpha^2 < \infty$ 깨짐 → 수렴 안 함.
- **Momentum/Adam**: 이를 변형한 supermartingale 구조 분석 가능.
- **Non-convex**: Strong convexity 없어도 local 수렴 가능 (saddle points, local minima analysis).

**연결**: 현대 deep learning 수렴 이론의 근간. Neural ODE (Chen 2018), NTK (Jacot 2018), implicit regularization 등 모두 supermartingale 관점 활용.

</details>

---

<div align="center">

◀ [01. 마팅게일의 정의](./01-martingale-definition.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [03. Optional Stopping Theorem](./03-optional-stopping.md)

</div>
