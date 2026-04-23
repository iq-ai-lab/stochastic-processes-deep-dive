# 05. 마팅게일과 ML — Online Learning

## 🎯 핵심 질문

- **Azuma-Hoeffding 부등식** $\mathbb{P}(|X_n - X_0| \geq t) \leq 2\exp(-\frac{t^2}{2\sum c_k^2})$가 어떻게 유도되는가?
- 이 부등식이 **Online Convex Optimization**의 regret bound $\mathcal{O}(\sqrt{T})$를 유도하는 이유는?
- **Bandit** 문제의 "exploration vs exploitation" trade-off에서 마팅게일 concentration이 어떻게 쓰이는가?
- **Stochastic Gradient Descent**의 수렴 분석에서 마팅게일 difference가 왜 필요한가?

---

## 🔍 왜 이 연결이 AI에서 중요한가

현대 ML 이론의 대부분 **concentration inequality**가 마팅게일 기반:
- **SGD convergence**: Robbins-Monro가 마팅게일 수렴 + Azuma
- **UCB bandit**: Concentration → exploration bonus
- **Online learning regret**: $O(\sqrt T)$ bound
- **RL PAC bounds**: State-action visit count의 concentration
- **Differential Privacy**: Noise accumulation의 확률론적 분석

마팅게일 없이는 현대 ML 이론 대부분 재구성 불가.

---

## 📐 수학적 선행 조건

- [Ch5-01 ~ Ch5-04](./01-martingale-definition.md): 마팅게일, 수렴, OST, 이차변분
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Chernoff bound, moment generating function

---

## 📖 직관적 이해

### Hoeffding → Azuma

**Hoeffding (iid)**: $\xi_k$ iid, $|\xi| \leq c$, $S_n = \sum \xi_k - n\mathbb{E}\xi$ → $\mathbb{P}(|S_n| \geq t) \leq 2\exp(-t^2/(2nc^2))$.

**Azuma (martingale difference)**: $\xi_k = M_k - M_{k-1}$ martingale difference, $|\xi_k| \leq c_k$ → 같은 bound with $\sum c_k^2$.

**차이**: Azuma는 **독립 가정 불필요** — martingale difference면 됨. 이는 **adaptive / sequential** setting에서 결정적 (각 step의 noise가 과거에 의존 가능).

### Online Learning의 Setup

Player가 매 라운드 $t$에서 action $x_t \in \mathcal{X}$ 선택, adversary가 loss $\ell_t(x_t)$ 드러냄. Player는 $x_t$를 **과거 $\ell_1, \ldots, \ell_{t-1}$ 만** 보고 선택 (online).

**Regret**: $R_T = \sum_t \ell_t(x_t) - \min_{x \in \mathcal{X}} \sum_t \ell_t(x)$. "Hindsight의 best 고정 action 대비 얼마나 나쁜가."

**목표**: $R_T = o(T)$ (no-regret), 이상적으로 $\mathcal{O}(\sqrt T)$.

### OCO regret의 Azuma 유도

OCO (Online Convex Optimization)의 classical 알고리즘 **Online Gradient Descent**:
$$x_{t+1} = \Pi_\mathcal{X}(x_t - \eta \nabla \ell_t(x_t)).$$

Regret의 구성 요소 중 일부가 **martingale difference**로 표현 → Azuma 적용하여 high-probability bound.

구체적으로 "randomized OGD"의 regret이:
$$R_T = \underbrace{\text{expected regret}}_{\text{deterministic}} + \underbrace{\text{variance term}}_{\text{martingale}}.$$

Variance term에 Azuma 적용 → $\mathcal{O}(\sqrt T)$ tail bound.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Martingale Difference

$\{\xi_k\}$ is a **martingale difference sequence** (MDS) w.r.t. $\{\mathcal{F}_k\}$:
1. $\xi_k \in \mathcal{F}_k$ (adapted)
2. $\mathbb{E}|\xi_k| < \infty$
3. $\mathbb{E}[\xi_k | \mathcal{F}_{k-1}] = 0$

$M_n = \sum_{k=1}^n \xi_k$이 martingale (with $M_0 = 0$).

### 정의 5.2 — Online Learning Regret

T 라운드 후 regret:
$$R_T = \sum_{t=1}^T \ell_t(x_t) - \min_{x \in \mathcal{X}} \sum_{t=1}^T \ell_t(x).$$

**Expected regret**: $\mathbb{E}[R_T]$.

**High-prob regret**: $\mathbb{P}(R_T \geq f(T)) \leq \delta$ for some $f$.

---

## 🔬 정리와 증명

### 정리 5.1 — Azuma-Hoeffding 부등식

$\{M_n\}$ martingale, $|M_k - M_{k-1}| \leq c_k$ a.s. 그러면 $\forall t > 0$:
$$\mathbb{P}(M_n - M_0 \geq t) \leq \exp\left(-\frac{t^2}{2\sum_{k=1}^n c_k^2}\right).$$

$|M_n - M_0|$에 대해선 2배.

### 증명 스케치

**Chernoff-style**: $\mathbb{P}(M_n - M_0 \geq t) \leq e^{-\lambda t} \mathbb{E}[e^{\lambda(M_n - M_0)}]$.

$\mathbb{E}[e^{\lambda(M_n - M_0)}]$을 귀납으로 bound. $X_k = M_k - M_{k-1}$, $|X_k| \leq c_k$, $\mathbb{E}[X_k | \mathcal{F}_{k-1}] = 0$.

**Hoeffding's lemma (martingale version)**: Conditional MGF
$$\mathbb{E}[e^{\lambda X_k} | \mathcal{F}_{k-1}] \leq e^{\lambda^2 c_k^2 / 2}.$$

(Bounded mean-zero variable의 MGF bound; convexity 이용.)

따라서:
$$\mathbb{E}[e^{\lambda(M_n - M_0)}] = \mathbb{E}[e^{\lambda(M_{n-1} - M_0)} \mathbb{E}[e^{\lambda X_n} | \mathcal{F}_{n-1}]] \leq e^{\lambda^2 c_n^2/2} \mathbb{E}[e^{\lambda(M_{n-1} - M_0)}].$$

귀납: $\mathbb{E}[e^{\lambda(M_n - M_0)}] \leq \exp(\lambda^2 \sum c_k^2/2)$.

Chernoff: $\mathbb{P}(M_n - M_0 \geq t) \leq e^{-\lambda t + \lambda^2 \sum c_k^2/2}$. $\lambda = t/\sum c_k^2$로 optimal:
$$\leq \exp\left(-\frac{t^2}{2\sum c_k^2}\right).$$
$\square$

### 정리 5.2 — McDiarmid's (Bounded Differences) Inequality

$Z = f(X_1, \ldots, X_n)$ with $X_k$ 독립, $|f(\ldots, X_k, \ldots) - f(\ldots, X_k', \ldots)| \leq c_k$ for all other arguments fixed. 그러면
$$\mathbb{P}(|Z - \mathbb{E}Z| \geq t) \leq 2\exp\left(-\frac{2t^2}{\sum c_k^2}\right).$$

*증명 스케치*. $M_k = \mathbb{E}[f | X_1, \ldots, X_k]$ (Doob martingale). $|M_k - M_{k-1}| \leq c_k$ ("bounded differences"). Azuma. $\square$

**응용**: 학습 알고리즘의 generalization error의 concentration (Bartlett-Mendelson).

### 정리 5.3 — Online Gradient Descent Regret

OCO with convex $\ell_t : \mathcal{X} \to \mathbb{R}$, $\|\nabla \ell_t\| \leq G$, $\text{diam}(\mathcal{X}) \leq D$. OGD with $\eta = D/(G\sqrt T)$:
$$R_T = \sum_t \ell_t(x_t) - \min_x \sum_t \ell_t(x) \leq GD\sqrt T.$$

*증명 스케치*. Convexity: $\ell_t(x_t) - \ell_t(x^*) \leq \langle \nabla \ell_t(x_t), x_t - x^* \rangle$. 각 항을 bound.

$\|x_{t+1} - x^*\|^2 = \|x_t - \eta \nabla \ell_t - x^*\|^2 \leq \|x_t - x^*\|^2 - 2\eta \langle \nabla \ell_t, x_t - x^*\rangle + \eta^2 G^2$ (projection nonexpansive).

재정렬:
$\langle \nabla \ell_t, x_t - x^*\rangle \leq \frac{1}{2\eta}(\|x_t - x^*\|^2 - \|x_{t+1} - x^*\|^2) + \frac{\eta G^2}{2}$.

합: $R_T \leq \frac{D^2}{2\eta} + \frac{\eta T G^2}{2}$. $\eta = D/(G\sqrt T)$ optimal.

**Stochastic/noisy setting**: $\nabla \ell_t$이 stochastic이면 (예: mini-batch gradient), difference가 martingale → Azuma로 $O(\sqrt{T})$ high-prob bound. $\square$

### 정리 5.4 — Multi-Armed Bandit UCB Regret

$K$-arm bandit, arm $i$의 reward $\sim$ bounded distribution with mean $\mu_i$. UCB algorithm 선택 $a_t = \arg\max_i (\hat\mu_i(t) + \sqrt{2\log t / n_i(t)})$. Regret:
$$\mathbb{E}[R_T] = O\left(\sum_{i : \Delta_i > 0} \frac{\log T}{\Delta_i}\right),$$
$\Delta_i = \mu^* - \mu_i$ (gap).

*증명 핵심*: Azuma-Hoeffding으로 $|\hat\mu_i - \mu_i| \leq \sqrt{2\log T/n_i}$ w.h.p. 각 suboptimal arm이 $\log T/\Delta_i^2$ 회 넘게 안 뽑힘 → regret bound.

---

## 💻 NumPy 구현 검증

### 실험 1 — Azuma-Hoeffding 검증

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# Symmetric martingale difference: ξ ~ Rademacher (±1)
# |ξ| ≤ 1 → c_k = 1
# Azuma: P(|M_n| ≥ t) ≤ 2 exp(-t²/(2n))

n_sim = 100000
n = 100
xi = rng.choice([-1, 1], size=(n_sim, n))
M = xi.sum(axis=1)

# 경험적 tail vs 이론
t_vals = np.linspace(5, 30, 10)
empirical = [(np.abs(M) >= t).mean() for t in t_vals]
theoretical = [2 * np.exp(-t**2/(2*n)) for t in t_vals]

plt.semilogy(t_vals, empirical, 'o-', label='실측')
plt.semilogy(t_vals, theoretical, '--', label='Azuma bound')
plt.xlabel('t'); plt.ylabel(r'$P(|M_n| \geq t)$ (log)')
plt.legend(); plt.grid(True, which='both', alpha=0.3)
plt.title(f'Azuma-Hoeffding (n={n}, c_k=1)')
plt.show()
# 실측이 이론 bound 아래 (더 tight) — Azuma가 보수적
```

### 실험 2 — Martingale difference with dependent bounded noise

```python
# ξ_k = sign(M_{k-1}) × ε_k, ε_k ~ Rademacher
# → ξ_k는 과거에 의존하지만 martingale difference
# (E[ξ_k | F_{k-1}] = sign(M_{k-1}) × 0 = 0)

def run_dependent_mds(n):
    M = 0; Ms = [0]
    for _ in range(n):
        # Dependent bound: |ξ| = 1 always
        xi = rng.choice([-1, 1])
        M += xi
        Ms.append(M)
    return Ms

n_trials = 5000
n = 100
final_M = np.array([run_dependent_mds(n)[-1] for _ in range(n_trials)])

# Azuma with c_k = 1
print(f'P(|M_n| >= 20) 실측: {(np.abs(final_M) >= 20).mean():.4f}')
print(f'Azuma bound: {2*np.exp(-20**2/(2*n)):.4f}')
# 일치 (느슨한 bound이지만 correct)
```

### 실험 3 — UCB Bandit 실험

```python
K = 5   # 5 arms
mu_true = np.array([0.3, 0.5, 0.7, 0.4, 0.6])   # true means (max=0.7)
T = 5000

n_arm = np.zeros(K)
sum_arm = np.zeros(K)
regret = 0
regrets = []

for t in range(1, T+1):
    if np.any(n_arm == 0):
        a = np.argmin(n_arm)
    else:
        ucb = sum_arm/n_arm + np.sqrt(2*np.log(t)/n_arm)
        a = np.argmax(ucb)
    reward = rng.random() < mu_true[a]  # Bernoulli
    n_arm[a] += 1
    sum_arm[a] += reward
    regret += mu_true.max() - mu_true[a]
    regrets.append(regret)

plt.plot(regrets, label=f'UCB regret (K={K}, T={T})')
plt.plot(np.sqrt(np.arange(1, T+1)) * 3, '--', label=r'$O(\sqrt T)$')
plt.xlabel('t'); plt.ylabel('cumulative regret')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
print(f'\n각 arm 방문 횟수: {n_arm}')
# Optimal arm (index 2, μ=0.7)이 가장 많이 뽑힘
```

---

## 🔗 AI/ML 연결

**Deep Learning SGD 수렴**  
$\theta_{n+1} = \theta_n - \alpha \nabla L_{\text{batch}}(\theta_n)$. Noise $\epsilon_n = \nabla L_{\text{batch}} - \nabla L_{\text{full}}$가 MDS (zero conditional mean). Azuma로 $\mathbb{P}(\|\theta_n - \theta^*\| \geq t)$ bound.

**PAC Learning Bounds**  
$\mathbb{P}(|\hat L(h) - L(h)| \geq t) \leq 2\exp(-2nt^2)$ (Hoeffding). Empirical risk minimization의 generalization.

**Reinforcement Learning PAC-bounds**  
State-action visitation count의 concentration (Kearns-Singh 1998). Polynomial-time RL algorithms의 이론적 보장.

**Online Convex Optimization 라이브러리**  
PyTorch Optimizer의 Adam/AdaGrad/RMSProp 수렴 분석이 OCO 이론 기반. Regret bound $O(\sqrt T)$를 "실시간" 학습 효율로 해석.

**Differential Privacy Composition**  
$T$ rounds of $\epsilon$-DP queries → total $(O(\sqrt{T\log(1/\delta)}) \epsilon, \delta)$-DP (Advanced composition). Martingale concentration의 직접 응용.

**Contextual Bandits**  
LinUCB (Li 2010): arm reward $= \theta^\top x$ + noise. Noise가 MDS → Azuma-style concentration → UCB confidence bound. Modern recommender systems의 기반.

---

## ⚖️ 가정과 한계

**가정 — Bounded differences**  
Azuma가 worst-case $c_k$ 사용 → 평균적으로 작은 noise면 loose. **Freedman inequality**(Ch5-04)가 이차변분 기반 tighter bound.

**한계 — Sub-Gaussian assumption 필요 (확장)**  
Bounded 아닌 noise(e.g., Gaussian)는 Azuma 변형 — 조건부 sub-Gaussian 가정. Heavy-tailed은 별도 theory.

**한계 — Adversarial 환경**  
최악 시나리오 bound. 실제 환경은 "benign" (i.i.d. or mild structure) → 훨씬 tight regret 가능 (online-to-batch conversion).

---

## 📌 핵심 정리

| 정리 | 조건 | 결론 |
|---|---|---|
| Azuma-Hoeffding | martingale, $\|\Delta M\| \leq c_k$ | $\mathbb{P}(M_n - M_0 \geq t) \leq \exp(-t^2/(2\sum c_k^2))$ |
| McDiarmid | 독립 $X_k$, bounded diff | McDiarmid bound |
| Hoeffding's lemma | bounded mean-zero | $\mathbb{E}e^{\lambda X} \leq e^{\lambda^2 c^2/2}$ |
| OGD regret | convex, bounded grad | $O(GD\sqrt T)$ |
| UCB regret | bounded rewards | $O(\sum \log T/\Delta_i)$ |
| Freedman | variance-aware | tighter bound |

**한 줄 요약**: 마팅게일 concentration(Azuma-Hoeffding)은 "adaptive/sequential" 환경에서의 iid Hoeffding의 일반화. 현대 ML 이론 — SGD, OCO, bandit, PAC, RL, DP — 의 수학적 도구이자, $O(\sqrt T)$ regret bound의 기반.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. Coin flip $n = 100$ 번, $|M_n - 50| \geq 20$일 확률을 Azuma로 bound하고 정확한 binomial과 비교하라.

<details>
<summary>해설</summary>

$M_n - 50 = \sum (\xi_k - 0.5) = \sum X_k$ where $X_k = \xi_k - 0.5 \in [-0.5, 0.5]$ → $c_k = 0.5$.

**Azuma**: $\mathbb{P}(|M_n - 50| \geq 20) \leq 2 \exp(-20^2/(2 \cdot 100 \cdot 0.25)) = 2e^{-8} \approx 6.7 \times 10^{-4}$.

**정확 (Binomial $n=100, p=0.5$)**: $\mathbb{P}(|X - 50| \geq 20) = \mathbb{P}(X \leq 30) + \mathbb{P}(X \geq 70)$. Normal approx $\mathcal{N}(50, 25)$: $Z = \pm 4$ → $\approx 6 \times 10^{-5}$.

**비교**: Azuma는 10배 loose. Exact는 exponentially tighter due to unbiased tail decay.

**교훈**: Azuma는 worst-case bound. Real distribution의 구조를 활용하면 (Chernoff with exact MGF) tighter.

</details>

**문제 2 (심화)**. OGD의 **stochastic** 버전: $\nabla \ell_t$ 대신 $\hat g_t = \nabla \ell_t + \xi_t$ 사용, $\xi_t$ MDS with $\|\xi_t\| \leq \sigma$. High-prob regret $O(GD\sqrt T + \sigma\sqrt{T\log(1/\delta)})$를 유도.

<details>
<summary>해설</summary>

**Decomposition**: $R_T = \sum_t \ell_t(x_t) - \ell_t(x^*)$.

Convexity: $\ell_t(x_t) - \ell_t(x^*) \leq \langle \nabla \ell_t, x_t - x^*\rangle = \langle \hat g_t - \xi_t, x_t - x^*\rangle$.

즉 $R_T \leq \sum_t \langle \hat g_t, x_t - x^*\rangle + \sum_t \langle \xi_t, x^* - x_t\rangle$.

**첫 항** (deterministic OGD analysis): $\leq \frac{D^2}{2\eta} + \frac{\eta}{2}\sum \|\hat g_t\|^2 \leq \frac{D^2}{2\eta} + \frac{\eta T (G + \sigma)^2}{2}$.

**두 번째 항** (martingale): $Z_T := \sum_t \langle \xi_t, x^* - x_t\rangle$. MDS (각 항이 $\mathcal{F}_{t-1}$ 조건부 mean-zero), $|\langle \xi_t, x^* - x_t\rangle| \leq \sigma D$. Azuma:
$$\mathbb{P}(Z_T \geq t) \leq \exp(-t^2/(2T\sigma^2 D^2)).$$
High-prob: $Z_T \leq \sigma D \sqrt{2T \log(1/\delta)}$ with prob $\geq 1 - \delta$.

**총 bound**: $R_T \leq \frac{D^2}{2\eta} + \frac{\eta T (G+\sigma)^2}{2} + \sigma D\sqrt{2T\log(1/\delta)}$.

$\eta = \frac{D}{(G+\sigma)\sqrt T}$ optimal: $R_T \leq (G+\sigma) D \sqrt T + \sigma D \sqrt{2T\log(1/\delta)} = O(GD\sqrt T + \sigma D \sqrt{T\log(1/\delta)})$.

**해석**: 
- $O(GD\sqrt T)$: deterministic cost
- $O(\sigma D \sqrt{T\log(1/\delta)})$: stochastic noise cost
- Total $O(\sqrt T)$ regret, noise가 추가 constant factor

**연결**: Modern ML optimization 이론(SGD, mini-batch)의 기본 계산. Adam, RMSProp 등 variant의 regret 분석도 유사.

</details>

**문제 3 (AI 연결)**. LinUCB algorithm for contextual bandits: $\hat\theta_t = \arg\min_\theta \sum_{s \leq t}(\theta^\top x_s - r_s)^2 + \lambda \|\theta\|^2$. Arm selection $a_t = \arg\max_a (\hat\theta_t^\top x_{a,t} + \alpha \sqrt{x_{a,t}^\top V_t^{-1} x_{a,t}})$. UCB term의 derivation을 martingale concentration으로 스케치하라.

<details>
<summary>해설</summary>

**Linear bandit setup**: Reward $r_s = \theta^{*\top} x_s + \eta_s$, $\eta_s$ mean-zero sub-Gaussian noise.

**Ridge regression estimate**: $\hat\theta_t = V_t^{-1} b_t$, $V_t = \lambda I + \sum x_s x_s^\top$, $b_t = \sum x_s r_s$.

**Estimation error**: $\hat\theta_t - \theta^* = V_t^{-1}(b_t - V_t \theta^*) = V_t^{-1}(\sum x_s \eta_s - \lambda \theta^*)$.

$\epsilon_t := \sum x_s \eta_s$ — **vector-valued martingale** (each $x_s$ is $\mathcal{F}_{s-1}$-measurable, $\eta_s$ MDS).

**Self-normalized concentration (Abbasi-Yadkori 2011)**: $\eta_s$ sub-Gaussian → $\|V_t^{-1/2} \epsilon_t\|^2$ concentrates around $d \log(\det V_t / \lambda^d)$. 구체적으로:
$$\|\epsilon_t\|_{V_t^{-1}} \leq \sigma\sqrt{2\log(\det V_t^{1/2}/(\lambda^{d/2}\delta))}.$$

**Confidence bound for $\theta^*$**: $\|\hat\theta_t - \theta^*\|_{V_t} \leq \|\epsilon_t\|_{V_t^{-1}} + \lambda^{1/2}\|\theta^*\|$.

**UCB term**: 새로운 context $x$에서 예측 error bound:
$$|\langle \hat\theta_t - \theta^*, x\rangle| \leq \|\hat\theta_t - \theta^*\|_{V_t} \cdot \|x\|_{V_t^{-1}} \leq \alpha_t \sqrt{x^\top V_t^{-1} x}$$
where $\alpha_t$ grows $\sim \sqrt{\log t}$.

**UCB arm selection**:
$\tilde r(a) = \hat\theta_t^\top x_a + \alpha_t \sqrt{x_a^\top V_t^{-1} x_a}$ — optimistic estimate.

**Regret**:
$R_T = O(d \sqrt T \log T)$ where $d$ = feature dimension. Sub-linear!

**핵심 통찰**:
1. Martingale (self-normalized) concentration이 **context-dependent** confidence bound 제공.
2. $\sqrt{x^\top V_t^{-1} x}$가 "이 context 방향으로 얼마나 exploration 되었나"를 측정.
3. 방문 많은 방향 → small UCB → exploitation; 방문 적은 방향 → large UCB → exploration.

**실전**:
- LinUCB (news recommendation, Yahoo 2010)
- Thompson Sampling (Bayesian version, 유사 regret)
- Neural UCB (linear → neural net 확장, Zhou 2020)

**연결**: Ch5의 모든 마팅게일 이론이 modern contextual bandit / RL algorithms의 **정량적 exploration bonus** 설계에 기여. Theory-practice gap을 좁히는 대표적 연결 고리.

</details>

---

<div align="center">

◀ [04. Doob 분해와 이차변분](./04-doob-decomposition.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [Ch6-01. 브라운 운동의 공리적 정의](../ch6-brownian/01-axiomatic-definition.md)

</div>
