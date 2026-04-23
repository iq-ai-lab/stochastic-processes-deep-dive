# 01. 마팅게일의 정의

## 🎯 핵심 질문

- 마팅게일 $\mathbb{E}[X_{n+1} | \mathcal{F}_n] = X_n$는 **"공정한 게임"**인가 — 도박 이론에서 어떻게 유래했는가?
- **Sub**·**super** 마팅게일은 각각 "유리"·"불리"한 게임의 어느 쪽인가?
- $X$가 마팅게일이면 $X^2$, $|X|$가 왜 **submartingale**인가 — Jensen 부등식의 역할?
- **Integrability**와 **adaptedness**가 왜 정의의 필수 조건인가?

---

## 🔍 왜 마팅게일이 AI에서 중요한가

**Online Learning의 regret bound**: SGD, bandit, online convex optimization의 이론이 마팅게일 concentration(Azuma-Hoeffding)에 의존. Ch5-05 참조.

**RL policy evaluation**: TD learning의 convergence 분석에서 value function이 martingale-like.

**Stochastic Approximation**: 학습률 schedule의 convergence — Robbins-Monro 정리가 마팅게일 수렴에 기반.

**Conditional Expectation Networks**: Neural ODE, Score function 추정이 본질적으로 "조건부 기댓값" — 마팅게일 관점에서 학습 동역학 이해.

**Diffusion Model의 score**: $B_t, B_t^2 - t, e^{\lambda B_t - \lambda^2 t/2}$가 모두 BM 기반 마팅게일 → SDE Deep Dive의 이토 공식 근간.

---

## 📐 수학적 선행 조건

- [Ch1-04](../ch1-foundations/04-filtration.md): 필트레이션, 조건부 기댓값, adapted
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Tower property, Jensen 부등식

---

## 📖 직관적 이해

### "공정한 게임"

$X_n$ = 시각 $n$에서 도박꾼의 자산. 게임이 "공정"하다면 다음 라운드의 기댓값은 현재 자산과 같음:
$$\mathbb{E}[X_{n+1} | \text{현재까지 정보}] = X_n.$$

**예**: 동전 던지기에 \$1 거는 공정 게임. 이기면 +\$1, 지면 -\$1 → 평균 변화 = 0. 이것이 **simple random walk**, 마팅게일.

### Sub / Super 마팅게일

- **Submartingale**: $\mathbb{E}[X_{n+1} | \mathcal{F}_n] \geq X_n$ — **"유리한 게임"** (기댓값이 증가)
- **Supermartingale**: $\mathbb{E}[X_{n+1} | \mathcal{F}_n] \leq X_n$ — **"불리한 게임"** (기댓값이 감소)

**기억법**: Super는 "위", 실제로는 내려감 (감소하는 양). Sub는 "아래", 실제로는 올라감. (관습: 조금 반직관적)

**해석**: 카지노의 관점에서 게임은 항상 supermartingale (도박꾼 관점에서 하락). Life insurance 보험금은 submartingale (시간에 따라 일반적으로 증가).

### Jensen 부등식과의 관계

$\varphi$가 convex이고 $X$가 마팅게일이면 $\varphi(X)$는 **submartingale**.

**이유**: Jensen (조건부 버전) $\varphi(\mathbb{E}[X_{n+1} | \mathcal{F}_n]) \leq \mathbb{E}[\varphi(X_{n+1}) | \mathcal{F}_n]$. 마팅게일 가정 대입:
$$\varphi(X_n) \leq \mathbb{E}[\varphi(X_{n+1}) | \mathcal{F}_n].$$

따라서 $X^2, |X|, e^X$ 등이 submartingale (if integrable).

---

## ✏️ 엄밀한 정의

### 정의 1.1 — 마팅게일 (Martingale)

확률공간 $(\Omega, \mathcal{F}, \mathbb{P})$와 필트레이션 $\{\mathcal{F}_n\}_{n \geq 0}$가 주어졌을 때, 과정 $\{X_n\}$이 **마팅게일**이다:
1. **Adapted**: $X_n \in \mathcal{F}_n$ for all $n$
2. **Integrable**: $\mathbb{E}|X_n| < \infty$ for all $n$
3. **Martingale property**: $\mathbb{E}[X_{n+1} | \mathcal{F}_n] = X_n$ a.s.

### 정의 1.2 — Sub-/Super-마팅게일

같은 (1), (2) 조건에 대해:
- **Submartingale**: $\mathbb{E}[X_{n+1} | \mathcal{F}_n] \geq X_n$
- **Supermartingale**: $\mathbb{E}[X_{n+1} | \mathcal{F}_n] \leq X_n$

### 정의 1.3 — 연속시간 마팅게일

$T = [0, \infty)$ 위의 과정 $\{X_t\}$가 마팅게일이다:
- Adapted
- Integrable
- $\mathbb{E}[X_t | \mathcal{F}_s] = X_s$ for $s \leq t$.

---

## 🔬 정리와 증명

### 정리 1.1 — Tower Property를 이용한 다중 스텝

$\{X_n\}$이 마팅게일이면 $\mathbb{E}[X_{n+k} | \mathcal{F}_n] = X_n$ for all $k \geq 0$.

*증명*. $k$에 대한 귀납. $k = 1$은 정의. $k \to k+1$:
$$\mathbb{E}[X_{n+k+1} | \mathcal{F}_n] = \mathbb{E}[\mathbb{E}[X_{n+k+1} | \mathcal{F}_{n+k}] | \mathcal{F}_n] = \mathbb{E}[X_{n+k} | \mathcal{F}_n] = X_n.$$
$\square$

### 정리 1.2 — 기댓값 불변

마팅게일이면 $\mathbb{E}[X_n] = \mathbb{E}[X_0]$ for all $n$.

*증명*. Tower + 마팅게일 성질. $\square$

### 정리 1.3 — $X^2$와 $|X|$가 Submartingale

$X$가 $L^1$-martingale이고 $\varphi$ convex이며 $\mathbb{E}|\varphi(X_n)| < \infty$이면 $\varphi(X)$는 submartingale.

*증명*. Jensen (조건부 버전):
$$\mathbb{E}[\varphi(X_{n+1}) | \mathcal{F}_n] \geq \varphi(\mathbb{E}[X_{n+1} | \mathcal{F}_n]) = \varphi(X_n).$$
$\square$

**따름 정리**: 
- $X_n^2$ submartingale ($L^2$ 가정) — Doob 분해(Ch5-04)의 출발점
- $|X_n|$ submartingale
- $\exp(X_n)$ submartingale

### 정리 1.4 — 마팅게일 변환 (Martingale Transform)

$\{X_n\}$ 마팅게일, $\{H_n\}$ predictable (i.e., $H_n \in \mathcal{F}_{n-1}$), 유계. 그러면
$$(H \cdot X)_n := \sum_{k=1}^n H_k (X_k - X_{k-1})$$
도 마팅게일.

*증명*. Adapted (합이 adapted) + integrable (유계 H, integrable X 증분). 마팅게일 성질:
$$\mathbb{E}[H_{n+1}(X_{n+1} - X_n) | \mathcal{F}_n] = H_{n+1} \mathbb{E}[X_{n+1} - X_n | \mathcal{F}_n] = 0.$$
따라서 $\mathbb{E}[(H \cdot X)_{n+1} | \mathcal{F}_n] = (H \cdot X)_n$. $\square$

**해석**: "공정한 게임에 predictable 베팅 전략"은 여전히 공정. **"Cannot beat a fair game"** — 가장 유명한 확률 이론 결과 중 하나.

### 정리 1.5 — Examples of Martingales

**예 1 — Simple Random Walk**: $\xi_1, \xi_2, \ldots$ iid with $\mathbb{E}\xi = 0$. $S_n = \sum_{k=1}^n \xi_k$. 마팅게일:
$$\mathbb{E}[S_{n+1} | \mathcal{F}_n] = S_n + \mathbb{E}[\xi_{n+1}] = S_n.$$

**예 2 — Doob's Martingale**: $X$ integrable 확률변수, $Z_n = \mathbb{E}[X | \mathcal{F}_n]$. Tower:
$$\mathbb{E}[Z_{n+1} | \mathcal{F}_n] = \mathbb{E}[\mathbb{E}[X | \mathcal{F}_{n+1}] | \mathcal{F}_n] = \mathbb{E}[X | \mathcal{F}_n] = Z_n.$$

**예 3 — Brownian Motion** (연속시간): $B_t, B_t^2 - t, \exp(\lambda B_t - \lambda^2 t/2)$ 모두 마팅게일 (Ch6, SDE).

**예 4 — Exponential Martingale**: $\mathbb{E}[e^{\lambda \xi}] = e^{\psi(\lambda)}$ — $M_n = e^{\lambda S_n - n \psi(\lambda)}$는 마팅게일.

### 정리 1.6 — 마팅게일 중간정리 (Optional)

$\{X_n\}$ 마팅게일, $\tau$ bounded stopping time. 그러면 $\mathbb{E}[X_\tau] = \mathbb{E}[X_0]$.

(일반 정지시각의 경우는 Ch5-03 Optional Stopping Theorem에서)

---

## 💻 NumPy 구현 검증

### 실험 1 — Simple random walk의 마팅게일 성질

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
N = 1000
n_paths = 10000

# Random walk paths
xi = rng.choice([-1, 1], size=(n_paths, N))
S = np.cumsum(xi, axis=1)

# E[S_n] should be 0 for all n
E_S_n = S.mean(axis=0)
print(f'E[S_1] = {E_S_n[0]:.4f}')
print(f'E[S_500] = {E_S_n[499]:.4f}')
print(f'E[S_1000] = {E_S_n[-1]:.4f}')
# 모두 ~0 (정리 1.2)

# Var(S_n) = n (iid 분산 합)
Var_S_n = S.var(axis=0)
print(f'\nVar(S_n) / n (몇 지점):')
for n in [10, 100, 1000]:
    print(f'n={n}: {Var_S_n[n-1]/n:.4f}')
# 모두 ~1
```

### 실험 2 — $S_n^2 - n$이 마팅게일인가

```python
# 이론: E[S_{n+1}² - (n+1) | F_n] = S_n² + 1 - n - 1 = S_n² - n
# Simple random walk: E[(S_n + ξ)²] = S_n² + 2 S_n E[ξ] + E[ξ²] = S_n² + 1

# 경험적 검증: S_n² - n의 평균이 시간 독립
for n in [10, 100, 500, 1000]:
    val = S[:, n-1]**2 - n
    print(f'E[S_{n}² - {n}] = {val.mean():.4f}')
# 모두 ~0 → S_n² - n이 마팅게일 (E[0] = 0 for all n)
```

### 실험 3 — $S_n^2$이 submartingale

```python
E_S_squared = (S**2).mean(axis=0)
print(f'E[S_n²] (몇 지점):')
for n in [10, 100, 500, 1000]:
    print(f'n={n}: {E_S_squared[n-1]:.4f}')   
# 1, 10, 100, 500, 1000 - 증가 → S² submartingale
# E[S_n²] = n (정확히)

plt.plot(np.arange(1, N+1), E_S_squared, label=r'$\mathbb{E}[S_n^2]$')
plt.plot(np.arange(1, N+1), np.arange(1, N+1), '--', label='n (이론)')
plt.xlabel('n'); plt.ylabel(r'$\mathbb{E}[S_n^2]$')
plt.legend(); plt.title(r'$S_n^2$는 submartingale (증가 경향)')
plt.grid(True, alpha=0.3); plt.show()
```

---

## 🔗 AI/ML 연결

**Value function as martingale**  
고정 정책 $\pi$ 하 $V^\pi(s) = \mathbb{E}[\sum_{t \geq 0} \gamma^t r_t | S_0 = s]$. 만약 $Z_t = V^\pi(S_t) + \sum_{k < t} \gamma^k r_k$라면 $Z_t$는 적절히 normalize된 martingale (with discounting). TD error의 zero mean property가 이에 기인.

**Score matching과 tower property**  
Denoising score matching (Vincent 2011): $\mathbb{E}[s_\theta(X_t) | X_t] = \nabla \log p_t(X_t)$를 학습. 조건부 기댓값 → martingale 구조로 해석.

**Online Convex Optimization**  
$f_t$가 adversarial convex function, $x_t$가 learner의 선택. Regret $R_T = \sum_t f_t(x_t) - \min_x \sum_t f_t(x)$. 마팅게일 성질 + Azuma로 $R_T = O(\sqrt{T})$ bound (Ch5-05).

**Self-supervised contrastive**  
Tower property on hidden representations: $\mathbb{E}[Z_t | \mathcal{F}_s] = Z_s$. InfoNCE loss의 expected gradient가 zero at optimum (mini-batch 내 martingale-like).

---

## ⚖️ 가정과 한계

**가정 — Integrability**  
$\mathbb{E}|X_n| < \infty$ 필수. 없으면 조건부 기댓값 정의 안 됨. Heavy-tailed process(Cauchy random walk)는 마팅게일이 아님 (지수 moment 없음).

**한계 — Filtration 선택에 의존**  
같은 과정도 필트레이션 선택에 따라 마팅게일 / 아님 달라짐. 자연 필트레이션이 가장 일반적이지만 larger filtration에서는 성질 잃을 수 있음.

**한계 — 마팅게일이 분포의 equivalence class**  
Scale + shift으로 무한히 많은 마팅게일. 마팅게일 성질만으로 specific 과정 특정 불가.

---

## 📌 핵심 정리

| 개념 | 정의 / 요약 |
|---|---|
| Martingale | $\mathbb{E}[X_{n+1}\|\mathcal{F}_n] = X_n$ |
| Submartingale | $\mathbb{E}[X_{n+1}\|\mathcal{F}_n] \geq X_n$ |
| Supermartingale | $\mathbb{E}[X_{n+1}\|\mathcal{F}_n] \leq X_n$ |
| $\varphi$ convex | $\varphi(X_n)$ submartingale (Jensen) |
| $X^2, \|X\|, e^X$ | submartingale (if integrable) |
| Doob martingale | $Z_n = \mathbb{E}[X\|\mathcal{F}_n]$ |
| Martingale transform | Predictable bet on fair game → fair game |
| Tower | $\mathbb{E}[X_n] = \mathbb{E}[X_0]$ |

**한 줄 요약**: 마팅게일은 "공정한 게임"의 수학적 추상. Sub/super로 유리/불리 게임을 구분, Jensen으로 convex 변환이 submartingale로 변환. 이 구조가 ML regret bound, RL value function, diffusion score 등 현대 AI의 여러 이론적 기반.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. $X_n = \prod_{k=1}^n (1 + \xi_k)$, $\xi_k$ iid with $\mathbb{E}\xi = 0$, $|\xi_k| < 1$. $X_n$이 마팅게일임을 보여라.

<details>
<summary>해설</summary>

Adapted, integrable (bounded).
$$\mathbb{E}[X_{n+1} | \mathcal{F}_n] = X_n \mathbb{E}[1 + \xi_{n+1} | \mathcal{F}_n] = X_n (1 + 0) = X_n.$$

**해석**: "곱 형태" 마팅게일 — compound return. Log-scale $\log X_n = \sum \log(1 + \xi_k)$은 합 형태이지만 martingale 아님 (Jensen으로 $\mathbb{E}[\log(1+\xi)] < \log \mathbb{E}[1+\xi] = 0$) — supermartingale.

**GBM과 연결**: $X_n \approx e^{S_n - n\sigma^2/2}$ 에서 $e^{\lambda B_t - \lambda^2 t/2}$ martingale과 연결.

</details>

**문제 2 (심화)**. $X$가 마팅게일, $\tau$가 정지시각. 정지 정리 (Ch5-03) 없이 $Y_n = X_{\min(n, \tau)}$가 마팅게일임을 보여라.

<details>
<summary>해설</summary>

$Y_n$ adapted (각 시각 $n$에서 $X_{\min(n, \tau)} \in \mathcal{F}_n$, $\{\tau \leq n\} \in \mathcal{F}_n$이므로).

Integrable: $|Y_n| \leq \max_{k \leq n}|X_k|$, 각 $|X_k|$ integrable.

Martingale property:
$$Y_{n+1} - Y_n = (X_{n+1} - X_n) \mathbf{1}_{\{\tau > n\}}.$$

$\{\tau > n\} = \Omega \setminus \{\tau \leq n\} \in \mathcal{F}_n$ (predictable indicator).

$$\mathbb{E}[Y_{n+1} - Y_n | \mathcal{F}_n] = \mathbf{1}_{\{\tau > n\}} \mathbb{E}[X_{n+1} - X_n | \mathcal{F}_n] = \mathbf{1}_{\{\tau > n\}} \cdot 0 = 0.$$

따라서 $Y$는 마팅게일. $\square$

**의의**: "정지된 마팅게일도 마팅게일". 정지시각에서 시스템을 멈춰도 공정성 유지. 이것이 Optional Stopping Theorem의 예비 단계.

</details>

**문제 3 (AI 연결)**. TD(0) update $V_\theta(s_t) \leftarrow V_\theta(s_t) + \alpha(r_t + \gamma V_\theta(s_{t+1}) - V_\theta(s_t))$에서 **TD error** $\delta_t = r_t + \gamma V_\theta(s_{t+1}) - V_\theta(s_t)$가 optimum에서 mean-zero martingale-like 성질을 갖는 이유를 설명하라.

<details>
<summary>해설</summary>

**Optimum의 정의**: $V^*(s) = \mathbb{E}_\pi[r_t + \gamma V^*(s_{t+1}) | S_t = s]$ (Bellman).

**TD error at $V^*$**:
$\delta_t^* = r_t + \gamma V^*(s_{t+1}) - V^*(s_t)$.

**조건부 기댓값**:
$\mathbb{E}[\delta_t^* | \mathcal{F}_t] = \mathbb{E}[r_t + \gamma V^*(s_{t+1}) | \mathcal{F}_t] - V^*(s_t) = V^*(s_t) - V^*(s_t) = 0$.

즉 **optimum에서 $\delta_t^*$는 martingale difference** — $\mathcal{F}_t$ 조건부 평균이 0.

**함의**:
1. **Unbiased update**: $\mathbb{E}[\alpha \delta_t^*] = 0$ → SGD의 unbiasedness.
2. **Variance analysis**: Martingale CLT 적용 가능 → 수렴률 분석.
3. **TD(λ) convergence**: Robbins-Monro + martingale convergence로 수렴.

**SGD에서의 martingale structure**:
일반적인 SGD gradient $\nabla_\theta L_{\text{mini-batch}}$가 $\mathbb{E}[\nabla L_{\text{full}} | \theta_t] = \nabla L_{\text{full}}$이면 unbiased. 이는 각 mini-batch gradient를 **martingale-like structure around full gradient**로 보는 관점.

**실전 함의**:
- **Target network**: $V^*$ 대신 $V_{\text{target}}$를 사용 → martingale difference 성질 깨짐 → slow moving target으로 안정화.
- **Double Q-learning**: Bias 줄이기 위한 구조 — martingale 관점에서 overestimation 문제 해결.
- **Advantage estimation**: $A(s, a) = Q(s, a) - V(s)$에서 baseline 빼기 → variance 감소, martingale 성질 유지.

**연결**: Ch5-02 (martingale 수렴), Ch5-05 (Azuma for RL regret)에서 자세히.

</details>

---

<div align="center">

◀ [Ch4-04. Birth-Death 과정](../ch4-continuous-markov/04-birth-death.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [02. 마팅게일 수렴 정리](./02-convergence-theorem.md)

</div>
