# 06. 에르고딕 정리(Ergodic Theorem)

## 🎯 핵심 질문

- **에르고딕 정리**: 왜 **시간평균 = 공간평균**인가 — $\frac{1}{n}\sum_{k=1}^n f(X_k) \to \mathbb{E}_\pi[f]$가 a.s. 성립하는 조건은?
- 이 정리가 **MCMC의 이론적 근거**인 이유는 — 샘플링의 정당성이 어떻게 나오는가?
- 에르고딕성과 기약성의 관계 — 기약·양재귀는 에르고딕을 보장하는가?
- **CLT 버전** $\sqrt{n}(\bar f_n - \mathbb{E}_\pi f) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$에서 **asymptotic variance** $\sigma^2$는 어떻게 계산되는가?

---

## 🔍 왜 이 정리가 AI에서 중요한가

**MCMC 정당성의 근간**: MCMC로 posterior expectation $\mathbb{E}_\pi[f]$를 추정할 때, 단일 장기 체인의 시간평균이 목표 기댓값에 수렴한다는 보장이 **에르고딕 정리**. 이 정리 없이 MCMC는 무의미.

**RL의 정책 평가**: Policy gradient $\nabla J = \mathbb{E}_{s \sim d^\pi}[\nabla \log \pi(a|s) Q(s, a)]$의 Monte Carlo 추정. 단일 trajectory를 쭉 돌리면서 평균내면 에르고딕 정리로 참값 수렴.

**Stochastic Gradient Descent의 수렴 분석**: SGD를 Markov chain으로 간주(노이즈 있는 업데이트). 수렴 해석 시 에르고딕 성질이 가정으로 들어감.

**CLT-기반 신뢰구간**: 에르고딕 CLT가 MCMC 추정의 **표준오차**를 제공 — Gelman-Rubin, ESS의 이론적 기반.

---

## 📐 수학적 선행 조건

- [Ch2-01 ~ Ch2-05](./01-markov-property.md): 마르코프, 재귀성, 정상분포, 스펙트럴
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): SLLN, CLT, Borel-Cantelli
- 함수해석: Poisson 방정식 $(I - P) g = f - \mathbb{E}_\pi f$

---

## 📖 직관적 이해

### 시간평균 vs 공간평균

단일 sample path $\omega$를 따라 $f$의 **시간평균**:
$$\bar{f}_n(\omega) = \frac{1}{n} \sum_{k=1}^n f(X_k(\omega)).$$

**공간평균** (정상분포에 대한 기댓값):
$$\mathbb{E}_\pi[f] = \sum_i \pi_i f(i).$$

**에르고딕 정리**: 기약·양재귀 체인에서 **시간평균 = 공간평균 a.s.**:
$$\bar{f}_n \to \mathbb{E}_\pi[f] \quad \text{a.s.}$$

> **비유**: 한 사람이 한 도시에서 오래 살면서 자기 동네의 평균 기온을 기록(시간평균)하는 것 vs 하루에 도시 모든 사람의 집 온도를 기록(공간평균). 에르고딕이면 두 평균이 같다 — "장기 관찰 ≈ 전체 분포에서 무작위 관찰".

### iid SLLN과의 차이

iid 경우 **SLLN**: $\frac{1}{n}\sum Z_i \to \mathbb{E}[Z]$ a.s. (Kolmogorov).  
마르코프 경우: $X_k$가 **독립이 아님**(시퀀스 의존) — 더 정교한 분석 필요. 그러나 결과는 유사.

**핵심 통찰**: 에르고딕 체인은 "충분히 긴 시간 후 초기 분포를 잊음" → 각 시각에서의 marginal이 정상 $\pi$에 가까워짐 → 평균적 관점에서 iid 유사.

### Asymptotic Variance

CLT 버전: $\sqrt{n}(\bar f_n - \mathbb{E}_\pi f) \xrightarrow{d} \mathcal{N}(0, \sigma^2_f)$.

$\sigma^2_f$는 **autocorrelation**을 반영: 
$$\sigma^2_f = \text{Var}_\pi(f) + 2\sum_{k \geq 1} \text{Cov}_\pi(f(X_0), f(X_k)).$$

양의 autocorrelation → $\sigma^2_f$ 커짐 → ESS 감소 (Ch7-05).

---

## ✏️ 엄밀한 정의

### 정의 6.1 — 에르고딕 (Ergodic)

확률공간 $(\Omega, \mathcal{F}, \mathbb{P})$ 위 시프트 불변 측도 $\mathbb{P}$가 **에르고딕**이다 ⇔ 모든 shift-invariant 사건 $A$ ($\theta^{-1} A = A$)가 $\mathbb{P}(A) \in \{0, 1\}$.

Markov chain 맥락: 기약·양재귀 체인은 자동으로 에르고딕.

### 정의 6.2 — Asymptotic Variance

$\sigma^2_f := \text{Var}_\pi(f) + 2 \sum_{k=1}^\infty \text{Cov}_\pi(f(X_0), f(X_k))$

($\text{Cov}_\pi$는 $X_0 \sim \pi$, $X_k$가 체인 진행 후 분포 — Markov 구조로 $\text{Cov}_\pi(f(X_0), f(X_k)) = \mathbb{E}_\pi[f(X_0) \mathbb{E}[f(X_k) | X_0]] - (\mathbb{E}_\pi f)^2$)

### 정의 6.3 — Poisson Equation

주어진 $f$에 대해 $g : E \to \mathbb{R}$를 $(I - P) g = f - \mathbb{E}_\pi f$의 해 (mean-zero component에 대해). 이 $g$를 이용해 CLT 증명.

---

## 🔬 정리와 증명

### 정리 6.1 (이산 Markov 에르고딕 정리 — SLLN 버전)

$\{X_n\}$이 기약·양재귀 이산 Markov 체인이고 $\pi$가 정상분포, $f : E \to \mathbb{R}$가 $\mathbb{E}_\pi |f| < \infty$이면
$$\frac{1}{n}\sum_{k=1}^n f(X_k) \to \mathbb{E}_\pi[f] \quad \text{a.s.} \quad (n \to \infty)$$
어떤 초기분포에서 시작해도.

### 증명 스케치 — "Regenerative" 기법

양재귀 상태 $x_0 \in E$ 선택 (기약이므로 임의). 첫 반환시각 $T_1 = T_{x_0}$, 그 이후 반환 시각들 $T_2, T_3, \ldots$. 각 "cycle" $T_j$에서 $T_{j+1} - 1$까지의 경로 조각을 $C_j$로 놓자.

**관찰**: 강한 마르코프 성질로 $C_j$들이 **iid** (상태 $x_0$에서 재시작, 독립). 각 cycle의 길이 $L_j = T_{j+1} - T_j$가 iid with mean $\mu = \mathbb{E}_{x_0}[T_{x_0}]$ — 양재귀로 유한.

Cycle 내 $f$ 합 $S_j = \sum_{k=T_j}^{T_{j+1}-1} f(X_k)$도 iid.  
$\mathbb{E}[S_j] = \mathbb{E}_{x_0}\left[\sum_{k=0}^{T_{x_0}-1} f(X_k)\right] = \mu \mathbb{E}_\pi[f]$ (정상분포 해석, 정리 3.3).

N cycle 후 합:
$$\sum_{k=1}^{T_N} f(X_k) = \sum_{j=1}^{N} S_j.$$

시각 $n$이 $T_N \leq n < T_{N+1}$ 구간에 있을 때
$$\frac{1}{n} \sum_{k=1}^n f(X_k) \approx \frac{N \cdot \mu \mathbb{E}_\pi[f]}{N \cdot \mu} = \mathbb{E}_\pi[f].$$

iid SLLN 적용으로 $\frac{S_j}{\bar L_j} \to \mathbb{E}_\pi[f]$ a.s. $\square$

> **핵심 아이디어**: Markov 체인을 iid cycle의 합으로 "분해" — iid 도구(SLLN, CLT) 적용 가능.

### 정리 6.2 (CLT 버전 — Asymptotic Normality)

같은 가정에 $\mathbb{E}_\pi[f^2] < \infty$와 Poisson 방정식의 해 $g$가 존재하면
$$\sqrt{n}(\bar f_n - \mathbb{E}_\pi f) \xrightarrow{d} \mathcal{N}(0, \sigma^2_f),$$
여기서 $\sigma^2_f$는 정의 6.2.

### 증명 스케치

Poisson 방정식 해 $g$를 이용해 **martingale decomposition**:
$$f(X_k) - \mathbb{E}_\pi f = g(X_k) - (Pg)(X_k) = \underbrace{g(X_k) - g(X_{k+1})}_{\text{telescoping}} + \underbrace{(g(X_{k+1}) - (Pg)(X_k))}_{=: M_{k+1} - M_k \text{ martingale diff}}.$$

Telescoping part는 $o(\sqrt n)$으로 사라짐. Martingale 부분에 martingale CLT (Durrett) 적용:
$$\frac{1}{\sqrt n} \sum (M_{k+1} - M_k) \xrightarrow{d} \mathcal{N}(0, \mathbb{E}_\pi[(g(X_1) - (Pg)(X_0))^2]).$$

계산하면 이것이 $\sigma^2_f$. $\square$

### 정리 6.3 (Reversible 체인에서 $\sigma^2$의 스펙트럴 표현)

Reversible $(P, \pi)$, $f$가 $\pi$-orthogonal to constants ($\mathbb{E}_\pi f = 0$). 스펙트럴 분해 $f = \sum_k c_k u_k$ (고유함수 $u_k$ with $\lambda_k$). 그러면
$$\sigma^2_f = \sum_k c_k^2 \frac{1 + \lambda_k}{1 - \lambda_k}.$$

*증명 스케치.* Reversible 체인의 autocovariance는 $\text{Cov}_\pi(f(X_0), f(X_k)) = \sum_l c_l^2 \lambda_l^k$. 합:
$$\sigma^2_f = \sum_l c_l^2 + 2 \sum_{k \geq 1} \sum_l c_l^2 \lambda_l^k = \sum_l c_l^2 \left(1 + \frac{2 \lambda_l}{1 - \lambda_l}\right) = \sum_l c_l^2 \frac{1 + \lambda_l}{1 - \lambda_l}.$$
$\square$

> **함의**: $\lambda_k$가 1에 가까우면 denominator $1 - \lambda_k$ 작음 → $\sigma^2_f$ 커짐. **Spectral gap이 작으면 추정 오차 커짐**. 이것이 ESS = $n / \text{IAT}$(integrated autocorrelation time)의 이론적 기반.

### 정리 6.4 (Effective Sample Size, ESS)

IAT $\tau_f = \sigma^2_f / \text{Var}_\pi(f)$. ESS = $n / \tau_f$. 이는 "상관된 $n$ 샘플이 iid 몇 개에 해당하는가"의 측정.

---

## 💻 NumPy 구현 검증

### 실험 1 — 에르고딕 SLLN 직접 확인

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# 3-state 기약·비주기 체인
P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5],
])

# 정상분포
eigvals, eigvecs = np.linalg.eig(P.T)
pi = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1))])
pi = pi / pi.sum()

# f(state) = state + 1
f_vals = np.array([1.0, 2.0, 3.0])
true_Ef = pi @ f_vals
print(f'공간평균 E_π[f] = {true_Ef:.4f}')

# 시간평균 수렴 관찰
N = 100_000
state = 0
running_avg = np.zeros(N)
cumsum = 0.0
for n in range(N):
    cumsum += f_vals[state]
    running_avg[n] = cumsum / (n + 1)
    state = rng.choice(3, p=P[state])

plt.plot(running_avg, label='시간평균 (1/n) Σ f(X_k)')
plt.axhline(true_Ef, color='k', linestyle='--', label=r'$\mathbb{E}_\pi[f]$')
plt.xscale('log')
plt.xlabel('n'); plt.ylabel('평균')
plt.legend(); plt.grid(True, alpha=0.3)
plt.title('에르고딕 정리 — 시간평균 → 공간평균')
plt.show()

print(f'시간평균 (n=1e5): {running_avg[-1]:.4f}, 이론: {true_Ef:.4f}')
```

### 실험 2 — CLT로 asymptotic variance 측정

```python
# 독립 trial들로 sqrt(n) * (sample mean - true mean)의 분포 확인
n_trials = 5000
N = 10_000

sample_means = np.zeros(n_trials)
for t in range(n_trials):
    state = rng.choice(3, p=pi)
    s = 0.0
    for _ in range(N):
        s += f_vals[state]
        state = rng.choice(3, p=P[state])
    sample_means[t] = s / N

# 이론 σ² (reversible이 아니라 직접 계산)
rescaled = np.sqrt(N) * (sample_means - true_Ef)
print(f'Var(√n (f̄ - Ef)) = {rescaled.var():.4f}')
# → 이론 σ²_f (정의 6.2)에 근접

plt.hist(rescaled, bins=50, density=True, alpha=0.5, label='실측')
x = np.linspace(rescaled.min(), rescaled.max(), 200)
sigma = rescaled.std()
plt.plot(x, np.exp(-x**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi)), 
         'r-', label=r'$\mathcal{N}(0, \hat\sigma^2)$')
plt.legend(); plt.title('에르고딕 CLT 검증')
plt.show()
```

### 실험 3 — IAT와 ESS 계산

```python
# 단일 chain의 autocorrelation → IAT
state = 0
N = 100_000
f_path = np.zeros(N)
for n in range(N):
    f_path[n] = f_vals[state]
    state = rng.choice(3, p=P[state])

f_centered = f_path - f_path.mean()
var_f = (f_centered**2).mean()

# autocovariance R(k) = E[f_0 f_k] - μ²
max_lag = 200
R = np.array([np.mean(f_centered[:-k] * f_centered[k:]) if k > 0 else var_f
              for k in range(max_lag + 1)])

# IAT = 1 + 2 Σ R(k)/R(0)
# 실전에서는 R이 0 이하로 떨어지기 시작하는 지점까지 합산 (Geyer 1992)
rho = R / R[0]
cutoff = np.argmax(rho < 0) if np.any(rho < 0) else max_lag
IAT = 1 + 2 * rho[1:cutoff].sum()
ESS = N / IAT
print(f'IAT ≈ {IAT:.2f}, ESS = {ESS:.0f} (out of {N})')
# → 작은 IAT = 체인이 빠르게 decorrelate, 큰 ESS = 효율적 MCMC
```

---

## 🔗 AI/ML 연결

**MCMC 추정의 표준오차**  
$$\hat{\mathbb{E}}_\pi[f] = \bar f_n, \quad \text{SE}(\hat{\mathbb{E}}_\pi[f]) = \sigma_f / \sqrt{n} = \sigma_f / \sqrt{\text{ESS}}.$$
MCMC diagnostics(Ch7-05)가 ESS를 계산하는 것은 실질적 sample size 추정. 이론적으로 iid $n$ 샘플과 동등한 정밀도.

**Policy Gradient의 variance**  
REINFORCE 등의 Monte Carlo gradient estimator는 **장기 경로에 대한 에르고딕 평균**. Autocorrelation이 많으면 (같은 state 반복 방문) variance 커짐 → **baseline subtraction**과 **GAE**가 asymptotic variance 감소 장치.

**Replay Buffer와 on-policy vs off-policy**  
Off-policy (Q-learning with replay) = 과거 trajectory에서 iid sampling → 효율적. On-policy (A2C) = 현재 $\pi$에서의 에르고딕 평균 → Markov 구조 유지되지만 상관된 샘플.

**Diffusion model의 evaluation**  
FID score는 generated vs real image distribution 간 거리. MCMC가 아니지만 "generative 분포의 expectation" 추정 측면에서 유사 에르고딕 구조. Sampling 많이 할수록 $\hat{\text{FID}}$가 참값에 수렴.

**Hamiltonian MC의 ESS 개선**  
HMC(Ch7-04)가 RWMH 대비 ESS 수배~수십배 개선 — **spectral gap 개선** 때문. 정리 6.3의 $\sigma_f^2 \propto \frac{1+\lambda_k}{1-\lambda_k}$에서 $\lambda$ 감소 → 제곱 error 감소.

---

## ⚖️ 가정과 한계

**가정 — $\mathbb{E}_\pi|f| < \infty$**  
Heavy-tailed $f$ (예: $f(x) = 1/x$ near 0)에서 에르고딕 정리 failure. 실전 MCMC에서 importance ratio가 heavy-tail이면 (likelihood 비율이 큰 변동) SLLN 불완전 수렴.

**가정 — 기약·양재귀**  
기약성 깨짐 → 일부 상태 방문 안함 → 시간평균이 공간평균이 아닌 **local 평균**만. MCMC 실전에서 "여러 mode 중 한 개에만 갇힘" — tempering, parallel tempering 등으로 해결.

**한계 — 수렴 속도 vs 유한 샘플**  
정리는 asymptotic. 유한 $n$에서 얼마나 가까운가는 **mixing time**(Ch2-04)에 의존. $n < t_{\text{mix}}$이면 에르고딕 근사 나쁨.

**한계 — CLT의 적용성**  
$\sigma^2_f < \infty$가 가정. Null recurrent (예: 2D random walk)에서 $\sigma^2 = \infty$ → CLT 성립 안함. Variance를 유한하게 하려면 양재귀 + $\mathbb{E}_\pi f^2 < \infty$ 필요.

---

## 📌 핵심 정리

| 결과 | 수식 / 의미 |
|---|---|
| SLLN (Markov) | 기약·양재귀 + $\mathbb{E}_\pi\|f\| < \infty$ ⇒ $\bar f_n \to \mathbb{E}_\pi f$ a.s. |
| CLT | + $\mathbb{E}_\pi f^2 < \infty$ ⇒ $\sqrt n (\bar f_n - \mathbb{E}_\pi f) \xrightarrow{d} \mathcal{N}(0, \sigma_f^2)$ |
| $\sigma_f^2$ | $\text{Var}_\pi f + 2\sum_{k \geq 1} \text{Cov}_\pi(f(X_0), f(X_k))$ |
| Reversible spectral | $\sigma_f^2 = \sum_k c_k^2 \frac{1+\lambda_k}{1-\lambda_k}$ |
| ESS | $n/\tau_f = n \cdot \text{Var}_\pi f / \sigma_f^2$ |
| MCMC 근거 | 시간평균 = 공간평균 → 샘플링의 정당성 |

**한 줄 요약**: "시간평균 = 공간평균"은 기약·양재귀의 Markov 체인에서 자동으로 성립. 이것이 **MCMC의 이론적 근거**이며, CLT 버전이 추정 오차의 scale $\sigma_f^2$를 제공.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. 기약·비주기가 아니면 에르고딕 정리가 깨지는 예를 들어라.

<details>
<summary>해설</summary>

**비기약 예**: $P = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ (두 absorbing state). 시작이 0이면 $X_n \equiv 0$ → $\bar f_n = f(0)$, 시작이 1이면 $f(1)$. 두 개의 서로 다른 한계 → **"공간평균"이 유일하지 않음**(정상분포 두 개). 시간평균은 초기 분포에 따라 달라짐, SLLN의 본래 의미와 다름.

**주기 예** (유한상태에서): 2-state flip-flop $P = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$. 이는 기약이지만 주기 2. Marginal은 수렴 안 함, 하지만 **시간평균은 수렴**! 시간평균 = $\lim \frac{1}{n}\sum f(X_k) = \frac{1}{2}(f(0) + f(1)) = \mathbb{E}_\pi f$ ($\pi = (1/2, 1/2)$). 즉 에르고딕 정리는 **주기 상관없이** 성립(기약·양재귀만 있으면). 주기는 $P^n$ 수렴에만 영향.

**교훈**: SLLN 에르고딕은 기약·양재귀만 필요. $P^n$의 분포 수렴에는 비주기도 필요. 두 수렴을 혼동하지 말 것.

</details>

**문제 2 (심화)**. Reversible 체인 $(P, \pi)$에서 $f$가 특정 고유함수 $Pf = \lambda f$ ($|\lambda| < 1$)에 해당할 때, $\sigma_f^2$와 $\text{Var}_\pi f$의 비율 $\frac{\sigma_f^2}{\text{Var}_\pi f}$를 $\lambda$의 함수로 구하라. $\lambda \to 1$일 때 어떻게 되나?

<details>
<summary>해설</summary>

$f$가 고유함수이고 $\mathbb{E}_\pi f = 0$ 가정 (정리 5 문제 2 참조).

Autocovariance: $\text{Cov}_\pi(f(X_0), f(X_k)) = \mathbb{E}_\pi[f(X_0) f(X_k)] = \mathbb{E}_\pi[f(X_0) (P^k f)(X_0)] = \lambda^k \text{Var}_\pi f$.

$\sigma_f^2 = \text{Var}_\pi f + 2\sum_{k \geq 1} \lambda^k \text{Var}_\pi f = \text{Var}_\pi f \left(1 + \frac{2\lambda}{1 - \lambda}\right) = \text{Var}_\pi f \cdot \frac{1 + \lambda}{1 - \lambda}$.

**비율**: $\frac{\sigma_f^2}{\text{Var}_\pi f} = \frac{1 + \lambda}{1 - \lambda}$.

**$\lambda \to 1$**: 비율 $\to \infty$. 이는 **느리게 mixing하는 고유함수는 매우 큰 asymptotic variance**를 가짐. ESS = $n \cdot \frac{1 - \lambda}{1 + \lambda}$ → 0 as $\lambda \to 1$.

**실전 함의**: 체인의 spectral gap이 작으면($\lambda_2 \to 1$), $\lambda_2$에 해당하는 모드의 expectation 추정이 매우 비효율 → MCMC를 매우 오래 돌려야 함. HMC가 gradient로 $\lambda_2$ 줄이는 이유.

</details>

**문제 3 (AI 연결)**. RL 학습에서 "on-policy 샘플"의 시간 상관성이 학습 효율에 미치는 영향을 asymptotic variance 관점에서 분석하라. Replay buffer와 n-step 반환이 이 효율을 어떻게 개선하는가?

<details>
<summary>해설</summary>

**On-policy 상관성 문제**:
Policy $\pi$ 하 Markov chain of states, transitions. Gradient estimator $\hat g = \frac{1}{N}\sum_t \nabla \log \pi(A_t|S_t) Q(S_t, A_t)$. 에르고딕 CLT로 $\hat g$의 variance $\propto \sigma_g^2 / N$, 그러나 $\sigma_g^2$는 **per-step variance보다 $\frac{1+\rho}{1-\rho}$ 배 큼** ($\rho$ = autocorrelation).

예: $\rho = 0.9$면 $\sigma_g^2 / \text{Var}_\pi g = 19$. **ESS = $N/19$** — 효율 19배 감소.

**Replay Buffer의 효과**:
Buffer에서 uniform random sampling → temporal correlation 제거. iid 근사 → $\sigma_g^2 \to \text{Var}_\pi g$, ESS = $N$. 단, target policy $\pi_\theta$가 바뀜에 따라 buffer 데이터가 off-policy 됨 → importance sampling correction 필요 (DQN, SAC).

**n-step 반환**:
$n$-step return $R_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(S_{t+n})$. $n$ 크면 Monte Carlo (unbiased, high variance), 작으면 TD (biased, low variance). **TD(λ)**가 adaptive 중간 지점 제공.

**GAE (Generalized Advantage Estimation)**:
$A_t^{GAE(\lambda)} = \sum_l (\gamma\lambda)^l \delta_{t+l}$ — bias-variance trade-off의 최적화. $\lambda$ 튜닝으로 asymptotic variance 조절.

**종합**:
1. Replay buffer: temporal correlation 제거 → ESS ≈ N
2. N-step / GAE: variance reduction via bias trade-off
3. Target network: moving target 완화 → 수렴 안정화

이 모든 장치들이 "**에르고딕 asymptotic variance 최소화**"를 향한 engineering. 이론적 한계는 에르고딕 정리의 $\sigma_f^2$.

</details>

---

<div align="center">

◀ [05. Reversibility와 Detailed Balance](./05-detailed-balance.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [Ch3-01. Poisson 과정의 3가지 동치 정의](../ch3-poisson/01-three-equivalent-definitions.md)

</div>
