# 05. MCMC 수렴 진단과 혼합 시간(Mixing Time)

## 🎯 핵심 질문

- **Mixing time** $t_{\text{mix}}(\epsilon)$의 정의 — 체인이 정상분포에 얼마나 빨리 도달하는가?
- Spectral gap과의 관계 $t_{\text{mix}} \sim 1/(1 - |\lambda_2|)$?
- 실전 진단 도구: **Gelman-Rubin $\hat R$**, **ESS**(Effective Sample Size), **trace plot**의 사용법?
- MCMC의 "failure modes" — stuck in mode, high autocorrelation, non-stationarity 등 어떻게 식별?

---

## 🔍 왜 진단이 AI에서 중요한가

**MCMC가 수렴 안 했는데 수렴했다고 착각 위험**: 잘못된 posterior 추정 → 잘못된 prediction, 잘못된 uncertainty. Bayesian inference의 신뢰성 근간.

**Stan/PyMC/NumPyro의 routine diagnostics**: $\hat R < 1.01$, ESS $> 400$ — 표준 검증 기준.

**Reproducibility**: MCMC 결과 공유 시 diagnostics 포함 필수 (Gelman 2017 "Bayesian Data Analysis" 표준).

---

## 📐 수학적 선행 조건

- [Ch2-04](../ch2-discrete-markov/04-convergence-rate.md): Spectral gap, mixing time
- [Ch2-06](../ch2-discrete-markov/06-ergodic-theorem.md): Ergodic theorem, autocorrelation
- [Ch7-01 ~ Ch7-04](./01-mcmc-idea.md): MCMC algorithms

---

## 📖 직관적 이해

### "수렴했다"의 의미

Two aspects:
1. **Distribution convergence**: $\mu_n \to \pi$ in TV. Mixing time.
2. **Ergodic convergence**: $\hat f_n \to \mathbb{E}_\pi f$. 시간평균 수렴.

두 measures 모두 spectral gap에 의존하지만 서로 다른 aspect 측정.

### Gelman-Rubin $\hat R$

여러 chain을 서로 다른 초기점에서 돌려 비교:
- **Within-chain variance** $W$ vs **between-chain variance** $B$.
- 수렴하면 $W \approx B/n$ → $\hat R = \sqrt{(n-1)/n + B/(nW)} \to 1$.

**해석**: 다른 초기점에서 시작한 chain들이 "같은 분포로 수렴" = 모두 정상분포에 도달.

**기준**: $\hat R < 1.01$ 권장 (Vehtari 2021). 구버전 $\hat R < 1.1$는 too loose.

### Effective Sample Size (ESS)

$\sigma_f^2 = \text{Var}_\pi(f)(1 + 2\sum \rho_k)$ (Ch2-06).

$\text{ESS} = \frac{n}{1 + 2\sum \rho_k} = \frac{n \cdot \text{Var}_\pi(f)}{\sigma_f^2}$.

"상관된 $n$ 샘플이 실효 iid 몇 개인가". ESS < 400이면 추정 신뢰 낮음.

### Trace plot

Sample $x^{(1)}, x^{(2)}, \ldots$ vs iteration. 시각적으로:
- **Good mixing**: "hairy caterpillar" — rapid fluctuation around mean.
- **Bad mixing**: Slow drift, stuck in regions, visible autocorrelation.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Mixing Time

$t_{\text{mix}}(\epsilon) := \min\{n : \max_x \|P^n(x, \cdot) - \pi\|_{TV} \leq \epsilon\}$.

**$\epsilon = 1/4$** 가장 common. $t_{\text{rel}} = 1/\gamma$, $t_{\text{mix}}(\epsilon) = O(t_{\text{rel}} \log(1/\epsilon))$.

### 정의 5.2 — Gelman-Rubin $\hat R$

$m$ chains, $n$ samples each. For scalar $\theta$:
- Within-chain: $W = \frac{1}{m}\sum_j s_j^2$ ($s_j^2$ = chain $j$ 내 variance)
- Between-chain: $B = \frac{n}{m-1}\sum_j (\bar\theta_j - \bar\theta_{..})^2$
- Marginal variance estimate: $\hat V = \frac{n-1}{n} W + \frac{1}{n} B$.
- $\hat R = \sqrt{\hat V / W}$.

수렴: $\hat R \to 1$.

### 정의 5.3 — Effective Sample Size (ESS)

$\hat{\text{ESS}} = \frac{n}{1 + 2\sum_{k=1}^K \hat\rho_k}$, $K$ = cutoff (autocorrelation sign change).

$\hat\rho_k = \hat\gamma_k / \hat\gamma_0$, $\hat\gamma_k = \frac{1}{n}\sum (x_i - \bar x)(x_{i+k} - \bar x)$.

### 정의 5.4 — Integrated Autocorrelation Time (IAT)

$\tau = 1 + 2\sum \rho_k$. ESS = $n/\tau$.

---

## 🔬 정리와 증명

### 정리 5.1 — Spectral Gap과 Mixing Time

Reversible $\pi$-chain with spectral gap $\gamma$, $\pi_{\min} = \min_x \pi(x)$:
$$\frac{1}{2\gamma} \log\frac{1}{2\epsilon} \leq t_{\text{mix}}(\epsilon) \leq \frac{1}{\gamma} \log\frac{1}{\epsilon\pi_{\min}}.$$

(Ch2-04 정리 4.5 요약.)

### 정리 5.2 — $\hat R$의 수렴 보장

$m \geq 2$ independent chains from over-dispersed initialization. $\hat R \to 1$ as $n \to \infty$.

*증명 직관*. CLT로 각 chain의 $\bar\theta_j \to \mu$. $W \to \text{Var}_\pi$, $B \to n \text{Var}_\pi$ (between-chain variance scales with $n$ as chains converge). $\hat V/W \to 1$. $\square$

### 정리 5.3 — ESS의 CLT

$\sqrt{\text{ESS}} (\hat f - \mathbb{E}_\pi f) \xrightarrow{d} \mathcal{N}(0, \text{Var}_\pi f)$.

즉 ESS가 **실효 iid 샘플 수**로서 CLT variance 결정.

---

## 💻 NumPy 구현 검증

### 실험 1 — $\hat R$ 계산

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# Target: N(0, 1). RW-MH with poor step size.
def log_pi(x): return -0.5 * x**2

def mh_chain(x0, step, n_iter):
    x = x0; samples = [x]
    for _ in range(n_iter):
        y = x + rng.normal(0, step)
        if np.log(rng.random()) < log_pi(y) - log_pi(x):
            x = y
        samples.append(x)
    return np.array(samples)

# 4 chains from different initializations
init_points = [-5, -2, 2, 5]
n_iter = 5000
chains = np.array([mh_chain(x0, step=1.5, n_iter=n_iter) for x0 in init_points])

def gelman_rubin(chains):
    m, n = chains.shape
    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)
    overall_mean = chain_means.mean()
    
    W = chain_vars.mean()
    B = n * chain_vars.var(ddof=0) * m / (m-1)   # between variance
    # Actually: B = n * sum((chain_mean - overall_mean)^2) / (m-1)
    B = n * ((chain_means - overall_mean)**2).sum() / (m - 1)
    
    V_hat = (n - 1)/n * W + (1/n) * B
    return np.sqrt(V_hat / W)

# R-hat at various truncation lengths
n_vals = [100, 500, 2000, 5000]
for n in n_vals:
    Rhat = gelman_rubin(chains[:, :n])
    print(f'n = {n}: R-hat = {Rhat:.4f}')
# 처음엔 > 1 (not converged), 나중엔 ~1
```

### 실험 2 — ESS 계산

```python
def autocorr(x, max_lag=100):
    x = x - x.mean()
    var = x.var()
    return np.array([np.mean(x[:-k]*x[k:])/var if k > 0 else 1 
                     for k in range(max_lag)])

def effective_sample_size(x):
    rho = autocorr(x, max_lag=min(100, len(x)//3))
    # Geyer's positive sequence estimator
    K = 1
    while K + 1 < len(rho) and rho[K] + rho[K+1] > 0:
        K += 2
    tau = 1 + 2 * rho[1:K+1].sum()
    return len(x) / tau

# 하나의 chain
single_chain = mh_chain(x0=0, step=1.5, n_iter=10000)
ess = effective_sample_size(single_chain[1000:])   # exclude burn-in
print(f'Chain length: {len(single_chain) - 1000}, ESS: {ess:.1f}')

# Effect of step size
for step in [0.1, 0.5, 1.5, 5.0]:
    chain = mh_chain(x0=0, step=step, n_iter=10000)
    ess = effective_sample_size(chain[1000:])
    print(f'step={step}: ESS = {ess:.0f}')
# 적정 step size에서 ESS 최대
```

### 실험 3 — Trace plot 시각화

```python
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

# Good mixing
chain_good = mh_chain(x0=0, step=1.5, n_iter=3000)
axes[0].plot(chain_good); axes[0].set_title('Good mixing (step=1.5)'); axes[0].grid(True, alpha=0.3)

# Bad: slow drift (small step)
chain_bad1 = mh_chain(x0=0, step=0.1, n_iter=3000)
axes[1].plot(chain_bad1); axes[1].set_title('Bad: slow mixing (step=0.1)'); axes[1].grid(True, alpha=0.3)

# Bad: high rejection (large step)
chain_bad2 = mh_chain(x0=0, step=10.0, n_iter=3000)
axes[2].plot(chain_bad2); axes[2].set_title('Bad: high rejection (step=10)'); axes[2].grid(True, alpha=0.3)

for ax in axes:
    ax.set_xlabel('iteration'); ax.set_ylabel('x')
plt.tight_layout(); plt.show()
```

---

## 🔗 AI/ML 연결

**ArviZ (Python MCMC diagnostics library)**  
$\hat R$, ESS, trace plots, rank plots, energy diagnostics (HMC)의 표준 구현. Stan/PyMC output 분석 필수 도구.

**Bayesian Workflow (Gelman et al. 2020)**  
체계적 validation: prior predictive → fit → posterior predictive + diagnostics + sensitivity analysis. 각 단계에서 MCMC 진단.

**Probabilistic Programming의 "Divergences"**  
HMC/NUTS에서 "divergent transitions" (leapfrog step이 너무 큼) 감지 → reparameterization 힌트. Funnel distributions 등.

**Large-Scale BNN training**  
SG-MCMC의 진단 어려움 — $\hat R$ 계산이 parameter 많을 때 부담. Summary statistics (marginal test quantities)에 focus.

---

## ⚖️ 가정과 한계

**한계 — $\hat R$이 false positive 가능**  
"여러 chain 일치" but 모두 **같은 local mode**에 빠짐 → $\hat R \approx 1$이지만 실제 수렴 안 함. Over-dispersed initialization 필수.

**한계 — ESS 추정의 편향**  
Autocorrelation 추정이 noisy — short chain에서 ESS overestimated 가능. Multiple chain ESS가 더 신뢰.

**한계 — 고차원 diagnostics**  
$10^6$ parameters에서 모든 parameter에 $\hat R$ 계산 비용. Summary quantities에 집중.

**현대 권장사항** (Vehtari et al. 2021):
- $\hat R < 1.01$ (not 1.1)
- ESS > 400 per chain
- No divergent transitions in HMC
- Split-$\hat R$ + rank normalization

---

## 📌 핵심 정리

| 진단 도구 | 정의 | 기준 |
|---|---|---|
| Mixing time | $\min\{n : \|\mu_n - \pi\|_{TV} \leq \epsilon\}$ | — |
| Spectral gap | $\gamma = 1 - \|\lambda_2\|$ | 클수록 빠른 수렴 |
| $\hat R$ | $\sqrt{\hat V/W}$ | $< 1.01$ |
| ESS | $n/(1 + 2\sum \rho_k)$ | $> 400$ |
| IAT | $1 + 2\sum \rho_k$ | 작을수록 좋음 |
| Trace plot | $x^{(n)}$ vs $n$ | Hairy caterpillar |

**한 줄 요약**: MCMC diagnostics는 spectral gap 기반 mixing time 이론의 실전 적용. $\hat R$(여러 chain 일치)과 ESS(autocorrelation으로부터 실효 샘플 수)가 표준 체크. 진단 통과 없이는 MCMC 결과 신뢰 불가 — Bayesian AI의 reproducibility 근간.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. Bimodal target $\pi = 0.5 \mathcal{N}(-5, 1) + 0.5 \mathcal{N}(5, 1)$에 RW-MH 적용. Single chain으로 $\hat R$이 $\approx 1$이어도 수렴 안 할 수 있음을 설명하라.

<details>
<summary>해설</summary>

**상황**: 단일 chain이 한 mode (e.g., around $-5$)에 갇힘. Chain이 "locally converged" — 그 mode의 정상분포만 따름.

**$\hat R$ 계산 with single chain**: $\hat R$은 정의상 여러 chain 필요. 하지만 "split-$\hat R$" — single chain을 두 half로 분할해 계산하면 두 half 모두 같은 mode에 머무름 → $\hat R \approx 1$.

**문제**: "Local convergence ≠ global convergence". 다른 mode를 놓침.

**올바른 diagnosis**:
1. **Over-dispersed multiple chains**: Initialize 각 chain을 서로 다른 mode 근처에서. 일부 chain은 $-5$ 근처, 다른 일부 $+5$ 근처. 그러면 **$\hat R \gg 1$** 드러남.

2. **Prior predictive checks**: True distribution의 특성 (bimodal 알려진) vs sampler output.

3. **Multi-modal aware samplers**:
   - Parallel tempering: 여러 temperature chains, swap moves.
   - Simulated tempering.
   - Mode-hopping MCMC.

**현실 BNN**:
- Posterior가 often multi-modal (e.g., permutation symmetries in NN).
- Deep Ensembles (multiple independent training runs) = modes 탐색 proxy.
- "Ensemble of chains from different inits" 표준.

**교훈**: $\hat R$은 "다른 chain들이 같은 분포로 수렴"을 시사. 하지만 **"초기 분포가 충분히 spread"**되어 전체 target support 탐색해야 validity.

</details>

**문제 2 (심화)**. ESS를 공식 $\sigma_f^2 = \text{Var}_\pi f (1 + 2\sum \rho_k)$로 derive하라 (Ch2-06과 연결).

<details>
<summary>해설</summary>

**정상 chain + $X_0 \sim \pi$ 가정**.

$\hat f_n = \frac{1}{n} \sum_{k=1}^n f(X_k)$.

$\text{Var}(\hat f_n) = \frac{1}{n^2} \sum_{k, l} \text{Cov}(f(X_k), f(X_l))$.

정상 chain: $\text{Cov}(f(X_k), f(X_l)) = \text{Var}_\pi f \cdot \rho_{|k-l|}$ where $\rho_h = \text{Corr}(f(X_0), f(X_h))$.

$\text{Var}(\hat f_n) = \frac{\text{Var}_\pi f}{n^2} \sum_{k, l} \rho_{|k-l|}$.

**$\sum_{k, l} \rho_{|k-l|}$** 평가:
$= \sum_{k, l} \rho_{|k-l|}$ where $k, l \in \{1, \ldots, n\}$. Group by $h = |k - l|$:
$= n \rho_0 + 2 \sum_{h=1}^{n-1} (n - h) \rho_h$
$\approx n \left(1 + 2\sum_{h \geq 1} \rho_h\right)$ for large $n$ (assuming $\sum |\rho_h| < \infty$).

$\text{Var}(\hat f_n) \approx \frac{\text{Var}_\pi f}{n}(1 + 2\sum \rho_h) = \frac{\sigma_f^2}{n}$.

**ESS 정의**: "iid 샘플 $N$개의 variance = $\text{Var}_\pi f / N$". 이 variance가 $\sigma_f^2/n$과 같으려면 $N = n \cdot \text{Var}_\pi f / \sigma_f^2 = n / (1 + 2\sum\rho_h)$.

$\text{ESS} = n / \tau$, $\tau = 1 + 2\sum \rho_h$ = **integrated autocorrelation time**.

**CLT**:
$\sqrt{n}(\hat f_n - \mathbb{E}_\pi f) \xrightarrow{d} \mathcal{N}(0, \sigma_f^2)$.
$\sqrt{\text{ESS}}(\hat f_n - \mathbb{E}_\pi f) \xrightarrow{d} \mathcal{N}(0, \text{Var}_\pi f)$ — iid-like rate.

**실전 추정**:
$\hat \rho_h = \hat\gamma_h / \hat\gamma_0$. $\hat\tau$에서 **cutoff $K$** 선택이 tricky — Geyer's positive sequence, monotone sequence estimator.

**Ch2-06 연결**: Ergodic CLT $\sqrt n(\hat f - \mathbb{E}_\pi f) \xrightarrow{d} \mathcal{N}(0, \sigma_f^2)$가 이 derivation의 이론적 정당성. ESS는 MCMC variance를 iid terms로 재표현.

</details>

**문제 3 (AI 연결)**. Large-scale BNN training with SG-MCMC. Millions of parameters에서 진단의 challenges와 실전 workarounds를 논하라.

<details>
<summary>해설</summary>

**Challenges in high-dim BNN**:

1. **Parameter-wise $\hat R$ infeasible**: 10^6 parameters → 10^6 $\hat R$ 계산 비용. 또한 많은 parameters의 marginals이 uninformative (symmetry).

2. **Permutation symmetry**: 같은 function을 여러 weight configurations가 represent → posterior multimodal, $\hat R \gg 1$ but model predictions 동일.

3. **Sub-sampling noise**: SG-MCMC의 mini-batch가 non-reversible → standard $\hat R$ 이론 깨짐.

4. **Long chains infeasible**: Each iteration expensive → $n < 10^4$ 제한.

**실전 workarounds**:

**(1) Functional diagnostics instead of parametric**:
- Test predictions $f(\theta, x_{\text{test}})$의 $\hat R$ 계산.
- Log-likelihood $\ell(\theta)$의 $\hat R$.
- Function space이 much lower-dim than parameter space.

**(2) Multiple chains with deep ensembles**:
- Train 5-10 models with different seeds.
- Treat each as "sample from posterior" (imperfect).
- Summary statistics of predictions.

**(3) Calibration diagnostics**:
- Expected Calibration Error (ECE)
- Reliability diagrams
- Out-of-distribution detection performance

**(4) Approximate ESS via variance**:
- Compute running variance of predictions.
- Compare to iid bound: if variance plateau → convergence.

**(5) Divergence checks (HMC-specific)**:
- SG-HMC's friction term can be monitored.
- Energy conservation as proxy for correctness.

**(6) Prior predictive check**:
- Before training, verify that prior over $\theta$ gives reasonable prior over predictions.
- $\hat R$ of prior samples = baseline.

**Modern BNN tools**:
- **Laplace approximation**: Gaussian around MAP, easy to diagnose.
- **SWAG** (Stochastic Weight Averaging-Gaussian): Gaussian from SGD trajectory.
- **Hamiltonian flows**: Neural network approximations to HMC.

**Open problems**:
- Principled MCMC for modern deep nets (scale + diagnostic).
- Parameter symmetries in posteriors (modes equivalent functionally).
- "Convergence" 개념 자체가 over-parameterized regime에서 재정의 필요.

**연결**: Bayesian deep learning이 "exact MCMC + modern scale"의 open frontier. Classical diagnostics이 직접 적용 안 되고, domain-specific summaries + calibration + functional convergence로 대체.

</details>

---

<div align="center">

◀ [04. Hamiltonian Monte Carlo (HMC)](./04-hamiltonian-mc.md) &nbsp;·&nbsp; [📚 README](../README.md)

</div>

---

<div align="center">

### 🎉 Stochastic Processes Deep Dive 완주!

**7 chapters · 35 documents · 80+ theorems · Total ~18,000 lines of rigorous mathematics**

**다음 여정**: [SDE Deep Dive](https://github.com/iq-ai-lab/sde-deep-dive) — 이토 적분·Fokker-Planck·Anderson 시간반전·Score-SDE로 이어지는 **확률미분방정식의 심층 탐구**. Ch6-05의 이차변분 $\langle B\rangle_t = t$이 그대로 $(dB)^2 = dt$로, Ch7의 Langevin이 Score-based generative model로 확장됩니다.

</div>
