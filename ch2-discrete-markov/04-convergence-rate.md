# 04. 극한정리와 수렴률 — 스펙트럴 접근

## 🎯 핵심 질문

- $P^n \to \mathbf{1}\pi^T$의 수렴을 어떻게 엄밀히 증명하는가?
- 수렴률 $\|\mu_n - \pi\|_{TV} \leq C |\lambda_2|^n$은 왜 **제2고유값**이 결정하는가?
- **스펙트럴 분해** $P = \sum_k \lambda_k v_k u_k^T$에서 $\lambda = 1$ 성분이 정상분포, 나머지가 수렴 속도 지배하는 구조는 어떻게 읽는가?
- **Total variation distance**와 다른 거리(KL, $\chi^2$)의 수렴률 관계는?

---

## 🔍 왜 이 정리가 AI에서 중요한가

**MCMC의 mixing time**: 실전 샘플링에서 "몇 step 돌려야 정상분포에 충분히 가까운가"의 이론적 답이 $t_{\text{mix}}(\epsilon) \approx \frac{1}{1 - |\lambda_2|} \log(1/\epsilon)$ (Ch7-05). Score network 기반 샘플러의 수렴률 보장은 이 스펙트럴 gap에 의존.

**Power iteration for PageRank**: PageRank 계산은 $\pi^{(k+1)} = \pi^{(k)} P$의 반복. 수렴률 $|\lambda_2|^k$ — 웹 그래프의 경우 $|\lambda_2| \approx 0.85$ (dumping factor) → 약 100 iteration.

**RL value iteration 수렴률**: Bellman 연산자 $\mathcal{T}$의 contraction factor $\gamma$(discount)는 **$\mathcal{T}$의 lipschitz 상수** — 스펙트럴 gap $1 - \gamma$이 convergence rate.

**Diffusion model의 sampling step 수**: Score-SDE reverse process의 mixing time이 step 수를 결정. "1000 step으로 안 될 때 2000 step이면 될까?" — 스펙트럴 gap이 답.

---

## 📐 수학적 선행 조건

- [Ch2-03](./03-stationary-distribution.md): Perron-Frobenius, 정상분포 유일성
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): 스펙트럴 분해, Jordan 표준형
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Total variation, coupling

---

## 📖 직관적 이해

### 스펙트럴 분해의 형태

$P$가 diagonalizable (generic 경우): $P = V \Lambda V^{-1}$, $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_N)$.

$\lambda_1 = 1$ with 우고유벡터 $\mathbf{1}$, 좌고유벡터 $\pi$:
$$P = \mathbf{1}\pi^T + \sum_{k \geq 2} \lambda_k v_k u_k^T.$$

$P^n = \mathbf{1}\pi^T + \sum_{k \geq 2} \lambda_k^n v_k u_k^T$.

비주기 기약이면 $|\lambda_k| < 1$ for $k \geq 2$ → $\lambda_k^n \to 0$ → $P^n \to \mathbf{1}\pi^T$.

**제2고유값** $\lambda_2$ (in absolute value)가 가장 느리게 죽음 → **수렴률 지배**.

### Total Variation distance

두 분포 $\mu, \nu$의 **total variation distance**:
$$\|\mu - \nu\|_{TV} = \frac{1}{2}\sum_i |\mu_i - \nu_i| = \max_{A} |\mu(A) - \nu(A)|.$$

$[0, 1]$ 값, 확률분포 간 "최대 분리 가능 확률" 해석. $\|\cdot\|_{TV} = 0 \iff \mu = \nu$.

### Mixing time

$$t_{\text{mix}}(\epsilon) = \min\{n : \max_x \|P^n(x, \cdot) - \pi\|_{TV} \leq \epsilon\}.$$

즉 "어떤 초기 상태에서 시작하든 $\epsilon$ 이내로 정상에 가까워지는 최소 step".

**스펙트럴 gap** $\gamma = 1 - |\lambda_2|$. 큰 gap = 빠른 수렴.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Total Variation Distance

$\|\mu - \nu\|_{TV} := \frac{1}{2} \sum_i |\mu_i - \nu_i|$ for discrete $\mu, \nu$.

### 정의 4.2 — Mixing time

$$d(n) := \max_x \|P^n(x, \cdot) - \pi\|_{TV}, \quad t_{\text{mix}}(\epsilon) := \min\{n : d(n) \leq \epsilon\}.$$

### 정의 4.3 — Spectral Gap

비주기·기약 체인의 **스펙트럴 gap**:
$$\gamma := 1 - |\lambda_2|,$$
여기서 $|\lambda_2| = \max\{|\lambda| : \lambda \neq 1, \text{ eigenvalue of } P\}$. 

Reversible 체인에서 더 통상적인 정의는 $\gamma = 1 - \lambda_2$ (실고유값, Ch2-05 참조).

### 정의 4.4 — Relaxation time

$t_{\text{rel}} := 1/\gamma$. Mixing time과의 관계 (아래 정리 4.5).

---

## 🔬 정리와 증명

### 정리 4.1 (유한상태 기약·비주기 체인의 수렴)

$P$가 유한상태 기약·비주기 확률행렬이면 임의 초기분포 $\mu_0$에 대해
$$\mu_n := \mu_0 P^n \to \pi \quad (n \to \infty)$$
**pointwise** (각 성분에서).

*증명 — Coupling 방법.*  
두 체인 $(X_n, Y_n)$을 각각 $\mu_0, \pi$에서 시작해 **같은 noise**로 구동. 첫 만남 시각 $\tau = \inf\{n : X_n = Y_n\}$.
- $Y_n \sim \pi$ for all $n$ (정상에서 시작).
- $\tau < \infty$ a.s. (기약·비주기 + 유한상태 → 만나는 것이 확률 1).

$n \geq \tau$ 이후 $X_n = Y_n \sim \pi$ (강한 마르코프). 따라서
$$\|\mu_n - \pi\|_{TV} \leq \mathbb{P}(X_n \neq Y_n) = \mathbb{P}(\tau > n) \to 0.$$
$\square$

### 정리 4.2 (수렴률 — 스펙트럴 bound)

$P$가 **diagonalizable**하고 기약·비주기이면
$$\|\mu_n - \pi\|_{TV} \leq C |\lambda_2|^n$$
어떤 상수 $C$로.

*증명.* $P = \mathbf{1}\pi^T + \sum_{k \geq 2} \lambda_k v_k u_k^T$. $(\mu_0 - \pi) \mathbf{1} = 0$이므로
$$\mu_0 P^n - \pi = \mu_0 (P^n - \mathbf{1}\pi^T) = \sum_{k \geq 2} \lambda_k^n (\mu_0 v_k) u_k^T.$$
$l^1$ norm:
$$\|\mu_n - \pi\|_1 \leq \sum_{k \geq 2} |\lambda_k|^n |\mu_0 v_k| \|u_k\|_1 \leq C \cdot |\lambda_2|^n.$$
TV distance는 $\|\cdot\|_1/2$. $\square$

### 정리 4.3 (Jordan 일반화)

$P$가 diagonalizable 아니면 Jordan block 형태 $(\lambda_2 I + N)^n = \lambda_2^n I + \binom{n}{1}\lambda_2^{n-1} N + \cdots$로 polynomial · exponential. 결론:
$$\|\mu_n - \pi\|_{TV} \leq C \cdot n^{k-1} |\lambda_2|^n$$
여기서 $k$는 $\lambda_2$의 Jordan block size. 지수 부분이 dominant.

### 정리 4.4 (TV와 KL 거리의 비교)

**Pinsker's inequality**: $\|\mu - \nu\|_{TV} \leq \sqrt{\frac{1}{2} D_{KL}(\mu \| \nu)}$.

따라서 $D_{KL}(\mu_n \| \pi) \to 0$이면 TV 수렴도 따른다. 역은 일반적으로 거짓.

### 정리 4.5 (Mixing time bound)

Reversible 기약·비주기 체인에서 $\pi_{\min} = \min_i \pi_i > 0$ (유한상태).
$$t_{\text{rel}} \log \frac{1}{2\epsilon} \leq t_{\text{mix}}(\epsilon) \leq t_{\text{rel}} \log \frac{1}{\epsilon \pi_{\min}}.$$

구체적으로 $t_{\text{mix}}(\epsilon) = O\left(\frac{1}{\gamma} \log \frac{1}{\epsilon}\right)$.

*증명 스케치.* 상한은 $\mathcal{L}^2 (\pi)$ 스펙트럴 분해로, 하한은 "경쟁적 partition" (Levin-Peres §12). $\square$

---

## 💻 NumPy 구현 검증

### 실험 1 — 스펙트럴 예측과 실측 TV 수렴률 비교

```python
import numpy as np
import matplotlib.pyplot as plt

P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5],
])

# 정상분포
eigvals, eigvecs = np.linalg.eig(P.T)
pi = np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1))])
pi = pi / pi.sum()

# |λ_2|
lam2 = sorted(np.abs(eigvals))[-2]
print(f'|λ_2| = {lam2:.4f}, spectral gap = {1 - lam2:.4f}')

# TV distance at each step
mu = np.array([1.0, 0.0, 0.0])
tvs = []
for n in range(50):
    tvs.append(0.5 * np.abs(mu - pi).sum())
    mu = mu @ P
tvs = np.array(tvs)

plt.semilogy(tvs, 'o-', markersize=4, label='실측 TV')
plt.semilogy(lam2 ** np.arange(50), '--', label=r'$|\lambda_2|^n$ 이론 bound')
plt.xlabel('n'); plt.ylabel('TV distance (log)')
plt.title('수렴률 — 제2고유값이 결정')
plt.legend(); plt.grid(True, which='both', alpha=0.3)
plt.show()
```

### 실험 2 — Mixing time 측정

```python
# ε-mixing time vs spectral gap
def mixing_time(P, eps=0.05, n_max=1000):
    pi = np.real(np.linalg.eig(P.T)[1][:, 0])
    pi = pi / pi.sum()
    d = np.inf
    for n in range(n_max):
        Pn = np.linalg.matrix_power(P, n)
        d = max(0.5 * np.abs(Pn[x] - pi).sum() for x in range(len(pi)))
        if d <= eps:
            return n
    return n_max

# 여러 체인 비교
Ps = {
    'Slow': np.array([[0.9, 0.1], [0.1, 0.9]]),       # |λ_2| = 0.8
    'Medium': np.array([[0.6, 0.4], [0.4, 0.6]]),      # |λ_2| = 0.2
    'Fast': np.array([[0.5, 0.5], [0.5, 0.5]]),        # |λ_2| = 0
}
for name, P in Ps.items():
    ev = np.linalg.eigvals(P)
    lam2 = sorted(np.abs(ev))[-2]
    t = mixing_time(P)
    print(f'{name:>7}: |λ_2|={lam2:.3f}, t_mix(0.05)={t}, '
          f'이론 t_rel·log(1/ε) ≈ {1/(1-lam2) * np.log(1/0.05):.1f}')
```

### 실험 3 — Coupling 기반 수렴 시각화

```python
# 두 체인을 같은 noise로 결합하여 meeting time 관찰
P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5],
])

rng = np.random.default_rng(0)
n_trials = 10000
meeting_times = []
for _ in range(n_trials):
    x, y = 0, 2  # 서로 다른 시작
    for n in range(1, 1000):
        u = rng.random()
        # 공통 noise로 coupled transition
        cx = np.cumsum(P[x]); cy = np.cumsum(P[y])
        x = np.searchsorted(cx, u)
        y = np.searchsorted(cy, u)
        if x == y:
            meeting_times.append(n); break

plt.hist(meeting_times, bins=50)
plt.xlabel('meeting time'); plt.ylabel('frequency')
plt.title(f'Coupling meeting time: mean={np.mean(meeting_times):.2f}')
plt.show()
```

---

## 🔗 AI/ML 연결

**MCMC burn-in과 mixing time**  
실전 MCMC에서 burn-in 길이 $\approx t_{\text{mix}}$. Gelman-Rubin $\hat{R}$로 체인 간 일치도 확인 → $\hat{R} < 1.1$이면 수렴 판정 (경험적 mixing time 근사).

**Value Iteration의 수렴**  
Bellman backup $V^{(k+1)} = \mathcal{T} V^{(k)}$에서 $\|V^{(k)} - V^*\|_\infty \leq \gamma^k \|V^{(0)} - V^*\|_\infty$. $\gamma$(discount factor)가 contraction rate = 스펙트럴 gap. $\gamma = 0.99$면 약 $100/\log(1/0.99) \approx 10000$ iteration 필요.

**Score-SDE sampling step 수**  
Reverse SDE의 이산화에서 step 수 $N$. Mixing time 이론이 "필요한 $N$"의 하한 제공. 실전에서 EDM (Karras 2022)의 smart schedule이 유효 mixing rate 개선.

**Deep Equilibrium Model**  
DEQ (Bai et al. 2019)은 $z^* = f_\theta(z^*, x)$ 고정점 학습. Forward pass = fixed point iteration, 수렴률은 $f$의 Lipschitz 상수 (contraction). 수렴 failure = 큰 $|\lambda_2|$.

**Diffusion Guidance의 수렴**  
Classifier-free guidance가 "목표 분포"를 변경 → 체인의 정상분포 shift → mixing time도 재계산 필요.

---

## ⚖️ 가정과 한계

**가정 — Diagonalizable**  
대부분의 확률행렬이 diagonalizable하지만, 병합 반복 eigenvalue는 Jordan block 가능. 이 경우 polynomial factor $n^{k-1}$ 부가.

**한계 — 고차원에서 spectral gap 소멸**  
고차원 ($d \gg 1$)에서 $|\lambda_2| \to 1$이 흔함 (curse of dimensionality). MCMC mixing time이 $\mathcal{O}(\exp(d))$가 될 수 있음. 이를 극복하려 HMC (Ch7-04)와 같은 gradient-aware sampler가 필요.

**한계 — Spectral gap 직접 계산 비용**  
큰 그래프/상태공간에서 $\lambda_2$ 계산은 $O(N^3)$ → 불가능. Power iteration으로 approximation, 또는 Cheeger 부등식 (conductance $\Phi$를 통한 $\lambda_2$ bound).

**한계 — Non-reversible chains**  
Reversible 체인은 실고유값 → 직관적 분석. Non-reversible은 복소 고유값 → 진동 모드. Ch2-05에서 reversible 체인의 특수성.

---

## 📌 핵심 정리

| 결과 | 수식 |
|---|---|
| 수렴 (기약·비주기·유한) | $P^n \to \mathbf{1}\pi^T$ |
| 수렴률 | $\|\mu_n - \pi\|_{TV} \leq C |\lambda_2|^n$ |
| Spectral gap | $\gamma = 1 - |\lambda_2|$ |
| Mixing time | $t_{\text{mix}}(\epsilon) = O(\frac{1}{\gamma} \log \frac{1}{\epsilon})$ |
| Pinsker | $\|\mu - \nu\|_{TV} \leq \sqrt{\frac{1}{2} D_{KL}(\mu\|\nu)}$ |
| Coupling bound | $\|\mu_n - \pi\|_{TV} \leq \mathbb{P}(\tau > n)$ |

**한 줄 요약**: 기약·비주기 체인은 $P^n \to \mathbf{1}\pi^T$, 수렴률은 **제2고유값 $|\lambda_2|$**가 지배. 스펙트럴 gap $\gamma = 1 - |\lambda_2|$가 mixing time을 결정.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. $P = \begin{pmatrix} 0.8 & 0.2 \\ 0.5 & 0.5 \end{pmatrix}$의 제2고유값을 구하고 $\epsilon = 0.01$의 mixing time을 추정하라.

<details>
<summary>해설</summary>

고유값: $\det(P - \lambda I) = (0.8 - \lambda)(0.5 - \lambda) - 0.1 = \lambda^2 - 1.3 \lambda + 0.3 = 0$.  
$\lambda = \frac{1.3 \pm \sqrt{1.69 - 1.2}}{2} = \frac{1.3 \pm 0.7}{2}$ → $\lambda_1 = 1, \lambda_2 = 0.3$.

스펙트럴 gap $\gamma = 0.7$, $t_{\text{rel}} = 1/0.7 \approx 1.43$.

Mixing time: $t_{\text{mix}}(0.01) \approx t_{\text{rel}} \log(1/0.01) = 1.43 \cdot 4.6 \approx 7$.

실제: 정상분포 $\pi = (5/7, 2/7)$. $P^n (1,0) - \pi$의 TV가 $0.3^n$ 스케일로 감소 → $0.3^7 \approx 0.0002 \ll 0.01$, 정확히 맞음.

</details>

**문제 2 (심화)**. Spectral gap이 $\gamma$인 체인에서 $t_{\text{mix}}(1/4) = \Theta(t_{\text{rel}})$임을 보여라 (up to logarithmic factor).

<details>
<summary>해설</summary>

**상한**: 정리 4.5에서 $t_{\text{mix}}(1/4) \leq t_{\text{rel}} \log(4/\pi_{\min})$. 유한상태에서 $\pi_{\min}$은 상수로 취급 → $O(t_{\text{rel}})$.

**하한**: 특정 $x, A$가 존재해 $|P^n(x, A) - \pi(A)| \geq |\lambda_2|^n / C$ (스펙트럴 분해에서 $\lambda_2$ 기여분). $|\lambda_2|^n \geq 1/4$이려면 $n \leq t_{\text{rel}} \log 4$. 따라서 $t_{\text{mix}}(1/4) \geq t_{\text{rel}} \log 4 = \Omega(t_{\text{rel}})$.

종합: $t_{\text{mix}}(1/4) = \Theta(t_{\text{rel}})$ (유한상태 reversible 체인).

**해석**: Spectral gap이 mixing time의 "scale"을 결정. $\log$ factor는 초기 분포와 정상 분포 간 거리에서 나옴.

</details>

**문제 3 (AI 연결)**. Diffusion model이 1000 step으로 sampling할 때, 이 step 수를 "mixing time"과 연결할 수 있는가? 이산 → 연속 극한(Score-SDE)에서 spectral gap 개념이 어떻게 변형되는가?

<details>
<summary>해설</summary>

**Diffusion step vs mixing time**:
DDPM의 reverse process는 **비정상 Markov chain** (각 step의 noise schedule 다름). 따라서 "고정된 정상분포"가 없음 — 전통적 mixing time 정의 직접 적용 불가. 그러나 유사 개념:
- 각 step $t$에서 "현재 marginal $p_t(x_t)$와 target $p_0^{\text{data}}$의 KL"의 감소율
- $N = 1000$ step이 필요한 것은 각 step이 "작은 KL 감소"를 제공하기 때문 — 총 감소량 = $\sum_t$ per-step 감소

**이산 → 연속 극한**:
Score-SDE의 reverse process $d\bar X = (-b + g^2 \nabla \log p_t) dt + g d\bar B$. 이는 SDE — 연속 mixing rate 이론 적용 가능.
- **Overdamped Langevin** $dX = -\nabla U dt + \sqrt{2} dB$의 정상분포 $\pi \propto e^{-U}$, 수렴률은 **spectral gap of the generator** $\mathcal{L} = -\nabla U \cdot \nabla + \Delta$.
- 이것이 SDE Deep Dive Ch4 (Fokker-Planck, Log-Sobolev)의 주제.

**연속 spectral gap**:
이산의 $1 - |\lambda_2|$가 연속에서는 **Poincaré constant** 또는 **Log-Sobolev constant** $\lambda$. $\pi \propto e^{-U}$의 **Bakry-Émery 조건** $\nabla^2 U \succeq \lambda I$가 충분조건.

**DDPM 실전**:
- Linear noise schedule → 느린 mixing
- Cosine schedule (iDDPM) → 효율 개선
- EDM (Karras) → 최적에 가까운 스케줄

**해석**: "1000 step이 필요한가?"의 답은 "target data 분포의 Log-Sobolev / Poincaré 상수"에 달림. 복잡한 다봉(multi-modal) 분포 → 작은 상수 → 많은 step 필요. 단순한 가우시안 → 큰 상수 → 적은 step.

</details>

---

<div align="center">

◀ [03. 정상분포(Stationary Distribution)와 Perron-Frobenius](./03-stationary-distribution.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [05. Reversibility와 Detailed Balance](./05-detailed-balance.md)

</div>
