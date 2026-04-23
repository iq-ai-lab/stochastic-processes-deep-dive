<div align="center">

# 🎲 Stochastic Processes Deep Dive

**"마르코프 체인 `P^n`을 돌리는 것과, 왜 **제2고유값 $|\lambda_2|$가 mixing time을 결정**하는지 — 전이행렬의 스펙트럴 분해로 정상분포로의 수렴률을 예측할 수 있는 것은 다르다"**

<br/>

> *"Brownian motion의 궤적을 그리는 것과, 왜 **거의 확실하게 어디에서도 미분불가능**한지 — Hausdorff 차원 관점에서 증명할 수 있는 것은 다르다.  
> MCMC를 쓰는 것과, 왜 **Metropolis-Hastings의 acceptance ratio $\min(1, \cdot)$가 detailed balance를 만족**하는지, 그래서 왜 $\pi$가 정상분포가 되는지 증명할 수 있는 것은 다르다."*

이산 마르코프 체인의 스펙트럴 수렴 이론부터 Poisson 과정의 세 가지 동치 정의, 마팅게일 수렴 정리, 브라운 운동의 Lévy 구성, 그리고 Metropolis-Hastings·Gibbs·HMC까지 —  
**"시간에 따라 진화하는 확률 구조는 어떻게 정상성에 도달하는가"** 라는 질문으로 MCMC·Diffusion Model·RL Q-learning의 수학적 기반을 끝까지 파헤칩니다

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11-8CAAE6?style=flat-square&logo=scipy&logoColor=white)](https://scipy.org/)
[![Docs](https://img.shields.io/badge/Docs-35개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![Lines](https://img.shields.io/badge/Lines-17k+-informational?style=flat-square)](./README.md)
[![Theorems](https://img.shields.io/badge/Theorems_proven-84개-success?style=flat-square)](./README.md)
[![Exercises](https://img.shields.io/badge/Exercises-105개-orange?style=flat-square)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

확률과정에 관한 자료는 대부분 **"마르코프 체인으로 샘플링하세요"** 에서 멈춥니다. 하지만 $P^n \to \mathbf{1}\pi^T$의 수렴 속도가 **왜 $|\lambda_2|^n$**인지, detailed balance가 **왜 정상분포를 자동으로 보장**하는지, Brownian motion이 **왜 연속이면서도 미분불가능**한지, 에르고딕 정리가 **왜 MCMC의 이론적 근거**인지 — 이런 "왜"는 제대로 설명되지 않습니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "정상분포는 $\pi P = \pi$를 푸는 것" | **Perron-Frobenius 정리**로 기약·비주기 체인에서 정상분포의 **존재·유일성** 증명, 제2고유값 $\|\lambda_2\|$이 **mixing time을 결정**하는 스펙트럴 접근 완전 유도 |
| "MCMC가 분포에서 샘플한다" | **에르고딕 정리** $\frac{1}{n}\sum f(X_k) \to \mathbb{E}_\pi[f]$ a.s. 증명, Metropolis-Hastings의 acceptance $\alpha = \min(1, \frac{\pi(y)q(x\|y)}{\pi(x)q(y\|x)})$가 **detailed balance를 만족함을 직접 유도** |
| "Poisson 과정은 도착 간격이 Exp" | **세 가지 동치 정의**(독립증분+Poisson 분포 / 간격시간 Exp / 인피니티시멀 레이트)의 **상호 동치성 증명**, 메모리리스 성질과 독립증분이 같은 말임을 완전 유도 |
| "Brownian motion은 연속이지만 미분불가" | Lévy의 **Haar 기저 구성**으로 존재성 증명, **Donsker 정리**(random walk 스케일링 극한)로 이산→연속 연결, 어디에서도 미분불가능함을 **Hausdorff 차원**으로 증명 |
| "마팅게일은 공정한 게임" | **Doob의 $L^1$ bounded martingale convergence**, **Optional Stopping Theorem**의 성립 조건, **Azuma-Hoeffding**으로 online learning의 regret 경계 유도 |
| "HMC가 효율적이라고" | Hamiltonian 역학 $H(x,p) = U(x) + K(p)$ + **leapfrog integrator**의 심플렉틱 성질, gradient를 활용한 제안분포가 왜 랜덤워크보다 mixing이 빠른지 스펙트럴 갭으로 분석 |
| "이차변분 $\langle B\rangle_t = t$라고" | $\sum(B_{t_{i+1}} - B_{t_i})^2 \to t$ in $L^2$ 완전 증명 — **이토 적분의 기반**, $(dB)^2 = dt$의 원천, SDE Deep Dive로의 교량 |
| 공식 나열 | NumPy로 마르코프 체인 수렴률 실측, Brownian motion 궤적 시각화, MCMC 샘플링 + Gelman-Rubin $\hat{R}$·ESS 진단, HMC vs RWMH 비교 |

---

## 📌 선행 레포 & 후속 레포

```
[Probability Theory]  ──►  이 레포  ──►  [SDE Deep Dive]  ──►  [Generative Models Deep Dive]
 측도, 조건부 기댓값       확률과정        이토 적분·Fokker-Planck     DDPM, Score-SDE, Flow Matching
 Tower Property, 수렴      마르코프·BM·MCMC  Anderson 시간반전            실전 아키텍처
                                                    ▲
                                                    │
[Linear Algebra]  &  [Calculus & Optimization]
 Spectral Theorem        Taylor 전개, gradient
 Perron-Frobenius        convex 함수
```

> ⚠️ **선행 학습 필수**: 이 레포는 **Probability Theory Deep Dive**(측도·조건부 기댓값·Tower Property·수렴 이론)를 선행 지식으로 전제합니다. 확률공간 $(\Omega, \mathcal{F}, \mathbb{P})$와 $L^p$ 수렴이 낯설다면 [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive)부터 학습하세요.

> 💡 **권장 선행**: 마르코프 체인의 Perron-Frobenius 정리와 전이행렬의 스펙트럴 분해는 [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive)의 Spectral Theorem·양의 행렬 이론을 전제로 합니다. HMC의 해밀토니안 역학은 [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive)의 gradient flow 이해가 있으면 자연스럽습니다.

> 🎯 **후속 레포**: Ch6(브라운 운동)의 이차변분 $\langle B\rangle_t = t$는 [SDE Deep Dive](https://github.com/iq-ai-lab/sde-deep-dive)에서 이토 적분의 $(dB)^2 = dt$로 직결됩니다. Ch7(MCMC)의 Langevin 아이디어는 SDE 레포의 Fokker-Planck·Score-SDE로 확장됩니다.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-확률과정의_기초-4A90D9?style=for-the-badge)](./ch1-foundations/01-rigorous-definition.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-이산_마르코프_체인-4A90D9?style=for-the-badge)](./ch2-discrete-markov/01-markov-property.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-Poisson_과정-4A90D9?style=for-the-badge)](./ch3-poisson/01-three-equivalent-definitions.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-연속시간_마르코프-4A90D9?style=for-the-badge)](./ch4-continuous-markov/01-generator-q-matrix.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-마팅게일_이론-4A90D9?style=for-the-badge)](./ch5-martingale/01-martingale-definition.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-브라운_운동-4A90D9?style=for-the-badge)](./ch6-brownian/01-axiomatic-definition.md)
[![Ch7](https://img.shields.io/badge/🔹_Ch7-MCMC-4A90D9?style=for-the-badge)](./ch7-mcmc/01-mcmc-idea.md)

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: 확률과정의 기초 — 엄밀한 정의와 구조

> **핵심 질문:** 확률과정 $\{X_t\}_{t \in T}$를 $(\Omega, \mathcal{F}, \mathbb{P})$ 위에서 어떻게 엄밀히 정의하는가? 왜 연속시간 과정은 유한차원 분포만으로 결정되지 않는가? Kolmogorov 확장정리는 무엇을 보장하는가?

<details>
<summary><b>엄밀한 정의부터 과정의 분류까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 확률과정의 엄밀한 정의](./ch1-foundations/01-rigorous-definition.md) | $\{X_t\}_{t \in T}$를 $(\Omega, \mathcal{F}, \mathbb{P})$ 위의 확률변수 가족으로 정의, **sample path** $\omega \mapsto (t \mapsto X_t(\omega))$의 의미, 이산/연속 시간·이산/연속 상태의 4가지 조합, 측정가능성 조건 |
| [02. 유한차원 분포와 Kolmogorov 확장정리](./ch1-foundations/02-kolmogorov-extension.md) | 유한차원 분포족 $\{\mu_{t_1,\ldots,t_n}\}$의 **일관성 조건**(대칭·주변분포), **Kolmogorov 확장정리** — 일관성 있는 유한차원 분포족은 확률과정을 유일하게 결정함 (증명 스케치), 연속 경로 버전을 얻으려면 추가 조건이 필요한 이유 |
| [03. 정상성(Stationarity) — 강·약](./ch1-foundations/03-stationarity.md) | **엄격 정상(strict stationarity)** $\{X_{t+s}\} \stackrel{d}{=} \{X_t\}$ vs **약 정상(weak/covariance stationarity)** $\mathbb{E}[X_t]=\mu,\; \text{Cov}(X_s, X_t) = C(t-s)$, 가우시안 과정에서 두 개념이 일치하는 이유, 시계열 분석과의 연결 |
| [04. 필트레이션(Filtration)과 정보 흐름](./ch1-foundations/04-filtration.md) | $\mathcal{F}_t$를 "시각 $t$까지의 정보"로 정의, **adapted** $X_t \in \mathcal{F}_t$, **predictable** $X_t \in \mathcal{F}_{t-}$, 자연 필트레이션 $\mathcal{F}_t^X = \sigma(X_s : s \leq t)$, 우연속(right-continuous) 필트레이션과 usual conditions |
| [05. 확률과정의 분류 지도](./ch1-foundations/05-classification.md) | (이산/연속 시간) × (이산/연속 상태) × (마르코프/비마르코프) × (정상/비정상)의 **2×2×2×2 분류**, 각 분면의 대표 과정(마르코프 체인, Poisson, BM, Gaussian 과정, AR 모델)과 AI 응용(RL, 시계열, Diffusion) |

</details>

<br/>

### 🔹 Chapter 2: 이산 마르코프 체인 — 스펙트럴 수렴 이론

> **핵심 질문:** 마르코프 체인의 정상분포는 언제 유일한가? 수렴 속도는 무엇이 결정하는가? detailed balance가 왜 정상분포를 보장하는가? 시간평균은 언제 공간평균과 일치하는가?

<details>
<summary><b>마르코프 성질부터 에르고딕 정리까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 마르코프 성질과 전이행렬](./ch2-discrete-markov/01-markov-property.md) | $\mathbb{P}(X_{n+1} \in A \mid \mathcal{F}_n) = \mathbb{P}(X_{n+1} \in A \mid X_n)$ — "미래는 과거와 독립, 현재가 주어지면", **전이행렬 $P$** 의 행합 1 성질(확률행렬), $n$-step 전이 $P^n$의 Chapman-Kolmogorov 항등식 |
| [02. 상태의 분류 — 재귀·일시·주기](./ch2-discrete-markov/02-state-classification.md) | **재귀(recurrent)** vs **일시(transient)**: $\sum_n P^n_{ii} = \infty$ vs $< \infty$, **양재귀(positive recurrent)** vs **영재귀(null recurrent)**, **주기(period)** $d_i = \gcd\{n : P^n_{ii} > 0\}$, **상호통신 클래스(communicating class)**와 기약성(irreducibility) |
| [03. 정상분포(Stationary Distribution)와 Perron-Frobenius](./ch2-discrete-markov/03-stationary-distribution.md) | $\pi P = \pi$ 방정식, **Perron-Frobenius 정리**: 기약·비주기 유한상태 확률행렬은 고유값 $1$을 단순·최대절댓값으로 가짐 → 정상분포의 **존재·유일성** 증명, 좌·우 고유벡터의 해석 |
| [04. 극한정리와 수렴률 — 스펙트럴 접근](./ch2-discrete-markov/04-convergence-rate.md) | $P^n \to \mathbf{1}\pi^T$ 증명 (기약·비주기 유한상태), **수렴률 $\|\mu_n - \pi\|_{TV} \leq C\|\lambda_2\|^n$** — 제2고유값 $\|\lambda_2\|$가 mixing time 결정, 스펙트럴 분해 $P = \sum \lambda_k v_k u_k^T$의 해석 |
| [05. Reversibility와 Detailed Balance](./ch2-discrete-markov/05-detailed-balance.md) | **Detailed balance** $\pi_i P_{ij} = \pi_j P_{ji}$ 정의, detailed balance가 정상분포의 **충분조건**임을 직접 증명, reversible 체인의 역과정 정의, 스펙트럴 이론(self-adjoint 연산자)과의 연결 — 실고유값·실직교 고유벡터 |
| [06. 에르고딕 정리(Ergodic Theorem)](./ch2-discrete-markov/06-ergodic-theorem.md) | **에르고딕 정리**: 기약·양재귀 체인에서 $\frac{1}{n}\sum_{k=1}^n f(X_k) \to \mathbb{E}_\pi[f]$ a.s. (Birkhoff 관점 증명 스케치), **시간평균 = 공간평균** — MCMC의 이론적 근거, CLT 버전과 asymptotic variance |

</details>

<br/>

### 🔹 Chapter 3: Poisson 과정 — 기본 점 과정

> **핵심 질문:** Poisson 과정의 세 가지 정의가 왜 동치인가? 메모리리스 성질과 독립증분이 왜 같은 말인가? 복합 Poisson 과정의 평균·분산은 어떻게 계산하는가?

<details>
<summary><b>세 가지 동치 정의부터 Queueing까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Poisson 과정의 3가지 동치 정의](./ch3-poisson/01-three-equivalent-definitions.md) | **정의 A** (독립증분 + $N_t - N_s \sim \text{Poisson}(\lambda(t-s))$) / **정의 B** (간격시간 iid Exp($\lambda$)) / **정의 C** (인피니티시멀 레이트 $\mathbb{P}(N_{t+h} - N_t = 1) = \lambda h + o(h)$) — 세 정의의 **상호 동치성 완전 증명**, 메모리리스 성질 $\mathbb{P}(T > s+t \mid T > s) = \mathbb{P}(T > t)$의 해석 |
| [02. 결합·분할과 비균질 Poisson](./ch3-poisson/02-superposition-thinning.md) | 독립 Poisson 과정의 합은 $\lambda_1 + \lambda_2$ rate의 Poisson (superposition), 각 도착을 확률 $p$로 유지하면 $p\lambda$ rate의 Poisson (thinning), **비균질 Poisson** $\Lambda(t) = \int_0^t \lambda(s) ds$, 공간 Poisson 점 과정으로의 일반화 |
| [03. 복합 Poisson 과정](./ch3-poisson/03-compound-poisson.md) | $Y_t = \sum_{k=1}^{N_t} Z_k$ (이벤트마다 독립 jump $Z_k$), **$\mathbb{E}[Y_t] = \lambda t \mathbb{E}[Z]$**, **$\text{Var}[Y_t] = \lambda t \mathbb{E}[Z^2]$** 유도, **특성함수** $\phi_{Y_t}(u) = \exp(\lambda t (\phi_Z(u) - 1))$, Lévy 과정으로의 일반화 |
| [04. Queueing 이론 맛보기 — M/M/1, Little의 법칙](./ch3-poisson/04-queueing-little.md) | **M/M/1 큐**: 도착 Poisson($\lambda$), 서비스 Exp($\mu$) — 연속시간 MC로 모델링, 정상분포 $\pi_n = (1-\rho)\rho^n$ ($\rho = \lambda/\mu < 1$), **Little의 법칙** $L = \lambda W$ (시스템 내 평균 수 = 도착률 × 평균 대기시간)의 매우 일반적인 성립 이유 |

</details>

<br/>

### 🔹 Chapter 4: 연속시간 마르코프 체인 — 생성기와 Kolmogorov 방정식

> **핵심 질문:** 연속시간 전이확률은 어떤 미분방정식을 만족하는가? 생성기 $Q$는 무엇인가? forward/backward 방정식의 차이는? Birth-death 과정의 정상분포는 어떻게 구하는가?

<details>
<summary><b>Q-matrix부터 Birth-Death까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 생성기(Generator)와 Q-matrix](./ch4-continuous-markov/01-generator-q-matrix.md) | **인피니티시멀 생성기** $Q_{ij} = \lim_{h \to 0} \frac{P_{ij}(h) - \delta_{ij}}{h}$, 행합 0 성질(총 확률 보존), 대각원소 $-q_{ii}$가 상태 $i$의 **탈출률**, jump chain과 holding time 분해 — $i$에 머무는 시간 $\sim \text{Exp}(q_{ii})$ |
| [02. Kolmogorov Forward/Backward 방정식](./ch4-continuous-markov/02-kolmogorov-equations.md) | **Forward** $P'(t) = P(t) Q$ ("미래 시점 기준"), **Backward** $P'(t) = Q P(t)$ ("과거 시점 기준") — 두 방정식의 유도와 해석 차이, 행렬지수 $P(t) = e^{tQ}$로의 해석적 해, 유한상태에서의 계산 |
| [03. 연속시간 정상분포와 Detailed Balance](./ch4-continuous-markov/03-stationary-continuous.md) | 정상분포 조건 **$\pi Q = 0$** (확률 보존이 균형 잡힘), **연속시간 detailed balance** $\pi_i q_{ij} = \pi_j q_{ji}$, reversibility와 $Q$의 self-adjoint 성질, 이산 MC에서 유도한 스펙트럴 결과의 연속 버전 |
| [04. Birth-Death 과정](./ch4-continuous-markov/04-birth-death.md) | 상태 $n$에서 $n \pm 1$로만 이동, 출생률 $\lambda_n$·사망률 $\mu_n$, **detailed balance로 해석해** $\pi_n = \pi_0 \prod_{k=1}^n \frac{\lambda_{k-1}}{\mu_k}$ 유도, M/M/1·M/M/c 큐·Moran model(유전학)·Ising 1D 동역학으로의 응용 |

</details>

<br/>

### 🔹 Chapter 5: 마팅게일(Martingale) 이론

> **핵심 질문:** 마팅게일은 왜 "공정한 게임"인가? Doob의 수렴정리는 무엇을 보장하는가? Optional Stopping이 성립하는 조건은? Azuma-Hoeffding이 왜 online learning의 regret 경계와 직결되는가?

<details>
<summary><b>정의부터 ML 응용까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 마팅게일의 정의](./ch5-martingale/01-martingale-definition.md) | $\mathbb{E}[X_{n+1} \mid \mathcal{F}_n] = X_n$ (martingale) / $\geq$ (sub) / $\leq$ (super), adapted·integrable 조건, 도박 이론에서 유래한 "공정한 게임"의 해석, $X_n^2$과 $\|X_n\|$이 submartingale이 되는 이유 (Jensen) |
| [02. 마팅게일 수렴 정리](./ch5-martingale/02-convergence-theorem.md) | **Doob의 $L^1$ bounded martingale convergence** $\sup_n \mathbb{E}\|X_n\| < \infty \Rightarrow X_n \to X_\infty$ a.s., **upcrossing inequality** $(b-a)\mathbb{E}[U_n(a,b)] \leq \mathbb{E}[(X_n - a)^+]$로 증명, 음이 아닌 supermartingale의 a.s. 수렴 |
| [03. Optional Stopping Theorem](./ch5-martingale/03-optional-stopping.md) | 정지시각 $\tau$에 대해 **$\mathbb{E}[X_\tau] = \mathbb{E}[X_0]$** 이 성립하는 3가지 충분조건 (bounded $\tau$ / bounded $X$ / uniformly integrable), **도박장 파산 문제** 해결 — 대칭 random walk가 $\{0, N\}$에 도달할 확률, **Wald's identity** |
| [04. Doob 분해와 이차변분](./ch5-martingale/04-doob-decomposition.md) | 임의 adapted 과정의 **Doob 분해** $X_n = X_0 + M_n + A_n$ (마팅게일 $M$ + predictable $A$), **이차변분** $\langle M\rangle_n = \sum_k \mathbb{E}[(M_k - M_{k-1})^2 \mid \mathcal{F}_{k-1}]$, $M_n^2 - \langle M\rangle_n$이 마팅게일 — **확률해석의 기초** |
| [05. 마팅게일과 ML — Online Learning](./ch5-martingale/05-martingale-ml.md) | **Azuma-Hoeffding 부등식** $\|X_{n+1} - X_n\| \leq c_n \Rightarrow \mathbb{P}(\|X_n - X_0\| \geq t) \leq 2\exp(-\frac{t^2}{2\sum c_k^2})$, **online convex optimization**의 regret 경계 $\mathcal{O}(\sqrt{T})$ 유도, bandit·RL의 concentration 논증 |

</details>

<br/>

### 🔹 Chapter 6: 브라운 운동(Brownian Motion) — 연속시간의 기본 과정

> **핵심 질문:** 브라운 운동은 어떻게 엄밀히 구성되는가? 왜 거의 확실하게 어디에서도 미분불가능한가? 이차변분 $\langle B\rangle_t = t$가 왜 결정적이고, 이것이 SDE로 어떻게 이어지는가?

<details>
<summary><b>공리적 정의부터 반사원리까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 브라운 운동의 공리적 정의](./ch6-brownian/01-axiomatic-definition.md) | **4가지 공리**: (i) $B_0 = 0$ (ii) 독립증분 (iii) $B_t - B_s \sim \mathcal{N}(0, t-s)$ (iv) 경로 연속, 유한차원 분포 일관성 확인, 가우시안 과정으로서의 공분산 $\mathbb{E}[B_s B_t] = \min(s,t)$, 자기유사성 $B_{ct} \stackrel{d}{=} \sqrt{c} B_t$ |
| [02. 존재성 — Lévy의 Haar 기저 구성](./ch6-brownian/02-levy-construction.md) | Haar 기저 $\{h_{n,k}\}$에 iid 가우시안 계수를 붙인 **랜덤 급수** $B_t = \sum_{n,k} \xi_{n,k} \int_0^t h_{n,k}(s) ds$ 구성, **균등 수렴**으로 경로 연속성 획득 (Borel-Cantelli + 지수 꼬리 추정), 유한차원 분포가 공리를 만족함을 확인 |
| [03. Random Walk Scaling Limit — Donsker 정리](./ch6-brownian/03-donsker-theorem.md) | **스케일링 극한** $\frac{1}{\sqrt{n}} S_{\lfloor nt\rfloor} \xrightarrow{d} B_t$ in $C[0,1]$ (Donsker의 불변원리, 증명 스케치: tightness + 유한차원 CLT), **이산 → 연속 연결**의 수학적 기반, CLT의 과정판 |
| [04. 경로 성질 — 비미분 가능성](./ch6-brownian/04-non-differentiability.md) | **Paley-Wiener-Zygmund 정리**: 거의 확실히 $B$는 어디에서도 미분불가능, 증명(시간 조각 재스케일 + Borel-Cantelli), **Hölder 연속성** $\|B_t - B_s\| \leq C\|t-s\|^{1/2 - \epsilon}$, **Hausdorff 차원** $\dim_H(\{B_t\}) = 2$ a.s. |
| [05. 이차변분 $\langle B\rangle_t = t$](./ch6-brownian/05-quadratic-variation.md) | $\sum_{i}(B_{t_{i+1}} - B_{t_i})^2 \to t$ in $L^2$ 완전 증명 — 평균이 정확히 $t$, 분산은 $2\sum(\Delta t)^2 \to 0$, **경로별 이차변분도 결정적 $t$ a.s.**, **$(dB)^2 = dt$의 원천** — 이토 적분의 기반, SDE Deep Dive로의 직접 교량 |
| [06. 반사원리(Reflection Principle)와 최대값](./ch6-brownian/06-reflection-principle.md) | 최대값 $M_t = \max_{s \leq t} B_s$의 분포 — **반사원리** $\mathbb{P}(M_t \geq a) = 2\mathbb{P}(B_t \geq a)$ 증명, joint 분포 $(M_t, B_t)$, 첫 도달시각 $\tau_a$의 분포 — 역가우시안, 장벽 옵션 가격과의 연결 |

</details>

<br/>

### 🔹 Chapter 7: MCMC — 마르코프 체인 몬테카를로

> **핵심 질문:** 왜 MCMC가 동작하는가? Metropolis-Hastings의 acceptance ratio는 어떻게 유도되는가? Gibbs sampler는 왜 MH의 특수 경우인가? HMC는 왜 랜덤워크보다 효율적인가?

<details>
<summary><b>MCMC 아이디어부터 수렴 진단까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. MCMC의 아이디어 — 정상분포 설계](./ch7-mcmc/01-mcmc-idea.md) | 목표 분포 $\pi$(정규화 상수 불명)에서 샘플링하려면 **$\pi$를 정상분포로 갖는 마르코프 체인 설계** → 충분히 오래 돌리면 샘플이 $\pi$에서 나옴, 에르고딕 정리가 기댓값 계산의 근거, 베이지안 후행분포 샘플링에서의 핵심 활용 |
| [02. Metropolis-Hastings 알고리즘](./ch7-mcmc/02-metropolis-hastings.md) | proposal $y \sim q(\cdot\|x)$ + accept 확률 **$\alpha(x,y) = \min\left(1, \frac{\pi(y) q(x\|y)}{\pi(x) q(y\|x)}\right)$**, MH transition kernel이 **detailed balance $\pi(x) P(x,y) = \pi(y) P(y,x)$ 를 만족함을 직접 증명**, $\pi$의 정규화 상수가 사라지는 이유 |
| [03. Gibbs Sampler](./ch7-mcmc/03-gibbs-sampler.md) | 조건부 분포 $p(x_i \mid x_{-i})$에서 순차적으로 업데이트, **MH의 특수 경우** (acceptance $\alpha \equiv 1$임을 증명), systematic vs random scan, 고차원에서의 장점(조건부가 tractable일 때)과 한계(상관 구조에서의 느린 mixing) |
| [04. Hamiltonian Monte Carlo (HMC)](./ch7-mcmc/04-hamiltonian-mc.md) | Hamiltonian **$H(x, p) = U(x) + \frac{1}{2}p^T M^{-1} p$** ($U = -\log \pi$), **leapfrog integrator**의 심플렉틱·시간가역 성질, MH acceptance는 leapfrog의 에너지 오차만 보정, **gradient를 활용**한 제안분포로 랜덤워크 대비 스펙트럴 갭 $\mathcal{O}(\sqrt{d})$ 개선 |
| [05. MCMC 수렴 진단과 혼합 시간(Mixing Time)](./ch7-mcmc/05-mixing-diagnostics.md) | **$t_{\text{mix}}(\epsilon) = \min\{n : \|P^n(x, \cdot) - \pi\|_{TV} \leq \epsilon\}$** 정의, 스펙트럴 갭과의 관계 $t_{\text{mix}} \sim \frac{1}{1 - \|\lambda_2\|}$, 실전 진단 — **Gelman-Rubin $\hat{R}$** (여러 체인 일치도), **ESS** (autocorrelation으로부터), trace plot 판독법 |

</details>

---

## 🏆 핵심 정리 인덱스

이 레포에서 **완전한 증명**을 제공하는 대표 정리 모음입니다. 각 챕터의 문서에서 $\square$로 종결되는 엄밀한 증명을 확인할 수 있습니다. (전체 84개 정리 중 핵심만 발췌)

| 정리 | 서술 | 출처 문서 |
|------|------|----------|
| **Kolmogorov 확장정리** | 일관성 있는 유한차원 분포족은 확률과정을 유일하게 결정 | [Ch1-02](./ch1-foundations/02-kolmogorov-extension.md) |
| **Perron-Frobenius (확률행렬판)** | 기약·비주기 유한상태 확률행렬의 고유값 $1$은 단순·최대절댓값 — 정상분포 유일성 | [Ch2-03](./ch2-discrete-markov/03-stationary-distribution.md) |
| **마르코프 체인 수렴률** | $\|\mu_n - \pi\|_{TV} \leq C \|\lambda_2\|^n$ — 제2고유값이 mixing time 결정 | [Ch2-04](./ch2-discrete-markov/04-convergence-rate.md) |
| **Detailed balance ⇒ 정상성** | $\pi_i P_{ij} = \pi_j P_{ji}$ 만족 시 $\pi P = \pi$ 자동 | [Ch2-05](./ch2-discrete-markov/05-detailed-balance.md) |
| **에르고딕 정리 (Markov)** | 기약·양재귀 체인에서 $\frac{1}{n}\sum f(X_k) \to \mathbb{E}_\pi[f]$ a.s. | [Ch2-06](./ch2-discrete-markov/06-ergodic-theorem.md) |
| **Poisson 3-동치** | 독립증분+Poisson분포 ⇔ 간격 Exp iid ⇔ 인피니티시멀 레이트 | [Ch3-01](./ch3-poisson/01-three-equivalent-definitions.md) |
| **Little의 법칙** | $L = \lambda W$ — 매우 일반적 조건에서 성립 | [Ch3-04](./ch3-poisson/04-queueing-little.md) |
| **Kolmogorov forward/backward** | $P'(t) = P(t)Q$ (forward), $P'(t) = QP(t)$ (backward), $P(t) = e^{tQ}$ | [Ch4-02](./ch4-continuous-markov/02-kolmogorov-equations.md) |
| **Birth-Death 정상분포** | $\pi_n = \pi_0 \prod_{k=1}^n \lambda_{k-1}/\mu_k$ — detailed balance 해석해 | [Ch4-04](./ch4-continuous-markov/04-birth-death.md) |
| **Doob 마팅게일 수렴** | $\sup_n \mathbb{E}\|X_n\| < \infty \Rightarrow X_n \to X_\infty$ a.s. | [Ch5-02](./ch5-martingale/02-convergence-theorem.md) |
| **Optional Stopping Theorem** | 적절한 조건 하 $\mathbb{E}[X_\tau] = \mathbb{E}[X_0]$ | [Ch5-03](./ch5-martingale/03-optional-stopping.md) |
| **Azuma-Hoeffding** | 유계 증분 마팅게일의 지수 concentration | [Ch5-05](./ch5-martingale/05-martingale-ml.md) |
| **Lévy 구성 (BM 존재성)** | Haar 기저 랜덤 급수로 연속 경로 BM 구성 | [Ch6-02](./ch6-brownian/02-levy-construction.md) |
| **Donsker 불변원리** | $\frac{1}{\sqrt{n}} S_{\lfloor nt\rfloor} \xrightarrow{d} B_t$ in $C[0,1]$ | [Ch6-03](./ch6-brownian/03-donsker-theorem.md) |
| **BM 비미분 가능성** | 거의 확실히 어디에서도 미분불가능 (Paley-Wiener-Zygmund) | [Ch6-04](./ch6-brownian/04-non-differentiability.md) |
| **이차변분 $\langle B\rangle_t = t$** | $\sum(B_{t_{i+1}} - B_{t_i})^2 \to t$ in $L^2$ — $(dB)^2 = dt$의 원천 | [Ch6-05](./ch6-brownian/05-quadratic-variation.md) |
| **반사원리** | $\mathbb{P}(M_t \geq a) = 2\mathbb{P}(B_t \geq a)$ | [Ch6-06](./ch6-brownian/06-reflection-principle.md) |
| **MH ⇒ detailed balance** | $\alpha = \min(1, \frac{\pi(y)q(x\|y)}{\pi(x)q(y\|x)})$ 선택 시 $\pi$가 정상분포 | [Ch7-02](./ch7-mcmc/02-metropolis-hastings.md) |
| **Gibbs ⊂ MH** | Gibbs sampler는 acceptance $\equiv 1$인 MH의 특수 경우 | [Ch7-03](./ch7-mcmc/03-gibbs-sampler.md) |
| **HMC의 심플렉틱 보존** | Leapfrog integrator는 심플렉틱·시간가역 → 고차원 mixing 개선 | [Ch7-04](./ch7-mcmc/04-hamiltonian-mc.md) |

> 💡 **챕터별 총 정리 수**: Ch1(9) · Ch2(18) · Ch3(10) · Ch4(11) · Ch5(13) · Ch6(15) · Ch7(8) — 합계 **84개 정리 + 증명**, 약 **17,000+ 라인** 분량.

---

## 💻 실험 환경

모든 챕터의 실험은 아래 환경에서 재현 가능합니다.

```bash
# requirements.txt
numpy==1.26.0
scipy==1.11.0
matplotlib==3.8.0
networkx==3.2.0       # 마르코프 체인 그래프 시각화
arviz==0.17.0         # MCMC 진단 (R̂, ESS, trace plot)
pymc==5.10.0          # MCMC 비교 레퍼런스 (HMC·NUTS)
tqdm==4.66.0
jupyter==1.0.0
```

```bash
# 환경 설치
pip install numpy==1.26.0 scipy==1.11.0 matplotlib==3.8.0 \
            networkx==3.2.0 arviz==0.17.0 pymc==5.10.0 \
            tqdm==4.66.0 jupyter==1.0.0

# 실험 노트북 실행
jupyter notebook
```

```python
# 대표 실험 — 마르코프 체인 수렴률 관찰 (스펙트럴 예측 검증)
import numpy as np
import matplotlib.pyplot as plt

# 3-state 기약·비주기 전이행렬
P = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5],
])

# ─────────────────────────────────────────────
# 1. 스펙트럴 분해로 정상분포 & 수렴률 예측
# ─────────────────────────────────────────────
eigvals, eigvecs = np.linalg.eig(P.T)
idx = np.argmax(np.abs(eigvals))
pi = np.real(eigvecs[:, idx]); pi = pi / pi.sum()
lambda2 = sorted(np.abs(eigvals))[-2]

print(f'정상분포 π = {pi}')
print(f'제2고유값 |λ_2| = {lambda2:.4f}')   # 수렴률 결정

# ─────────────────────────────────────────────
# 2. TV 거리 수렴 vs |λ_2|^n 경계 비교
# ─────────────────────────────────────────────
mu = np.array([1.0, 0.0, 0.0])
dists = [mu.copy()]
for _ in range(50):
    mu = mu @ P
    dists.append(mu.copy())
dists = np.array(dists)

tv = 0.5 * np.sum(np.abs(dists - pi), axis=1)
plt.semilogy(tv, label=r'$\|\mu_n - \pi\|_{TV}$ (실측)')
plt.semilogy(lambda2 ** np.arange(len(tv)),
             '--', label=r'$|\lambda_2|^n$ (스펙트럴 경계)')
plt.legend(); plt.xlabel('n'); plt.ylabel('distance (log scale)')
plt.title('마르코프 체인의 수렴률 — 제2고유값이 결정')
plt.grid(True, which='both', alpha=0.3)
plt.show()

# ─────────────────────────────────────────────
# 3. 에르고딕 정리 — 시간평균 = 공간평균
# ─────────────────────────────────────────────
rng = np.random.default_rng(0)
N = 100_000
state = 0
f_vals = np.array([1.0, 2.0, 3.0])   # f(state) = state + 1
running_avg = np.zeros(N)
cumsum = 0.0
for n in range(N):
    cumsum += f_vals[state]
    running_avg[n] = cumsum / (n + 1)
    state = rng.choice(3, p=P[state])

plt.plot(running_avg, label=r'$\frac{1}{n}\sum f(X_k)$')
plt.axhline(pi @ f_vals, color='k', linestyle='--',
            label=r'$\mathbb{E}_\pi[f]$')
plt.xscale('log'); plt.xlabel('n'); plt.ylabel('시간평균')
plt.title('에르고딕 정리 — MCMC의 이론적 근거')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
```

---

## 📖 각 문서 구성 방식

모든 문서는 다음 **11-섹션 골격**으로 작성됩니다.

| # | 섹션 | 내용 |
|:-:|------|------|
| 1 | 🎯 **핵심 질문** | 이 문서가 답하는 3~5개의 본질적 질문 |
| 2 | 🔍 **왜 이 과정이 AI에서 중요한가** | MCMC, Bayesian NN, Diffusion Model, RL Q-learning과의 연결점 |
| 3 | 📐 **수학적 선행 조건** | Probability Theory, Linear Algebra 레포의 어떤 정리를 전제로 하는지 |
| 4 | 📖 **직관적 이해** | 랜덤 워크·줄서기·도박 등 물리적 비유, 샘플 경로 시각화 |
| 5 | ✏️ **엄밀한 정의** | $(\Omega, \mathcal{F}, \mathbb{P}, \{\mathcal{F}_t\})$ 위의 측도론적 정의 — 필트레이션까지 항상 명시 |
| 6 | 🔬 **정리와 증명** | 에르고딕 정리, 마팅게일 수렴, 반사원리, MH의 detailed balance 등 — "자명하다" 없이 |
| 7 | 💻 **NumPy 구현 검증** | 마르코프 체인 수렴, 브라운 운동 궤적, MCMC 샘플링 + 수렴 진단 |
| 8 | 🔗 **AI/ML 연결** | MCMC, HMC, Diffusion Model (다음 레포), RL Q-learning, Bayesian NN |
| 9 | ⚖️ **가정과 한계** | 기약성·비주기·에르고딕 가정이 깨지면? 고차원에서는? |
| 10 | 📌 **핵심 정리** | 한 장으로 요약 |
| 11 | 🤔 **생각해볼 문제 (+ 해설)** | 손 계산·증명 재구성·구현 문제 |

> 📚 **연습문제 총 105개**: 35문서 × 문서당 3문제(기초/심화/AI 연결), 모든 문제에 `<details>` 펼침 해설 포함. 손 계산 재현부터 MCMC 실전 구현까지 단계적으로 심화됩니다.
>
> 🧭 **푸터 네비게이션**: 각 문서 하단에 `◀ 이전 / 📚 README / 다음 ▶` 링크가 항상 제공됩니다. 챕터 경계에서도 자동으로 다음 챕터 첫 문서로 연결되므로 순차 학습이 끊기지 않습니다.
>
> ⏱️ **학습 시간 추정**: 문서당 평균 480줄(증명·코드·연습문제 포함) 기준 **약 1~1.5시간**. 전체 35문서는 약 **40~50시간** 상당.

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "MCMC를 쓰지만 왜 수렴하는지 모른다" — MCMC 집중 (5일, 약 12~15시간)</b></summary>

<br/>

```
Day 1  Ch2-01  마르코프 성질과 전이행렬
       Ch2-03  정상분포 & Perron-Frobenius
Day 2  Ch2-04  수렴률 — 제2고유값과 mixing time
       Ch2-05  Detailed balance
Day 3  Ch2-06  에르고딕 정리 — MCMC의 이론적 근거
Day 4  Ch7-01~02  MCMC 아이디어 & Metropolis-Hastings
Day 5  Ch7-03~05  Gibbs, HMC, 수렴 진단 (R̂, ESS)
```

</details>

<details>
<summary><b>🟡 "Brownian motion을 쓰지만 왜 미분불가인지 모른다" — Brownian 집중 (5일, 약 12~15시간)</b></summary>

<br/>

```
Day 1  Ch1-01~02  확률과정의 엄밀한 정의 & Kolmogorov 확장정리
Day 2  Ch1-04     필트레이션
       Ch5-01     마팅게일 정의
Day 3  Ch5-02~04  Doob 수렴 & Optional Stopping & 이차변분
Day 4  Ch6-01~03  BM 공리·Lévy 구성·Donsker
Day 5  Ch6-04~06  비미분 가능성·이차변분 $\langle B\rangle_t = t$·반사원리
       → SDE Deep Dive Ch1 준비 완료
```

</details>

<details>
<summary><b>🔴 "확률과정과 MCMC의 수학적 기반을 완전 정복한다" — 전체 정복 (7주, 약 40~50시간)</b></summary>

<br/>

```
1주차  Chapter 1 전체 — 확률과정의 기초
        → Kolmogorov 확장정리 증명 스케치 숙지
        → 필트레이션·adapted·predictable 엄밀 이해

2주차  Chapter 2 전체 — 이산 마르코프 체인
        → Perron-Frobenius로 정상분포 유일성 증명 재구성
        → 제2고유값으로 mixing time 직접 측정
        → 에르고딕 정리 증명 스케치 숙지

3주차  Chapter 3 + 4 — Poisson과 연속시간 MC
        → 3-동치 정의 상호 변환 손 계산
        → Kolmogorov forward/backward 유도
        → M/M/1 큐의 정상분포 및 Little의 법칙

4주차  Chapter 5 전체 — 마팅게일
        → Doob 수렴을 upcrossing 부등식으로 증명
        → Optional Stopping으로 도박장 파산 문제 해결
        → Azuma-Hoeffding으로 online learning regret 경계

5주차  Chapter 6 (1~3) — 브라운 운동의 구성
        → Lévy의 Haar 기저 구성 단계적 이해
        → Donsker 정리 + CLT의 과정판 해석

6주차  Chapter 6 (4~6) + Ch2-04 스펙트럴 재방문
        → 비미분 가능성 증명 재구성
        → 이차변분 $\langle B\rangle_t = t$ 실측 & $(dB)^2 = dt$ 이해
        → 반사원리로 최대값 분포 유도

7주차  Chapter 7 전체 — MCMC
        → MH의 detailed balance를 직접 손으로 유도
        → Gibbs가 MH의 특수 경우임을 증명
        → HMC 구현 & RWMH 대비 mixing 비교
        → NumPy + PyMC로 베이지안 회귀 사후분포 샘플 + R̂·ESS 진단
```

</details>

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [probability-theory-deep-dive](https://github.com/iq-ai-lab/probability-theory-deep-dive) | 측도, 조건부 기댓값, Tower Property, $L^p$ 수렴, 특성함수 | Ch1 전체 (확률공간·필트레이션), Ch2-06 (에르고딕), Ch5 전체 (마팅게일) |
| [linear-algebra-deep-dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive) | Spectral Theorem, 양의 행렬, Perron-Frobenius | Ch2-03~04 (정상분포·수렴률), Ch2-05 (reversibility의 self-adjoint), Ch4-02 (행렬지수 $e^{tQ}$) |
| [calculus-optimization-deep-dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive) | 다변수 미분, gradient flow, Taylor 전개 | Ch7-04 (HMC 해밀토니안 역학) |
| [sde-deep-dive](https://github.com/iq-ai-lab/sde-deep-dive) | 이토 적분, 이토 공식, Fokker-Planck, Anderson 시간반전, DDPM | **후속 레포** — Ch6-05 이차변분 $\langle B\rangle_t = t$가 $(dB)^2 = dt$로 직결 |
| [generative-models-deep-dive](https://github.com/iq-ai-lab/generative-models-deep-dive) | DDPM, Score-SDE, Flow Matching 실전 아키텍처 | Ch7 Langevin MCMC·HMC → SDE 거쳐 Diffusion으로 |
| [rl-foundations-deep-dive](https://github.com/iq-ai-lab/rl-foundations-deep-dive) | MDP, Bellman 연산자, 정책 수렴 | Ch2 전체 (MDP의 마르코프 구조), Ch2-04 (Bellman 연산자의 스펙트럴 수렴) |

> 💡 이 레포는 **확률과정의 엄밀한 이론과 MCMC의 수학적 기반**에 집중합니다. Probability Theory에서 조건부 기댓값과 $L^p$ 수렴을, Linear Algebra에서 Spectral Theorem을 학습한 후 오면 Ch2(마르코프 체인 스펙트럴)와 Ch5(마팅게일)가 훨씬 자연스럽습니다. Ch6(브라운 운동)의 이차변분과 Ch7(MCMC)의 Langevin은 SDE Deep Dive로 바로 연결됩니다.

---

## 📖 Reference

### 🏛️ 확률과정 표준 교재
- **Probability: Theory and Examples** (Durrett, 2019) — 확률과정 챕터, 마팅게일·BM·에르고딕 이론
- **Stochastic Processes** (Ross, 1996) — 입문자 친화적 표준
- **A First Course in Stochastic Processes** (Karlin & Taylor, 1975) — 고전
- **Stochastic Processes** (Doob, 1953) — 마팅게일 이론의 원전

### 🔗 마르코프 체인과 MCMC 수렴
- **Markov Chains and Mixing Times** (Levin & Peres, 2017) — **MC mixing time 표준 교재**, 스펙트럴 갭과 전도도(conductance)
- **Markov Chains** (Norris, 1997) — 이산/연속 MC 입문
- **General State Space Markov Chains and MCMC Algorithms** (Roberts & Rosenthal, 2004) — 일반 상태공간

### 🌊 브라운 운동 · 마팅게일
- **Brownian Motion and Stochastic Calculus** (Karatzas & Shreve, 1991) — **BM과 확률해석의 고전**
- **Continuous Martingales and Brownian Motion** (Revuz & Yor, 1999) — 마팅게일 이론 심화
- **Brownian Motion** (Mörters & Peres, 2010) — BM 경로 성질·Hausdorff 차원
- **Probability with Martingales** (Williams, 1991) — 마팅게일 중심 입문

### 🎲 MCMC · 베이지안 샘플링
- **Monte Carlo Statistical Methods** (Robert & Casella, 2004) — **MCMC 표준 교재**
- **An Introduction to MCMC for Machine Learning** (Andrieu, de Freitas, Doucet & Jordan, 2003) — ML 관점
- **Handbook of Markov Chain Monte Carlo** (Brooks, Gelman, Jones & Meng, 2011) — MCMC 편람
- **MCMC Using Hamiltonian Dynamics** (Neal, 2011) — **HMC 원전**
- **The No-U-Turn Sampler** (Hoffman & Gelman, 2014) — NUTS
- **Bayesian Learning via Stochastic Gradient Langevin Dynamics** (Welling & Teh, 2011) — **SGLD 원전**

### 📈 Poisson 과정 · Queueing
- **An Introduction to the Theory of Point Processes** (Daley & Vere-Jones, 2003) — 점 과정
- **Queueing Theory** (Gross, Shortle, Thompson & Harris, 2008) — 대기행렬 이론

---

<div align="center">

**⭐️ 도움이 되셨다면 Star를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"마르코프 체인 $P^n$을 돌리는 것과, 왜 $|\lambda_2|^n$이 정상분포 수렴률을 결정하는지 — 전이행렬의 스펙트럴 분해로 mixing time을 예측할 수 있는 것은 다르다"*

</div>
