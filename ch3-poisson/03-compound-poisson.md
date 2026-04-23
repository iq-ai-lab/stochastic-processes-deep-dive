# 03. 복합 Poisson 과정

## 🎯 핵심 질문

- **복합 Poisson** $Y_t = \sum_{k=1}^{N_t} Z_k$ — 각 이벤트가 단순 카운트가 아닌 **random jump** $Z_k$를 동반할 때 과정의 분포는?
- 평균 $\mathbb{E}[Y_t] = \lambda t \mathbb{E}[Z]$, 분산 $\text{Var}[Y_t] = \lambda t \mathbb{E}[Z^2]$가 왜 이런 깔끔한 형태인가?
- **특성함수** $\phi_{Y_t}(u) = \exp(\lambda t (\phi_Z(u) - 1))$를 어떻게 유도하고, 왜 **Lévy process**의 대표 예시인가?
- **보험 위험**, **jump-diffusion 금융 모델** 등 실제 응용에서 어떻게 쓰이는가?

---

## 🔍 왜 이 과정이 AI에서 중요한가

**Financial Transformer / Jump Diffusion**: 주식 가격의 sudden jump (뉴스, 실적) 모델링에 compound Poisson 항 추가 (Merton jump-diffusion). 딥러닝 기반 옵션 가격모형도 이를 반영.

**이벤트 기반 RL**: 불규칙 간격으로 reward 획득 — 각 이벤트에서 random 크기의 reward $Z_k$. Compound Poisson이 총 reward 분포를 기술.

**Insurance / Risk NN**: 보험 청구 모델링 (각 사고 발생 + 청구 금액). Actuarial AI에서 compound Poisson이 core.

**Lévy process로서의 generative model**: Variational Autoencoder의 Lévy prior 확장, Lévy flight-based optimization (simulated annealing 변형).

---

## 📐 수학적 선행 조건

- [Ch3-01 ~ Ch3-02](./01-three-equivalent-definitions.md): Poisson 과정, Superposition
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): 특성함수, 모멘트 생성함수, Wald의 항등식
- 독립 확률변수의 합 분포

---

## 📖 직관적 이해

### 정의와 예

이벤트 시점은 Poisson 과정 $\{N_t\}$ (rate $\lambda$), 각 이벤트에서 iid jump $Z_k \sim F_Z$ (독립, $N$과 독립):
$$Y_t = \sum_{k=1}^{N_t} Z_k.$$

**예 1**: 보험 청구. $N_t$ = 청구 건수, $Z_k$ = 청구 금액 → $Y_t$ = 시각 $t$까지 총 청구액.

**예 2**: 웹 트래픽. $N_t$ = 방문자 수, $Z_k$ = 각 방문자의 페이지 열람 수 → $Y_t$ = 총 페이지뷰.

### 왜 평균·분산 공식이 간단한가 — Wald 아이디어

"랜덤 수의 iid 합". $N_t$가 고정이면 $Y_t | N_t = n \sim \sum_{k=1}^n Z_k$. 기댓값:
$$\mathbb{E}[Y_t] = \mathbb{E}[\mathbb{E}[Y_t | N_t]] = \mathbb{E}[N_t \cdot \mathbb{E}[Z]] = \lambda t \mathbb{E}[Z].$$

분산은 "조건부 분산 공식":
$$\text{Var}(Y_t) = \mathbb{E}[\text{Var}(Y_t | N_t)] + \text{Var}(\mathbb{E}[Y_t | N_t]).$$
$\text{Var}(Y_t | N_t) = N_t \text{Var}(Z)$, $\mathbb{E}[Y_t|N_t] = N_t \mathbb{E}[Z]$. 대입:
$$\text{Var}(Y_t) = \mathbb{E}[N_t] \text{Var}(Z) + (\mathbb{E}[Z])^2 \text{Var}(N_t) = \lambda t (\text{Var}(Z) + (\mathbb{E}[Z])^2) = \lambda t \mathbb{E}[Z^2].$$

**깔끔한 형태**: $\mathbb{E}[Z^2] = \text{Var}(Z) + (\mathbb{E}Z)^2$. Poisson의 $\text{Var}(N_t) = \mathbb{E}[N_t]$ 성질이 결합해서 단일 term.

### 특성함수 유도

$\phi_{Y_t}(u) = \mathbb{E}[e^{iuY_t}] = \mathbb{E}[\mathbb{E}[e^{iu\sum Z_k} | N_t]]$. iid 독립:
$$= \mathbb{E}[(\phi_Z(u))^{N_t}] = \sum_n \frac{(\lambda t)^n e^{-\lambda t}}{n!} \phi_Z(u)^n = \exp(\lambda t (\phi_Z(u) - 1)).$$

**Lévy process의 전형**: 특성함수가 exponential 형태 $e^{t\Psi(u)}$, $\Psi(u) = \lambda(\phi_Z(u) - 1)$ — **Lévy exponent**.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — 복합 Poisson 과정

$\{N_t\}$가 rate $\lambda$ Poisson, $\{Z_k\}_{k \geq 1}$가 iid $F_Z$, $N$과 $\{Z_k\}$ 독립. 
$$Y_t := \sum_{k=1}^{N_t} Z_k$$
(빈 합은 0). $\{Y_t\}_{t \geq 0}$가 **복합 Poisson 과정**.

### 정의 3.2 — Lévy process

$\{X_t\}$가 **Lévy process**이다:
1. $X_0 = 0$
2. 독립증분
3. 정상증분 ($X_{t+s} - X_s \stackrel{d}{=} X_t$)
4. 우연속 경로

**대표 예**:
- Brownian motion: 가우시안, 연속
- Poisson process: Poisson 카운트, integer jump size 1
- Compound Poisson: 임의 jump size 분포
- $\alpha$-stable process: heavy-tailed

### 정의 3.3 — Lévy-Khintchine 공식

모든 Lévy process의 특성함수는 $\mathbb{E}[e^{iu X_t}] = e^{t\Psi(u)}$ 형태로, **Lévy exponent**:
$$\Psi(u) = ib u - \frac{1}{2}\sigma^2 u^2 + \int (e^{iuz} - 1 - iuz\mathbf{1}_{|z|<1}) \nu(dz),$$
$b$ = drift, $\sigma^2$ = 가우시안 성분, $\nu$ = **Lévy measure** (jump 강도).

Compound Poisson: $\nu(dz) = \lambda F_Z(dz)$, $\sigma = 0$. BM: $\sigma^2 > 0, \nu = 0$.

---

## 🔬 정리와 증명

### 정리 3.1 — 평균과 분산

$\mathbb{E}[Z] < \infty$, $\mathbb{E}[Z^2] < \infty$이면:
$$\mathbb{E}[Y_t] = \lambda t \mathbb{E}[Z], \quad \text{Var}(Y_t) = \lambda t \mathbb{E}[Z^2].$$

*증명.* 직관적 이해 섹션의 Wald-style 계산 참조. $\square$

### 정리 3.2 — 특성함수

$$\phi_{Y_t}(u) = \exp\left(\lambda t (\phi_Z(u) - 1)\right).$$

*증명.* 직관적 이해 섹션 참조. $\square$

### 정리 3.3 — 복합 Poisson은 Lévy process

$\{Y_t\}$는 독립·정상증분 + 우연속 경로 → Lévy process.

*증명.* $N$과 $Z$의 독립증분 구조에서 $Y_{t+s} - Y_t = \sum_{k=N_t+1}^{N_{t+s}} Z_k$가 $\mathcal{F}_t$와 독립, $\sum_{k=1}^{N_{t+s} - N_t} Z_k \stackrel{d}{=} Y_s$. $\square$

### 정리 3.4 — Lévy measure 형태

Compound Poisson $Y_t = \sum Z_k$의 Lévy exponent:
$$\Psi(u) = \lambda(\phi_Z(u) - 1) = \lambda \int (e^{iuz} - 1) F_Z(dz) = \int (e^{iuz} - 1) \nu(dz),$$
$\nu(dz) = \lambda F_Z(dz)$. $\nu$가 **유한 측도** (전체 질량 $\lambda$) — 이는 compound Poisson의 **특징**. 일반 Lévy는 $\nu(\mathbb{R}) = \infty$ 가능 (infinite activity, e.g., $\alpha$-stable, Variance Gamma).

### 정리 3.5 — 복합 Poisson의 중첩 (Superposition)

두 독립 복합 Poisson $Y^{(1)}_t, Y^{(2)}_t$ (rates $\lambda_1, \lambda_2$, jump distributions $F_1, F_2$)의 합:
$$Y_t = Y^{(1)}_t + Y^{(2)}_t$$
역시 복합 Poisson. Rate $\lambda = \lambda_1 + \lambda_2$, jump distribution
$$F(dz) = \frac{\lambda_1 F_1(dz) + \lambda_2 F_2(dz)}{\lambda_1 + \lambda_2}$$
(mixture).

*증명.* 특성함수:
$$\phi_{Y_t}(u) = \phi_{Y^{(1)}_t}(u) \phi_{Y^{(2)}_t}(u) = e^{t[\lambda_1 (\phi_{Z^{(1)}}(u) - 1) + \lambda_2 (\phi_{Z^{(2)}}(u) - 1)]}.$$
이를 rate $\lambda = \lambda_1 + \lambda_2$, mixture characteristic function $\phi_Z(u) = \frac{\lambda_1 \phi_{Z^{(1)}} + \lambda_2 \phi_{Z^{(2)}}}{\lambda_1 + \lambda_2}$로 묶으면 $e^{t \lambda(\phi_Z(u) - 1)}$. 복합 Poisson 형태. $\square$

### 정리 3.6 — CLT for 복합 Poisson

$\mathbb{E}[Z^2] < \infty$일 때
$$\frac{Y_t - \lambda t \mathbb{E}[Z]}{\sqrt{\lambda t \mathbb{E}[Z^2]}} \xrightarrow{d} \mathcal{N}(0, 1) \quad (t \to \infty).$$

*증명.* CLT with random number of summands (Anscombe). $N_t/t \to \lambda$ a.s., 독립으로 $Y_t = \sum_{k=1}^{N_t} Z_k$의 CLT는 $N_t$가 deterministic인 것처럼. $\square$

---

## 💻 NumPy 구현 검증

### 실험 1 — 복합 Poisson 시뮬레이션

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
lam, T = 2.0, 20.0
# Z ~ Uniform(1, 3)
Z_sample = lambda n: rng.uniform(1, 3, n)

# Poisson 시각 생성
events = []
t = 0
while True:
    t += rng.exponential(1/lam)
    if t > T: break
    events.append(t)
events = np.array(events)
Z = Z_sample(len(events))

# Y_t 구성 (piecewise constant)
t_grid = np.linspace(0, T, 500)
Y_grid = np.array([Z[events <= tg].sum() for tg in t_grid])

plt.plot(t_grid, Y_grid, label=r'$Y_t$')
plt.plot(t_grid, lam * t_grid * 2, '--', label=r'$\lambda t \mathbb{E}[Z]$ (이론 평균)')
plt.xlabel('t'); plt.ylabel(r'$Y_t$')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
```

### 실험 2 — 평균·분산 검증

```python
n_sim, lam, T = 10000, 2.0, 5.0
mu_Z, m2_Z = 2.0, 4.333  # Z ~ Uniform(1,3): E[Z]=2, E[Z²]=13/3

Y_samples = np.zeros(n_sim)
for i in range(n_sim):
    N = rng.poisson(lam * T)
    Y_samples[i] = rng.uniform(1, 3, N).sum() if N > 0 else 0

print(f'실측 E[Y_T]: {Y_samples.mean():.4f}, 이론: {lam*T*mu_Z:.4f}')
print(f'실측 Var[Y_T]: {Y_samples.var():.4f}, 이론: {lam*T*m2_Z:.4f}')

# 히스토그램과 CLT 비교
normalized = (Y_samples - lam*T*mu_Z) / np.sqrt(lam*T*m2_Z)
plt.hist(normalized, bins=50, density=True, alpha=0.5)
x = np.linspace(-4, 4, 200)
plt.plot(x, np.exp(-x**2/2)/np.sqrt(2*np.pi), 'r-', label='N(0,1)')
plt.legend(); plt.title('CLT for 복합 Poisson')
plt.show()
```

### 실험 3 — 특성함수 검증

```python
# 특성함수 실측 vs 이론
u_vals = np.linspace(-2, 2, 40)
n_sim, lam, T = 50000, 1.5, 2.0

# Z ~ N(1, 0.5)
Y_samples = np.array([rng.normal(1, 0.5, rng.poisson(lam*T)).sum() if rng.poisson(lam*T)>0 else 0
                      for _ in range(n_sim)])

# 이론 특성함수
phi_Z = lambda u: np.exp(1j*u - 0.25*u**2/2)   # N(1, 0.5²) char func
phi_Y_theory = np.exp(lam * T * (phi_Z(u_vals) - 1))

# 실측
phi_Y_empirical = np.mean(np.exp(1j * np.outer(u_vals, Y_samples)), axis=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(u_vals, phi_Y_theory.real, 'r-', label='이론 Re')
axes[0].plot(u_vals, phi_Y_empirical.real, 'b--', label='실측 Re')
axes[1].plot(u_vals, phi_Y_theory.imag, 'r-', label='이론 Im')
axes[1].plot(u_vals, phi_Y_empirical.imag, 'b--', label='실측 Im')
for ax in axes: ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlabel('u')
plt.suptitle(r'$\phi_{Y_t}(u)$ 검증'); plt.tight_layout(); plt.show()
```

---

## 🔗 AI/ML 연결

**Merton Jump Diffusion**  
$dS_t = \mu S_t dt + \sigma S_t dB_t + S_{t^-} d(Y_t)$, $Y_t$ = 복합 Poisson (log-normal jumps). Black-Scholes의 연속 경로 가정을 완화. **Neural SDE** with jump component를 학습하는 모델(Chen 2018, Kidger 2020)에서 core.

**Insurance AI — Cramér-Lundberg model**  
보험회사 자본 $U_t = u + ct - Y_t$ ($c$ = premium rate, $Y_t$ = compound Poisson claims). 파산 확률 $\psi(u) = \mathbb{P}(\inf_t U_t < 0)$ 추정에 DL(Deep Learning). Actuarial Neural Network이 이 구조 활용.

**Event-based RL**  
Reward가 sparse 이벤트로 도착 ($Z_k$ = reward size). Total return $Y_T$의 분포 분석에 복합 Poisson theory. Variance reduction techniques가 $\text{Var}(Y_T) = \lambda T \mathbb{E}[Z^2]$를 직접 최소화.

**Lévy flights in optimization**  
Heavy-tailed proposals (Cauchy, $\alpha$-stable) in simulated annealing → Compound Poisson은 "discrete jump 버전" Lévy. Large discrete jump로 local optimum 탈출. Bayesian deep learning에서 Lévy prior 실험 중.

---

## ⚖️ 가정과 한계

**가정 — 독립 jump**  
$Z_k$가 iid. 현실에서는 jump 크기가 **자기상관**(market regime 따라 변화). GARCH-Jump 모델이 이를 확장.

**한계 — 유한 moments**  
$Z$가 heavy-tail (e.g., Pareto)이면 $\mathbb{E}[Z^2] = \infty$ → CLT 깨짐. $\alpha$-stable 극한으로 — 더 일반 Lévy.

**한계 — 단일 source**  
Multiple correlated sources가 상호작용 → **multi-variate Lévy process** 필요. Cross-excitation (Hawkes) 등.

---

## 📌 핵심 정리

| 결과 | 수식 |
|---|---|
| 정의 | $Y_t = \sum_{k=1}^{N_t} Z_k$, Poisson $N$ + iid $Z$ |
| 평균 | $\mathbb{E}[Y_t] = \lambda t \mathbb{E}[Z]$ |
| 분산 | $\text{Var}(Y_t) = \lambda t \mathbb{E}[Z^2]$ |
| 특성함수 | $\exp(\lambda t (\phi_Z(u) - 1))$ |
| Lévy exponent | $\lambda (\phi_Z(u) - 1)$ |
| Lévy measure | $\nu = \lambda F_Z$ (유한 측도) |
| 중첩 | 복합 Poisson끼리 합도 복합 Poisson (rate 합, mixture jump) |

**한 줄 요약**: 복합 Poisson은 "Poisson 시각 + 각 시각 iid jump"로 구성된 가장 단순한 **jump Lévy process**. 평균·분산·특성함수가 깔끔한 닫힌 형태로 주어지며, 보험·금융의 jump 모델링 + 딥러닝 기반 이벤트 시뮬레이션의 기반.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. 보험 청구: Poisson rate $\lambda = 2$/일, 청구 금액 $Z \sim \text{Exp}(1/1000)$ (평균 $1000. 7일 동안 총 청구액의 평균과 표준편차는?

<details>
<summary>해설</summary>

$\lambda T = 2 \times 7 = 14$.  
$\mathbb{E}[Z] = 1000$, $\mathbb{E}[Z^2] = 2 \cdot 1000^2 = 2 \times 10^6$ (지수의 2차 모멘트).

$\mathbb{E}[Y_7] = 14 \times 1000 = \$14,000$.  
$\text{Var}[Y_7] = 14 \times 2 \times 10^6 = 2.8 \times 10^7$.  
$\text{SD}[Y_7] = \sqrt{2.8 \times 10^7} \approx \$5,292$.

**계수 분석**: SD/Mean ≈ 0.38 — compound Poisson은 일반적으로 high variance (iid 합보다).

</details>

**문제 2 (심화)**. 복합 Poisson $Y_t$의 **Laplace transform** $\mathbb{E}[e^{-sY_t}]$을 유도하고, $Z$가 Exp($\mu$)일 때 명시적으로 계산하라.

<details>
<summary>해설</summary>

Laplace는 특성함수의 $s = -iu$ 대입 (real variable version):
$$\mathbb{E}[e^{-sY_t}] = \exp(\lambda t (\mathbb{E}[e^{-sZ}] - 1)) = \exp(\lambda t (L_Z(s) - 1)),$$
$L_Z(s) = \mathbb{E}[e^{-sZ}]$.

$Z \sim \text{Exp}(\mu)$ (rate $\mu$, mean $1/\mu$):
$L_Z(s) = \mu/(\mu + s)$.

따라서
$$\mathbb{E}[e^{-sY_t}] = \exp\left(\lambda t \left(\frac{\mu}{\mu + s} - 1\right)\right) = \exp\left(-\frac{\lambda t s}{\mu + s}\right).$$

**응용**: 이 Laplace transform의 inverse가 $Y_t$ 밀도를 제공. Exp jumps 경우 Gamma 분포의 가중합(Poisson mixture) 형태.

**Ruin theory 연결**: Cramér-Lundberg에서 파산 확률의 Laplace equation이 위 결과로 풀림 — actuarial science의 classical 결과.

</details>

**문제 3 (AI 연결)**. Jump-diffusion SDE $dS_t = \mu dt + \sigma dB_t + dY_t$ (BM + 복합 Poisson jump)의 학습에서 "continuous part"와 "jump part"를 분리하는 이유와 어려움을 논하라.

<details>
<summary>해설</summary>

**분리 이유**:
1. **수학 구조 차이**: BM 부분은 이토 공식(Ch2 SDE 레포), jump는 discrete event. 학습 loss 구성이 완전히 다름.
2. **Observable scale**: Continuous diffusion은 작은 fluctuation, jumps는 큰 discrete events — 다른 physical interpretation.
3. **모델 flexibility**: $\mu, \sigma$는 continuous dynamics, $\lambda, F_Z$는 jump dynamics — 별도 파라미터화로 더 flexible.

**분리의 어려움**:
1. **식별성(identifiability)**: 큰 BM variance는 작은 jumps와 구별 어려움 ("diffusion vs small jumps"). 데이터로부터 $\sigma$와 jump의 Lévy measure 분리 추정 non-trivial.
2. **Missing data**: Discrete observations에서 "jump 있었는가 없었는가"를 판별 (Merton 모델의 EM 알고리즘).
3. **Neural parametrization**: $v_\theta^{\text{diff}}(x, t)$ (drift NN), $\sigma_\theta(x, t)$ (diffusion NN), $\lambda_\theta(x, t)$ (jump rate NN), $f_\theta^{\text{jump}}(z | x, t)$ (jump density NN) — 많은 components, joint training 불안정.

**실전 접근**:
- **Bates model**: $\sigma$ 상수, jump 포아송 → analytical option pricing (결과)
- **Neural SDE with jumps** (Kidger 2020): flow-based jump density + neural drift
- **Variance filtering**: 관찰된 quadratic variation $\langle X\rangle_t = \sigma^2 t + \sum Z_k^2$을 이용해 jump 추출

**Deep learning 기회**:
- Transformer-based jump detection (Tick data의 anomaly detection)
- Score-SDE with jumps (generative model의 discrete latent)

**연결**: SDE Deep Dive Ch1-05 (Stratonovich)와 Ch2-02 ((dB)² = dt)에서 continuous 부분의 엄밀 분석; Ch3의 Lévy process 확장에서 jump 통합.

</details>

---

<div align="center">

◀ [02. 결합·분할과 비균질 Poisson](./02-superposition-thinning.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [04. Queueing 이론 맛보기 — M/M/1, Little의 법칙](./04-queueing-little.md)

</div>
