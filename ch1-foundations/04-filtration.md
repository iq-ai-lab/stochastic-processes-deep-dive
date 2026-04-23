# 04. 필트레이션(Filtration)과 정보 흐름

## 🎯 핵심 질문

- "시각 $t$까지 관찰 가능한 정보"를 수학적으로 어떻게 형식화하는가 — **필트레이션 $\{\mathcal{F}_t\}$**는 정확히 무엇인가?
- **Adapted**와 **predictable**은 어떻게 다르고, 왜 이 둘을 구별해야 하는가 — 이토 적분 $\int H dB$에서 $H$는 왜 progressively measurable이어야 하는가?
- 왜 연속시간에서 필트레이션에 **usual conditions**(right-continuous + complete)를 요구하는가 — 이를 갖추지 않으면 무엇이 깨지는가?
- **자연 필트레이션** $\mathcal{F}_t^X$와 **augmented/right-continuous 확장**의 관계는?

---

## 🔍 왜 이 구조가 AI에서 중요한가

**강화학습의 online 학습**에서 "시각 $t$의 정책은 시각 $t$까지의 관찰만 사용할 수 있다"는 제약이 본질이다. 이를 수학적으로 정당화하는 언어가 바로 필트레이션. Policy gradient의 unbiasedness는 **predictable representation**($A_{t+1}$이 $\mathcal{F}_t$-measurable)에 의존.

**시계열 예측**의 "forward-looking information leak"(테스트 셋 오염)은 필트레이션 관점에서 "$X_{t+1}$을 예측하는데 $\mathcal{F}_{t+\delta}$ 정보를 썼다"는 오류 — adapted 조건 위반. 실전에서 흔한 버그.

**확률해석 기반**: 이토 적분 $\int_0^t H_s dB_s$가 well-defined 확률변수이려면 $H$가 **progressively measurable**(특히 predictable) 해야 한다. 이는 BM 레퍼런스 필트레이션 $\mathcal{F}_t^B$ 위에서 이토 적분을 구성하는 Ch1(SDE Deep Dive)의 출발점. 또한 **Diffusion forward process**의 noise가 adapted라는 가정이 reverse SDE(Anderson 1982) 구성의 핵심.

**마팅게일 이론 전체의 언어**: Ch5 전체, Ch2의 optional stopping, Ch6의 BM 마팅게일 성질 모두 필트레이션에 의존. 필트레이션 없이는 "공정한 게임"의 개념 자체를 정의할 수 없다.

---

## 📐 수학적 선행 조건

- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): 조건부 기댓값 $\mathbb{E}[\cdot | \mathcal{G}]$, $\sigma$-대수, tower property
- [Ch1-01](./01-rigorous-definition.md): 확률과정의 정의, measurable process
- 측도론: 완비화(completion of $\sigma$-algebra)

---

## 📖 직관적 이해

### "정보의 흐름"이란

실험을 관찰하면서 시간이 흐름에 따라 "알 수 있는 사건"은 점점 많아진다. 시각 0에는 아무것도 몰랐지만, 시각 1에는 $X_1$의 값을 알고, 시각 2에는 $(X_1, X_2)$를 안다. 이 "시각 $t$에 알 수 있는 사건"의 모임이 $\sigma$-대수 $\mathcal{F}_t$다.

$t \leq s$이면 $\mathcal{F}_t \subseteq \mathcal{F}_s$ — **정보는 잃지 않고 쌓이기만** 한다. 이 단조증가 $\sigma$-대수 가족이 **필트레이션**.

> **비유**: 도서관이 매일 새 책을 받는 상황. 시각 $t$의 장서 목록 = $\mathcal{F}_t$. 과거 책은 버리지 않으므로 단조증가. "사건 $A$가 $\mathcal{F}_t$-measurable"이란 "$A$가 시각 $t$까지의 장서로 결정 가능"이라는 의미.

### Adapted vs Predictable — 미묘하지만 결정적 차이

- **Adapted**: 각 $t$에서 $X_t$가 "지금까지 관찰 가능" — $X_t \in \mathcal{F}_t$
- **Predictable**: $X_t$가 "$t$ **직전** 시점까지의 정보로 미리 알 수 있음" — $X_t \in \mathcal{F}_{t-}$

이산시간에서는 predictable이 "one-step ahead knowable" — 예를 들어 베팅 전략 $H_{n+1}$이 $\mathcal{F}_n$ (시각 $n$까지의 관찰)로 결정되면 predictable. 연속시간에서 predictable은 더 복잡한 정의(정의 4.4).

**핵심 구별**: 이토 적분 $\int_0^t H_s dB_s$에서 "시각 $s$에 베팅 $H_s$를 건 직후 BM이 $dB_s$만큼 움직였다" — $H_s$는 **$dB_s$를 보고 조정할 수 없어야** 한다(미래 정보로 베팅 금지). 이것이 predictable/progressively measurable 요구의 직관.

### Usual conditions의 필요

$\mathcal{F}_t$가 "우연속"(right-continuous)하고 "영집합을 포함"(completion)하면 기술적 성질이 많이 간결해진다:
- 정지시각(stopping time)이 closed set에서 hit time으로 표현 가능
- Modification 조정이 자유로움
- 마팅게일 이론의 Doob-Meyer 분해 등 주요 정리의 표준 전제

---

## ✏️ 엄밀한 정의

### 정의 4.1 — 필트레이션 (Filtration)

확률공간 $(\Omega, \mathcal{F}, \mathbb{P})$ 위의 **필트레이션**은 단조증가 $\sigma$-대수족
$$\{\mathcal{F}_t\}_{t \in T}, \quad \mathcal{F}_s \subseteq \mathcal{F}_t \text{ for } s \leq t, \quad \mathcal{F}_t \subseteq \mathcal{F}.$$

$T = \mathbb{N}$이면 이산 필트레이션, $T = [0, \infty)$이면 연속 필트레이션.

### 정의 4.2 — Adapted process

확률과정 $\{X_t\}$가 필트레이션 $\{\mathcal{F}_t\}$에 **adapted**라는 것은 각 $t$에 대해 $X_t$가 $\mathcal{F}_t$-measurable이라는 의미.

### 정의 4.3 — 자연 필트레이션 (Natural Filtration)

확률과정 $\{X_t\}$의 **자연 필트레이션**은
$$\mathcal{F}_t^X := \sigma(X_s : s \leq t, s \in T).$$
"시각 $t$까지의 $X$ 관찰로 결정되는 사건들". $X$는 자동으로 $\mathcal{F}_t^X$-adapted.

### 정의 4.4 — Predictable process (이산 / 연속)

**이산시간**: $\{H_n\}_{n \geq 1}$이 $\{\mathcal{F}_n\}$-predictable이라는 것은 각 $n \geq 1$에 대해 $H_n$이 $\mathcal{F}_{n-1}$-measurable.

**연속시간**: $[0, \infty) \times \Omega$ 위의 **predictable $\sigma$-algebra** $\mathcal{P}$는 left-continuous adapted process를 생성하는 $\sigma$-algebra. $\{H_t\}$가 predictable이라는 것은 $(t, \omega) \mapsto H_t(\omega)$가 $\mathcal{P}$-measurable.

### 정의 4.5 — Progressively measurable

$\{X_t\}_{t \geq 0}$가 **progressively measurable**이라는 것은 각 $T \geq 0$에 대해
$$X : [0, T] \times \Omega \to \mathbb{R}$$
가 $\mathcal{B}([0, T]) \otimes \mathcal{F}_T$-measurable. (Joint measurability with adaptedness.)

> 계층: **predictable ⊂ progressively measurable ⊂ adapted & jointly measurable**. 이토 적분은 predictable 적분자에 대해 정의되며, 모든 predictable process는 progressively measurable.

### 정의 4.6 — 우연속 필트레이션 & Usual Conditions

- $\mathcal{F}_{t+} := \bigcap_{s > t} \mathcal{F}_s$. 필트레이션이 **우연속**이라는 것은 $\mathcal{F}_t = \mathcal{F}_{t+}$.
- **완비(complete)**: $\mathcal{F}_0$가 모든 $\mathbb{P}$-영집합 $N$을 포함.
- 필트레이션이 **usual conditions**를 만족한다는 것은 우연속 + 완비.

### 정의 4.7 — 좌연속 필트레이션

$\mathcal{F}_{t-} := \sigma\left(\bigcup_{s < t} \mathcal{F}_s\right)$. 일반적으로 $\mathcal{F}_{t-} \subsetneq \mathcal{F}_t$.

---

## 🔬 정리와 증명

### 정리 4.1 (자연 필트레이션에서 adaptedness)

$\{X_t\}$는 자동으로 자연 필트레이션 $\mathcal{F}_t^X$에 adapted.

*증명.* $\mathcal{F}_t^X = \sigma(X_s : s \leq t) \ni X_t$ (서로 같은 $X_t$가 그 정의 안에 있음). $\square$

### 정리 4.2 (Doob's 측정가능성 — conditional expectation의 필트레이션 버전)

$X$가 integrable 확률변수이고 $\{\mathcal{F}_t\}$가 필트레이션이면
$$Z_t := \mathbb{E}[X | \mathcal{F}_t]$$
는 필트레이션 $\{\mathcal{F}_t\}$에 대한 **마팅게일**이다: $\mathbb{E}[Z_t | \mathcal{F}_s] = Z_s$ for $s \leq t$.

*증명.* Tower property: $\mathbb{E}[Z_t | \mathcal{F}_s] = \mathbb{E}[\mathbb{E}[X | \mathcal{F}_t] | \mathcal{F}_s] = \mathbb{E}[X | \mathcal{F}_s] = Z_s$ (since $\mathcal{F}_s \subseteq \mathcal{F}_t$). $\square$

이 예가 **모든 마팅게일의 원형**: "현재의 정보로 미래의 기댓값을 내다보는" 구조(Ch5에서 자세히).

### 정리 4.3 (Predictable process의 이토 적분 well-definedness — 직관)

이산시간 $\int H dB := \sum_{n \geq 1} H_n (B_n - B_{n-1})$가 마팅게일이려면 $H_n$이 $\mathcal{F}_{n-1}$-measurable(즉 predictable)이어야 한다.

*증명 아이디어.* $H$가 predictable이면
$$\mathbb{E}[H_n (B_n - B_{n-1}) | \mathcal{F}_{n-1}] = H_n \mathbb{E}[B_n - B_{n-1} | \mathcal{F}_{n-1}] = H_n \cdot 0 = 0$$
(BM 증분이 마팅게일 증분). 따라서 부분합이 마팅게일. 반대로 $H_n$이 $\mathcal{F}_n$-measurable이어서 $dB_n$을 "보고" 결정되면 위 조건부 기댓값이 0이 아닐 수 있고, 마팅게일성이 깨짐. $\square$

> **연결**: SDE Deep Dive Ch1-04(이토 적분의 마팅게일 성질) 참조.

### 정리 4.4 (우연속 필트레이션과 정지시각)

필트레이션이 **우연속**이면, closed set $F \subseteq \mathbb{R}$에 대해 첫 hit time
$$\tau_F := \inf\{t \geq 0 : X_t \in F\}$$
는 **정지시각**이다 — 즉 $\{\tau_F \leq t\} \in \mathcal{F}_t$ for all $t$.

*증명 스케치.* 경로 연속 $X$에 대해 $\{\tau_F \leq t\} = \{\inf_{s \leq t} \text{dist}(X_s, F) = 0\}$. 이는 수열 $\{s_n\}$ 유리수 조밀으로 근사, $\mathcal{F}_t^+$에서 measurable. 우연속 조건으로 $\mathcal{F}_t^+ = \mathcal{F}_t$. $\square$

우연속 없이는 $\{\tau_F \leq t\}$만 확인할 수 있고 $\{\tau_F < t\}$ 같은 strict inequality 버전에서 문제. Doob-Meyer 분해 등에서 이 기술적 단순화가 중요.

### 정리 4.5 (Brownian filtration — 자연 필트레이션 augmentation의 우연속성)

$B$가 BM이고 $\mathcal{F}_t^B$가 자연 필트레이션, $\mathcal{F}_t = \sigma(\mathcal{F}_t^B \cup \mathcal{N})$ (영집합 포함 augmentation)이면 $\{\mathcal{F}_t\}$는 **우연속**(Brownian filtration은 usual conditions 만족).

이는 비자명한 정리(Karatzas-Shreve 2.7.7) — BM의 특수한 성질(독립증분)에 의존. 일반 과정에서는 자연 필트레이션 augmentation이 자동으로 우연속은 아님.

### 명제 4.6 (Predictable ⊂ Progressively measurable ⊂ Adapted)

정의로부터 직접:
- Predictable $\sigma$-algebra $\mathcal{P}$는 left-continuous adapted process로 생성되며, left-continuous adapted는 progressively measurable.
- Progressively measurable ⇒ adapted: 각 $t$에서 $X_t(\omega) = X(t, \omega)$가 $\mathcal{F}_t$-measurable은 progressive measurability의 특수 경우.
- 역포함은 모두 strict (일반적으로 adapted이지만 progressively measurable이 아닌 process, 또는 progressively measurable이지만 predictable이 아닌 process 존재).

---

## 💻 NumPy 구현 검증

### 실험 1 — Adapted vs non-adapted 트레이딩 전략 비교

```python
import numpy as np
rng = np.random.default_rng(42)

# BM random walk 근사
T, N = 1.0, 1000
dt = T / N
dB = rng.standard_normal((10_000, N)) * np.sqrt(dt)
B = np.concatenate([np.zeros((10_000, 1)), np.cumsum(dB, axis=1)], axis=1)

# (1) Adapted: H_n = sign(B_{n-1}) — 과거 정보로만 결정
H_adapted = np.sign(B[:, :-1])  # shape (10_000, N)
pnl_adapted = np.sum(H_adapted * dB, axis=1)
print(f'Adapted 전략 평균 손익: {pnl_adapted.mean():.4f} (이론: 0)')
print(f'Adapted 전략 표준편차:  {pnl_adapted.std():.4f}')

# (2) Non-adapted (미래 정보 사용): H_n = sign(dB_n)
# 이는 predictable 조건 위반 — 시각 n에서 아직 못 본 dB_n을 보고 결정
H_cheating = np.sign(dB)
pnl_cheat = np.sum(H_cheating * dB, axis=1)
print(f'Cheating 전략 평균 손익: {pnl_cheat.mean():.4f}')  # 양수!
# → non-adapted 전략은 기댓값이 양수인 "arbitrage" — 현실 불가능.
#   이토 적분이 predictable 요구하는 수학적 이유가 여기서 드러남.
```

### 실험 2 — 자연 필트레이션 vs augmented 필트레이션

```python
# σ-대수는 측정가능 함수의 모임으로 실증적으로 비교
# 자연 필트레이션: F_t^X = σ(X_s : s ≤ t)
# Augmented: F_t = σ(F_t^X ∪ 영집합)

# 영집합의 예: "BM이 정확히 π라는 사건"
# F_t^B에서 이는 measurable이지만 확률 0
# augmentation은 모든 확률 0 사건의 부분집합도 measurable로 만듦

# 수치적으로는 같은 결과를 주지만, 이론적 구분이 필요
# 수치 확인: P(|B_0.5 - π| < 10^{-4})
samples = rng.standard_normal(1_000_000) * np.sqrt(0.5)
prob = np.mean(np.abs(samples - np.pi) < 1e-4)
print(f'P(|B_0.5 - π| < 10^{{-4}}) = {prob:.2e}')   # 거의 0
# 이 사건은 F_0.5^B에서 measurable & 거의 영집합.
# augmentation은 이 사건의 "임의 부분집합"도 measurable로 만드는 기술적 확장.
```

### 실험 3 — 이산 마팅게일 — conditional expectation

```python
# Z_n = E[X | F_n]가 마팅게일인지 확인 (정리 4.2)
N = 1000
X = rng.standard_normal(10_000)   # 최종 값

# F_n = σ(ε_1, ..., ε_n) — 이산 필트레이션 시뮬레이션
# 간단히 X = Σ ε_k로 구성, Z_n = Σ_{k ≤ n} ε_k (부분합)
eps = rng.standard_normal((10_000, N))
X_sum = eps.sum(axis=1)
Z = np.cumsum(eps, axis=1)   # Z_n은 F_n-measurable

# E[Z_{n+1} | F_n] = Z_n 검증: 경험적으로
# 특정 n에서 Z_n 값이 같은 path 집단 내 Z_{n+1} 평균 = Z_n인가
# 간단화 — 상관관계 확인
n_check = 500
print(f'Corr(Z_{n_check}, Z_{n_check+1}) = {np.corrcoef(Z[:, n_check], Z[:, n_check+1])[0,1]:.4f}')
print(f'E[Z_{n_check+1} - Z_{n_check}] = {np.mean(Z[:, n_check+1] - Z[:, n_check]):.4f}')
# 증분 평균이 0 → 마팅게일 성질의 간접 검증
```

---

## 🔗 AI/ML 연결

**Policy Gradient의 unbiasedness**  
REINFORCE: $\nabla_\theta J = \mathbb{E}[\sum_t \nabla_\theta \log \pi_\theta(A_t | S_t) R_t]$의 unbiasedness는 $\nabla \log \pi_\theta$가 $\mathcal{F}_{t-}$-measurable(predictable), $R_t$ 미래 보상은 $\mathcal{F}_t$ 이후 — 이 **predictable × martingale increment** 구조가 gradient의 분산 축소(baseline) 이론의 토대. Reward-to-go baseline이 bias 없는 이유가 여기서 나온다.

**시계열 Cross-Validation**  
일반 K-fold CV는 시계열에서 필트레이션 위반(test 데이터가 train 이후 시점이어야 함). **Walk-forward validation**은 $\mathcal{F}_t$-adapted 예측만 허용하여 이 문제 해결. 실전 ML 파이프라인의 "data leakage" 버그는 필트레이션 위반의 대부분.

**이토 적분과 Diffusion 학습**  
Score network $s_\theta(X_t, t)$는 **adapted to $\mathcal{F}_t^X$**(시각 $t$의 noisy input만 사용) — 이것이 DSM loss의 well-definedness의 숨은 전제. 만약 $s_\theta$가 $X_{t+\delta}$를 봤다면 훈련 시 cheating, 추론에서 완전히 다른 함수 학습됨.

**Filtering(Kalman Filter, Particle Filter)**  
상태추정 $\hat{X}_t = \mathbb{E}[X_t | \mathcal{Y}_t]$ ($\mathcal{Y}_t$ = 측정 필트레이션)의 정의 자체가 필트레이션. 이 $\hat{X}$은 정리 4.2에 의해 마팅게일 — Kalman 필터의 수학적 근간.

**Transformer의 causal mask**  
Decoder의 causal attention mask는 $A_t$가 $\mathcal{F}_t = \sigma(X_1, \ldots, X_t)$에 adapted이도록 강제. mask를 빼면 bidirectional(BERT처럼)이 되어 필트레이션 의미가 없어짐.

---

## ⚖️ 가정과 한계

**가정 — 필트레이션 선택**  
같은 과정에 여러 필트레이션을 부여할 수 있다(자연 필트레이션, augmented, 더 큰 필트레이션). 마팅게일성은 필트레이션에 의존 — 작은 필트레이션에서 마팅게일이 큰 것에서는 아닐 수도 있음. 따라서 "어떤 필트레이션인지"를 명시하는 것이 중요.

**한계 — Usual conditions의 가정**  
우연속은 BM·Poisson 같은 "잘 행동하는" 과정에서 자동 성립(완비화 후). 그러나 모든 과정에서 그런 것은 아님. Point process, Semimartingale 중 일부는 예외.

**한계 — Observability in real systems**  
실제 관측 가능한 정보(예: 노이즈 낀 측정)는 $\mathcal{F}_t^X$보다 작을 수 있음. Filtering 문제는 이 **"관측 필트레이션"**과 "상태 필트레이션"의 차이에서 발생.

**주의 — Non-anticipating ≠ Markov**  
Adapted는 "미래 정보 안 씀"이지 "과거 전체를 씀/안 씀"과 무관. 즉 adapted process는 Markov이지 않을 수 있다. 두 개념을 혼동하지 말 것.

---

## 📌 핵심 정리

| 개념 | 정의 |
|------|------|
| 필트레이션 | 단조증가 $\sigma$-대수족 $\{\mathcal{F}_t\}$ |
| Adapted | 각 $t$에서 $X_t \in \mathcal{F}_t$ |
| Predictable | 이산: $H_n \in \mathcal{F}_{n-1}$; 연속: predictable $\sigma$-algebra measurable |
| Progressively measurable | $(t, \omega) \mapsto X_t(\omega)$가 $\mathcal{B}([0,T]) \otimes \mathcal{F}_T$-measurable |
| 자연 필트레이션 | $\mathcal{F}_t^X = \sigma(X_s : s \leq t)$ |
| Usual conditions | 우연속 + 완비 |
| 계층 | predictable ⊂ progressively measurable ⊂ adapted |

**한 줄 요약**: 필트레이션은 "시간 따라 커지는 정보 구조"이며, adapted는 "미래 정보 안 씀", predictable은 "한 스텝 앞까지 내다볼 수 있음"을 뜻한다. 이토 적분·마팅게일·정지시각 등 모든 확률해석 언어가 이 구조 위에 세워진다.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. $X_t = B_{2t}$ ($B$는 BM)가 $\mathcal{F}_t^B$-adapted인가? $\mathcal{F}_{2t}^B$-adapted인가?

<details>
<summary>해설</summary>

$\mathcal{F}_t^B = \sigma(B_s : s \leq t)$이므로 $X_t = B_{2t}$는 $t \neq 0$에서 $\mathcal{F}_t^B$-measurable이 **아니다** — $B_{2t}$는 $\mathcal{F}_{2t}^B$에는 있지만 $\mathcal{F}_t^B$에는 없음(시각 $2t > t$의 관찰).

따라서 $X$는 $\mathcal{F}_t^B$에 adapted가 아니지만, $\mathcal{F}_{2t}^B$를 재정의(새 필트레이션 $\mathcal{G}_t = \mathcal{F}_{2t}^B$)하면 adapted. 즉 **시간 재척도(rescaling) 하에서는 새 필트레이션**이 필요.

**교훈**: 같은 과정도 필트레이션 선택에 따라 adapted 여부가 달라진다. "이 과정이 adapted"라는 말은 "이 필트레이션에 대해"라고 명시해야 한다.

</details>

**문제 2 (심화)**. 연속시간에서 $X_t(\omega) = B_{t+1}(\omega)$(미래 값)이 $\mathcal{F}_t^B$에 adapted가 아님은 분명하다. 하지만 $X_t = \mathbb{E}[B_{t+1} | \mathcal{F}_t^B]$는 adapted인가? $X_t$를 명시적으로 계산하고 이것이 어떤 마팅게일의 예인지 논하라.

<details>
<summary>해설</summary>

**계산**: BM의 독립증분으로
$$\mathbb{E}[B_{t+1} | \mathcal{F}_t^B] = B_t + \mathbb{E}[B_{t+1} - B_t | \mathcal{F}_t^B] = B_t + 0 = B_t.$$
따라서 $X_t = B_t$ — adapted(자명).

**마팅게일 예**: 이는 정리 4.2 $Z_t = \mathbb{E}[X | \mathcal{F}_t]$의 특수 경우는 아니지만(상수 $X$가 아니기 때문), **BM 자체가 마팅게일**임을 재확인하는 예. 즉 "미래의 BM 값의 최선 예측은 현재 값" — BM이 "평균적으로 제자리"라는 공정한 게임 성격.

**연결**: Ch5-01(마팅게일 정의)와 Ch6(BM이 마팅게일)에서 $B_t, B_t^2 - t, \exp(\lambda B_t - \lambda^2 t/2)$이 모두 마팅게일임을 자세히 다룸.

</details>

**문제 3 (AI 연결)**. DDPM의 reverse process 학습 시 loss $\|\epsilon - \epsilon_\theta(X_t, t)\|^2$에서 $X_t$는 $\mathcal{F}_t$에 어떻게 adapted되는가? 만약 실수로 $\epsilon_\theta$가 $X_{t-1}$(미래 덜 노이즈된 값)을 입력으로 받는다면 훈련과 추론은 어떻게 깨지는가?

<details>
<summary>해설</summary>

**Adapted 구조**: DDPM에서 forward $X_t = \sqrt{\bar\alpha_t} X_0 + \sqrt{1 - \bar\alpha_t}\epsilon$. 학습 시 각 샘플에 대해 $(X_t, t, \epsilon)$ triple을 생성하고 $\epsilon_\theta(X_t, t)$가 $\epsilon$을 예측. $\epsilon_\theta$는 **$X_t$와 $t$만**을 입력으로 받으므로 $\mathcal{F}_t^X$에 adapted (함수값이 $X_{s \leq t}$에만 의존; 사실 단일 시점 $X_t$에만 의존하는 **Markov**).

**Cheating 버그**: 만약 $\epsilon_\theta(X_t, X_{t-1}, t)$처럼 $X_{t-1}$(더 noisy 아닌 값)을 입력으로 받으면:
- **훈련**: $X_{t-1}$에서 $X_t$로의 변환은 deterministic + 한 단계 노이즈 — $\epsilon$이 $X_t - \sqrt{1 - \beta_t}X_{t-1}$에서 거의 완전히 읽힘 → **훈련 손실이 0에 수렴**.
- **추론 시점**: Reverse sampling $X_T \to X_{T-1} \to \cdots \to X_0$에서 시각 $t$에서 $X_{t-1}$를 구하려는데, 이미 $X_{t-1}$을 입력으로 요구 → **circular dependency**, 불가능.

**필트레이션 관점**: $\epsilon_\theta$가 $\mathcal{F}_t$ 이외의(미래) 정보를 쓰면 이토 적분/SDE 관점에서 non-adapted. 학습은 될 수 있지만 **추론 시 adapted 복제 불가능**.

**교훈**: DDPM/Score-SDE의 "시간 조건부" 학습 구조는 implicit하게 adapted filtering constraint를 반영. 실전 구현에서 "입력을 잘못 설계"하면 학습이 너무 쉬워 보이지만 추론이 붕괴 — 이는 필트레이션 위반의 전형적 증상.

</details>

---

<div align="center">

◀ [03. 정상성(Stationarity) — 강·약](./03-stationarity.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [05. 확률과정의 분류 지도](./05-classification.md)

</div>
