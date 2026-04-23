# 02. 유한차원 분포와 Kolmogorov 확장정리

## 🎯 핵심 질문

- 확률과정을 "어떻게 만들 수 있는가" — 유한차원 분포족 $\{\mu_{t_1,\ldots,t_n}\}$만 주어져도 과정이 존재하는가?
- **일관성 조건**(대칭·주변) 두 가지는 각각 무엇을 의미하고, 왜 둘 다 필요한가?
- Kolmogorov 확장정리는 **무엇을 보장하고, 무엇을 보장하지 않는가** — 경로 연속성은 왜 별도의 정리(Kolmogorov continuity theorem)가 필요한가?
- 상태공간이 **Polish** 조건을 요구하는 이유는 무엇인가?

---

## 🔍 왜 이 정리가 AI에서 중요한가

**Diffusion Model의 forward process**를 "연속시간 확률과정"으로 기술할 때, 우리는 각 시각 $t$의 marginal $p_t$와 joint $p_{s,t}$만 주로 다룬다. 그러나 "경로 전체의 분포 $\mathbb{P}_X$가 존재한다"는 사실은 자명하지 않다. Kolmogorov 확장정리는 **유한차원 분포만 일관되게 주어지면 경로 측도가 유일하게 존재한다**는 근본적 보장을 제공한다. 이 보장 없이는 다음이 불가능하다:

- **Score-SDE**(Song 2021)의 path measure 해석과 reverse SDE 구성(Anderson 1982)
- **Gaussian Process**(베이지안 회귀)의 커널만으로 과정 정의
- **Wiener measure**(BM의 path measure)의 존재성 보장 — 이토 적분의 바탕
- **Normalizing flow**의 continuous-time 확장(FFJORD)에서 ODE 해의 측도 정의

또한 Transformer의 시계열 학습에서 "실제 데이터 분포"를 **fdd 가족**으로 추상화하는 것이 이 정리의 직접적 응용이다.

---

## 📐 수학적 선행 조건

- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Carathéodory 확장정리, premeasure, $\sigma$-additivity
- 측도론: Polish space, product $\sigma$-algebra, cylinder set
- [Ch1-01](./01-rigorous-definition.md): 확률과정, sample path, 유한차원 분포

---

## 📖 직관적 이해

### 문제 설정

확률과정 $\{X_t\}_{t \in T}$를 구성하려면 전체 sample path의 분포 $\mathbb{P}_X$ on $E^T$를 줘야 한다. 그런데 실제로 우리가 "알고 있는" 정보는 보통 **유한한 시각에서의 결합분포**뿐이다:
$$\mu_{t_1, \ldots, t_n}(\cdot) = \text{Law}(X_{t_1}, \ldots, X_{t_n}).$$

**질문**: fdd 가족만 주면 과정이 존재하는가? 즉, "마르코프 체인 $P$", "BM의 가우시안 fdd", "포아송의 독립증분 fdd"만으로 경로 측도를 정의할 수 있는가?

**답**: "일관성" 조건을 만족하면 **YES** (Polish 상태공간 가정 하).

### 두 가지 일관성 조건

fdd 가족이 합리적이려면 다음을 만족해야 한다:

**(C1) 대칭(Symmetry)**: 인덱스를 permute해도 같은 분포
$$\mu_{t_{\sigma(1)}, \ldots, t_{\sigma(n)}}(A_{\sigma(1)} \times \cdots \times A_{\sigma(n)}) = \mu_{t_1, \ldots, t_n}(A_1 \times \cdots \times A_n)$$
(단, 같은 시각 집합에서 인덱스만 바꿈)

**(C2) 주변(Marginal)**: 한 인덱스를 "잊으면" 작은 fdd와 일치
$$\mu_{t_1, \ldots, t_n}(A_1 \times \cdots \times A_{n-1} \times E) = \mu_{t_1, \ldots, t_{n-1}}(A_1 \times \cdots \times A_{n-1})$$

**직관**: (C1)은 "순서는 의미 없음", (C2)는 "작은 시각 집합에서 본 분포는 큰 것의 marginal"이라는 상식적 요구.

### Kolmogorov 확장정리의 역할

위 두 조건을 만족하는 fdd 가족이 있으면, 이로부터 **유일한** $(E^T, \mathcal{E}^{\otimes T}, \mathbb{P})$ 위의 확률측도가 결정되어, $X_t(\omega) = \omega(t)$로 하는 coordinate 과정이 바로 그 fdd를 갖는다.

> **비유**: 건물을 짓는데 각 층 평면도(유한차원 분포)를 넘겼다고 하자. 각 층들이 "아래층 평면도에 호환되는가"(C2)와 "호수만 다른 같은 층은 같은 평면"(C1)을 만족하면 전체 건물(전역 측도)이 유일하게 완성된다. 이는 "로컬 정보 → 글로벌 존재"의 전형적 예시.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — 일관성 있는 fdd 가족

$T$를 임의의 지표집합, $(E, \mathcal{E})$를 가측공간이라 하자. 유한 튜플 $(t_1, \ldots, t_n)$ (서로 다른 $t_i$)마다 $E^n$ 위의 확률측도 $\mu_{t_1, \ldots, t_n}$이 지정된 가족 $\{\mu_{t_1, \ldots, t_n}\}$가 **일관성이 있다**(consistent)는 것은 (C1) 대칭과 (C2) 주변 조건을 모두 만족한다는 의미다.

### 정의 2.2 — Cylinder set

$T$의 유한 부분집합 $\{t_1, \ldots, t_n\}$과 Borel $A \in \mathcal{E}^{\otimes n}$에 대해
$$C = \{\omega \in E^T : (\omega(t_1), \ldots, \omega(t_n)) \in A\}$$
를 cylinder set이라 한다. cylinder set의 모임은 대수(algebra)를 이루며, 이 대수로 생성되는 $\sigma$-대수 $\mathcal{E}^{\otimes T}$가 product $\sigma$-algebra.

### 정의 2.3 — Polish space

거리공간 $(E, d)$가 **완비**(complete)이고 **분리가능**(separable)하면 **Polish space**라 한다. 대표 예: $\mathbb{R}^d$, $C([0,1])$, $\ell^2$, 가산 이산공간 등.

---

## 🔬 정리와 증명

### 정리 2.1 (Kolmogorov 확장정리, Existence Theorem)

$T$를 임의의 지표집합, $(E, \mathcal{E})$를 Polish space 위의 Borel $\sigma$-대수라 하자. 일관성을 만족하는 fdd 가족 $\{\mu_{t_1, \ldots, t_n}\}$이 주어지면, **$(E^T, \mathcal{E}^{\otimes T})$ 위에 유일한 확률측도 $\mathbb{P}$가 존재**하여, coordinate 과정 $X_t(\omega) = \omega(t)$가 주어진 fdd를 갖는다:
$$\mathbb{P}(X_{t_1} \in A_1, \ldots, X_{t_n} \in A_n) = \mu_{t_1, \ldots, t_n}(A_1 \times \cdots \times A_n).$$

### 증명 스케치

**Step 1: Cylinder 대수 위의 premeasure 정의**  
Cylinder set $C = \{(\omega(t_1), \ldots, \omega(t_n)) \in A\}$에 대해
$$\mathbb{P}_0(C) := \mu_{t_1, \ldots, t_n}(A).$$
일관성 (C1), (C2) 덕분에 **같은 cylinder set을 다른 유한 인덱스로 표현해도 같은 값이 나오는 well-definedness**가 성립.

**Step 2: Finitely additive임을 확인**  
Cylinder 대수가 대수임을 확인하고, 위 $\mathbb{P}_0$가 **유한가법적 premeasure**임을 증명. (각 유한 인덱스에서의 확률측도 성질이 그대로 이관됨)

**Step 3: $\sigma$-additivity (핵심 단계)**  
$C_n \downarrow \emptyset$이면 $\mathbb{P}_0(C_n) \downarrow 0$을 보여야 한다.  
*모순법*: $\mathbb{P}_0(C_n) \geq \epsilon > 0$이라 가정. Polish 상태공간이므로 각 마다 **inner-regular**한 compact approximation $K_n \subset C_n$을 잡을 수 있고(Ulam의 정리), Tychonoff 정리로 $\prod K_n$이 $E^T$의 compact set, 따라서 유한교집합 성질로 $\bigcap_n C_n \neq \emptyset$이라는 모순.  
**이 부분이 "Polish" 가정의 유일한 쓰임**이다.

**Step 4: Carathéodory 확장**  
$\sigma$-additive premeasure이면 Carathéodory 확장정리에 의해 $\mathcal{E}^{\otimes T}$로 유일 확장. $\square$

> **주의**: 증명은 "일관성 ⇒ 유한가법 premeasure ⇒ (Polish 덕분에) $\sigma$-additive ⇒ Carathéodory 확장"의 4단계 구조. Polish 가정이 깨지면 Step 3가 무너진다.

### 정리 2.2 (경로 연속성은 보장되지 않음)

Kolmogorov 확장정리로 얻은 측도 $\mathbb{P}$는 $E^T$ 전체 위에서 살지만, **sample path의 연속성·가측성에 대해 아무 말도 하지 않는다**.

*증명.* Ch1-01 정리 1.2의 예에서 $X_t(\omega) \equiv 0$과 $Y_t(\omega) = \mathbf{1}_{\{t = \omega\}}$가 같은 fdd(모든 유한 $t_1, \ldots, t_n$에서 $\mathbb{P}(X_{t_i} = 0) = 1$)를 갖지만 경로는 전혀 다르다. fdd는 경로 연속성을 결정하지 않는다. $\square$

### 정리 2.3 (Kolmogorov Continuity Theorem — 연속 modification)

$T = [0, T_*]$이고 확률과정 $\{X_t\}$가 다음을 만족한다고 하자:
$$\mathbb{E}\left[|X_t - X_s|^\alpha\right] \leq C |t-s|^{1+\beta}, \quad \forall s, t, \quad \text{some } \alpha, \beta, C > 0.$$
그러면 $X$의 modification $\tilde{X}$가 존재하여 **a.s. $\gamma$-Hölder 연속 경로**를 가진다 ($\gamma < \beta/\alpha$).

### 증명 스케치

다이아딕 시각 $D_n = \{k/2^n : k\}$ 위에서 $|X_{(k+1)/2^n} - X_{k/2^n}|$의 최대값을 Chebyshev 부등식 + Borel-Cantelli로 통제. 구체적으로
$$\mathbb{P}(|X_t - X_s| \geq 2^{-n\gamma}) \leq C 2^{-n(1 + \beta - \gamma \alpha)}$$
이고, 오른쪽 합이 수렴하면 거의 확실히 유한 번만 큰 점프. Dyadic에서 Hölder가 연속적으로 확장되고, dense set에서의 일관성으로 전체 $[0, T_*]$로. $\square$

> **활용**: BM은 $\mathbb{E}|B_t - B_s|^{2n} = c_n |t-s|^n$이므로 $\alpha = 2n, \beta = n - 1$로 $\gamma < 1/2 - 1/(2n)$인 모든 $\gamma$에 대해 Hölder — "BM은 거의 $1/2$-Hölder"의 근거.

### 명제 2.4 (Polish가 필요한 이유 — 반례)

Polish 가정 없이 임의 가측공간에서는 정리 2.1이 성립하지 않는다. 예: 비가산 $\Omega$와 trivial $\sigma$-대수 조합에서 cylinder premeasure가 $\sigma$-additive가 아닌 구성이 가능(기술적 예는 Dudley 2002 §12 참조). AI 응용에서 실제로 쓰는 상태공간($\mathbb{R}^d$, 유한 집합, $C([0,1])$)은 모두 Polish이므로 실전에서는 문제 없음.

---

## 💻 NumPy 구현 검증

### 실험 1 — 일관성 조건 수치적 확인 (BM의 fdd)

```python
import numpy as np
rng = np.random.default_rng(0)

# BM의 이론 fdd: (B_0.3, B_0.5, B_0.9)는 N(0, Σ), Σ_ij = min(t_i, t_j)
t = np.array([0.3, 0.5, 0.9])
Sigma = np.minimum.outer(t, t)
n_samples = 200_000
L = np.linalg.cholesky(Sigma)
samples_3 = (rng.standard_normal((n_samples, 3))) @ L.T   # (B_0.3, B_0.5, B_0.9)

# (C2) Marginal consistency: 첫 2좌표를 marginalize → (B_0.3, B_0.5)의 fdd
Sigma_12 = Sigma[:2, :2]
L12 = np.linalg.cholesky(Sigma_12)
samples_2 = (rng.standard_normal((n_samples, 2))) @ L12.T

# 두 방식의 (B_0.3, B_0.5) 분포가 같은지 — 공분산 비교
cov_from_3 = np.cov(samples_3[:, :2].T)
cov_from_2 = np.cov(samples_2.T)
print('3점에서 projection:'); print(cov_from_3)
print('2점 직접 생성:');       print(cov_from_2)
print('이론 Σ_12:');           print(Sigma_12)
# → 세 값 모두 일치 (오차는 샘플링 변동)
```

### 실험 2 — Kolmogorov continuity로 BM의 Hölder 지수 관찰

```python
# BM 경로의 dyadic 차이 최대값 ~ 2^{-n/2} √n (log 보정)
N = 2**16
dt = 1.0 / N
B = np.concatenate([[0], np.cumsum(rng.standard_normal(N) * np.sqrt(dt))])

results = []
for n in range(6, 15):
    step = 2**(16 - n)
    inds = np.arange(0, N + 1, step)
    diffs = np.abs(np.diff(B[inds]))
    # 최대 차이 ~ √dt · √(2 log 2^n)
    results.append((n, diffs.max(), np.sqrt(2**(-n)) * np.sqrt(2 * np.log(2**n))))

print(f'{"n":>3}  {"max |ΔB|":>10}  {"이론 ~√dt·√(2 log 2^n)":>25}')
for n, m, th in results:
    print(f'{n:>3}  {m:>10.5f}  {th:>25.5f}')
# 경험 최대값이 이론 상한 경계와 유사 스케일(~1/2-Hölder)임을 확인
```

### 실험 3 — 비일관된 fdd는 과정으로 존재하지 않음

```python
# 의도적으로 (C2) 주변 일관성을 깨는 fdd 시도
# μ_{t1, t2} = N(0, I), μ_{t1} = N(0, 2) (주변이 맞지 않음)
# → 이런 fdd 가족으로는 과정 존재 불가

Sigma_pair = np.eye(2)              # (X_{t1}, X_{t2}) ~ N(0, I)
var_t1_from_pair = Sigma_pair[0, 0]  # = 1
var_t1_claimed = 2.0                 # μ_{t1}이 주장하는 분산

if var_t1_from_pair != var_t1_claimed:
    print(f'주변 일관성 깨짐: '
          f'{var_t1_from_pair=} ≠ {var_t1_claimed=}')
    print('→ Kolmogorov 확장정리 적용 불가, 과정 존재 불가')
```

---

## 🔗 AI/ML 연결

**Gaussian Process (GP) 회귀**  
GP는 "모든 유한 부분집합 $\{x_1, \ldots, x_n\}$에서 joint 가우시안"으로 정의된다. 커널 $k(x, x')$로 공분산 $\Sigma_{ij} = k(x_i, x_j)$를 주면 자동으로 일관성이 만족되고(대칭성·marginal), Polish 상태공간($\mathbb{R}$)이므로 Kolmogorov가 GP의 **존재성을 보장**한다. 이 존재 보장 없이는 "사전 분포로서의 GP"가 의미 없음.

**Score-SDE의 path measure**  
$dX_t = f(X_t, t) dt + g(t) dB_t$가 생성하는 path measure $\mathbb{P}_X$는 fdd 가족(각 유한 시각에서의 조건부 가우시안)으로부터 Kolmogorov 확장으로 얻어진다. Reverse SDE를 **시간반전 path measure**로 해석할 때 이 존재 보장이 핵심.

**Continuous-time Normalizing Flow**  
FFJORD는 ODE flow $dx/dt = v_\theta(x, t)$를 이용해 $x_0 \sim p_0$에서 $x_1$의 분포를 생성. path measure로 해석할 때 Kolmogorov가 적용 — 각 시각 marginal은 flow ODE로 계산.

**Neural Process (NP)**  
각 시각에 대한 prediction이 "fdd consistent"하도록 학습되어야 이론적으로 GP 같은 과정을 근사한다(Garnelo et al. 2018). NP의 consistency loss는 본질적으로 (C1), (C2)를 깨지 않도록 하는 정규화.

---

## ⚖️ 가정과 한계

**가정 1 — Polish 상태공간**  
이 가정은 실전 ML에서 거의 문제 없다($\mathbb{R}^d$, $\mathbb{Z}^d$, $C([0,1])$ 모두 Polish). 다만 **분포값 과정**(distribution-valued process, 예: McKean-Vlasov 한계 interaction particle system)에서는 $E$가 확률측도 공간이 되어 Polish이긴 하지만 거리 구조가 Wasserstein 등으로 복잡.

**가정 2 — 일관성**  
실제 ML에서 "비일관된 fdd 가족"을 실수로 생성하기 쉽다(서로 다른 모델이 서로 다른 marginal을 주는 경우 등). 이를 방지하려면 fdd를 **단일 생성 메커니즘**(커널·SDE·transition kernel)으로 통일하는 것이 안전.

**한계 — 경로 정칙성**  
Kolmogorov 확장은 "과정이 존재함"만 보장하고, 경로가 연속인지·적분 가능한지는 추가 조건 필요(정리 2.3). 특히 BM의 연속 경로 버전은 Lévy의 직접 구성(Ch6-02)이 더 직접적.

**한계 — 무한 인덱스 $T$**  
$T$가 비가산이면 product $\sigma$-algebra $\mathcal{E}^{\otimes T}$가 "유한 좌표로 결정되는 사건"만 포함. 예: $\{\omega : \omega$는 연속$\}$는 이 $\sigma$-algebra에 **없을** 수 있다. 따라서 "BM이 연속 경로를 갖는다"를 Kolmogorov 구성 공간 위에서 측정할 수 없을 수 있고, modification을 잘 잡아야 한다.

---

## 📌 핵심 정리

| 개념 | 요약 |
|------|------|
| fdd | 유한 시각에서의 결합분포 |
| (C1) 대칭 | 인덱스 permutation 불변 |
| (C2) 주변 | 작은 부분집합 fdd = 큰 것의 marginal |
| Kolmogorov 확장 | 일관성 + Polish ⇒ path measure 유일 존재 |
| Kolmogorov continuity | $\mathbb{E}|X_t - X_s|^\alpha \leq C|t-s|^{1+\beta}$ ⇒ Hölder 연속 modification |
| 보장 안 됨 | 경로 연속성·joint measurability (별도 추가 조건) |

**한 줄 요약**: "유한차원 분포를 일관되게 주면 과정은 존재"(Kolmogorov 확장)한다. 그러나 "경로가 연속"인 것은 별도의 Hölder 모멘트 조건(Kolmogorov continuity)으로 얻는다.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. 가우시안 과정 $\{X_t\}$가 mean function $m(t)$와 covariance function $k(s, t)$로 주어진다고 하자. $k$가 **양정치 커널**이면 일관성이 자동으로 만족됨을 보여라.

<details>
<summary>해설</summary>

임의 $t_1, \ldots, t_n$에 대해 $\Sigma_{ij} = k(t_i, t_j)$가 대칭·양반정치 행렬이므로 $(m(t_1), \ldots, m(t_n))$ 평균과 $\Sigma$ 공분산의 다변량 가우시안 $\mu_{t_1, \ldots, t_n}$가 well-defined.

- **(C1) 대칭**: permutation $\sigma$에 대해 $\Sigma_{\sigma(i)\sigma(j)}$를 공분산으로 하는 가우시안은 원래 가우시안의 좌표 교체 — 같은 분포.
- **(C2) 주변**: $n$차원 가우시안의 첫 $n-1$ 좌표 marginal은 서브공분산 $\Sigma_{1:n-1, 1:n-1}$의 $(n-1)$차원 가우시안. 이는 정확히 $\mu_{t_1, \ldots, t_{n-1}}$과 일치.

따라서 양정치 커널만 있으면 일관성 자동. Polish 상태공간 $\mathbb{R}$이므로 Kolmogorov에 의해 과정 존재. 이것이 **GP를 커널만으로 정의**할 수 있는 이론적 근거.

</details>

**문제 2 (심화)**. Ch1-01 정리 1.2의 $Y_t(\omega) = \mathbf{1}_{\{t = \omega\}}$ 예에서 $Y$의 모든 fdd는 $X \equiv 0$의 fdd와 **같다**. Kolmogorov 확장정리가 경로 공간 $\mathbb{R}^T$ 위에서 단일 측도를 주는데, 왜 "$Y$의 경로"와 "$X$의 경로"가 다를 수 있는가?

<details>
<summary>해설</summary>

Kolmogorov가 주는 측도는 **coordinate 과정** $\omega \mapsto \omega(t)$만 결정하며, 이는 같은 $(\Omega, \mathcal{F}, \mathbb{P})$ 위에서 정의된 두 과정이 꼭 일치한다는 것이 아니다. **같은 fdd를 갖지만 서로 indistinguishable이 아닌 두 과정**은 동일 path measure의 **서로 다른 modification**일 뿐이다.

구체적으로, $Y_t(\omega) = \mathbf{1}_{\{t = \omega\}}$에 대한 path measure가 coordinate 공간에서 보는 분포는 "모든 시각에서 $0$이 a.s."인 분포 — 즉 $X \equiv 0$과 동일. $Y$는 **원래 $(\Omega, \mathcal{F}, \mathbb{P}) = ([0,1], \mathcal{B}, \text{Leb})$** 위에서 "각 $\omega$마다 단일 점프를 가진 경로"지만, 이 구체적 경로 구조는 **product $\sigma$-대수가 포착하지 못한다**(비가산 인덱스 T에서 "경로 연속성" 같은 사건은 product $\sigma$-algebra에 들어있지 않을 수 있음 — 한계 참조).

**교훈**: path measure는 fdd로 결정되지만, 구체적 경로 구조는 modification 수준에서 결정. 경로 연속성을 얻으려면 Kolmogorov continuity theorem(정리 2.3)으로 적절한 modification을 잡아야 함.

</details>

**문제 3 (AI 연결)**. Continuous-time normalizing flow $dx/dt = v_\theta(x, t)$에서 marginal $p_t$가 Fokker-Planck과 유사한 continuity equation $\partial_t p_t + \nabla \cdot (v_\theta p_t) = 0$을 만족한다(Neural ODE). 이 flow로 유도되는 path measure가 존재함을 보장하는 이론적 근거는? 또한 Kolmogorov 확장에서 "Polish" 조건이 쓰이는 부분을 지적하라.

<details>
<summary>해설</summary>

**존재 근거**: $v_\theta$가 Lipschitz이면 Picard 반복으로 ODE 해 $x_t = \phi_t(x_0)$가 일의적으로 존재. 이는 각 $x_0$마다 **연속 경로** $t \mapsto \phi_t(x_0)$를 결정 — **deterministic path**. 초기 분포 $p_0$를 주면 path measure는 "deterministic flow $\phi_t$의 pushforward"로 정의:
$$\mathbb{P}_X = \int \delta_{\phi_\bullet(x_0)} p_0(dx_0).$$
이는 $C([0, 1]; \mathbb{R}^d)$ 위의 측도로, fdd 가족 $\mu_{t_1, \ldots, t_n}(A_1 \times \cdots \times A_n) = \mathbb{P}_0(\{x_0 : \phi_{t_i}(x_0) \in A_i\})$가 일관성을 자동으로 만족. Kolmogorov 확장으로 $\mathbb{R}^{[0,1]}$ 위의 측도를 얻고, $\phi$의 연속성 덕에 $C([0,1]; \mathbb{R}^d)$ 위의 측도로 축소.

**Polish 조건**: 상태공간 $\mathbb{R}^d$는 Polish. 이 조건이 쓰이는 곳은 정리 2.1 증명의 Step 3 — inner-regular compact approximation으로 $\sigma$-additivity를 얻는 단계. 상태가 비 Polish(예: 함수공간 $L^0$의 거리 없는 버전)이면 이 논증이 무너진다.

**실전 함의**: FFJORD의 확률 해석(likelihood 계산)은 위 path measure의 marginal $p_t$를 ODE 해의 변수변환으로 계산. Kolmogorov의 존재 보장이 "이런 계산이 legitimate"임을 뒷받침.

</details>

---

<div align="center">

◀ [01. 확률과정의 엄밀한 정의](./01-rigorous-definition.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [03. 정상성(Stationarity) — 강·약](./03-stationarity.md)

</div>
