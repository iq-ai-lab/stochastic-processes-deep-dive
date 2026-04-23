# 02. 존재성 — Lévy의 Haar 기저 구성

## 🎯 핵심 질문

- BM이 **존재하는가** — 공리를 만족하는 구체적 구성은 무엇인가?
- Lévy가 제시한 **Haar 기저 기반 random series** 구성이 왜 작동하는가?
- **균등 수렴** (uniform convergence)이 연속 경로 보장에 어떻게 기여하는가?
- Kolmogorov 확장정리를 이용한 대안 존재 증명과의 비교?

---

## 🔍 왜 이 구성이 중요한가

**수학적 정당성**: BM을 쓰기 전에 "존재함을 증명"하는 것은 기초. 공리가 모순 없는지 확인.

**실전 시뮬레이션**: Lévy 구성은 **실전 BM 샘플링 알고리즘**의 기반 (Brownian bridge, circulant embedding).

**Gaussian process 생성**: 임의 공분산 $k(s,t)$의 GP 생성에 basis expansion 기법. Lévy 구성은 BM의 경우.

**Fractional BM 구성**: BM의 아이디어를 general Hurst parameter로 확장. Wavelet-based 구성 (Daubechies, Mallat).

---

## 📐 수학적 선행 조건

- [Ch1-02](../ch1-foundations/02-kolmogorov-extension.md): Kolmogorov 확장, continuity theorem
- [Ch6-01](./01-axiomatic-definition.md): BM 공리
- 함수해석 기초: $L^2$ orthonormal basis, series convergence
- Borel-Cantelli lemma

---

## 📖 직관적 이해

### Key aidea: "GP를 Gaussian series로 쓰기"

$B_t$를 함수 $[0, 1] \to \mathbb{R}$로 보자. $L^2[0, 1]$의 orthonormal basis $\{\phi_n\}$에 iid 가우시안 계수 $\xi_n$:
$$B_t = \sum_n \xi_n \int_0^t \phi_n(s) ds.$$

**왜 이것이 BM인가**:
- Linear in $\xi_n$ → Gaussian process
- 공분산 계산 후 $\min(s, t)$ 확인 → BM fdd

**연속성 문제**: Series가 균등 수렴하는가? Gaussian 계수의 sup에 대한 tail bound (Borel-Cantelli) 필요.

### Haar 기저의 선택

**Haar wavelets** $\{h_{n, k}\}$: 다이아딕 step functions:
$$h_{n, k}(t) = 2^{n/2} \cdot \begin{cases} 1 & t \in [k \cdot 2^{-n}, (k + 1/2) \cdot 2^{-n}) \\ -1 & t \in [(k+1/2) \cdot 2^{-n}, (k+1) \cdot 2^{-n}) \\ 0 & \text{else} \end{cases}$$

**$\int_0^t h_{n, k}$의 모양**: Triangular function (tent) with peak at $(k + 1/2) \cdot 2^{-n}$.

**장점**: 계수 $\xi_{n, k}$가 서로 다른 $n$에서 "다른 시간 scale" 기여 → 자연스러운 multi-scale 구조.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Haar System on $[0, 1]$

$\phi_0 \equiv 1$ (상수).
$n \geq 0, 0 \leq k < 2^n$에 대해
$$h_{n, k}(t) = 2^{n/2} (\mathbf{1}_{[k 2^{-n}, (k+1/2) 2^{-n})} - \mathbf{1}_{[(k+1/2) 2^{-n}, (k+1) 2^{-n})}).$$

$\{\phi_0\} \cup \{h_{n, k} : n \geq 0, 0 \leq k < 2^n\}$가 $L^2[0, 1]$의 orthonormal basis.

### 정의 2.2 — Schauder System (Integral of Haar)

$\Phi_0(t) = t$ (integral of $\phi_0$).
$\Phi_{n, k}(t) = \int_0^t h_{n, k}(s) ds$ — "tent" function, supports on $[k 2^{-n}, (k+1) 2^{-n}]$, peak $2^{-n/2-1}$ at $(k + 1/2) 2^{-n}$.

### 정의 2.3 — Lévy's Construction

iid $\mathcal{N}(0, 1)$: $\xi_0, \xi_{n, k}$ for $n \geq 0, 0 \leq k < 2^n$. 정의:
$$B_t := \xi_0 t + \sum_{n=0}^\infty \sum_{k=0}^{2^n - 1} \xi_{n, k} \Phi_{n, k}(t).$$

---

## 🔬 정리와 증명

### 정리 2.1 — Lévy's Construction의 결과가 BM

정의 2.3의 $B_t$가:
(a) 거의 확실히 연속
(b) Gaussian process with covariance $\min(s, t)$
(c) 독립증분 + $\mathcal{N}(0, t-s)$ 증분 + $B_0 = 0$

따라서 표준 BM (up to modification).

### 증명 스케치

**Step 1 — 공분산 계산**:
$\int_0^t \phi_n(s) ds \cdot \int_0^s \phi_n(u) du$를 각 orthonormal basis 성분에 대해 계산. Parseval:
$$\mathbb{E}[B_s B_t] = \sum_n \left(\int_0^s \phi_n(u) du\right) \left(\int_0^t \phi_n(u) du\right).$$

Orthonormal basis에서 $\int_0^t \phi_n(s) ds$가 $\mathbf{1}_{[0, t]}$의 basis expansion의 $n$번째 계수. Parseval:
$$\sum_n \left(\int_0^s \phi_n\right) \left(\int_0^t \phi_n\right) = \langle \mathbf{1}_{[0,s]}, \mathbf{1}_{[0,t]}\rangle_{L^2} = \int_0^1 \mathbf{1}_{[0,s]} \mathbf{1}_{[0,t]} du = \min(s, t).$$

**Step 2 — Gaussian property**:
Linear in Gaussian $\xi$ → Gaussian process (모든 fdd Gaussian).

**Step 3 — 연속성 (핵심)**:
$$B_t = \xi_0 t + \sum_{n=0}^\infty Z_n(t), \quad Z_n(t) := \sum_{k=0}^{2^n - 1} \xi_{n, k} \Phi_{n, k}(t).$$

$\sup_t |\Phi_{n, k}(t)| \leq 2^{-n/2 - 1}$ (tent 높이). 각 $n$에 대해 여러 $k$들의 support가 **disjoint** (서로 다른 dyadic intervals). 따라서 
$$\sup_t |Z_n(t)| = \max_k |\xi_{n, k}| \cdot 2^{-n/2 - 1}.$$

$\max_{0 \leq k < 2^n} |\xi_{n, k}|$의 tail: iid standard normals의 max. **Gaussian tail bound**:
$$\mathbb{P}(\max_k |\xi_{n, k}| > \sqrt{c n}) \leq 2^n \cdot 2 e^{-cn/2} = 2 \cdot 2^{n(1 - c/(2 \log 2))}.$$

$c > 2 \log 2$로 선택 → $\sum_n 2^n e^{-cn/2} < \infty$ → Borel-Cantelli: a.s. $\max_k |\xi_{n, k}| \leq \sqrt{cn}$ for large $n$.

따라서 a.s.
$$\sup_t |Z_n(t)| \leq \sqrt{cn} \cdot 2^{-n/2 - 1} = C n^{1/2} 2^{-n/2}.$$

$\sum_n n^{1/2} 2^{-n/2} < \infty$ (exponential decay beats polynomial) → series uniformly convergent a.s.

**Uniform limit of continuous functions is continuous** → $B_t$ 연속 a.s. $\square$

### 정리 2.2 — Lévy's 구성이 독립증분을 가짐

Step 1에서 공분산 $\min(s, t)$ → Gaussian process with this covariance. Independent increments 자동:
$$\text{Cov}(B_s, B_t - B_s) = \min(s, t) - \min(s, s) = s - s = 0.$$

Gaussian + uncorrelated → independent. $\square$

### 정리 2.3 — Lévy vs Kolmogorov 구성

**Kolmogorov 접근**: fdd → Kolmogorov 확장 → 과정 존재, but 연속 path 보장 없음. Kolmogorov continuity theorem으로 continuous modification 별도 확인.

**Lévy 접근**: 직접 연속 경로 과정 구성. Uniform convergence로 연속성이 **자동**.

**장점 비교**: Lévy는 explicit & constructive, Kolmogorov는 abstract & general.

---

## 💻 NumPy 구현 검증

### 실험 1 — Lévy's 구성으로 BM 샘플링

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

def schauder_triangle(t, n, k):
    """Φ_{n,k}(t) — tent function"""
    a = k * 2**(-n)
    m = (k + 0.5) * 2**(-n)
    b = (k + 1) * 2**(-n)
    if t <= a or t >= b:
        return 0
    elif t <= m:
        return 2**(n/2) * (t - a)
    else:
        return 2**(n/2) * (b - t)

def levy_BM(t_grid, n_levels, rng):
    """Lévy 구성으로 BM 샘플 생성"""
    xi_0 = rng.standard_normal()
    B = xi_0 * t_grid  # 초기 term
    for n in range(n_levels):
        for k in range(2**n):
            xi = rng.standard_normal()
            for i, t in enumerate(t_grid):
                B[i] += xi * schauder_triangle(t, n, k)
    return B

# Multi-resolution 시각화
t_grid = np.linspace(0, 1, 1000)
fig, axes = plt.subplots(1, 4, figsize=(16, 3))
for i, n_lev in enumerate([0, 2, 5, 10]):
    rng_ = np.random.default_rng(0)
    B = levy_BM(t_grid, n_lev, rng_)
    axes[i].plot(t_grid, B)
    axes[i].set_title(f'Levels 0..{n_lev}')
    axes[i].grid(True, alpha=0.3)
plt.suptitle("Lévy's Construction — multi-level Haar basis")
plt.tight_layout(); plt.show()
# n 레벨 증가 → 더 fine detail 출현, BM 경로 approach
```

### 실험 2 — 공분산 검증

```python
n_sim = 2000
n_levels = 10
N_t = 50
t_grid = np.linspace(0, 1, N_t)

samples = np.zeros((n_sim, N_t))
for s in range(n_sim):
    rng_ = np.random.default_rng(s)
    samples[s] = levy_BM(t_grid, n_levels, rng_)

# 실측 공분산 vs min(s, t)
cov_emp = np.cov(samples, rowvar=False)
cov_theory = np.minimum.outer(t_grid, t_grid)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.imshow(cov_emp); plt.title('실측'); plt.colorbar()
plt.subplot(1, 2, 2); plt.imshow(cov_theory); plt.title('min(s,t)'); plt.colorbar()
plt.show()
# 거의 일치 (약간 noise)
```

### 실험 3 — 표준 독립증분 BM과 비교

```python
# Standard BM: independent increments
def standard_BM(t_grid, rng):
    dt = np.diff(np.concatenate([[0], t_grid]))
    return np.cumsum(rng.standard_normal(len(t_grid)) * np.sqrt(dt))

# 두 방법의 variance
lev_samples = [levy_BM(t_grid, 10, np.random.default_rng(i)) for i in range(1000)]
std_samples = [standard_BM(t_grid, np.random.default_rng(i + 10000)) for i in range(1000)]
lev_samples = np.array(lev_samples)
std_samples = np.array(std_samples)

print(f'Levy B_1의 분산: {lev_samples[:, -1].var():.4f}')
print(f'Standard B_1의 분산: {std_samples[:, -1].var():.4f}')
print(f'이론 t = 1')
# 둘 다 1에 근접 — 같은 BM 분포
```

---

## 🔗 AI/ML 연결

**Wavelet-based Neural Networks**  
BM의 wavelet 구조 (Haar / Daubechies)가 CNN의 scale invariance와 유사. Multi-resolution analysis (MRA)가 현대 CNN의 이론적 motivation.

**Neural Process / Implicit Neural Representation**  
$f_\theta(t) = \sum_n \xi_n \psi_n(t)$의 neural basis version. SIREN, INR 등이 이 pattern.

**Fractional BM 생성 (Mandelbrot)**  
Hurst $H \neq 1/2$의 BM 확장. Cholesky decomposition이나 circulant embedding → wavelet-based (Daubechies). Financial data, turbulence modeling.

**Brownian Bridge for Sequence Generation**  
두 point 사이의 BM (conditioned). Interpolation between keyframes, diffusion-based in-between.

**Sampling methods for GPs**  
High-dim GP 샘플링에서 Karhunen-Loève decomposition (eigenfunction expansion) = Lévy 구성의 일반화. Computational cost 감소.

---

## ⚖️ 가정과 한계

**한계 — 유한 level 근사**  
실전 시뮬에서 finite $n_{\max}$. Truncation error $O(n_{\max}^{1/2} 2^{-n_{\max}/2})$.

**한계 — Lévy는 $[0, 1]$ domain**  
$[0, T]$로 확장하려면 multiple intervals join 또는 scaling.

**대안 시뮬레이션**:
1. **Standard increments** (simplest): $\sqrt{dt}$ Gaussian
2. **Brownian bridge**: 보간 기반
3. **Spectral**: FFT로 Gaussian field 생성 (효율)
4. **Lévy's**: 계층적, multi-resolution

---

## 📌 핵심 정리

| 결과 | 요약 |
|---|---|
| Haar basis | $\{\phi_0\} \cup \{h_{n, k}\}$ ONB of $L^2[0,1]$ |
| Schauder | $\Phi_{n, k} = \int h_{n, k}$, tent functions |
| Lévy 구성 | $B_t = \xi_0 t + \sum \xi_{n,k} \Phi_{n, k}(t)$ |
| 공분산 | Parseval → $\min(s, t)$ |
| 연속성 | Uniform convergence + Borel-Cantelli |
| 독립증분 | Gaussian uncorrelated → indep |

**한 줄 요약**: BM은 Haar orthonormal basis + iid Gaussian 계수의 random series로 explicit하게 구성. 이 **Lévy construction**이 존재성을 보증하고 wavelet-based simulation의 기반 제공.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. Schauder 함수 $\Phi_{n, k}$의 최대값과 $L^2$-norm을 계산하라.

<details>
<summary>해설</summary>

$\Phi_{n, k}$는 $[k2^{-n}, (k+1)2^{-n}]$ 위의 tent:
- 상승: $[k2^{-n}, (k+1/2)2^{-n}]$, slope $2^{n/2}$
- 하강: $[(k+1/2)2^{-n}, (k+1)2^{-n}]$, slope $-2^{n/2}$

**Max**: Peak at $t = (k+1/2)2^{-n}$. Height = $2^{n/2} \cdot 2^{-n-1} = 2^{-n/2 - 1}$.

**$L^2$-norm**: $\|\Phi_{n,k}\|^2 = \int \Phi^2$. Tent area under square = $\frac{1}{3} \cdot \text{base} \cdot \text{height}^2 = \frac{1}{3} \cdot 2^{-n} \cdot 2^{-n-2} = \frac{1}{3} \cdot 2^{-2n-2} / ... $

더 정확히: $\Phi(t)$ linear, $\int \Phi^2 dt = 2 \int_a^m (2^{n/2}(t-a))^2 dt = 2 \cdot 2^n \cdot \frac{1}{3}(m-a)^3 = \frac{2}{3} 2^n \cdot 2^{-3n-3} = \frac{1}{12} 2^{-2n}$.

즉 $\|\Phi_{n, k}\|_2 = 2^{-n}/\sqrt{12}$.

(Parseval 계산에서 이 값이 중요.)

</details>

**문제 2 (심화)**. Lévy's 구성에서 uniform convergence를 증명하는 데 왜 **$\sum n^{1/2} 2^{-n/2} < \infty$**가 결정적인가?

<details>
<summary>해설</summary>

**Key step**: $\sup_t |Z_n(t)| \leq C n^{1/2} 2^{-n/2}$ a.s.로 얻은 후, series $\sum Z_n$의 uniform 수렴을 보이려면 Weierstrass M-test.

**M-test 요구**: $\sum_n M_n < \infty$ where $M_n = \sup_t |Z_n(t)|$. 

여기서 $M_n \leq C n^{1/2} 2^{-n/2}$. $\sum n^{1/2} 2^{-n/2}$의 수렴:
- Ratio test: $\frac{(n+1)^{1/2} 2^{-(n+1)/2}}{n^{1/2} 2^{-n/2}} = \sqrt{\frac{n+1}{n}} \cdot 2^{-1/2} \to 2^{-1/2} < 1$ → 수렴.

**대안 Root test**: $\limsup M_n^{1/n} = 2^{-1/2} < 1$ → 수렴.

**만약 Gaussian 계수가 아니었다면**: $\max_k |\xi_{n,k}|$ tail이 heavier (e.g., Cauchy) → $M_n$ growth rate 더 빠름 → 수렴 실패 가능성.

**의미**: Gaussian의 **exponential tail** decay가 $\sqrt{n}$의 polynomial growth를 dominate하여 $2^{-n/2}$ decay가 살아남음. 이것이 BM의 연속 경로의 수학적 근원.

**연결**: Kolmogorov continuity에서도 동일 원리 — 4차 moment $\mathbb{E}|B_t - B_s|^4 = O(|t-s|^2)$가 **연속 modification** 보장.

</details>

**문제 3 (AI 연결)**. Wavelet-based Karhunen-Loève expansion이 Gaussian process regression 계산 복잡도를 어떻게 개선하는가?

<details>
<summary>해설</summary>

**Standard GP regression 복잡도**:
- Covariance matrix $K \in \mathbb{R}^{N \times N}$ 계산: $O(N^2)$
- Inverse / Cholesky: $O(N^3)$
- Prediction: $O(N^2)$ per test point

$N = 10^4$에서 $O(N^3) = 10^{12}$ → 불가능.

**Karhunen-Loève (KL) expansion**:
Process $X_t = \sum_n \lambda_n^{1/2} \xi_n \phi_n(t)$, $\lambda_n$이 covariance operator 고유값, $\phi_n$이 고유함수.

**BM의 경우** (또는 유사 process): Wavelet-based $\phi_n$ → 빠른 수렴 + sparse representation.

**Sparse GP**:
- $M \ll N$ inducing points → covariance rank reduction to $O(M)$
- **Variational Sparse GP** (Titsias 2009): ELBO minimization, complexity $O(NM^2)$

**Wavelet-specific benefits**:
- Multi-resolution: different scales of variation
- Sparsity: 대부분의 계수가 작음 → truncation에 robust
- Fast Wavelet Transform (FWT): $O(N)$ (not $O(N \log N)$ like FFT)

**실전**:
- Gaussian Process의 wavelet preconditioner
- Deep GP + wavelet basis (Damianou 2013)

**연결**: KL = Lévy construction의 general version. $L^2$ ONB에 iid Gaussian → process 구성. BM은 Haar basis with specific $\lambda$, general GP는 다른 basis. 현대 scalable GP 방법의 이론적 기반.

</details>

---

<div align="center">

◀ [01. 브라운 운동의 공리적 정의](./01-axiomatic-definition.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [03. Random Walk Scaling Limit — Donsker 정리](./03-donsker-theorem.md)

</div>
