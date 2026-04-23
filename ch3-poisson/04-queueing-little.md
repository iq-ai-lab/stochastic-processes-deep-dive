# 04. Queueing 이론 맛보기 — M/M/1, Little의 법칙

## 🎯 핵심 질문

- **M/M/1 큐**의 정상분포 $\pi_n = (1-\rho)\rho^n$ ($\rho = \lambda/\mu$)은 어떻게 유도되는가?
- 언제 큐가 **안정**(stable) 상태에 도달하는가 — $\rho < 1$의 의미는?
- **Little의 법칙** $L = \lambda W$ — "시스템 내 평균 수 = 도착률 × 평균 체류시간"이 왜 거의 모든 큐잉 시스템에서 성립하는가?
- 이 결과들이 LLM inference, 웹 서버, 네트워크 설계에 어떻게 응용되는가?

---

## 🔍 왜 큐잉 이론이 AI에서 중요한가

**LLM Inference Latency**: GPU 서비스에 요청이 Poisson으로 도착, 처리 시간 분포 → M/M/1 또는 일반 G/G/c 모델. Little의 법칙으로 "동시 처리 중 요청 수 = 도착률 × 평균 latency".

**Batching in Deep Learning**: Dynamic batching에서 "현재 대기열 길이 vs batch 크기 decision". 큐잉 이론이 optimal scheduling의 이론적 기반.

**Token throughput**: Transformer serving에서 token/sec throughput과 per-request latency의 trade-off. Universal scalability law가 큐잉 식의 확장.

**Microservice architecture**: ML serving pipeline (preprocessing → inference → postprocessing)의 end-to-end latency 분석에 M/M/c 네트워크 이론.

---

## 📐 수학적 선행 조건

- [Ch3-01 ~ Ch3-02](./01-three-equivalent-definitions.md): Poisson 과정, Exp 메모리리스
- [Ch2-03, Ch2-05](../ch2-discrete-markov/03-stationary-distribution.md): 정상분포, detailed balance
- 기본 확률: Geometric 분포, 가산상태 재귀

---

## 📖 직관적 이해

### M/M/1의 의미

**M/M/1** = Memoryless arrivals / Memoryless service / 1 server.
- 도착: Poisson rate $\lambda$
- 서비스 시간: iid $\text{Exp}(\mu)$
- 서버 1대, 무한 대기열

**상태** = 시스템 내 고객 수 (대기 + 서비스 중) $\in \{0, 1, 2, \ldots\}$. 연속시간 Markov chain.

### Utilization $\rho$

$\rho = \lambda / \mu$ = **서버 사용률**.
- $\rho < 1$: 안정 — 대기열 유한 (정상분포 존재)
- $\rho = 1$: 임계 — 영재귀
- $\rho > 1$: 불안정 — 대기열 무한 증가

$\rho \to 1$일 때 대기열이 폭발적으로 증가 — **congestion**의 수학적 근거.

### 정상분포 $\pi_n = (1-\rho)\rho^n$

**Geometric 분포**. 평균 $L = \rho/(1-\rho)$. $\rho = 0.9$면 $L = 9$ — "90% 사용률이면 평균 9명 대기".

$\rho \to 1$에서 $L \to \infty$. **미세한 추가 부하가 latency를 폭발**시킴.

### Little의 법칙

$$L = \lambda W.$$
- $L$ = 평균 시스템 내 고객 수
- $\lambda$ = 평균 도착률  
- $W$ = 평균 체류 시간 (대기 + 서비스)

**범용성**: 이 식은 M/M/1뿐 아니라 **거의 모든 큐잉 시스템**에서 성립 (정상성 가정만 필요, Poisson 가정 불필요).

---

## ✏️ 엄밀한 정의

### 정의 4.1 — M/M/1 큐

$\{X_t\}_{t \geq 0}$ on $E = \{0, 1, 2, \ldots\}$:
- 도착 Poisson rate $\lambda$
- 서비스 Exp rate $\mu$ (상태 $\geq 1$일 때)
- $X_t$ 증가 rate $\lambda$ (도착), 감소 rate $\mu$ (서비스 완료, if $X_t \geq 1$)

이는 **birth-death process** (Ch4-04): 출생 $\lambda$, 사망 $\mu$ (상수).

### 정의 4.2 — Utilization

$\rho := \lambda / \mu$.

### 정의 4.3 — Little의 정의

정상 상태의 큐잉 시스템에서:
- $L$: 시스템 내 평균 고객 수 (long-run average)
- $\lambda$: 평균 도착률 ($\lim_{T \to \infty} \frac{1}{T} \times$ $T$까지 도착 수)
- $W$: 평균 체류시간 ($\lim_{N \to \infty} \frac{1}{N} \sum_{i=1}^N W_i$, $W_i = $ 고객 $i$의 체류시간)

**Little의 법칙**: $L = \lambda W$.

---

## 🔬 정리와 증명

### 정리 4.1 — M/M/1 정상분포

$\rho = \lambda/\mu < 1$이면 M/M/1 큐의 정상분포는
$$\pi_n = (1 - \rho) \rho^n, \quad n = 0, 1, 2, \ldots$$

*증명 (detailed balance)*. 상태 $n$과 $n+1$ 사이의 flow 균형:
$$\pi_n \cdot \lambda = \pi_{n+1} \cdot \mu.$$
($n$에서 $n+1$로 가는 rate = 출생 rate $\lambda$; $n+1$에서 $n$으로는 서비스 rate $\mu$.)

$\pi_{n+1} = (\lambda/\mu) \pi_n = \rho \pi_n$. 반복: $\pi_n = \rho^n \pi_0$. 정규화 $\sum \pi_n = 1$:
$$\pi_0 \sum_{n \geq 0} \rho^n = \pi_0 \cdot \frac{1}{1 - \rho} = 1 \Rightarrow \pi_0 = 1 - \rho.$$
$\square$

**주의**: $\rho \geq 1$이면 $\sum \rho^n = \infty$ → 정상분포 없음 (불안정).

### 정리 4.2 — 평균 성능 지표

정상 상태 M/M/1 ($\rho < 1$):
$$L = \mathbb{E}[X] = \frac{\rho}{1 - \rho}, \quad W = \frac{1}{\mu - \lambda}, \quad L_q = \frac{\rho^2}{1 - \rho}, \quad W_q = \frac{\rho}{\mu - \lambda}.$$

($L_q$ = 대기열 고객 수 (서비스 중 제외), $W_q$ = 대기시간 (서비스 시간 제외))

*유도*.
$L = \sum_{n \geq 0} n (1-\rho) \rho^n = (1-\rho) \cdot \frac{\rho}{(1-\rho)^2} = \frac{\rho}{1-\rho}$ (geometric series).

$W$: Little의 법칙 $L = \lambda W \Rightarrow W = L/\lambda = \frac{\rho/(1-\rho)}{\lambda} = \frac{1}{\mu - \lambda}$.

### 정리 4.3 (Little's Law — General)

정상 상태의 큐잉 시스템에서 (Poisson 가정 불필요, 서비스 분포 임의) $L, \lambda, W$가 각각 유한하면
$$L = \lambda W.$$

*증명 스케치 (path 관점)*. 시간 $[0, T]$ 동안:
- 시스템 내 총 "고객-시간" = $\int_0^T X_t dt \approx L \cdot T$
- 도착 총 수 = $A(T) \approx \lambda T$
- 각 고객의 체류시간 합 = $\sum_{i} W_i \approx A(T) \cdot W$

**두 계산이 같은 것**: $\int X_t dt = \sum W_i$ (각 고객이 체류 동안 "단위 시간"씩 기여).

$L T = \lambda T \cdot W \Rightarrow L = \lambda W$. $\square$

**놀라운 일반성**: 서비스 규칙 (FCFS, LIFO, priority), 서버 수, 분포 형태 무관. 정상성과 유한 평균만 필요.

### 정리 4.4 — M/M/1 체류시간 분포

정상 상태 M/M/1에서 고객의 체류시간 $W \sim \text{Exp}(\mu - \lambda)$.

*증명 스케치*. FCFS 가정. 도착한 고객이 기존 $n$명과 서비스 완료 대기. 메모리리스로 기존 $n$명 남은 서비스 + 자신 서비스 = $(n+1)$ exp waiting. Total $= \sum_{k=1}^{n+1} \text{Exp}(\mu) | n$. Unconditional mix by $\pi_n = (1-\rho)\rho^n$:

$W$의 Laplace transform = $\mathbb{E}[e^{-sW}] = \sum_n \pi_n (\mu/(\mu+s))^{n+1} = \ldots = (\mu - \lambda)/(\mu - \lambda + s)$.

역변환 → Exp($\mu - \lambda$). $\square$

### 정리 4.5 — PASTA 성질 (Poisson Arrivals See Time Averages)

Poisson 도착 과정에서 "도착 시점의 시스템 분포" = "시간 평균 분포" = 정상분포 $\pi$.

*증명 스케치*. 도착 시점은 Poisson (history와 독립). 정지시각에서 상태 조건부 분포 = 정상분포 (Poisson의 memoryless + 강한 마르코프). $\square$

**실전**: 도착 시점의 평균 대기열 길이 = $L$ (정상분포 평균). 시뮬레이션 검증에 유용.

---

## 💻 NumPy 구현 검증

### 실험 1 — M/M/1 시뮬레이션 및 정상분포 검증

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
lam, mu, T = 0.8, 1.0, 1000.0   # ρ = 0.8
rho = lam / mu

# Event-driven 시뮬레이션
X = 0   # 현재 시스템 내 고객 수
t = 0.0
times = [0.0]; X_path = [0]
next_arrival = rng.exponential(1/lam)
next_service = np.inf

while t < T:
    if next_arrival < next_service:
        t = next_arrival
        X += 1
        if X == 1:
            next_service = t + rng.exponential(1/mu)
        next_arrival = t + rng.exponential(1/lam)
    else:
        t = next_service
        X -= 1
        if X > 0:
            next_service = t + rng.exponential(1/mu)
        else:
            next_service = np.inf
    times.append(t); X_path.append(X)

times = np.array(times[:-1]); X_path = np.array(X_path[:-1])
durations = np.diff(times)

# 시간가중 분포
max_n = int(np.max(X_path))
pi_empirical = np.zeros(max_n + 1)
for n in range(max_n + 1):
    pi_empirical[n] = np.sum(durations[X_path[:-1] == n]) / times[-1]

# 이론
pi_theory = (1 - rho) * rho**np.arange(max_n + 1)

print(f'{"n":>3} {"pi_emp":>10} {"pi_theory":>10}')
for n in range(min(10, max_n + 1)):
    print(f'{n:>3} {pi_empirical[n]:>10.4f} {pi_theory[n]:>10.4f}')

# 평균 L 비교
L_empirical = (X_path[:-1] * durations).sum() / times[-1]
L_theory = rho / (1 - rho)
print(f'\n평균 L: 실측 {L_empirical:.4f}, 이론 {L_theory:.4f}')
```

### 실험 2 — Little의 법칙 검증

```python
# 위 시뮬에서 각 고객의 체류시간 기록
# (간단화: 시뮬레이션을 다시 구성하며 W 기록)
import collections
arrivals = []
departures = []

X = 0; t = 0.0
next_arr = rng.exponential(1/lam)
next_srv = np.inf
queue = collections.deque()

while t < T:
    if next_arr < next_srv:
        t = next_arr
        arrivals.append(t)
        queue.append(t)
        X += 1
        if X == 1:
            next_srv = t + rng.exponential(1/mu)
        next_arr = t + rng.exponential(1/lam)
    else:
        t = next_srv
        arr_time = queue.popleft()
        departures.append((arr_time, t))
        X -= 1
        if X > 0:
            next_srv = t + rng.exponential(1/mu)
        else:
            next_srv = np.inf

W_samples = np.array([d - a for a, d in departures])
W_mean = W_samples.mean()
L_from_Little = lam * W_mean

print(f'W (평균 체류시간): 실측 {W_mean:.4f}, 이론 {1/(mu-lam):.4f}')
print(f'L = λW: 실측 {L_from_Little:.4f}')
print(f'이론 L = ρ/(1-ρ): {rho/(1-rho):.4f}')
# 세 값 모두 일치 → Little의 법칙 검증
```

### 실험 3 — $\rho$ 변화에 따른 $L$의 폭발

```python
rho_vals = np.linspace(0.1, 0.95, 20)
L_theory = rho_vals / (1 - rho_vals)
W_theory = 1 / (1 - rho_vals)   # μ = 1 가정

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(rho_vals, L_theory, 'o-')
plt.xlabel(r'$\rho$'); plt.ylabel('L (평균 고객 수)')
plt.title('Utilization vs Queue Length'); plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.semilogy(rho_vals, W_theory, 'o-')
plt.xlabel(r'$\rho$'); plt.ylabel('W (체류시간, log)')
plt.title('$\\rho \\to 1$에서 latency 폭발'); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()
# → ρ = 0.9면 L = 9, W = 10
#    ρ = 0.95면 L = 19, W = 20 — 급속 폭발
```

---

## 🔗 AI/ML 연결

**LLM Inference Serving**  
GPT-4/Claude API 요청 → Poisson arrival 근사. 서버 GPU 처리 → Exp(또는 일반) 서비스. **Dynamic batching**의 최적 batch 크기는 현재 큐 길이의 함수. vLLM, TensorRT-LLM이 이 이론을 적용.

**Latency vs Throughput Trade-off**  
Throughput ↑ ($\lambda$ ↑) → $\rho$ ↑ → latency $W$ ↑. 실전 operation에서 $\rho \leq 0.7$을 유지하여 latency 제어. Auto-scaling은 $\rho$ 감시 + 서버 추가.

**Network Queue in Distributed Training**  
Allreduce operation에서 각 GPU의 gradient 전송이 network queue에 들어감. $M/M/c$ 모델로 분석 가능. RDMA, NCCL 최적화가 이 이론 활용.

**Dataflow Pipeline Optimization**  
Preprocessing → Training → Validation pipeline의 각 stage를 큐로 모델링. Little의 법칙으로 end-to-end latency 분석. TensorFlow's `tf.data.Dataset` autotuning이 큐 balance 조정.

**Real-time ML Serving**  
TF Serving, TorchServe의 request scheduler. Priority queue (Type-of-Service) ⇒ M/M/c/k 등 변형. Fail rate, SLA 보장을 위한 capacity planning.

---

## ⚖️ 가정과 한계

**가정 — Poisson 도착**  
현실: burst arrivals (휴일 트래픽, 플래시 세일)이 Poisson 깨짐. **M/G/1**(general service), **GI/GI/1**(general arrivals) 일반화.

**가정 — Exp 서비스**  
현실: LLM inference는 prompt length에 비례(deterministic component), output generation까지 추가 (random). **Pollaczek-Khinchine** 공식이 M/G/1 시스템의 평균 대기시간 제공.

**한계 — Infinite buffer**  
Real 시스템은 유한 buffer (rejected requests). **M/M/1/K** 모델 (K=buffer size). Block rate 계산.

**한계 — $\rho \geq 1$**  
실전에서 $\rho > 0.9$이면 variance 매우 큼 — 평균값만으로는 risk 과소평가. Percentile (p99 latency) 중요. **Heavy-tailed service time**에서 특히 심각.

---

## 📌 핵심 정리

| 결과 | 수식 |
|---|---|
| M/M/1 정상분포 | $\pi_n = (1-\rho)\rho^n$, $\rho = \lambda/\mu$ |
| 안정 조건 | $\rho < 1$ |
| 평균 고객 수 | $L = \rho/(1-\rho)$ |
| 평균 체류시간 | $W = 1/(\mu - \lambda)$ |
| 체류시간 분포 | $\text{Exp}(\mu - \lambda)$ |
| Little의 법칙 | $L = \lambda W$ (매우 일반적) |
| PASTA | Poisson arrivals see time averages |

**한 줄 요약**: M/M/1은 $\rho = \lambda/\mu < 1$일 때 안정 상태로 수렴, 정상분포가 geometric. Little의 법칙 $L = \lambda W$는 거의 모든 큐잉 시스템에 적용되는 보편적 결과로, LLM inference·web serving의 capacity planning의 기반.

---

## 🤔 생각해볼 문제 (+ 해설)

**문제 1 (기초)**. LLM 서비스: 요청 rate $\lambda = 0.8$/sec, 서비스 rate $\mu = 1$/sec. 평균 대기열 길이와 평균 응답시간은?

<details>
<summary>해설</summary>

$\rho = 0.8/1 = 0.8$.

$L = \rho/(1-\rho) = 0.8/0.2 = 4$명.  
$W = 1/(\mu - \lambda) = 1/0.2 = 5$초.  
$W_q = L_q/\lambda$, $L_q = \rho^2/(1-\rho) = 0.64/0.2 = 3.2$ → $W_q = 4$초 (대기시간).

**해석**: 평균 4명이 시스템에, 새 요청은 평균 5초 대기 후 응답 받음. 서비스 시간 1초 + 대기 4초.

**Little 검증**: $L = \lambda W = 0.8 \times 5 = 4$ ✓.

**Capacity planning**: $\rho$를 0.7로 낮추려면 $\mu = 0.8/0.7 = 1.14$/sec로 서비스 향상 (GPU 추가 등). 그러면 $W = 1/(1.14 - 0.8) = 2.94$초 — 40% 개선.

</details>

**문제 2 (심화)**. M/M/c 큐(c개 서버): 정상분포와 평균 대기시간 공식을 유도하라 (Erlang-C).

<details>
<summary>해설</summary>

**Birth-death 구조**: 상태 $n$에서 $n+1$로 rate $\lambda$, $n+1$에서 $n$으로 rate $\min(n+1, c) \mu$ (서버 바쁨 수).

**Detailed balance**:
- $n < c$: $\pi_n \lambda = \pi_{n+1} (n+1) \mu$ → $\pi_{n+1} = \pi_n \lambda/((n+1)\mu) = \pi_0 (c\rho)^{n+1}/(n+1)!$, $\rho = \lambda/(c\mu)$.
- $n \geq c$: $\pi_n \lambda = \pi_{n+1} c \mu$ → $\pi_{n+1} = \pi_n \rho$.

**정규화**: 
$$\pi_0 = \left[\sum_{n=0}^{c-1} \frac{(c\rho)^n}{n!} + \frac{(c\rho)^c}{c!(1-\rho)}\right]^{-1}.$$

**Erlang-C** (모든 서버 바쁨 확률):
$$C(c, c\rho) = \frac{(c\rho)^c / (c!(1-\rho))}{\sum_{n=0}^{c-1} (c\rho)^n/n! + (c\rho)^c/(c!(1-\rho))}.$$

**평균 대기시간** (응답 지연):
$W_q = C(c, c\rho) / (c\mu(1-\rho))$.

**응용**: 콜센터 에이전트 수 결정, LLM server GPU 풀 크기 결정. $c$ 증가 → $\rho$ 감소 → 급격히 $W_q$ 개선.

</details>

**문제 3 (AI 연결)**. LLM API의 P99 latency SLA를 만족시키려 할 때, 단순 "평균 $W$" 만으로는 부족한 이유와 실전 capacity planning 전략을 논하라.

<details>
<summary>해설</summary>

**평균만으로 부족한 이유**:

1. **Heavy tail**: M/M/1에서 $W \sim \text{Exp}(\mu - \lambda)$. P99 = $-\log(0.01)/(\mu - \lambda) = 4.6/(\mu - \lambda)$ — 평균의 4.6배.

2. **$\rho$가 높을수록 분산 증폭**: $\text{Var}(W) = 1/(\mu - \lambda)^2$. $\rho \to 1$에서 평균과 표준편차 모두 발산.

3. **실제 서비스 heterogeneity**: LLM 요청은 token length 크게 다름 → service time의 CV(coefficient of variation) 큼. P99이 훨씬 길어짐. M/G/1에서 P-K 공식: $W_q = \rho \mathbb{E}[S^2]/(2(1-\rho)\mathbb{E}[S])$, service time 분산이 2차 모멘트로 들어옴.

**실전 Capacity Planning 전략**:

1. **낮은 $\rho$ 유지**: 평균 $\rho \leq 0.6$을 목표. Tail latency를 안전 margin 하에.

2. **Auto-scaling**:
   - 현재 $\rho$ 모니터링 → 서버 추가/제거
   - Reactive (load 기반) + Predictive (trend 기반)

3. **Request prioritization**:
   - High-priority (paid tier) → 별도 큐 with higher $c$
   - Low-priority → shared pool with rate limits
   - **Weighted Fair Queueing** (WFQ) 구현

4. **Timeout + Retry**:
   - 예측 P99 × 1.5를 timeout으로
   - 실패 시 exponential backoff retry
   - Client-side caching for idempotent requests

5. **Batching optimization**:
   - Dynamic batch size를 큐 깊이 함수로
   - Continuous batching (vLLM) — 이미 서비스 중인 batch에 추가 요청 병합
   - Trade-off: batch 크기 ↑ → throughput ↑, per-request latency는 ↑

6. **Request admission control**:
   - 피크 시 어떤 요청은 거절 (503 Service Unavailable)
   - **Token bucket** / **Leaky bucket**으로 arrival shaping

**수학적 도구**:
- Chernoff bound로 P99 upper bound
- Percentile-based SLA → 구체적 $\mu, c$ 결정
- Queueing network 분석 (pipeline 단계별 독립 큐)

**현대 AI serving 시스템**:
- vLLM: continuous batching + paged attention
- TensorRT-LLM: kernel fusion + in-flight batching
- Triton Inference Server: dynamic batching + priority queue

**종합**: 큐잉 이론이 "평균 latency"를 분석하는 것을 넘어 "tail latency"로 확장 → Deep Learning 시스템 운영의 실전 과제.

</details>

---

<div align="center">

◀ [03. 복합 Poisson 과정](./03-compound-poisson.md) &nbsp;·&nbsp; [📚 README](../README.md) &nbsp;·&nbsp; 다음 ▶ [Ch4-01. 생성기(Generator)와 Q-matrix](../ch4-continuous-markov/01-generator-q-matrix.md)

</div>
