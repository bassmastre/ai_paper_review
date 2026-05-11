# TRN001 - LoRA: Low-Rank Adaptation of Large Language Models

> Metadata: see `../../index.md`

### Summary
- LoRA는 full fine-tuning에서 전체 파라미터 변화량 `ΔW`를 직접 학습하는 대신, 이를 저랭크 행렬분해 형태인 `ΔW = BA`로 제한하여 훨씬 적은 수의 파라미터만 학습하는 PEFT 방법이다.
- 기존 pretrained weight `W₀`는 freeze하고, Transformer 내부의 특정 linear projection layer에 LoRA branch를 붙여 `W = W₀ + BA`처럼 동작하게 만든다.
- 논문에서는 주로 Transformer attention layer의 projection matrix, 특히 `Wq`, `Wv` 등에 LoRA를 적용하여 full fine-tuning에 가까운 성능을 훨씬 적은 학습 파라미터와 저장 비용으로 달성할 수 있음을 보였다.

### Key Idea
- 핵심 가정은 downstream task adaptation에 필요한 weight update `ΔW`가 full-rank일 필요가 없고, 낮은 intrinsic rank를 가진다는 것이다.
- 즉, 모델 전체를 모든 방향으로 바꾸는 것이 아니라, task-specific 변화가 일어나는 제한된 저차원 부분공간만 학습해도 충분하다는 관점이다.
- LoRA는 기존 layer를 대체하지 않고, 기존 출력 `W₀x`에 작은 보정값 `BAx`를 더한다.
- `A ∈ R^{r × d_in}`, `B ∈ R^{d_out × r}`이며, `r`은 사용자가 정하는 LoRA rank 하이퍼파라미터다.
- `d_in`, `d_out`은 LoRA에서 임의로 정하는 값이 아니라, LoRA를 붙이는 기존 Transformer linear layer의 입력/출력 차원이다.
- 학습 시에는 `W₀`는 고정하고 `A`, `B`만 loss와 gradient를 통해 업데이트한다.
- 초기화는 보통 `A`는 random 또는 Gaussian/Kaiming 계열로 초기화하고, `B`는 0으로 초기화하여 학습 시작 시점에는 `BA = 0`, 즉 원래 pretrained model과 같은 출력에서 출발하게 한다.

### Contribution
- Full fine-tuning의 가장 큰 문제인 태스크별 전체 파라미터 저장 및 학습 비용을 크게 줄였다.
- 기존 adapter 방식과 달리 inference 시 `W₀ + BA`로 merge할 수 있어 추가 latency를 거의 만들지 않는다.
- 작은 rank `r`만으로도 여러 NLP task와 Transformer 계열 모델에서 full fine-tuning에 가까운 성능을 낼 수 있음을 광범위한 실험으로 보였다.
- LoRA는 단순한 행렬분해 테크닉을 넘어서, LLM의 downstream adaptation이 저차원 변화만으로 충분할 수 있다는 관점을 제시했다.
- 태스크별 adapter만 저장하고 교체할 수 있어, 하나의 base model에 여러 downstream task-specific delta를 붙여 관리할 수 있다.
- 데이터가 적을 때도 상대적으로 안정적인 경향이 있는데, 이는 LoRA의 직접 목적이라기보다 low-rank constraint와 frozen base model에서 오는 regularization 효과에 가깝다.

### Note
- LoRA의 본질은 “새로운 지식을 넣는 구조”라기보다, 기존 pretrained model의 행동을 작은 보정값으로 조정하는 방식이다.
- RAG가 외부 문서를 context로 넣어 지식을 보강하는 방식이라면, LoRA는 모델의 답변 습관, 포맷, 태스크 수행 패턴을 파라미터 차원에서 조정하는 방식이다.
- `ΔW = A`처럼 하나의 행렬만 학습하면 결국 기존 `W₀`와 같은 shape이어야 하므로 파라미터 절감 효과가 없다. `ΔW = BA`로 두어야 작은 두 행렬의 곱으로 기존 weight와 같은 shape의 update를 만들 수 있다.
- `A`, `B`를 encoder-decoder처럼 볼 수도 있지만, 엄밀히는 의미 표현을 압축/복원하려는 구조가 아니라 weight update를 저랭크로 제한하기 위한 re-parameterization이다.
- 중간에 `C`를 넣어 `ΔW = BCA`처럼 만드는 것도 가능하지만, 모두 선형이고 전부 학습한다면 `BA` 형태로 흡수될 수 있어 기본 LoRA에서는 큰 의미가 없다.
- 다만 multimodal setting에서는 vision/audio/sensor representation과 language representation을 정렬해야 하므로, projector, latent bridge, query encoder 같은 중간 구조가 자연스러워진다.
- LoRA의 기본 목적은 data efficiency가 아니라 parameter efficiency, memory efficiency, deployment efficiency다.
- 개인 환경에서는 논문 수준의 대규모 실험 재현보다는, Hugging Face `transformers` + `peft` + 필요 시 `bitsandbytes` 기반 QLoRA로 작은 모델에 적용해보는 것이 현실적이다.