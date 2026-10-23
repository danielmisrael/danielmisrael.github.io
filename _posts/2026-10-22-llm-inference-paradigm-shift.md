---
title: "The Structure of LLM Inference Revolutions"
date: 2026-10-22
permalink: /posts/2026/10/llm-inference-paradigm-shift/
tags:
  - machine learning
  - language models
  - parallel decoding
---

**TL;DR:** *LLM inference is undergoing a paradigm shift from batch parallelism to sequence parallelism.*

<aside class="notice" aria-label="Author's Note" markdown="1">
**Author's Note:** I do not have insider knowledge of industry inference stacks. My opinions may be completely wrong or patently obvious.
</aside>

---

In *The Structure of Scientific Revolutions*, Thomas Kuhn argued that science does not advance via a mere accumulation of discoveries.[^kuhn] Researchers work within a shared set of assumptions about what the important problems are and how to solve them: a paradigm. I believe we are in **a paradigm shift in LLM inference**. We are moving from **batch parallel decoding to sequence parallel decoding**.

<figure>
  <img src="/images/llm-inference-paradigm-shift/figure-01.gif" alt="Figure 1 — Across users or along a sequence">
  <figcaption markdown="1">
Figure 1. Batch parallelism parallelizes over the number of users. Sequence parallelism can generate tokens in parallel for a single user.
  </figcaption>
</figure>

Kuhn's account of scientific progress can be organized into five phases.[^kuhn-phases]

1. **Pre-paradigm.** Competing approaches have yet to settle into a shared framework.

2. **Normal science.** Researchers solve problems within the prevailing paradigm.

3. **Anomalies.** Results put the paradigm's assumptions under pressure.

4. **Paradigm shift.** Researchers reexamine those assumptions and develop a new framework.

5. **Post-revolution.** The new framework becomes the basis for normal science.

## Pre-paradigm

LLM serving inherited infrastructure built for general-purpose deep learning. Production systems such as TensorFlow Serving[^tensorflow-serving] and NVIDIA Triton Inference Server[^triton] handled model deployment, request batching, and efficient execution across machine-learning frameworks. Batching requests in this time was already common practice. However, the advent of autoregressive LLM inference posed new challenges. One request requires repeated model executions, retains a growing state (the KV cache), and finishes after a variable number of tokens.

It became clear that autoregressive decoding could be **memory-bandwidth-bound**. Each step reads model weights and the KV cache to generate only one new token per request. Pope and colleagues[^pope] explain how this bottleneck changes depending on the workload. At small batches and short contexts, loading weights dominates memory traffic. Batching spreads the weight-loading cost across more requests, but each request still has its own KV cache to read. As batches and contexts grow, those cache reads can become the dominant memory cost.

## Normal science

The response to the weight-loading bottleneck was **batch parallelism**. In this blog, I will examine the following paradigmatic assumption: **batch parallelism is the best way to maximize throughput in an LLM serving system**. Under Kuhn’s framework, what then proceeds in this paradigm is normal science. Several natural questions start to emerge: How do we keep the batch full? How do we fit more requests in memory? How do we keep other work from delaying decoding?

Batching amortizes weight reads across tokens from independent requests. A batch of B sequences allows a dense linear layer to perform roughly B times as much computation per weight load, turning matrix–vector computation into denser matrix–matrix computation. This raises arithmetic intensity and accelerator utilization. In the context of LLM serving, the systems problem is about maintaining a sufficiently large active batch despite having variable request lengths and memory occupied the KV cache of each request.

Orca[^orca] introduced iteration-level scheduling so completed requests could leave and new requests could join between decoding steps. vLLM's PagedAttention[^vllm] reduced KV-cache fragmentation, allowing more requests to fit in memory. SGLang[^sglang] developed KV-cache reuse across generation calls, avoiding redundant computation and storage for shared prefixes.

These advances made it easier to sustain large batches and use their memory efficiently. Kwon and colleagues state the premise in the opening of the vLLM paper[^vllm].

> High throughput serving of large language models (LLMs) requires batching sufficiently many requests at a time. However, existing systems struggle because the key-value cache (KV cache) memory for each request is huge and grows and shrinks dynamically. When managed inefficiently, this memory can be significantly wasted by fragmentation and redundant duplication, limiting the batch size.

In LLM inference (and basically all systems), the goal is to manage two key metrics: (1) **System throughput**, which measures how many tokens the whole server produces per second (2) **Interactivity**, which measures how quickly each user receives tokens. A larger batch often improves (1) at the cost of (2).

<figure>
  <img src="/images/llm-inference-paradigm-shift/figure-02.png" alt="Figure 2 — Four autoregressive curves on shared axes">
  <figcaption markdown="1">
Figure 2. Four autoregressive serving curves: Qwen3.5 (397B) and GLM-5.1 (754B), each on NVIDIA H200 and B300 GPUs using FP8 and 1k input / 1k output lengths. Data from public SemiAnalysis InferenceX comparison pages[^inferencex], collected October 21, 2026. Throughput includes input and output tokens per GPU; interactivity measures output tokens per second per user.
  </figcaption>
</figure>

Once a system Pareto dominates another system on the throughput-interactivity tradeoff, it becomes very hard to ignore. In fact, if you hold economic cost and token quality constant, it is completely irrational not to switch to the strictly more efficient system. Note that Pareto dominate means always better in both metrics, so if a system can hold onto even a single non-dominated point, there can still be doubt about switching.

## Anomalies

Under a scientific paradigm, a series of anomalies can accumulate. For example, Newtonian predictions for Mercury's orbital precession fell 43 arcseconds per century short of observations. It was hypothesized at the time that a hidden planet named Vulcan existed closer to the Sun and was pulling on Mercury, but decades of searching turned up nothing.[^mercury-vulcan] In an analogous manner, several anomalies call the batch parallel paradigm into question, but in each case it is always possible to explain it away.&#x20;

Sequence parallel decoding was first introduced by speculative decoding[^speculative-decoding]. A small draft model proposes a continuation and a target model verifies the proposed tokens in parallel. This did not disrupt the batching paradigm, because it was generally believed to only be helpful in high-interactivity, low batch size setting. For example, SmartSpec[^smartspec] found that speculation could make interacticity worse under higher request rates or low acceptance. Su and collaborators[^su-batching] found that longer speculation block size could slow decoding at larger batches. SPEED-Bench[^speed-bench] explained that as batching moves execution toward a compute-bound regime, the spare capacity that speculative decoding can exploits diminishes. As summarized by the MagicDec abstract[^magicdec] with regard to speculative decoding, “the conventional wisdom suggests that its efficacy is limited to small batch sizes.”

However, MagicDec was an early counterexample to the batch parallel paradigm. Its experiments showed that speculation could become *more* effective as the batch grew. Figure 3 holds the model, hardware, and drafting settings fixed and changes the context length. At 4k token context, larger batches reduce the speculative decoding speedup, but at 32k, the larger batches **increase** speedup. However, one can be dismissive because it only applies to long-context scenarios.

<figure>
  <img src="/images/llm-inference-paradigm-shift/figure-03.png" alt="At 4k context MagicDec speedup falls with batch size; at 32k context it rises">
  <figcaption markdown="1">
Figure 3. Short context and long context produce opposite batch-size trends. Measurements from MagicDec, Table 3(c)[^magicdec], using Llama-3.1-8B on eight A100s with eight-way tensor parallelism, self-speculation, two draft tokens, and a fixed draft KV budget of 512. The plotted batch sizes are available at both context lengths. Each speedup is relative to autoregressive decoding at the same batch and context.
  </figcaption>
</figure>

These results are not immediately intuitive. It should be easier to parallelize over **independent** users compared to highly interdependent tokens within a sequence. The reason these results are possible is because batch parallel decoding grows the KV cache in the number of users, and the KV cache can eventually become the bottleneck on throughput. Sequence parallel decoding makes it possible to increase throughput without adding more KV pressure. Put another way, speculative decoding can increase high-concurrency throughput by amortizing the KV cache footprint across multiple accepted tokens per step.

Figure 4 compares the number of tokens MagicDec generates per verification with the number needed to match autoregressive decoding. As context grows, the tokens per verification cycle stays roughly constant at 2.5, while the acceptance rate needed to beat pure autoregression falls. For this particular model, the interpolated crossover point where sequence parallelism achieves better throughput is just over 16K context length.

<figure>
  <img src="/images/llm-inference-paradigm-shift/figure-04.png" alt="Measured output per cycle and output needed to match autoregressive decoding, with a dotted line at their crossover">
  <figcaption markdown="1">
Figure 4. Measured output per cycle and the output needed to match AR, using MagicDec, Table 3(c)[^magicdec].
  </figcaption>
</figure>

MoESD[^moesd] found another example of sequence parallelism succeeding in sparse mixture-of-experts models. Each token uses only a few experts, so a batch can have tokens that activate many experts. At a moderate batch size, a decoding step may already require reading nearly all the expert weights, even though each expert processes only a few tokens. Verifying several proposed tokens per request increases arithmetic intensity with little additional weight-reading cost. At sufficiently high acceptance rate, this increases throughput. At larger batches, computation eventually becomes the bottleneck and speculative decoding achieves diminishing returns.

<figure>
  <img src="/images/llm-inference-paradigm-shift/figure-05.png" alt="Figure 5 — MoESD measured speedup by batch size">
  <figcaption markdown="1">
Figure 5. Speculation decoding's advantage as batch size varies for two sparse models: Qwen2 and Mixtral-8×7B. Measured points recovered from MoESD, Figure 2(a,c)[^moesd].
  </figcaption>
</figure>

The EAGLE family improved speculation by giving the drafter access to the target model's internal features. The original EAGLE[^eagle] predicts the target's next hidden feature vector from its final-layer features and the sampled token, then uses the target's output layer to turn that prediction into token probabilities. EAGLE-3[^eagle3] combines features from early, middle, and late layers, and trains on its own previous draft outputs so it can continue drafting accurately after the first step. In the experiment below, EAGLE-3 obtains a strong advantage over autoregression in even high concurrency settings.

<figure>
  <img src="/images/llm-inference-paradigm-shift/figure-06.png" alt="EAGLE and EAGLE-3 throughput gains in the same vLLM batch sweep">
  <figcaption markdown="1">
Figure 6. Throughput relative to AR without speculation, for Llama-3.1-8B on MT-Bench given by Table 5[^eagle3].
  </figcaption>
</figure>

While speculative decoding was producing these anomolies, an independent line of research was developing another way to parallelize over the sequence. A key advance was Austin and colleagues' work on discrete diffusion in 2021[^d3pm], which extended diffusion to discrete data. A model learns to reverse a noise corruption process, allowing it to generate text by updating multiple positions in parallel over successive denoising steps. MDLM in 2024[^mdlm] further narrowed the gap with autoregression, and LLaDA in 2025[^llada] scaled the approach to an 8B model with instruction-following capabilities. Thus, sequence parallel inference was of growing interest to the research community.

<figure>
  <img src="/images/llm-inference-paradigm-shift/figure-07.png" alt="Figure 7 — Cumulative citations to D3PM, SEDD, MDLM, and LLaDA on shared axes">
  <figcaption markdown="1">
Figure 7. Cumulative citing papers for D3PM[^d3pm], SEDD[^sedd], MDLM[^mdlm], and LLaDA[^llada], using the MLsanity citation index[^mlsanity] retrieved on October 22, 2026.
  </figcaption>
</figure>

While diffusion models presented opportunities for high interactivity, early open models often struggled to turn parallel generation into higher throughput.[^fast-dllm] Furthermore, they are trained completely differently from autoregressive models, and it still is not clear whether diffusion models can usurp autoregressive models in terms of output quality, so from the standpoint of LLM inference paradigms, they could still be dismissed.

## Paradigm shift

This is the phase I believe we are currently in. In my opinion, speculative diffusion (using a diffusion drafter in speculative decoding) produces results that can no longer be dismissed.

I believe diffusion drafters are fundamentally better than autoregressive drafters because they can achieve higher acceptance rates in a fixed amount of drafting time. Consider a larger or deeper draft model that can make better proposals. An autoregressive drafter uses the full depth of the model for every token it proposes. This creates pressure to keep the drafter shallow even when more capacity could improve acceptance rate. A diffusion drafter can instead do parallel computation on many proposed positions simultaneously, allowing a larger drafter to fit within the same latency budget. **Parallel drafting makes it possible to scale the drafter, which can increase the number of accepted tokens per verification.**

With this in mind, DFlash[^dflash] proposes a block of tokens using diffusion and gives every draft layer access to the target model's internal features, similar to EAGLE. Unlike EAGLE, they show that diffusion drafting latency can remain constant in the number of tokens drafted.

<figure>
  <img src="/images/llm-inference-paradigm-shift/figure-08.png" alt="Autoregressive drafting latency rises with draft length while DFlash latency remains nearly flat">
  <figcaption markdown="1">
Figure 8. Drafting latency versus proposed token count, redrawn from LMSYS's DFlash and Spec V2 figure[^dflash-sglang].
  </figcaption>
</figure>

Uno[^uno] builds upon their results, using the base model's weights with a diffusion adaptation for drafting and sharing the KV cache with verification. We can examine the corresponding throughput–interactivity frontier.

<figure>
  <img src="/images/llm-inference-paradigm-shift/figure-09.png" alt="Uno, DFlash, EAGLE-3, and autoregressive throughput versus interactivity without directional arrows">
  <figcaption markdown="1">
Figure 9. Qwen3-8B on one H200, with 1,024 input tokens and 8,192 output tokens, at concurrency 1–64. Data from Uno, Table 18[^uno].
  </figcaption>
</figure>

As I stated previously, if one method dominates another on throughput-interactivity, it will be impossible to argue not to change to that method. A fair critique is that the above results are only on an 8B model. Scaling the model parameters and using state of the art architectures perhaps increases compute intensity faster than it increases KV cache pressure. If so, batch parallelism would be favored in those regimes. Fortunately, SemiAnalysis's InferenceX provides comparisons of large open source models with speculative decoding turned on and off on various hardware configurations. Note that these include several speculative methods, not just diffusion drafting.

<figure>
  <img src="/images/llm-inference-paradigm-shift/figure-10.png" alt="Figure 10 — Arrows from speculation off to on, beside matched throughput gains by model">
  <figcaption markdown="1">
Figure 10. Left: arrows connect each of the 129 matched pairs from speculation off (open circle) to on (arrow tip), at fixed interactivity. Right: their paired throughput ratios at matched interactivity, grouped by model. Both vertical axes are logarithmic. Data: public InferenceX comparison pages[^inferencex], collected October 21, 2026.
  </figcaption>
</figure>

Thus, even for large models there is substantial evidence that sequence parallelism can improve throughput over batch parallelism alone. As diffusion drafters gain popularity and improve over time, I anticipate eventually that the highest throughput configuration at any level of interactivity will include some form sequence parallelism through speculative decoding.

Another work worth mentioning is DSpark[^dspark]. DSpark obtains even more improvement by using a special decoding head to improve causal consistency of diffusion drafts. Slight tangent, but I must shout out my labmates for exploring this idea before DSpark in *Breaking the Factorization Barrier in Diffusion Language Models*.[^codd] Anyway, DeepSeek deploys speculative diffusion in its V4 serving system[^dspark], replacing its previous single-token MTP setup. Under live user traffic, it improved the measured throughput–interactivity frontier.

## Post-revolution

I would consider the revolution complete when all major inference providers routinely use sequence-level parallelism to achieve both higher throughput and interactivity. At that point, it becomes ingrained as an assumption of LLM serving design, and the field returns to normal science within the new paradigm.

There will still be plenty of scientific questions in the new paradigm. For example, I personally do not think speculative decoding is very clean. There is a drafter that spends some amount of compute and a verifier that spends another amount. Why should there be exactly two levels? Why not a hierarchy of adaptive computation. There ought to be a different amount of computation allocated according to the difficulty of the prediction. The right abstraction might allow infinitely many possible compute budgets rather than a fixed small model paired with a fixed large one.

Another possibility is that there should simply be one large model. Why not pure diffusion? Diffusion is computationally similar to speculative decoding in the sense that a forward pass proposes tokens and a subsequent forward pass accepts or rejects and refines the proposals. Google's DiffusionGemma[^diffusiongemma] achieves high throughput and high interactivity.

<figure>
  <img src="/images/llm-inference-paradigm-shift/figure-11.png" alt="Figure 11 — DiffusionGemma and Gemma 4 throughput versus interactivity">
  <figcaption markdown="1">
Figure 11. Throughput–interactivity curves digitized from Figure 12(a) of the DiffusionGemma Technical Report[^diffusiongemma].
  </figcaption>
</figure>

However, critiques will argue that diffusion has yet to match autoregression in quality. In addition to improving output quality, I would also like to see a diffusion model designed that outright dominates autoregressive models and speculative decoding at even the highest concurrency settings. This has not been achieved yet, and there is still a lot of room for exploration.

What will emerge in a sequence-parallel decoding paradigm?

## References

[^kuhn]: Kuhn, Thomas S. *The Structure of Scientific Revolutions*. 2nd ed., University of Chicago Press, 1970. [Bibliographic record](https://search.worldcat.org/title/The-structure-of-scientific-revolutions/oclc/93075).

[^kuhn-phases]: Wikipedia contributors. “The Structure of Scientific Revolutions,” §“Phases.” [Article](https://en.wikipedia.org/wiki/The_Structure_of_Scientific_Revolutions#Phases). Accessed October 22, 2026.

[^tensorflow-serving]: Olston, Chris, and Noah Fiedel. “Latest Innovations in TensorFlow Serving.” Google Research, November 2, 2017. [Blog post](https://research.google/blog/latest-innovations-in-tensorflow-serving/).

[^triton]: Goodwin, David. “NVIDIA Triton Inference Server Boosts Deep Learning Inference.” NVIDIA Technical Blog, September 12, 2018. [Blog post](https://developer.nvidia.com/blog/nvidia-serves-deep-learning-inference/); [dynamic batching documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html).

[^pope]: Pope, Reiner, et al. “Efficiently Scaling Transformer Inference.” arXiv preprint, 2022. [arXiv:2211.05102](https://arxiv.org/abs/2211.05102); [memory costs and KV-cache analysis, §§2, 3.3, and 4.2](https://arxiv.org/html/2211.05102v1).

[^orca]: Yu, Gyeong-In, et al. “Orca: A Distributed Serving System for Transformer-Based Generative Models.” OSDI, 2022. [Paper](https://www.usenix.org/conference/osdi22/presentation/yu).

[^vllm]: Kwon, Woosuk, et al. “Efficient Memory Management for Large Language Model Serving with PagedAttention.” SOSP, 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180).

[^sglang]: Zheng, Lianmin, et al. “SGLang: Efficient Execution of Structured Language Model Programs.” arXiv preprint, 2023, revised 2024. [arXiv:2312.07104](https://arxiv.org/abs/2312.07104).

[^inferencex]: SemiAnalysis. “InferenceX: Speculative Decoding Comparisons.” Public benchmark comparison pages; data collected October 21, 2026. [Comparison pages](https://inferencex.semianalysis.com/compare-spec-decode).

[^mercury-vulcan]: Will, Clifford M. “New General Relativistic Contribution to Mercury’s Perihelion Advance.” Physical Review Letters 120, 191101, 2018. [Introduction](https://link.aps.org/accepted/10.1103/PhysRevLett.120.191101). For the unsuccessful search for Vulcan, see Pantelidou, C. “The story of Mercury and Vulcan, as told by Einstein.” Institute of Cosmos Sciences, University of Barcelona, p. 1. [Historical overview](https://serviastro.ub.edu/sites/serviastro/files/2019-10/mercurivulcaneng.pdf).

[^speculative-decoding]: Leviathan, Yaniv, Matan Kalman, and Yossi Matias. “Fast Inference from Transformers via Speculative Decoding.” ICML, 2023; first preprint 2022. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192).

[^smartspec]: Liu, Xiaoxuan, et al. “Optimizing Speculative Decoding for Serving Large Language Models Using Goodput.” arXiv preprint, 2024. SmartSpec version 1, rather than the later TurboSpec revision. [arXiv:2406.14066v1](https://arxiv.org/html/2406.14066v1).

[^su-batching]: Su, Qidong, Christina Giannoula, and Gennady Pekhimenko. “The Synergy of Speculative Decoding and Batching in Serving Large Language Models.” arXiv preprint, 2023. [arXiv:2310.18813](https://arxiv.org/html/2310.18813).

[^speed-bench]: Abramovich, Talor, et al. “SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding.” ICML, 2026. [arXiv:2604.09557](https://arxiv.org/html/2604.09557).

[^magicdec]: Sadhukhan, Ranajoy, et al. “MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding.” arXiv preprint, 2024, revised 2025. [arXiv:2408.11049](https://arxiv.org/abs/2408.11049); [full text, including Table 3(c)](https://arxiv.org/html/2408.11049).

[^moesd]: Huang, Zongle, et al. “MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE.” NeurIPS, 2025. [arXiv:2505.19645v4](https://arxiv.org/html/2505.19645v4); [Figure 2(a,c)](https://arxiv.org/html/2505.19645v4#S4.F2).

[^eagle]: Li, Yuhui, et al. “EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.” arXiv preprint, 2024. [arXiv:2401.15077](https://arxiv.org/abs/2401.15077).

[^eagle3]: Li, Yuhui, et al. “EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test.” arXiv preprint, 2025. [arXiv:2503.01840](https://arxiv.org/abs/2503.01840); [method, §3](https://arxiv.org/html/2503.01840#S3); [full text, including Table 5](https://arxiv.org/html/2503.01840).

[^d3pm]: Austin, Jacob, et al. “Structured Denoising Diffusion Models in Discrete State-Spaces.” NeurIPS, 2021. [arXiv:2107.03006](https://arxiv.org/abs/2107.03006); [MLsanity citation data](https://mlsanity.com/paper/2107.03006/).

[^mdlm]: Sahoo, Subham Sekhar, et al. “Simple and Effective Masked Diffusion Language Models.” NeurIPS, 2024. [arXiv:2406.07524](https://arxiv.org/abs/2406.07524); [MLsanity citation data](https://mlsanity.com/paper/2406.07524/).

[^llada]: Nie, Shen, et al. “Large Language Diffusion Models.” arXiv preprint, 2025. [arXiv:2502.09992](https://arxiv.org/abs/2502.09992); [MLsanity citation data](https://mlsanity.com/paper/2502.09992/).

[^sedd]: Lou, Aaron, Chenlin Meng, and Stefano Ermon. “Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution.” ICML, 2024; first preprint 2023. [arXiv:2310.16834](https://arxiv.org/abs/2310.16834); [MLsanity citation data](https://mlsanity.com/paper/2310.16834/).

[^mlsanity]: MLsanity. Citation index and methodology. Annual citation counts retrieved October 22, 2026. [About the index](https://mlsanity.com/about/); individual paper records are linked in the D3PM, SEDD, MDLM, and LLaDA references.

[^fast-dllm]: Wu, Chengyue, et al. “Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding.” arXiv preprint, 2025. [arXiv:2505.22618](https://arxiv.org/abs/2505.22618).

[^dflash]: Chen, Jian, Yesheng Liang, and Zhijian Liu. “DFlash: Block Diffusion for Flash Speculative Decoding.” ICML, 2026. [arXiv:2602.06036](https://arxiv.org/html/2602.06036).

[^dflash-sglang]: Z Lab, Modal, and SGLang Teams. “The next generation of speculative decoding: DFlash and Spec V2.” LMSYS, June 15, 2026. [Blog post](https://lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/).

[^uno]: Sahoo, Subham Sekhar, et al. “Unlocking Lossless Speedups in LLMs via Discrete Diffusion.” arXiv preprint, 2026. Uno results in Table 18. [arXiv:2609.04010](https://arxiv.org/abs/2609.04010).

[^dspark]: Cheng, Xin, et al. “DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation.” arXiv preprint, 2026. [arXiv:2607.05147](https://arxiv.org/abs/2607.05147); [Markov head, §3.1](https://arxiv.org/html/2607.05147#S3.SS1); [production deployment, §5.4](https://arxiv.org/html/2607.05147#S5.SS4).

[^codd]: Li, Ian, et al. “Breaking the Factorization Barrier in Diffusion Language Models.” arXiv preprint, 2026. CoDD; first submitted February 9, 2026. [arXiv:2603.00045](https://arxiv.org/abs/2603.00045).

[^diffusiongemma]: DiffusionGemma Team. “DiffusionGemma Technical Report.” Google DeepMind, 2026. [arXiv:2608.00146](https://arxiv.org/abs/2608.00146); [throughput–interactivity comparison, Figure 12](https://arxiv.org/html/2608.00146#S6.F12); [model card](https://huggingface.co/google/diffusiongemma-26B-A4B-it).
