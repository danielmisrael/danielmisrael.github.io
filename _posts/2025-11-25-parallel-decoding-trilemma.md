---
title: 'The Parallel Decoding Trilemma'
date: 2025-11-25
permalink: /posts/2025/11/parallel-decoding-trilemma/
tags:
  - machine learning
  - language models
  - parallel decoding
usemathjax: true
---

**TL;DR:** *Parallel decoding is a fight to increase speed while maintaining fluency and diversity.*

---

Recently, "Parallel Decoding" has become an active area of LLM research, especially with the proliferation of diffusion language models[^d3pm][^ratios][^simple][^llada][^dream]. The primary goal is to speed up LLM inference, but I think the tradeoffs involved are not fully understood by the community. I want to set the stage so that researchers understand this landscape and what will actually push the frontier.

I have written papers on this topic, specifically *Accelerating Diffusion LLMs via Adaptive Parallel Decoding*[^apd] and *Planned Diffusion*[^planned], and my opinions are heavily shaped by working on them.

## What's at Stake

Three properties are in tension:

1. **Speed** — How fast can we generate?
2. **Fluency** — How correct is the output?
3. **Diversity** — How much coverage do we have over all correct outputs?

---

## Speed

Let's define these notions, starting with the most straightforward: **speed**. Speed (or latency) is simply the time it takes for an LLM to generate a response.

We can increase speed through *parallel decoding*: generating multiple tokens simultaneously. This is faster because it allows us to use fewer forward evaluations through the model to reach the final answer. In technical terms, we are reducing the depth of the computational graph required to produce the response.

In this blog, I'm going to focus on **diffusion large language models (dLLMs)** as a new paradigm to improve speed. That said, I believe my analysis of the tradeoffs between speed, fluency, and diversity extends beyond this specific paradigm and offers general lessons for LLM research.

---

## Fluency and Diversity

Now let's examine **fluency** and **diversity** together, as they are intrinsically linked.

- **Fluency** means the LLM generates "correct" things
- **Diversity** means the LLM provides good coverage over *all* correct things

When evaluating LLMs, people usually prioritize fluency and neglect diversity. This leads, for example, to LLMs that tell the same 25 jokes over and over [^jokes] and contributes to an overall blandness of responses. I believe LLM generation diversity is underrated and critical, but I'll address that more later.

### Formal Definitions

Suppose there exists an ideal LLM distribution $p^\*$. This $p^\*$ is omniscient and god-like; its breadth of knowledge is immense, it can recall any fact, reason through any challenge, and can delight you in a multitude of ways even when given the exact same prompt repeatedly.

We are training a model $p$, and we can decompose the error between $p$ and $p^\*$ using **Total Variation Distance (TVD)**:

$$
\text{TVD}(p, p^*) = \mathcal{L}_{\text{fluency}}(p, p^*) + \mathcal{L}_{\text{diversity}}(p, p^*)
$$

where we define the error components by integrating over regions where one distribution assigns greater probability than another:

$$
\mathcal{L}_{\text{fluency}}(p, p^*) = \int_{\{x \mid p(x) > p^*(x)\}} \bigl(p(x) - p^*(x)\bigr) \, dx
$$

$$
\mathcal{L}_{\text{diversity}}(p, p^*) = \int_{\{x \mid p^*(x) > p(x)\}} \bigl(p^*(x) - p(x)\bigr) \, dx
$$

These definitions mirror those in the paper [*Assessing Generative Models via Precision and Recall*](https://arxiv.org/abs/1806.00035)[^precrecall]. In fact, when $p$ and $p^\*$ are Bernoulli distributions over binary outcomes, we recover the standard definitions of precision and recall. I rename these to *fluency* and *diversity* because they're more apt for our context (and easier to remember).

### Visualizing the Errors

This figure should make the concepts of diversity error and fluency error much more clear:

<img src="/images/combined_diversity_fluency.png" alt="Fluency vs Diversity Error" style="max-width: 700px; display: block; margin: 20px auto;">
*Three scenarios: balanced error (left), diversity error dominates when the model misses modes (center), and fluency error dominates when the model generates outside valid regions (right).*

To summarize:

- If $p^\*$ is multi-modal and $p$ only fits one mode → **diversity error**
- If $p$ generates data outside any mode of $p^\*$ → **fluency error**

Fluency error is the type of error associated with "hallucinations"[^hallucinations]. Diversity error will be associated with "mode collapse"[^modecollapse]. Of course, this is a theoretical framework, and one never actually has access to the god-like $p^\*$. These images are also a drastic oversimplication because LLM parameterize extremely high dimensional, multi-modal distributions that likely escape human intuition.

---

## Diffusion Language Models (dLLMs)

Before discussing the tradeoffs between the above three properties, I must first explain diffusion language models. If you are already familiar with them, you may skip this section. Structurally, a dLLM is a BERT-style[^bert] architecture. It takes tokens as input—some of which are `[MASK]` tokens—and predicts what should fill those masked positions. Diffusion language models predict the *marginal probabilities* over the `[MASK]` tokens given the existing tokens. 

<img src="/images/gptvsbert.png" alt="AR vs Masked Diffusion" style="max-width: 500px; display: block; margin: 0 auto;">

Here is a table summary of the differences between diffusion and autoregressive models:

|                  | **Autoregressive**         | **Diffusion**          |
|------------------|----------------------------|------------------------|
| **Architecture** | GPT                        | BERT                   |
| **Masking**      | Causal Attention Masking   | Masked Token Input     |
| **Training**     | Exact NLL                  | Denoising ELBO         |
| **Inference**    | Sequential                 | Parallel               |

Formally, consider a data point as a sequence of $n$ tokens $x = (x_1, \ldots, x_n)$. For sets of indices $\mathcal{Q}, \mathcal{O} \subseteq \{1,\ldots,n\}$ where $\mathcal{Q} \cap \mathcal{O} = \emptyset$, a masked diffusion language model $p_{\text{D}}(\cdot \mid \cdot; \theta)$ computes the marginal probabilities of tokens with query indices $\mathcal{Q}$ conditioned on tokens with observed indices $\mathcal{O}$:

$$
p_{\text{D}}(x_\mathcal{Q} \mid x_\mathcal{O}; \theta) = \prod_{i \in \mathcal{Q}} p_\theta(x_i \mid x_\mathcal{O})
$$

where $p_\theta$ is a learned conditional distribution parameterized by $\theta$.

**Importantly**, unlike autoregressive models which predict the next token sequentially, diffusion language models can make multiple predictions in parallel. This capability makes dramatically reduced latency possible[^mercury].

### Why "Diffusion"?

Unlike typical BERT models, dLLMs are full generative models trained over different masking probabilities. We define a data corruption process $q$ that stochastically converts a clean sequence $x^0$ to a noisy $x^t$, gradually converting clean tokens to `[MASK]` over time $t$:

$$
q_{t\mid 0}(x^t_i \mid x^0_i) = 
\begin{cases}
t, & \text{if } x^t_i = \text{[MASK]} \\
1-t, & \text{if } x^t_i = x^0_i \\
0 & \text{otherwise}
\end{cases}
$$

$$
q_{t\mid 0}(x^t \mid x^0) = \prod_i q_{t\mid 0}(x^t_i \mid x^0_i)
$$

Given this noise process, dLLMs are trained to maximize a lower bound on the log-likelihood:

$$
\log p_\theta(x^0) \geq \mathbb{E}_{t\sim U(0,1),\, x^t \sim q(x^t \mid x^0)} \left[ \frac{1}{t} \log p_{\text{D}} \left(x_{\mathbb{1}(x^t = \text{[MASK]})} \mid x_{\mathbb{1}(x^t \neq \text{[MASK]})}; \theta\right)\right]
$$

This objective corresponds to sampling a masking ratio randomly, and making predictions at the locations of the masked tokens. I won't derive this bound here, but the key insight is that this objective is *exactly equivalent* to training an **Any-Order Autoregressive Model**[^aoar], a model that learns an autoregressive joint distribution given random permutations of the data. 

Put this way, dLLMs aren't so intimidating: *it's just autoregression that allows arbitrary orders and parallel sampling.*

<img src="/images/rnd1_gen.gif" alt="Diffusion LLM generation" style="max-width: 600px; display: block; margin: 20px auto;">
*Diffusion LLM generating text by iteratively unmasking tokens. Credit: RND1[^rnd1]*

---


## The Trilemma

So, here is where we stand. Consider an LLM as a black box:

| **Input** | **Output** |
|-----------|------------|
| Data, Compute | Speed, Fluency, Diversity |

**I argue there is no free lunch.** For a fixed level of data and compute, you generate a finite "budget" of performance measured by speed, fluency, and diversity. You can trade fluency or diversity to gain speed, but you cannot improve all three simultaneously without increasing your compute and data inputs.

### Trading Fluency for Speed

Diffusion LLMs have a built-in mechanism for trading fluency for speed: you can use fewer denoising steps. Under fewer denoising steps, a diffusion model will sample more tokens in parallel per forward pass. In my paper[^apd], I show that sampling more tokens in parallel has a very direct impact on the fluency of the output.

<img src="/images/apd_tradeoff.png" alt="APD Speed vs Accuracy Tradeoff" style="max-width: 700px; display: block; margin: 0 auto;">
*Speed vs. accuracy tradeoff on GSM8K: more parallel tokens means faster generation but lower accuracy.*

On GSM8K (grade school math), accuracy drops in a very predictable manner while speed goes up. This seems fairly intuitive, but what exactly is going on? The paper [*Discrete Copula Diffusion*](https://arxiv.org/abs/2410.01949)[^dcd] has an excellent figure to explain this.

<img src="/images/dcd.png" alt="Discrete Copula Diffusion - Marginal vs Joint" style="max-width: 550px; display: block; margin: 0 auto;">
*Discrete Copula Diffusion: "puppy" and "scared" are each marginally likely, but jointly unlikely.*

When you sample multiple tokens in parallel, you necessarily only capture their *marginal* distribution. You cannot model the dependencies between them, because each token doesn't know what the other will commit to. In this example, "puppy" and "scared" are marginally likely, but jointly unlikely. Thus, more parallelism (and speed) leads to fewer dependencies captured, and hence lower fluency.

### Trading Diversity for Speed

Imagine in the example above if "barking" and "scared" had a marginal probability of 1. It would follow that their joint likelihood would also be 1, and so the product of marginal probabilities would be equivalent to the joint probability.

It turns out this logic extends to marginal probabilities near 1 as well. In *fast-dLLM* [^fastdllm], the authors use a strategy that samples tokens at positions that exceed some probability threshold. Specifically, they state that for a joint distribution $p(x)$ and marginal product distribution $q(x)$ that factorizes $p$, if each marginal in the product has probability greater than $1 - \epsilon$, then we can bound the total variation distance between the joint and the marginal:

$$
\text{TVD}(p, q) < \frac{3n-1}{2} \epsilon
$$

on a sequence of $n$ tokens to be sampled. For sufficiently small $\epsilon$, specifically $\epsilon \leq \frac{1}{n+1}$, this sampling method will be exactly equivalent to greedy decoding, meaning it will only sample the most likely sequence from $p$. I'm not being fully precise with the notation, and I encourage those interested to read the details, but the overall point should be clear: **one can significantly improve speed at the expense of diversity.**

I learned this working on Adaptive Parallel Decoding (APD)[^apd]. APD defines a multiplicative mixture between the dLLM marginals and a small autoregressive model and samples left to right from this distribution, only accepting parallel samples with high likelihood from both.

<img src="/images/apd_results.png" alt="APD Results" style="max-width: 650px; display: block; margin: 0 auto;">
*APD can dramatically increase speed with some expense to fluency and diversity.*

I also introduced a parameter $R$ that controls the tradeoff. As you can see from this plot, compared to before just modifying the number of denoising steps like before, we are able to achieve a much better tradeoff. This method can improve speed by a factor of almost **10x** without a significant drop in quality. It's hard to imagine that this is free, and indeed, I think the price paid is in diversity. A multiplicative mixture between distributions inherently reduces diversity (or entropy if you prefer).

<img src="/images/poe.png" alt="Product of Experts" style="max-width: 500px; display: block; margin: 0 auto;">
*A product-of-experts[^poe] decreases diversity*

In many cases, losing diversity may be a worthy tradeoff. Does coding really need a diversity of samples? Maybe reasoning does not require diversity.

### In defense of diversity

I claim that many writing tasks *will* require diversity, especially as we see the internet become saturated with "LLM slop" [^slop]. Also, parallel decoding and faster inference will only lead to more intelligent models if diversity can be preserved somehow. It would be great if we could plug parallel decoding methods into modern Reinforcement Learning (RL) frameworks, because the bottleneck of RL is typically the speed of inference. However, it is widely understood that RL methods work by essentially sharpening the distribution of the LLM, revealing capabilities that have probability mass under the base model [^rlbase]. Using our terminology, RL allows models to trade fluency error for diversity error (also note that they trade speed for fluency by outputting more tokens). If by parallel decoding, one removes all the diversity, then there will be almost nothing left for RL to sharpen.

### No Free Lunch

Let's revisit the claim that there is no free lunch. Is there a way to increase speed (i.e., parallel sampling) without sacrificing fluency or diversity? What about speculative decoding?

Speculative decoding [^specdec] seems to increase parallelism at zero cost. To clarify, I believe there is no free lunch *at a fixed level of compute and data*. Speculative decoding uses an auxiliary draft model, which I interpret as adding more compute to the input side of the ledger. It does seem, though, that there is something profound with speculative decoding. While I maintain that parallel sampling has no free lunch, perhaps parallel verification (i.e., likelihood computation) which can be used for sampling is free. TiDAR [^tidar] seems like a promising approach along these lines.

Instead of just adding compute to the inputs of the model, it is possible to also add data. This is what we show in *Planned Diffusion*[^planned], where we train a model to use control tags to first plan a response, then use diffusion to generate from that plan to increase parallelism.

<img src="/images/planned_diffusion.png" alt="Planned Diffusion" style="max-width: 700px; display: block; margin: 0 auto;">
*Planned Diffusion: first generate a plan autoregressively, then fill in spans in parallel with diffusion.*

This type of approach is trying to improve what we call "semantic parallelism" [^pasta]. If you believe as I do that there is no free lunch with respect to speed, fluency, and diversity, adding more data seems like the best solution. After all, just adding more data until the problem was fixed is how we got LLMs in the first place [^scaling].

---

## Concluding Remarks

To summarize, you can add compute and data into an LLM and get out a fixed amount of speed, fluency, and diversity:

- You can trade diversity or speed for fluency through **RL**
- You can trade fluency for speed via **fewer diffusion denoising steps**
- You can trade diversity for speed with **better sampling techniques**

To my knowledge, I don't know a way to trade fluency for diversity or speed—but maybe those are bad trades that are not even worth investigating.

There are now benchmarks that analyze the tradeoff between speed and quality in diffusion LLMs [^parallelbench], and I encourage readers to work on algorithms to get the best tradeoffs. However, I think it is a mistake to collapse fluency and diversity into a single metric. In fact, I worry that the LLM community as a whole has over-focused on fluency, and I find it likely that more diversity is needed to push the field forward.

> **I look forward to novel approaches navigating the parallel decoding trilemma.**


---

*Thanks for reading! Feel free to reach out with questions or comments.*

---

## Cite This Post

<div style="position: relative;">
<button onclick="navigator.clipboard.writeText(document.getElementById('bibtex').innerText); this.innerText='Copied!'; setTimeout(() => this.innerText='Copy', 2000);" style="position: absolute; top: 8px; right: 8px; padding: 4px 12px; background: #4a4a4a; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">Copy</button>
<pre id="bibtex" style="background: #2d2d2d; color: #f8f8f2; padding: 20px; border-radius: 8px; font-size: 15px; line-height: 1.5; overflow-x: auto;">@misc{israel2025trilemma,
  title        = {The Parallel Decoding Trilemma},
  author       = {Israel, Daniel},
  year         = {2025},
  month        = {November},
  howpublished = {Blog post},
  url          = {https://danielmisrael.github.io/posts/2025/11/parallel-decoding-trilemma/}
}</pre>
</div>

---

## References

[^apd]: Daniel Israel, Guy Van den Broeck, and Aditya Grover. "Accelerating Diffusion LLMs via Adaptive Parallel Decoding." *Advances in Neural Information Processing Systems 38 (NeurIPS)*, 2025. [arXiv:2506.00413](https://arxiv.org/abs/2506.00413)

[^planned]: Daniel Israel, Tian Jin, Ellie Cheng, Guy Van den Broeck, Aditya Grover, Suvinay Subramanian, and Michael Carbin. "Planned Diffusion." *arXiv preprint*, 2025. [arXiv:2510.18087](https://arxiv.org/abs/2510.18087)

[^d3pm]: Austin, Jacob, et al. "Structured Denoising Diffusion Models in Discrete State-Spaces." *NeurIPS*, 2021. [arXiv:2107.03006](https://arxiv.org/abs/2107.03006)

[^ratios]: Lou, Aaron, Chenlin Meng, and Stefano Ermon. "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution." *arXiv preprint*, 2023. [arXiv:2310.16834](https://arxiv.org/abs/2310.16834)

[^simple]: Sahoo, Subham, et al. "Simple and Effective Masked Diffusion Language Models." *NeurIPS*, 2024. [arXiv:2406.07524](https://arxiv.org/abs/2406.07524)

[^llada]: Nie, Shen, et al. "Large Language Diffusion Models." *arXiv preprint*, 2025. [arXiv:2502.09992](https://arxiv.org/abs/2502.09992)

[^dream]: Ye, Jiacheng, et al. "Dream 7B: Diffusion Large Language Models." *arXiv preprint*, 2025. [arXiv:2508.15487](https://arxiv.org/abs/2508.15487)

[^jokes]: Jentzsch, Sophie, and Kristian Kersting. "ChatGPT is Fun, but It is Not Funny! Humor is Still Challenging Large Language Models." *arXiv preprint*, 2023. [arXiv:2306.04563](https://arxiv.org/abs/2306.04563)

[^slop]: Paredes, Jose, et al. "More Articles Are Now Created by AI Than Humans." *Graphite*, 2024. [graphite.io](https://graphite.io/five-percent/more-articles-are-now-created-by-ai-than-humans)

[^rlbase]: Yue, Yang, et al. "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?" *arXiv preprint*, 2025. [arXiv:2504.13837](https://arxiv.org/abs/2504.13837)

[^fastdllm]: Wu, Chengyue, et al. "Fast-dLLM: Training-Free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding." *arXiv preprint*, 2025. [arXiv:2505.22618](https://arxiv.org/abs/2505.22618)

[^tidar]: Liu, Jingyu, et al. "TiDAR: Think in Diffusion, Talk in Autoregression." *arXiv preprint*, 2025. [arXiv:2511.08923](https://arxiv.org/abs/2511.08923)

[^specdec]: Leviathan, Yaniv, Matan Kalman, and Yossi Matias. "Fast Inference from Transformers via Speculative Decoding." *ICML*, 2023. [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)

[^parallelbench]: Kang, Wonjun, et al. "ParallelBench: Understanding the Trade-offs of Parallel Decoding in Diffusion LLMs." *arXiv preprint*, 2025. [arXiv:2510.04767](https://arxiv.org/abs/2510.04767)

[^mercury]: Khanna, Samar, et al. "Mercury: Ultra-Fast Language Models Based on Diffusion." *arXiv preprint*, 2025. [arXiv:2506.17298](https://arxiv.org/abs/2506.17298)

[^aoar]: Shih, Andy, Dorsa Sadigh, and Stefano Ermon. "Training and Inference on Any-Order Autoregressive Models the Right Way." *NeurIPS*, 2022. [arXiv:2205.13554](https://arxiv.org/abs/2205.13554)

[^precrecall]: Sajjadi, Mehdi SM, et al. "Assessing Generative Models via Precision and Recall." *NeurIPS*, 2018. [arXiv:1806.00035](https://arxiv.org/abs/1806.00035)

[^modecollapse]: Thanh-Tung, Hoang, and Truyen Tran. "Catastrophic Forgetting and Mode Collapse in GANs." *IJCNN*, 2020. [arXiv:1807.04015](https://arxiv.org/abs/1807.04015)

[^hallucinations]: Xu, Ziwei, Sanjay Jain, and Mohan Kankanhalli. "Hallucination is Inevitable: An Innate Limitation of Large Language Models." *arXiv preprint*, 2024. [arXiv:2401.11817](https://arxiv.org/abs/2401.11817)

[^poe]: Hinton, Geoffrey E. "Training Products of Experts by Minimizing Contrastive Divergence." *Neural Computation*, 2002. [PDF](https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf)

[^bert]: Devlin, Jacob, et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *NAACL*, 2019. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

[^rnd1]: Chandrasegaran, Keshigeyan, Armin W. Thomas, et al. "RND1: Simple, Scalable AR-to-Diffusion Conversion." *Radical Numerics*, 2025. [radicalnumerics.ai](https://www.radicalnumerics.ai/blog/rnd1)

[^dcd]: Liu, Anji, et al. "Discrete copula diffusion." *arXiv preprint*, 2024. [arXiv:2410.01949](https://arxiv.org/abs/2410.01949)

[^scaling]: Kaplan, Jared, et al. "Scaling laws for neural language models." *arXiv preprint*, 2020. [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)

[^pasta]: Jin, Tian, et al. "Learning to Keep a Promise: Scaling Language Model Decoding Parallelism with Learned Asynchronous Decoding." *arXiv preprint*, 2025. [arXiv:2502.11517](https://arxiv.org/abs/2502.11517)