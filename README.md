# This is the official repository of the paper ["Towards Rationality in Language and Multimodal Agents: A Survey"](https://arxiv.org/abs/2406.00252)

[![Arxiv](https://img.shields.io/badge/ArXiv-Full_Paper-B31B1B)](https://arxiv.org/abs/2406.00252)
[![Google Scholar](https://img.shields.io/badge/Google_Scholar-Cite_Our_Paper-4085F4)](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C39&q=Multi-Modal+and+Multi-Agent+Systems+Meet+Rationality%3A+A+Survey&btnG=)

Unlike **reasoning** that aims to draw conclusions from premises, **rationality** ensures that those conclusions are reliably consistent, have an orderability of preference, and are aligned with evidence from various sources and logical principles. It becomes increasingly important for human users applying these agents in critical sectors like health care and finance that expect consistent and reliable decision-making. This survey is the first to comprehensively explore the notion of rationality 🧠 in language and multimodal agents 🤖. 

- [Define Rationality](#define-rationality)
- [Towards Rationality in Agents](#towards-rationality-in-agents)
- [Evaluating Rationality in Agents](#evaluating-rationality-in-agents)

<p align="center">
<img src=header.png />
</p>

The fields of language and multimodal agents are rapidly evolving, so we highly encourage researchers who want to promote their amazing works on this dynamic repository to submit a pull request and make updates. 💜

We have a concurrent work **A Peek into Token Bias: Large Language Models Are Not Yet Genuine Reasoners** [Paper]() [Code](https://github.com/bowen-upenn/llm_logical_fallacies) at EMNLP 2024 Main, which reconceptualizes the evaluation of reasoning capabilities in LLMs into a general, statistically rigorous framework. Its findings reveal that LLMs are susceptible to **superficial token perturbations**, primarily relying on token biases rather than genuine reasoning. 

## Citations
This bunny 🐰 will be happy if you could cite our work. (Google Scholar is still indexing our old title)

    @misc{jiang2024multimodal,
          title={Multi-Modal and Multi-Agent Systems Meet Rationality: A Survey}, 
          author={Bowen Jiang and Yangxinyu Xie and Xiaomeng Wang and Weijie J. Su and Camillo J. Taylor and Tanwi Mallick},
          year={2024},
          eprint={2406.00252},
          archivePrefix={arXiv},
          primaryClass={cs.AI}
    }

## Define Rationality
**Rationality** is the quality of being guided by reason, characterized by logical thinking and decision-making that align with evidence and logical rules. Drawing on foundational works in cognitive science about rational decision-making, we present four necessary, though not sufficient, axioms we expect a rational agent or agent systems to fulfill:

- **Information grounding**
  
  A rational agent's decision-making should be grounded in physical and factual reality, incorporating information it perceives from multimodal formats and sources.
  
- **Logical Consistency**

  Logical consistency refers to an agent's ability to avoid self-contradictions in reasoning and ensure that its conclusions logically follow from its premises. A rational agent should deliver consistent decisions in its final responses, producing invariant decisions across equivalent representations of the same problem.
  
- **Invariance from Irrelevant Context**

  A rational agent should not be swayed by irrelevant contextual information, focusing instead on the logical essence of the problems and relevant data.

- **Orderability of Preference**.

  When comparing alternatives in a decision scenario, a rational agent should be able to rank the options based on the current state and ultimately select the most preferred one based on the expected outcomes.



## Towards Rationality in Agents
<p align="center">
<img src=tree.png />
</p>

**Bold fonts are used to mark work that involve multi-modalities.** In their original writings, most existing studies do not explicitly base their frameworks on rationality. Our analysis aims to reinterpret these works through the lens of our four axioms of rationality, offering a novel perspective that bridges existing methodologies with rational principles.

## 1. Advancing Information Grounding
<p align="center">
<img src=grounding.png />
</p>

### 1.1. Grounding on multimodal information

**CLIP: Learning transferable visual models from natural language supervision** [Paper](https://arxiv.org/abs/2103.00020) [Code](https://github.com/openai/CLIP) \
**iNLG: Imagination-guided open-ended text generation** [Paper](https://arxiv.org/abs/2210.03765) [Code](https://github.com/VegB/iNLG) \
**BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models** [Paper](https://arxiv.org/abs/2301.12597) [Code](https://github.com/salesforce/BLIP) \
**Vid2Seq: Large-scale pretraining of a visual language model for dense video captioning** [Paper](https://arxiv.org/abs/2302.14115) [Code](https://github.com/google-research/scenic/tree/main/scenic/projects/vid2seq) \
**MiniGPT-4: Enhancing vision-language understanding with advanced large language models** [Paper](https://arxiv.org/abs/2304.10592) [Code](https://github.com/Vision-CAIR/MiniGPT-4) \
**Flamingo: a visual language model for few-shot learning** [Paper](https://arxiv.org/abs/2204.14198) \
**OpenFlamingo: An open-source framework for training large autoregressive vision-language models** [Paper](https://arxiv.org/abs/2308.01390) [Code](https://github.com/mlfoundations/open_flamingo) \
**LLaVA: Visual Instruction Tuning** [Paper](https://arxiv.org/pdf/2304.08485) [Code](https://github.com/haotian-liu/LLaVA) \
**LLaVA 1.5: Improved Baselines with Visual Instruction Tuning** [Paper](https://arxiv.org/abs/2310.03744) [Code](https://github.com/haotian-liu/LLaVA) \
**CogVLM: Visual expert for pretrained language models** [Paper](https://arxiv.org/abs/2311.03079) [Code](https://github.com/THUDM/CogVLM) \
**GPT-4V(ision) System Card** [Paper](https://cdn.openai.com/papers/GPTV_System_Card.pdf) \
**Gemini: A Family of Highly Capable Multimodal Models** [Paper](https://arxiv.org/abs/2312.11805) \
**Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context** [Paper](https://arxiv.org/abs/2403.05530) \
**GPT-4o** [Website](https://openai.com/index/hello-gpt-4o/)
**JEPA: A Path Towards Autonomous Machine Intelligence** [Paper](https://openreview.net/pdf?id=BZ5a1r-kVsf) \
Voyager: An open-ended embodied agent with large language models [Paper](https://arxiv.org/abs/2305.16291) [Code](https://github.com/MineDojo/Voyager) \
Ghost in the Minecraft: Generally capable agents for open-world enviroments via large language models with text-based knowledge and memory [Paper](https://arxiv.org/abs/2305.17144) [Code](https://github.com/OpenGVLab/GITM) \
**Objective-Driven AI** [Slides](https://www.ece.uw.edu/wp-content/uploads/2024/01/lecun-20240124-uw-lyttle.pdf) \
**LWM: World Model on Million-Length Video And Language With RingAttention** [Paper](https://arxiv.org/abs/2402.08268) [Code](https://github.com/LargeWorldModel/LWM) \
**Sora: Video generation models as world simulators** [Website](https://openai.com/index/video-generation-models-as-world-simulators/) \
**IWM: Learning and Leveraging World Models in Visual Representation Learning** [Paper](https://arxiv.org/pdf/2403.00504) \
**CubeLLM: Language-Image Models with 3D Understanding** [Paper](https://arxiv.org/abs/2405.03685) [Code](https://github.com/NVlabs/Cube-LLM) 

#### 1.2 Expanding working memory from external knowledge retrieval and tool utilization
RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks [Paper](https://arxiv.org/abs/2005.11401) \
**Minedojo: Building open-ended embodied agents with internet-scale knowledge** [Paper](https://arxiv.org/abs/2206.08853) [Code](https://github.com/MineDojo/MineDojo) \
ReAct: Synergizing reasoning and acting in language models [Paper](https://arxiv.org/abs/2210.03629) [Code](https://github.com/ysymyth/ReAct) \
**RA-CM3: Retrieval-Augmented Multimodal Language Modeling** [Paper](https://arxiv.org/abs/2211.12561) \
**Chameleon: Plug-and-play compositional reasoning with large language models** [Paper](https://arxiv.org/abs/2304.09842) [Code](https://github.com/lupantech/chameleon-llm) \
Chain of knowledge: A framework for grounding large language models with structured knowledge bases [Paper](https://arxiv.org/abs/2305.13269) [Code](https://github.com/DAMO-NLP-SG/chain-of-knowledge) \
**SIRI: Towards Top-Down Reasoning: An Explainable Multi-Agent Approach for Visual Question Answering** [Paper](https://arxiv.org/abs/2311.17331) \
CooperKGC: Multi-Agent Synergy for Improving Knowledge Graph Construction [Paper](https://arxiv.org/abs/2312.03022) [Code](https://github.com/hongbinye/CooperKGC) \
**DoraemonGPT: Toward understanding dynamic scenes with large language models** [Paper](https://arxiv.org/abs/2401.08392) [Code](https://github.com/z-x-yang/DoraemonGPT) \
WildfireGPT: Tailored Large Language Model for Wildfire Analysis [Paper](https://arxiv.org/abs/2402.07877) [Code](https://github.com/jhutchison0/callm) \
**Chain-of-Action: Faithful and Multimodal Question Answering through Large Language Models** [Paper](https://arxiv.org/abs/2403.17359) \
CuriousLLM: Elevating Multi-Document QA with Reasoning-Infused Knowledge Graph Prompting [Paper](https://arxiv.org/abs/2404.09077) [Code](https://github.com/zukangy/KGP-CuriousLLM) \
Agent Hospital: A Simulacrum of Hospital with Evolvable Medical Agents [Paper](https://arxiv.org/pdf/2405.02957)
**Visual Programming: Compositional visual reasoning without training** [Paper](https://arxiv.org/abs/2211.11559) [Code](https://github.com/allenai/visprog) \
Parsel: Algorithmic Reasoning with Language Models by Composing Decompositions [Paper](https://arxiv.org/abs/2212.10561) [Code](https://github.com/ezelikman/parsel) \
Toolformer: Language Models Can Teach Themselves to Use Tools [Paper](https://arxiv.org/abs/2302.04761) [Code](https://github.com/lucidrains/toolformer-pytorch) \
BabyAGI [Code](https://github.com/yoheinakajima/babyagi) \
**ViperGPT: Visual inference via python execution for reasoning** [Paper](https://arxiv.org/abs/2303.08128) [Code](https://github.com/cvlab-columbia/viper) \
**HuggingGPT: Solving AI tasks with ChatGPT and its friends in Gugging Face** [Paper](https://arxiv.org/abs/2303.17580) [Code](https://github.com/microsoft/JARVIS?tab=readme-ov-file) \
**Chameleon: Plug-and-play compositional reasoning with large language models** [Paper](https://arxiv.org/abs/2304.09842) [Code](https://github.com/lupantech/chameleon-llm) \
AutoGPT: build & use AI agents [Code](https://github.com/Significant-Gravitas/AutoGPT?tab=readme-ov-file) \
**ToolAlpaca: Generalized tool learning for language models with 3000 simulated cases** [Paper](https://arxiv.org/abs/2306.05301) [Code](https://github.com/tangqiaoyu/ToolAlpaca) \
**AssistGPT: A general multi-modal assistant that can plan, execute, inspect, and learn** [Paper](https://arxiv.org/abs/2306.08640) [Code](https://github.com/showlab/assistgpt) \
**Avis: Autonomous visual information seeking with large language model agent** [Paper](https://arxiv.org/abs/2306.08129) \
**BuboGPT: Enabling visual grounding in multi-modal llms** [Paper](https://arxiv.org/abs/2307.08581) [Code](https://github.com/magic-research/bubogpt) \
MemGPT: Towards llms as operating systems [Paper](https://arxiv.org/abs/2310.08560) [Code](https://github.com/cpacker/MemGPT) \
MetaGPT: Meta programming for multi-agent collaborative framework [Paper](https://arxiv.org/abs/2308.00352) [Code](https://github.com/geekan/MetaGPT) \
**Agent LUMOS: Learning agents with unified data, modular design, and open-source llms** [Paper](https://arxiv.org/abs/2311.05657) [Code](https://github.com/allenai/lumos) \
AutoAct: Automatic Agent Learning from Scratch for QA via Self-Planning [Paper](https://arxiv.org/abs/2401.05268) [Code](https://github.com/zjunlp/AutoAct) \
Small LLMs Are Weak Tool Learners: A Multi-LLM Agent [Paper](https://arxiv.org/abs/2401.07324) [Code](https://github.com/X-PLUG/Multi-LLM-Agent) \
DeLLMa: A Framework for Decision Making Under Uncertainty with Large Language Models [Paper](https://arxiv.org/abs/2402.02392) [Code](https://github.com/DeLLMa/DeLLMa) \
ConAgents: Learning to Use Tools via Cooperative and Interactive Agents [Paper](https://arxiv.org/abs/2403.03031) [Code](https://github.com/shizhl/CoAgents) \
**Multi-Agent VQA: Exploring Multi-Agent Foundation Models in Zero-Shot Visual Question Answering** [Paper](https://arxiv.org/abs/2403.14783) [Code](https://github.com/bowen-upenn/Multi-Agent-VQA)


## 2. Advancing Logical Consistency
<p align="center">
<img src=consistency.png />
</p>

### 2.1. Consensus from reflection and multi-agent collaboration
CoT: Chain-of-thought prompting elicits reasoning in large language models [Paper](https://arxiv.org/abs/2201.11903) \
Self-Refine: Iterative refinement with self-feedback [Paper](https://arxiv.org/abs/2303.17651) [Code](https://github.com/madaan/self-refine) \
Reflexion: Language agents with verbal reinforcement learning [Paper](https://arxiv.org/abs/2303.11366) [Code](https://github.com/noahshinn/reflexion) \
FORD: Examining Inter-Consistency of Large Language Models Collaboration: An In-depth Analysis via Debate [Paper](https://arxiv.org/abs/2305.11595) [Code](https://github.com/Waste-Wood/FORD) \
Memorybank: Enhancing large language models with long-term memory [Paper](https://arxiv.org/abs/2305.10250) [Code](https://github.com/zhongwanjun/MemoryBank-SiliconFriend) \
LM vs LM: Detecting factual errors via cross examination [Paper](https://arxiv.org/abs/2305.13281) \
Multi-Agent Collaboration: Harnessing the Power of Intelligent LLM Agents [Paper](https://arxiv.org/abs/2306.03314) \
Improving factuality and reasoning in language models through multiagent debate [Paper](https://arxiv.org/abs/2305.14325) [Code](https://github.com/composable-models/llm_multiagent_debate) \
MAD: Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate [Paper](https://arxiv.org/abs/2305.19118) [Code](https://github.com/Skytliang/Multi-Agents-Debate) \
S3: Social-network Simulation System with Large Language Model-Empowered Agents [Paper](https://arxiv.org/abs/2307.14984) \
ChatDev: Communicative agents for software development [Paper](https://arxiv.org/abs/2307.07924) [Code](https://github.com/OpenBMB/ChatDev) \
ChatEval: Towards better llm-based evaluators through multi-agent debate [Paper](https://arxiv.org/abs/2308.07201) [Code](https://github.com/thunlp/ChatEval) \
AutoGen: Enabling next-gen llm applications via multi-agent conversation framework [Paper](https://arxiv.org/abs/2308.08155) [Code](https://microsoft.github.io/autogen/) \
Corex: Pushing the boundaries of complex reasoning through multi-model collaboration [Paper](https://arxiv.org/abs/2310.00280) [Code](https://github.com/QiushiSun/Corex) \
DyLAN: Dynamic llm-agent network: An llm-agent collaboration framework with agent team optimization [Paper](https://arxiv.org/abs/2310.02170) [Code](https://github.com/SALT-NLP/DyLAN) \
AgentCF: Collaborative learning with autonomous language agents for recommender systems [Paper](https://arxiv.org/abs/2310.09233) \
MetaAgents: Simulating interactions of human behaviors for llm-based task-oriented coordination via collaborative generative agents [Paper](https://arxiv.org/abs/2310.06500) \
Social Learning: Towards Collaborative Learning with Large Language Models [Paper](https://arxiv.org/abs/2312.11441) \
Enhancing Diagnostic Accuracy through Multi-Agent Conversations: Using Large Language Models to Mitigate Cognitive Bias [Paper](https://arxiv.org/abs/2401.14589) \
Combating Adversarial Attacks with Multi-Agent Debate [Paper](https://arxiv.org/abs/2401.05998) \
Debating with More Persuasive LLMs Leads to More Truthful Answers [Paper](https://arxiv.org/abs/2402.06782) [Code](https://github.com/ucl-dark/llm_debate) 

### 2.2 Consistent execution from symbolic reasoning and tool utilization
_Refer 1.2 for tool unitization_

**Binder: Binding language models in symbolic languages** [Paper](https://arxiv.org/abs/2210.02875) [Code](https://github.com/xlang-ai/Binder) \
Parsel: Algorithmic Reasoning with Language Models by Composing Decompositions [Paper](https://arxiv.org/abs/2212.10561) [Code](https://github.com/ezelikman/parsel) \
**Sparks of artificial general intelligence: Early experiments with gpt-4** [Paper](https://arxiv.org/abs/2303.12712) \
Logic-LM: Empowering large language models with symbolic solvers for faithful logical reasoning [Paper](https://arxiv.org/abs/2305.12295) [Code](https://github.com/teacherpeterpan/Logic-LLM) \
Minding Language Models' (Lack of) Theory of Mind: A Plug-and-Play Multi-Character Belief Tracker [Paper](https://arxiv.org/abs/2306.00924) [Code](https://github.com/msclar/symbolictom) \
Towards formal verification of neuro-symbolic multi-agent systems [Paper](https://www.ijcai.org/proceedings/2023/0800.pdf) \
**What’s Left? Concept Grounding with Logic-Enhanced Foundation Models** [Paper](https://arxiv.org/abs/2310.16035) [Code](https://github.com/joyhsu0504/LEFT) \
Ada: Learning adaptive planning representations with natural language guidance [Paper](https://arxiv.org/abs/2312.08566) \
Large language models are neurosymbolic reasoners [Paper](https://arxiv.org/abs/2401.09334) [Code](https://github.com/hyintell/llmsymbolic) \
**DoraemonGPT: Toward understanding dynamic scenes with large language models** [Paper](https://arxiv.org/abs/2401.08392) [Code](https://github.com/z-x-yang/DoraemonGPT) \
A Neuro-Symbolic Approach to Multi-Agent RL for Interpretability and Probabilistic Decision Making [Paper](https://arxiv.org/abs/2402.13440) \
Conceptual and Unbiased Reasoning in Language Models [Paper](https://arxiv.org/abs/2404.00205)

## 3. Advancing Invariance from Irrelevant Information
<p align="center">
<img src=invariance.png />
</p>

- [ ] TODO: We are continuing updating the sections below. Please refer our paper for the full details.
### 3.1. Representation invariance across modalities

### 3.2. Abstraction from symbolic reasoning and tool utilization

## 4. Advancing Orderability of Preference
<p align="center">
<img src=preference.png />
</p>
### 4.1. Learning preference from reinforcement learning
### 4.2. Maximizing utility functions and controlling conformal risks


## Evaluating Rationality in Agents
### 0. General Benchmarks or Evaluation Metrics
CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge [Paper](https://arxiv.org/pdf/1811.00937) [Data](https://www.tau-nlp.sites.tau.ac.il/commonsenseqa) \
LogiQA: a challenge dataset for machine reading comprehension with logical reasoning [Paper](https://arxiv.org/pdf/2007.08124) [Data](https://github.com/lgw863/LogiQA-dataset) \
Logiqa 2.0: an improved dataset for logical reasoning in natural language understanding [Paper](https://ieeexplore.ieee.org/document/10174688) [Data](https://github.com/csitfun/LogiQA2.0) \
HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering [Paper](https://arxiv.org/abs/1809.09600) [Data](https://hotpotqa.github.io) \
Measuring mathematical problem solving with the math dataset [Paper](https://arxiv.org/abs/2103.03874) [Data](https://github.com/hendrycks/math) \
HybridQA: A Dataset of Multi-Hop Question Answering over Tabular and Textual Data [Paper](https://arxiv.org/abs/2004.07347) [Data](https://hybridqa.github.io) \
Conceptual and Unbiased Reasoning in Language Models [Paper](https://arxiv.org/abs/2404.00205) \
Large language model evaluation via multi AI agents: Preliminary results [Paper](https://arxiv.org/abs/2404.01023) \
AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents [Paper](https://arxiv.org/abs/2401.13178) [Code](https://github.com/hkust-nlp/AgentBoard) \
**AgentBench: Evaluating LLMs as Agents** [Paper](https://arxiv.org/abs/2308.03688) [Code](https://github.com/THUDM/AgentBench?tab=readme-ov-file) \
Benchmark self-evolving: A multi-agent framework for dynamic llm evaluation [Paper](https://arxiv.org/abs/2402.11443) [Code](https://github.com/nanshineloong/self-evolving-benchmark) \
Llm-deliberation: Evaluating llms with interactive multi-agent negotiation games [Paper](https://arxiv.org/abs/2309.17234) [Code](https://github.com/S-Abdelnabi/LLM-Deliberation)

## 1. Evaluating Information Grounding
**A multi-task, multilingual, multimodal evaluation of chatgpt on reasoning, hallucination, and interactivity** [Paper](https://arxiv.org/abs/2302.04023) [Data](https://github.com/HLTCHKUST/chatgpt-evaluation) \
Hallucinations in large multilingual translation models [Paper](https://arxiv.org/abs/2303.16104) [Data](https://github.com/deep-spin/lmt_hallucinations) \
Evaluating attribution in dialogue systems: The BEGIN benchmark [Paper](https://arxiv.org/abs/2105.00071) [Data](https://github.com/google/BEGIN-dataset) \
HaluEval: A large-scale hallucination evaluation benchmark for large language models [Paper](https://arxiv.org/abs/2305.11747) [Data](https://github.com/RUCAIBox/HaluEval) \
DialDact: A benchmark for fact-checking in dialogue [Paper](https://arxiv.org/abs/2110.08222) [Data](https://github.com/salesforce/DialFact) \
FaithDial: A faithful benchmark for information-seeking dialogue [Paper](https://arxiv.org/abs/2204.10757) [Data](https://github.com/McGill-NLP/FaithDial) \
AIS: Measuring attribution in natural language generation models [Paper](https://arxiv.org/abs/2112.12870) [Data](https://github.com/google-research-datasets/attributed-qa) \
Why does ChatGPT fall short in providing truthful answers [Paper](https://arxiv.org/abs/2304.10513) \
FADE: Diving deep into modes of fact hallucinations in dialogue systems [Paper](https://arxiv.org/abs/2301.04449) [Code](https://github.com/souvikdgp16/FADE) \
Hallucinated but factual! inspecting the factuality of hallucinations in abstractive summarization [Paper](https://arxiv.org/abs/2109.09784)  \
Exploring and evaluating hallucinations in llm-powered code generation [Paper](https://arxiv.org/abs/2404.00971) \
EureQA: Deceiving semantic shortcuts on reasoning chains: How far can models go without hallucination [Paper](https://arxiv.org/abs/2311.09702) [Code](https://vincentleebang.github.io/eureqa.github.io/) \
TofuEval: Evaluating hallucinations of llms on topic-focused dialogue summarization [Paper](https://arxiv.org/abs/2402.13249) [Code](https://github.com/amazon-science/tofueval) \
**Object hallucination in image captioning** [Paper](https://arxiv.org/abs/1809.02156) [Code](https://github.com/LisaAnne/Hallucination) \
**Let there be a clock on the beach: Reducing object hallucination in image captioning** [Paper](https://arxiv.org/abs/2110.01705) [Code](https://github.com/furkanbiten/object-bias) \
**Evaluating object hallucination in large vision-language models** [Paper](https://arxiv.org/abs/2305.10355) [Code](https://github.com/RUCAIBox/POPE) \
**LLaVA-RLHF: Aligning large multimodal models with factually augmented RLHF** [Paper](https://arxiv.org/abs/2309.14525) [Code](https://github.com/llava-rlhf/LLaVA-RLHF)

# 2. Evaluating Logical Consistency
Mind the instructions: a holistic evaluation of consistency and interactions in prompt-based learning [Paper](https://arxiv.org/abs/2310.13486) \
Rethinking benchmark and contamination for language models with rephrased samples [Paper](https://arxiv.org/abs/2311.04850) [Code](https://github.com/lm-sys/llm-decontaminator) \
From Form(s) to Meaning: Probing the semantic depths of language models using multisense consistency [Paper](https://arxiv.org/abs/2404.12145) [Code](https://github.com/facebookresearch/multisense_consistency) \
Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity [Paper](https://arxiv.org/abs/2104.08786) [Code](https://github.com/yaolu/Ordered-Prompt) \
On sensitivity of learning with limited labelled data to the effects of randomness: Impact of interactions and systematic choices [Paper](https://arxiv.org/abs/2402.12817) \
Benchmark self-evolving: A multi-agent framework for dynamic llm evaluation [Paper](https://arxiv.org/abs/2402.11443) [Code](https://github.com/nanshineloong/self-evolving-benchmark) \
Exploring multilingual human value concepts in large language models: Is value alignment consistent, transferable and controllable across languages? [Paper](https://arxiv.org/abs/2402.18120) [Code](https://github.com/shaoyangxu/Multilingual-Human-Value-Concepts) \
**Fool your (vision and) language model with embarrassingly simple permutations** [Paper](https://arxiv.org/abs/2310.01651) [Code](https://github.com/ys-zong/FoolyourVLLMs) \
Large language models are not robust multiple choice selectors [Paper](https://arxiv.org/abs/2309.03882) [Code](https://github.com/chujiezheng/LLM-MCQ-Bias)

## 3. Evaluating Invariance from Irrelevant Information
Large language models can be easily distracted by irrelevant context [Paper](https://arxiv.org/abs/2302.00093) \
How easily do irrelevant inputs skew the responses of large language models? [Paper](https://arxiv.org/abs/2404.03302) [Code](https://github.com/Di-viner/LLM-Robustness-to-Irrelevant-Information) \
Lost in the middle: How language models use long context [Paper](https://arxiv.org/abs/2307.03172) [Code](https://github.com/nelson-liu/lost-in-the-middle) \
Making retrieval-augmented language models robust to irrelevant context [Paper](https://arxiv.org/abs/2310.01558) [Code](https://github.com/oriyor/ret-robust) \
Towards AI-complete question answering: A set of prerequisite toy tasks [Paper](https://arxiv.org/abs/1502.05698) [Code](https://github.com/qapitan/babi-marcus) \
CLUTRR: A diagnostic benchmark for inductive reasoning from text [Paper](https://arxiv.org/abs/1908.06177) [Code](https://github.com/facebookresearch/clutrr) \
Transformers as soft reasoners over language [Paper](https://arxiv.org/abs/2002.05867) [Code](https://github.com/allenai/ruletaker) \
Do prompt-based models really understand the meaning of their prompts? [Paper](https://arxiv.org/abs/2109.01247) [Code](https://github.com/awebson/prompt_semantics) \
**MileBench: Benchmarking MLLMs in long context** [Paper](https://arxiv.org/abs/2404.18532) [Code](https://github.com/MileBench/MileBench) \
**Mementos: A comprehensive benchmark for multimodal large language model reasoning over image sequences** [Paper](https://arxiv.org/abs/2401.10529) [Code](https://github.com/umd-huang-lab/Mementos) \
**Seedbench-2: Benchmarking multimodal large language models** [Paper](https://arxiv.org/abs/2311.17092) [Code](https://github.com/AILab-CVC/SEED-Bench) \
**DEMON: Finetuning multimodal llms to follow zero-shot demonstrative instructions** [Paper](https://arxiv.org/abs/2308.04152) [Code](https://github.com/DCDmllm/Cheetah)

