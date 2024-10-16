# This is the official repository of the paper ["Towards Rationality in Language and Multimodal Agents: A Survey"](https://arxiv.org/abs/2406.00252)

## [10/14/2024] Update: We are scheduling an update of the manuscript within this week. 

[![Arxiv](https://img.shields.io/badge/ArXiv-Full_Paper-B31B1B)](https://arxiv.org/abs/2406.00252)
[![Google Scholar](https://img.shields.io/badge/Google_Scholar-Cite_Our_Paper-4085F4)](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C39&q=Multi-Modal+and+Multi-Agent+Systems+Meet+Rationality%3A+A+Survey&btnG=)

This survey is the first to specifically examine how **multi-modal and multi-agent systems** 🤖 are advancing towards **rationality** 🧠 in cognitive science, identifying their advancements over single-agent and language-only baselines, and discuss open problems in evaluating rationality beyong accuracy.

- [Define Rationality](#define-rationality)
- [Towards Rationality through Multi-Modal and Multi-Agent Systems](#towards-rationality-through-multi-modal-and-multi-agent-systems)
- [Evaluating Rationality of Agents](#evaluating-rationality-of-agents)

The fields of multi-modal and multi-agent systems are rapidly evolving, so we highly encourage researchers who want to promote their amazing works on this dynamic repository to submit a pull request and make updates. 💜

We have a concurrent work ***A Peek into Token Bias: Large Language Models Are Not Yet Genuine Reasoners*** [Paper]() [Code](https://github.com/bowen-upenn/llm_logical_fallacies) to be released in the upcoming week as well. The survey critics the limitations of existing benchmarks on evaluating rationality, and this paper goes beyond accuracy and reconceptualize the evaluation of reasoning capabilities in LLMs into a general, statistically rigorous framework of testable hypotheses. It is designed to determine whether LLMs are capable of genuine reasoning or if they primarily rely on token bias. Our findings with statistical guarantees suggest that LLMs struggle with probabilistic reasoning.

## Citations
A bunny 🐰 will be happy if you could cite our work.

    @misc{jiang2024multimodal,
          title={Multi-Modal and Multi-Agent Systems Meet Rationality: A Survey}, 
          author={Bowen Jiang and Yangxinyu Xie and Xiaomeng Wang and Weijie J. Su and Camillo J. Taylor and Tanwi Mallick},
          year={2024},
          eprint={2406.00252},
          archivePrefix={arXiv},
          primaryClass={cs.AI}
    }

## Define Rationality
**Rationality** is the quality of being guided by reason, characterized by logical thinking and decision-making that align with evidence and logical rules. This quality is essential for effective problem-solving, as it ensures that solutions are well-founded and systematically derived. We define four axioms we expect a rational agent or agent systems should satisfy: 
- **Information grounding**
  
  The decision of a rational agent is grounded on the physical and factual reality. In order to make a sound decision, the agent must be able to integrate sufficient and accurate information from different sources and modalities grounded in reality without hallucination.
  
- **Orderability of preference**

  When comparing alternatives in a decision scenario, a rational agent can rank the options based on the current state and ultimately select the most preferred one based on the expected outcomes. The orderability of preferences ensures the agent can make consistent and logical choices when faced with multiple alternatives. LLM-based evaluations heavily rely on this property.
  
- **Independence from irrelevant context**

  The agent's preference should not be influenced by information irrelevant to the decision problem at hand. LLMs have been shown to exhibit irrational behavior when presented with irrelevant context, leading to confusion and suboptimal decisions. To ensure rationality, an agent must be able to identify and disregard irrelevant information, focusing solely on the factors that directly impact the decision-making processes.

- **Invariance across logically equivalent representations**.

  The preference of a rational agent remains invariant across equivalent representations of the decision problem, regardless of specific wordings or modalities.


## Towards Rationality through Multi-Modal and Multi-Agent Systems
<p align="center">
<img src=evol_tree.png />
</p>

Each field of research in the figure above, such as knowledge retrieval or neuro-symbolic reasoning, addresses one or more fundamental axioms for rational thinking. These requirements are typically intertwined; therefore, an approach that enhances one aspect of rationality often inherently improves others simultaneously.

We include all related works in our survey below, categorized by their fields. **Bold fonts are used to mark work that involve multi-modalities.** In their original writings, most existing studies do not explicitly base their frameworks on rationality. Our analysis aims to reinterpret these works through the lens of our four axioms of rationality, offering a novel perspective that bridges existing methodologies with rational principles.

### Knowledge Retrieval
_The parametric nature of LLMs fundamentally limits how much information they can hold. A multi-modal and/or multi-agent system can include planning agents in its framework, which is akin to the System 2 process that can determine how and where to retrieve external knowledge, and what specific information to acquire. Additionally, the system can have summarizing agents that utilize retrieved knowledge to enrich the system's language outputs with better factuality._

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

### Multi-Modal Foundation Models
_As a picture is worth a thousand words, multi-modal approaches aim to improve the information grounding across various channels like language and vision. By incorporating multi-modal agents, multi-agent systems can greatly expand their capabilities, enabling a richer, more accurate, and contextually aware interpretation of environment. MMFMs are also particularly adept at promoting invariance by processing multi-modal data in an unified representation. Specifically, their large-scale cross-modal pretraining stage seamlessly tokenizes both vision and language inputs into a joint hidden embedding space, learning cross-modal correlations through a data-driven approach._

_Generating output texts from input images requires only a single inference pass, which is quick and straightforward, aligning closely with the System 1 process of fast and automatic thinking in **[dual-process theories](https://en.wikipedia.org/wiki/Dual_process_theory)**. RLHF and visual instruction-tuning enable more multi-round human-agent interactions and collaborations with other agents. This opens the possibility of subsequent research on the System 2 process in MMFMs._

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

### Large World Models
**JEPA: A Path Towards Autonomous Machine Intelligence** [Paper](https://openreview.net/pdf?id=BZ5a1r-kVsf) \
Voyager: An open-ended embodied agent with large language models [Paper](https://arxiv.org/abs/2305.16291) [Code](https://github.com/MineDojo/Voyager) \
Ghost in the Minecraft: Generally capable agents for open-world enviroments via large language models with text-based knowledge and memory [Paper](https://arxiv.org/abs/2305.17144) [Code](https://github.com/OpenGVLab/GITM) \
**Objective-Driven AI** [Slides](https://www.ece.uw.edu/wp-content/uploads/2024/01/lecun-20240124-uw-lyttle.pdf) \
**LWM: World Model on Million-Length Video And Language With RingAttention** [Paper](https://arxiv.org/abs/2402.08268) [Code](https://github.com/LargeWorldModel/LWM) \
**Sora: Video generation models as world simulators** [Website](https://openai.com/index/video-generation-models-as-world-simulators/) \
**IWM: Learning and Leveraging World Models in Visual Representation Learning** [Paper](https://arxiv.org/pdf/2403.00504) \
**CubeLLM: Language-Image Models with 3D Understanding** [Paper](https://arxiv.org/abs/2405.03685) [Code](https://github.com/NVlabs/Cube-LLM) 

### Tool Utilizations
_A multi-agent system can coordinate agents understanding when and which tool to use, which modality of information the tool should expect, how to call the corresponding API, and how to incorporate outputs from the API calls, which anchors subsequent reasoning processes with more accurate information beyond their parametric memory. Besides, using tools require translating natural language queries into API calls with predefined syntax. Once the planning agent has determined the APIs and their input arguments, the original queries that may contain irrelevant contexts become invisible to the tools, and the tools will ignore any variance in the original queries as long as they share the equivalent underlying logic, promiting the invariance property._

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


### Web Agents
WebGPT: Browser-assisted question-answering with human feedback [Paper](https://arxiv.org/abs/2112.09332) \
WebShop: Towards scalable real-world web interaction with grounded language agents [Paper](https://arxiv.org/abs/2207.01206) [Code](https://github.com/princeton-nlp/WebShop) \
**Pix2Act: From pixels to UI actions: Learning to follow instructions via graphical user interfaces** [Paper](https://arxiv.org/abs/2306.00245) [Code](https://github.com/google-deepmind/pix2act) \
**WebGUM: Multimodal web navigation with instruction-finetuned foundation models** [Paper](https://arxiv.org/abs/2305.11854) [Code] \
Mind2Web: Towards a Generalist Agent for the Web [Paper](https://arxiv.org/abs/2306.06070) [Code](https://github.com/OSU-NLP-Group/Mind2Web) \
WebAgent: A real-world webagent with planning, long context understanding, and program synthesis [Paper](https://arxiv.org/abs/2307.12856) \
**CogAgent: A visual language model for GUI agents** [Paper](https://arxiv.org/abs/2312.08914) [Code](https://github.com/THUDM/CogAgent) \
**SeeAct: Gpt-4v (ision) is a generalist web agent, if grounded** [Paper](https://arxiv.org/abs/2401.01614) [Code](https://github.com/OSU-NLP-Group/SeeAct)


### LLM-Based Evaluation
ChatEval: Towards better llm-based evaluators through multi-agent debate [Paper](https://arxiv.org/abs/2308.07201) [Code](https://github.com/thunlp/ChatEval) \
Benchmarking foundation models with language-model-as-an-examiner [Paper](https://arxiv.org/abs/2306.04181) \
CoBBLEr: Benchmarking cognitive biases in large language models as evaluators [Paper](https://arxiv.org/abs/2309.17012) [Code](https://github.com/minnesotanlp/cobbler) \
Large Language Models are Inconsistent and Biased Evaluators [Paper](https://arxiv.org/pdf/2405.01724)


### Neuro-Symbolic Reasoning
_Neural-symbolic reasoning is another promising approach to achieving consistent ordering of preferences and invariance by combining the strengths of languages and symbolic logic in a multi-agent system. A multi-agent system incorporating symbolic modules can not only understand language queries but also solve them with a level of consistency, providing a faithful and transparent reasoning process based on well-defined rules that adhere to logical principles, which is unachievable by LLMs alone within the natural language space. Neuro-Symbolic modules also expect standardized input formats. This layer of abstraction enhances the independence from irrelevant contexts and maintains the invariance of LLMs when handling natural language queries._

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


### Self-Reflection, Multi-Agent Debate, and Collboration
_Due to the probabilistic outputs of LLMs, which resemble the rapid, non-iterative nature of human System 1 cognition, ensuring preference orderability and invariance is challenging. In contrast, algorithms that enable self-reflection and multi-agent systems that promote debate and consensus can slow down the thinking process and help align outputs more closely with the deliberate and logical decision-making typical of System 2 processes, thus enhancing rational reasoning in agents._

_Collaborative approaches allow each agent in a system to compare and rank its preference on choices from its own or from other agents through critical judgments. It helps enable the system to discern and output the most dominant decision as a consensus, thereby improving the orderability of preference. At the same time, through such a slow and critical thinking process, errors in initial responses or input prompts are more likely to be detected and corrected._

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


### Prompting Strategy and Memory
_Works in section are not all directly related to multi-modal or multi-agent systems_ \
CoT: Chain-of-thought prompting elicits reasoning in large language models [Paper](https://arxiv.org/abs/2201.11903) \
Language model cascades [Paper](https://arxiv.org/abs/2207.10342) [Code](https://github.com/google-research/cascades) \
ReAct: Synergizing reasoning and acting in language models [Paper](https://arxiv.org/abs/2210.03629) [Code](https://github.com/ysymyth/ReAct) \
Memorybank: Enhancing large language models with long-term memory [Paper](https://arxiv.org/abs/2305.10250) [Code](https://github.com/zhongwanjun/MemoryBank-SiliconFriend) \
Tree of thoughts: Deliberate problem solving with large language models [Paper](https://arxiv.org/abs/2305.10601) [Code](https://github.com/princeton-nlp/tree-of-thought-llm) \
**Beyond chain-of-thought, effective graph-of-thought reasoning in large language models** [Paper](https://arxiv.org/abs/2305.16582) [Code](https://github.com/Zoeyyao27/Graph-of-Thought) \
Graph of thoughts: Solving elaborate problems with large language models [Paper](https://arxiv.org/abs/2308.09687) [Code](https://github.com/spcl/graph-of-thoughts) \
MemoChat: Tuning llms to use memos for consistent long-range open-domain conversation [Paper](https://arxiv.org/abs/2308.08239) [Code](https://github.com/LuJunru/MemoChat) \
Retroformer: Retrospective large language agents with policy gradient optimization [Paper](https://arxiv.org/abs/2308.02151) [Code](https://github.com/weirayao/retroformer) \
FormatSpread: How I learned to start worrying about prompt formatting [Paper](https://arxiv.org/abs/2310.11324) [Code](https://github.com/msclar/formatspread) \
ADaPT: As-Needed Decomposition and Planning with Language Models [Paper](https://arxiv.org/abs/2311.05772) [Code](https://github.com/archiki/ADaPT) \
EureQA: How Easy is It to Fool Your Multimodal LLMs? An Empirical Analysis on Deceptive Prompts [Paper](https://arxiv.org/abs/2402.13220) \
Combating Adversarial Attacks with Multi-Agent Debate [Paper](https://arxiv.org/abs/2401.05998) \
MAD-Bench: How Easy is It to Fool Your Multimodal LLMs? An Empirical Analysis on Deceptive Prompts [Paper](https://arxiv.org/abs/2402.13220)


##
_This survey builds connections between multi-modal and multi-agent systems with rationality, guided by dual-process theories and the four axioms we expect a rational agent or agent systems should satisfy: information grounding, orderability of preference, independence from irrelevant context, and invariance across logically equivalent representations. Our findings suggest that the grounding can usually be enhanced by multi-modalities, world models, knowledge retrieval, and tool utilization. The remaining three axioms are typically intertwined, and we sometimes describe their collective characteristics informally using terms such as coherence, consistency, and trustworthiness. These axioms are simultaneously improved by achievements in multi-modalities, tool utilization, neuro-symbolic reasoning, self-reflection, and multi-agent collaborations. These fields of research, by either **delibration that slows down the "thinking" process** or **abstraction that boils down tasks to their logical essence**, mimic the "System 2" thinking in human cognition, thereby enhancing the rationality of multi-agent systems in decision-making scenarios, compared to single-agent language-only baselines that resemble the "System 1" process._


## Evaluating Rationality of Agents

[The choices of evaluation metrics are important](https://arxiv.org/abs/2304.15004). We find that most benchmarks predominantly focus on the accuracy of the final performance, ignoring the most interesting intermediate reasoning steps and the concept of rationality. **A promising direction is to create benchmarks specifically tailored to assess rationality, going beyond existing ones on accuracy.** These new benchmarks should avoid data contamination and emphasize tasks that demand consistent reasoning across diverse representations and domains. Besides, existing evaluations on rationality provide limited comparisons between multi-modal/multi-agent frameworks and single-agent baselines, thus failing to fully elucidate the advantages multi-modal/multi-agent frameworks can offer.

***🏳️‍🌈 Our Work 🏳️‍⚧️ A Peek into Token Bias: Large Language Models Are Not Yet Genuine Reasoners*** [Paper]() [Code](https://github.com/bowen-upenn/llm_logical_fallacies) \
is designed to determine whether LLMs are capable of genuine reasoning or if they primarily rely on token bias. We go beyond accuracy and reconceptualize the evaluation of reasoning capabilities in LLMs into a general, statistically rigorous framework of testable hypotheses. Our findings with statistical guarantees suggest that LLMs struggle with probabilistic reasoning, with apparent performance improvements largely attributable to token bias. To be released in the upcoming week.


### General Benchmarks or Evaluation Metrics
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


### Adapting Cognitive Psychology Experiments
Using cognitive psychology to understand GPT-3 [Paper](https://arxiv.org/abs/2206.14576) [Code](https://github.com/marcelbinz/GPT3goesPsychology) \
On the dangers of stochastic parrots: Can language models be too big? [Paper](https://dl.acm.org/doi/10.1145/3442188.3445922)


### Testing Grounding against Hallucination
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


### Testing the Orderability of Preference
Large language models are not robust multiple choice selectors [Paper](https://arxiv.org/abs/2309.03882) [Code](https://github.com/chujiezheng/LLM-MCQ-Bias) \
Leveraging large language models for multiple choice question answering [Paper](https://arxiv.org/abs/2210.12353) [Code](https://github.com/BYU-PCCL/leveraging-llms-for-mcqa)


### Testing the Principle of Invariance
Mind the instructions: a holistic evaluation of consistency and interactions in prompt-based learning [Paper](https://arxiv.org/abs/2310.13486) \
Rethinking benchmark and contamination for language models with rephrased samples [Paper](https://arxiv.org/abs/2311.04850) [Code](https://github.com/lm-sys/llm-decontaminator) \
From Form(s) to Meaning: Probing the semantic depths of language models using multisense consistency [Paper](https://arxiv.org/abs/2404.12145) [Code](https://github.com/facebookresearch/multisense_consistency) \
Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity [Paper](https://arxiv.org/abs/2104.08786) [Code](https://github.com/yaolu/Ordered-Prompt) \
On sensitivity of learning with limited labelled data to the effects of randomness: Impact of interactions and systematic choices [Paper](https://arxiv.org/abs/2402.12817) \
Benchmark self-evolving: A multi-agent framework for dynamic llm evaluation [Paper](https://arxiv.org/abs/2402.11443) [Code](https://github.com/nanshineloong/self-evolving-benchmark) \
Exploring multilingual human value concepts in large language models: Is value alignment consistent, transferable and controllable across languages? [Paper](https://arxiv.org/abs/2402.18120) [Code](https://github.com/shaoyangxu/Multilingual-Human-Value-Concepts) \
**Fool your (vision and) language model with embarrassingly simple permutations** [Paper](https://arxiv.org/abs/2310.01651) [Code](https://github.com/ys-zong/FoolyourVLLMs) \
Large language models are not robust multiple choice selectors [Paper](https://arxiv.org/abs/2309.03882) [Code](https://github.com/chujiezheng/LLM-MCQ-Bias)


### Testing Independence from Irrelevant Context
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
