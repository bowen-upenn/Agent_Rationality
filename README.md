# This is the official repository of the paper "Multi-Modal and Multi-Agent Systems Meet Rationality: A Survey"

This survey is the first to specifically examine the increasingly important relations between **rationality** and **multi-modal and multi-agent systems**, identifying their advancements over single-agent and single-modal baselines in terms of rationality, and discussing open problems and future directions. 

**Rationality** is the quality of being guided by reason, characterized by logical thinking and decision-making that align with evidence and logical rules. This quality is essential for effective problem-solving, as it ensures that solutions are well-founded and systematically derived. We define four axioms we expect a rational agent or agent systems should satisfy: 
- **Information grounding**
  
  The decision of a rational agent is grounded on the physical and factual reality. In order to make a sound decision, the agent must be able to integrate sufficient and accurate information from different sources and modalities grounded in reality without hallucination.
  
- **Orderability of preference**

  When comparing alternatives in a decision scenario, a rational agent can rank the options based on the current state and ultimately select the most preferred one based on the expected outcomes. The orderability of preferences ensures the agent can make consistent and logical choices when faced with multiple alternatives. LLM-based evaluations heavily rely on this property.
  
- **Independence from irrelevant context**

  The agent's preference should not be influenced by information irrelevant to the decision problem at hand. LLMs have been shown to exhibit irrational behavior when presented with irrelevant context, leading to confusion and suboptimal decisions. To ensure rationality, an agent must be able to identify and disregard irrelevant information, focusing solely on the factors that directly impact the decision-making processes.

- **Invariance across logically equivalent representations**.

  The preference of a rational agent remains invariant across equivalent representations of the decision problem, regardless of specific wordings or modalities.

<p align="center">
<img src=tree.png />
</p>

Each field of research in the figure above, such as knowledge retrieval or neuro-symbolic reasoning, addresses one or more fundamental axioms for rational thinking. These requirements are typically intertwined; therefore, an approach that enhances one aspect of rationality often inherently improves others simultaneously.

We include all related works in our survey below, categorized by their fields. **Bold fonts are used to mark work that involve multi-modalities.** In their original writings, most existing studies do not explicitly base their frameworks on rationality. Our analysis aims to reinterpret these works through the lens of our four axioms of rationality, offering a novel perspective that bridges existing methodologies with rational principles.

## Knowledge Retrieval
The parametric nature of LLMs fundamentally limits how much information they can hold. A multi-modal and/or multi-agent system can include planning agents in its framework, which is akin to the System 2 process that can determine how and where to retrieve external knowledge, and what specific information to acquire. Additionally, the system can have summarizing agents that utilize retrieved knowledge to enrich the system's language outputs with better factuality. 

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

## Multi-Modal Foundation Models
As a picture is worth a thousand words, multi-modal approaches aim to improve the information grounding across various channels like language and vision. By incorporating multi-modal agents, multi-agent systems can greatly expand their capabilities, enabling a richer, more accurate, and contextually aware interpretation of environment. MMFMs are also particularly adept at promoting invariance by processing multi-modal data in an unified representation. Specifically, their large-scale cross-modal pretraining stage seamlessly tokenizes both vision and language inputs into a joint hidden embedding space, learning cross-modal correlations through a data-driven approach. 

Generating output texts from input images requires only a single inference pass, which is quick and straightforward, aligning closely with the [System 1 process](https://en.wikipedia.org/wiki/Dual_process_theory) of fast and automatic thinking. RLHF and visual instruction-tuning enable more multi-round human-agent interactions and collaborations with other agents. This opens the possibility of subsequent research on the [System 2 process](https://en.wikipedia.org/wiki/Dual_process_theory) in MMFMs.

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

## Large World Models
**JEPA: A Path Towards Autonomous Machine Intelligence** [Paper](https://openreview.net/pdf?id=BZ5a1r-kVsf) \
Voyager: An open-ended embodied agent with large language models [Paper](https://arxiv.org/abs/2305.16291) [Code](https://github.com/MineDojo/Voyager) \
Ghost in the Minecraft: Generally capable agents for open-world enviroments via large language models with text-based knowledge and memory [Paper](https://arxiv.org/abs/2305.17144) [Code](https://github.com/OpenGVLab/GITM) \
**Objective-Driven AI** [Slides](https://www.ece.uw.edu/wp-content/uploads/2024/01/lecun-20240124-uw-lyttle.pdf) \
**LWM: World Model on Million-Length Video And Language With RingAttention** [Paper](https://arxiv.org/abs/2402.08268) [Code](https://github.com/LargeWorldModel/LWM) \
**Sora: Video generation models as world simulators** [Website](https://openai.com/index/video-generation-models-as-world-simulators/) \
**IWM: Learning and Leveraging World Models in Visual Representation Learning** [Paper](https://arxiv.org/pdf/2403.00504) \
**CubeLLM: Language-Image Models with 3D Understanding** [Paper](https://arxiv.org/abs/2405.03685) [Code](https://github.com/NVlabs/Cube-LLM) 

## Tool Utilizations
A multi-agent system can coordinate agents understanding when and which tool to use, which modality of information the tool should expect, how to call the corresponding API, and how to incorporate outputs from the API calls, which anchors subsequent reasoning processes with more accurate information beyond their parametric memory. Besides, using tools require translating natural language queries into API calls with predefined syntax. Once the planning agent has determined the APIs and their input arguments, the original queries that may contain irrelevant contexts become invisible to the tools, and the tools will ignore any variance in the original queries as long as they share the equivalent underlying logic, promiting the invariance property.

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


## Web Agents
WebGPT: Browser-assisted question-answering with human feedback [Paper](https://arxiv.org/abs/2112.09332) \
WebShop: Towards scalable real-world web interaction with grounded language agents [Paper](https://arxiv.org/abs/2207.01206) [Code](https://github.com/princeton-nlp/WebShop) \
**Pix2Act: From pixels to UI actions: Learning to follow instructions via graphical user interfaces** [Paper](https://arxiv.org/abs/2306.00245) [Code](https://github.com/google-deepmind/pix2act) \
**WebGUM: Multimodal web navigation with instruction-finetuned foundation models** [Paper](https://arxiv.org/abs/2305.11854) [Code] \
Mind2Web: Towards a Generalist Agent for the Web [Paper](https://arxiv.org/abs/2306.06070) [Code](https://github.com/OSU-NLP-Group/Mind2Web) \
WebAgent: A real-world webagent with planning, long context understanding, and program synthesis [Paper](https://arxiv.org/abs/2307.12856) \
**CogAgent: A visual language model for GUI agents** [Paper](https://arxiv.org/abs/2312.08914) [Code](https://github.com/THUDM/CogAgent) \
**SeeAct: Gpt-4v (ision) is a generalist web agent, if grounded** [Paper](https://arxiv.org/abs/2401.01614) [Code](https://github.com/OSU-NLP-Group/SeeAct)


## LLM-Based Evaluation
ChatEval: Towards better llm-based evaluators through multi-agent debate [Paper](https://arxiv.org/abs/2308.07201) [Code](https://github.com/thunlp/ChatEval) \
Benchmarking foundation models with language-model-as-an-examiner [Paper](https://arxiv.org/abs/2306.04181) \
CoBBLEr: Benchmarking cognitive biases in large language models as evaluators [Paper](https://arxiv.org/abs/2309.17012) [Code](https://github.com/minnesotanlp/cobbler) \
Large Language Models are Inconsistent and Biased Evaluators [Paper](https://arxiv.org/pdf/2405.01724)


## Neuro-Symbolic Reasoning
Neural-symbolic reasoning is another promising approach to achieving consistent ordering of preferences and invariance by combining the strengths of languages and symbolic logic in a multi-agent system. A multi-agent system incorporating symbolic modules can not only understand language queries but also solve them with a level of consistency, providing a faithful and transparent reasoning process based on well-defined rules that adhere to logical principles, which is unachievable by LLMs alone within the natural language space. Neuro-Symbolic modules also expect standardized input formats. This layer of abstraction enhances the independence from irrelevant contexts and maintains the invariance of LLMs when handling natural language queries.

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


## Self-Reflection, Multi-Agent Debate, and Collboration
Due to the probabilistic outputs of LLMs, which resemble the rapid, non-iterative nature of human System 1 cognition, ensuring preference orderability and invariance is challenging. In contrast, algorithms that enable self-reflection and multi-agent systems that promote debate and consensus can slow down the thinking process and help align outputs more closely with the deliberate and logical decision-making typical of System 2 processes, thus enhancing rational reasoning in agents.

Collaborative approaches allow each agent in a system to compare and rank its preference on choices from its own or from other agents through critical judgments. It helps enable the system to discern and output the most dominant decision as a consensus, thereby improving the orderability of preference. At the same time, through such a slow and critical thinking process, errors in initial responses or input prompts are more likely to be detected and corrected. 

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


## Prompting Strategy and Memory
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












