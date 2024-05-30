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

We include all related work in our survey below, categorized by their fields. **Bold fonts are used to mark work that involve multi-modalities.** In their original writings, most existing studies do not explicitly base their frameworks on rationality. Our analysis aims to reinterpret these works through the lens of our four axioms of rationality, offering a novel perspective that bridges existing methodologies with rational principles.

## Knowledge Retrieval
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
**JEPA: A Path Towards Autonomous Machine Intelligence** [Paper](https://openreview.net/pdf?id=BZ5a1r-kVsf）\
Voyager: An open-ended embodied agent with large language models [Paper](https://arxiv.org/abs/2305.16291) [Code](https://github.com/MineDojo/Voyager) \
Ghost in the Minecraft: Generally capable agents for open-world enviroments via large language models with text-based knowledge and memory [Paper](https://arxiv.org/abs/2305.17144) [Code](https://github.com/OpenGVLab/GITM) \
**Objective-Driven AI** [Slides](https://www.ece.uw.edu/wp-content/uploads/2024/01/lecun-20240124-uw-lyttle.pdf) \
**LWM: World Model on Million-Length Video And Language With RingAttention** [Paper](https://arxiv.org/abs/2402.08268) [Code](https://github.com/LargeWorldModel/LWM) \
**Sora: Video generation models as world simulators** [Website](https://openai.com/index/video-generation-models-as-world-simulators/) \
**IWM: Learning and Leveraging World Models in Visual Representation Learning** [Paper](https://arxiv.org/pdf/2403.00504) \
**CubeLLM: Language-Image Models with 3D Understanding** [Paper](https://arxiv.org/abs/2405.03685) [Code](https://github.com/NVlabs/Cube-LLM) 

## Using Tools
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


















