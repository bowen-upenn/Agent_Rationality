# This is the official repository of the paper "Multi-Modal and Multi-Agent Systems Meet Rationality: A Survey"

This survey is the first to specifically examine the increasingly important relations between **rationality** and **multi-modal and multi-agent systems**, identifying their advancements over single-agent and single-modal baselines in terms of rationality, and discussing open problems and future directions. We define four axioms we expect a rational agent or agent systems should satisfy: 
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

Each field of research in the figure above, such as knowledge retrieval or neuro-symbolic reasoning, addresses one or more fundamental axioms for rational thinking. These requirements are typically intertwined; therefore, an approach that enhances one aspect of rationality often inherently improves others simultaneously. We include all related work in our survey below, categorized by their fields. **Bold fonts are used to mark works that involve multi-modalities.** In their original writings, most existing studies do not explicitly base their frameworks on rationality. Our analysis aims to reinterpret these works through the lens of our four axioms of rationality, offering a novel perspective that bridges existing methodologies with rational principles.

### Knowledge Retrieval
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













