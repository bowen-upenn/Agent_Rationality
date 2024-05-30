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


Each field of research in the figure above, such as knowledge retrieval or neuro-symbolic reasoning, addresses one or more fundamental requirements for rational thinking. These rationality requirements are typically intertwined; therefore, an approach that enhances one aspect of rationality often inherently improves others simultaneously. We include all related work in our survey, categorized by their fields. In their original writings, most existing studies do not explicitly base their frameworks on rationality. Our analysis aims to reinterpret these works through the lens of our four axioms of rationality, offering a novel perspective that bridges existing methodologies with rational principles.
