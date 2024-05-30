# This is the official repository of the paper "Multi-Modal and Multi-Agent Systems Meet Rationality: A Survey"

This survey is the first to specifically examine the increasingly important relations between **rationality** and **multi-modal and multi-agent systems**, identifying their advancements over single-agent and single-modal baselines in terms of rationality, and discussing open problems and future directions. 

We define four axioms we expect a rational agent or agent systems should satisfy: 
- **Information grounding**
  
  The decision of a rational agent is grounded on the physical and factual reality. For example, a flight booking agent must accurately retrieve available airports without fabricating nonexistent ones. In order to make a sound decision, the agent must be able to integrate sufficient and accurate information from different sources and modalities grounded in reality without hallucination. While this requirement is generally not explicitly stated in the cognitive science literature when defining rationality, it is implicitly implied, as most humans have access to physical reality through multiple sensory signals.
  
- **Orderability of preference**

  When comparing alternatives in a decision scenario, a rational agent can rank the options based on the current state and ultimately select the most preferred one based on the expected outcomes. The orderability of preferences ensures the agent can make consistent and logical choices when faced with multiple alternatives. LLM-based evaluations heavily rely on this property.
  
- **Independence from irrelevant context**

  The agent's preference should not be influenced by information irrelevant to the decision problem at hand. LLMs have been shown to exhibit irrational behavior when presented with irrelevant context, leading to confusion and suboptimal decisions. To ensure rationality, an agent must be able to identify and disregard irrelevant information, focusing solely on the factors that directly impact the decision-making processes.

- **Invariance across logically equivalent representations**.

  The preference of a rational agent remains invariant across equivalent representations of the decision problem, regardless of specific wordings or modalities.
