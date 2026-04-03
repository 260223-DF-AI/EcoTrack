# EcoTrack
Image recognition, classifying bird images to determine species, which is then mapped to their extinction/vulnerability, which could be pulled from IUCN using an applied API key.
- **The Problem**: A conservation group needs to monitor endangered species in trail camera footage.
- **Vision Task**: Identify animal species (e.g., `Deer`, `Bear`, `Endangered Species X`).
- **Reasoning Task**: Use **Graph of Thought (GoT)** to analyze location/weather logs and **ReAct** to draft a conservation alert if an endangered species is identified in an unusual location.

We might be fine with the current bird images if all we need to do with the CNN is classify what kind of bird it is. The locational analysis seems to be up to a separate but connected LLM using Graph of Thought prompting that takes the bird and some provided location and evaluates how appropriate it is for that bird to be there. If we need to train the LLM on what an appropriate location is there might be data that can be sourced from the IUCN API about bird locations and migratory patterns.

# Resources

## Images
[Kaggle - Caltech](https://www.kaggle.com/datasets/veeralakrishna/200-bird-species-with-11788-images?resource=download)

## Endangered Species Lists
- [IUCN Red List](https://www.iucnredlist.org/)
- [GBIF species database](https://www.gbif.org/)
- [WWF species lists](https://www.worldwildlife.org/)

## Further Reading
- [Graph-of-Thought Prompting](https://wandb.ai/sauravmaheshkar/prompting-techniques/reports/Chain-of-thought-tree-of-thought-and-graph-of-thought-Prompting-techniques-explained---Vmlldzo4MzQwNjMx)
