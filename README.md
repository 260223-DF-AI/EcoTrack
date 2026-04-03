# EcoTrack

## Problem
- **The Problem**: A conservation group needs to monitor endangered species in trail camera footage.
- **Vision Task**: Identify animal species (e.g., `Deer`, `Bear`, `Endangered Species X`).
- **Reasoning Task**: Use **Graph of Thought (GoT)** to analyze location/weather logs and **ReAct** to draft a conservation alert if an endangered species is identified in an unusual location.

## Approach
Training a torchvision model to classify bird images among 200 classes. If the identified species is classified any worse than "Least Concern" according to IUCN, proceed to an LLM that determines if the provided location of the image is unusual for that species, issuing a "Conservtion Alert" for it.

## Key Features
- Torchvision-based CNN for classifying bird images between 200 species
    - Cross Entropy Loss for multiple classes
- Finetuned LLM with Graph-of-Thought prompting trained on endangered bird species locations
    - Should only need to train/finetune on endangered bird species and their expected locations
    - Binary/Ternary output? No warning, Warning, and potential "Semi-Warning"
        - Some classes might be families or groups of species and have varying statuses, and require human review. 


# Resources

## Images
[Kaggle - Caltech](https://www.kaggle.com/datasets/veeralakrishna/200-bird-species-with-11788-images?resource=download)

## Endangered Species Lists
- [IUCN Red List](https://www.iucnredlist.org/)
- [GBIF species database](https://www.gbif.org/)
- [WWF species lists](https://www.worldwildlife.org/)

## Further Reading
- [Graph-of-Thought Prompting](https://wandb.ai/sauravmaheshkar/prompting-techniques/reports/Chain-of-thought-tree-of-thought-and-graph-of-thought-Prompting-techniques-explained---Vmlldzo4MzQwNjMx)
