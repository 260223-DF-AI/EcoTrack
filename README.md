# EcoTrack

## Description
Fine-tuned a torchvision model to identify 90 different species and map the output to their respective endangered status according to IUCN. Species classified as endangered are provided as input to a large language model, along with geographic information. The LLM uses Graph of Thought and ReAct to determine whether the endangered animal is in an unusual location, which prompts a conservation alert.

## Problem
- **The Problem**: A conservation group needs to monitor endangered species in trail camera footage.
- **Vision Task**: Identify animal species (e.g., `Deer`, `Bear`, `Endangered Species X`).
- **Reasoning Task**: Use **Graph of Thought (GoT)** to analyze location/weather logs and **ReAct** to draft a conservation alert if an endangered species is identified in an unusual location.

## Approach
Fine-tuning a torchvision ResNet model to classify animal images among 90 classes. If the identified species is classified as worse than "Least Concern" according to the IUCN, proceed to an LLM that determines whether the provided image location and/or weather is unusual for that species, issuing a "Conservation Alert" when necessary.
Animals with different "subspecies" with different endangered statuses are all grouped according to "worst case scenario," assuming the provided image is the species at most risk. This will require human review for conservation alerts. We would rather a false positive for a conservation alert than a false negative.

## Key Features
- Torchvision-based CNN with residual connections for classifying animal images between 90 species
    - Cross Entropy Loss for multiple classes
    - ResNet50 base model 
    - LLM with Graph-of-Thought prompting trained on endangered bird species locations

## Limitations
- Due to the generality of the species trained on, the image classifier will not discern between specific kinds that might have different endangered statuses. We elected to err on the side of caution and raise an alert assuming the image is of the most threatened species of its family, preferring false positives over false negatives.

# Resources

## Endangered Status
- NE: NOT EVALUATED
- DD: DATA DEFICIENT
- LC: LEAST CONCERN
- NT: NEAR THREATENED
- VU: VULNERABLE
- EN: ENDANGERED
- CR: CRITICALLY ENDANGERED
- EW: EXTINCT IN THE WILD
- EX: EXTINCT
According to IUCN as of 4/15/2026

## Images
- [90 Animals, 5.4k Images](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals?select=animals)

## Endangered Species Lists
- [IUCN Red List](https://www.iucnredlist.org/)
- [GBIF species database](https://www.gbif.org/)
- [WWF species lists](https://www.worldwildlife.org/)



## Potential Resources (# Species, # Images)
- [22 Florida Species, 100k](https://www.crcv.ucf.edu/research/projects/florida-wildlife-camera-trap-dataset/)
- [11 Endangered Species, 28k](https://lote-animal.github.io/)
- [10 Species, 26k](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
- [100 Species, 40k](https://www.scidb.cn/en/detail?dataSetId=e2ebd46cb1304a82bab54a8873cb3004)


## Further Reading
- [Graph-of-Thought Prompting](https://wandb.ai/sauravmaheshkar/prompting-techniques/reports/Chain-of-thought-tree-of-thought-and-graph-of-thought-Prompting-techniques-explained---Vmlldzo4MzQwNjMx)
- [Graph-of-Thought Walkthrough](https://github.com/spcl/graph-of-thoughts)
- [ResNet Example 1](https://medium.com/@anglilian/image-classification-with-resnet-pytorch-1e48a4c33905)
- [ResNet Example 2](https://medium.com/@engr.akhtar.awan/how-to-fine-tune-the-resnet-50-model-on-your-target-dataset-using-pytorch-187abdb9beeb)
- [Transfer learning and unfreezing layers](https://www.tensorflow.org/tutorials/images/transfer_learning)


# Requirements

#### 1. Computer Vision (Moderation Engine)

- **Model Architecture**: Use a custom `nn.Module` or a fine-tuned ResNet backbone.
- **Optimization**: Implement **Mixed Precision (AMP)** and **Early Stopping** based on a validation set.
- **Deployment**: Launch the training job via **SageMaker Script Mode**. Artifacts must be registered in the **SageMaker Model Registry**.

#### 2. Advanced Prompting (Reasoning & Response)

- **Cognitive Blueprint**: You must implement at least two Prompting Paradigms (e.g., **CoT**, **ToT**, **GoT**, or **Dialog State**).
- **Orchestration**: Implement a **ReAct** (Reasoning + Acting) loop to finalize the decision based on vision + text inputs.
- **Safety**: Implement an **Input Sanitization** layer (Week 3 TUE) to prevent malicious prompt injection.

#### 3. Integrated Service (System API)

- **FastAPI**: A `POST /analyze` endpoint that accepts a file and text input.
- **Persistence**: Store all audit logs in a **PostgreSQL** database, including CNN confidence scores, LLM reasoning, and the final generated output.

---

### Delivery Schedule (16-Hour Pair Milestone)

| Milestone | Expected Deliverables |
| :--- | :--- |
| **Phase 1 (4h)** | DB schema design, FastAPI skeleton, and track selection finalized. |
| **Phase 2 (6h)** | SageMaker training job successful and model registered in registry. |
| **Phase 3 (4h)** | Prompt logic (CoT/ReAct) implemented and tested via Boto3. |
| **Phase 4 (2h)** | End-to-end integration test; Final documentation and demo preparation. |

---

### Success Criteria

1. A **Git repository** with professional branching and commit messages.
2. A **runnable FastAPI service** connecting vision models and LLM prompts.
3. A **SageMaker Model Registry Entry** confirming your model's versioning.
4. A **database record** showing the complete "audit trail" for a submitted request.



#### Possible other species
201 201.Empty_Forest
202 202.Empty_Sky
203 203.Person
204 204.Deer
205 205.Fox
206 206.Bear
207 207.Squirrel
208 208.Wolf
209 209.Coyote
210 210.Chipmunk
211 211.Elk
212 212.Raccoon
213 213.Oposum
214 214.Rabbit
215 215.Frog