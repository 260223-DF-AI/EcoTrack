# EcoTrack

## Description
Fine-tuned a torchvision model to identify bird species and map the output to their respective endangered status. Species classified as endangered are provided as input to a large language model, along with the image location. The LLM uses Graph of Thought and ReAct to determine whether the bird is in an unusual location, which prompts a conservation alert.

## Problem
- **The Problem**: A conservation group needs to monitor endangered species in trail camera footage.
- **Vision Task**: Identify animal species (e.g., `Deer`, `Bear`, `Endangered Species X`).
- **Reasoning Task**: Use **Graph of Thought (GoT)** to analyze location/weather logs and **ReAct** to draft a conservation alert if an endangered species is identified in an unusual location.

## Approach
Fine-tuning a torchvision ResNet model to classify bird images among 200 classes. If the identified species is classified as worse than "Least Concern" according to the IUCN, proceed to an LLM that determines whether the provided image location is unusual for that species, issuing a "Conservation Alert" when necessary.

## Key Features
- Torchvision-based CNN with residual connections for classifying bird images between 200 species
    - Cross Entropy Loss for multiple classes
    - Use torchvision.models.resnet
- Fine-tuned LLM with Graph-of-Thought prompting trained on endangered bird species locations
    - The Graph of Thought article under **Further Reading** mentions a `graph-of-thought` package we may be able to use depending on our LLM implementation.
    - Should only need to train/finetune on endangered bird species and their expected locations
    - Binary/Ternary output? No warning, Warning, and potential "Semi-Warning"
        - Some classes might be families or groups of species and have varying statuses, and require human review. 

## Limitations
- Does not seem to be able to classify birds in flight, from the few species I have tried comparing images in flight and not in flight with, our model classified most of them as "Not a Bird", sometimes it would be classified as the wrong bird specied, and very rarely would it correctly classified.


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
According to IUCN as of 4/2/2026
*If classes correlate to a family or group of species, default to the most vulnerable

## Images
[Kaggle - Caltech](https://www.kaggle.com/datasets/veeralakrishna/200-bird-species-with-11788-images?resource=download)

## Endangered Species Lists
- [IUCN Red List](https://www.iucnredlist.org/)
- [GBIF species database](https://www.gbif.org/)
- [WWF species lists](https://www.worldwildlife.org/)

## Bird Databases
- [Avibase](https://avibase.bsc-eoc.org/)

## Possible Resources
- [Endangered](https://ecos.fws.gov/ecp0/reports/ad-hoc-species-report?kingdom=V&kingdom=I&status=E&status=T&status=EmE&status=EmT&status=EXPE&status=EXPN&status=SAE&status=SAT&mapstatus=3&fcrithab=on&fstatus=on&fspecrule=on&finvpop=on&fgroup=on&header=Listed+Animals)
- [Bird model](https://huggingface.co/dennisjooo/Birds-Classifier-EfficientNetB2) 
- [General model](https://huggingface.co/google/efficientnet-b2)
- [Found on GitHub](https://github.com/Moddy2024/Bird-Classification?tab=readme-ov-file)
- [5 Mil Bird Images](https://ieee-dataport.org/documents/lasbird-large-scale-bird-recognition-dataset)
- [Bird Feature and Location Data](https://onlinelibrary.wiley.com/doi/full/10.1111/ele.13898)
- [Big Bird](https://rdm.uq.edu.au/files/6f45329e-eccc-4e1e-afac-8895ee4123ee)

## Non-Bird Resources (#, Endangered/Generic, # Images)
- [22 Generic Florida Species + Endangered Panther, 100k](https://www.crcv.ucf.edu/research/projects/florida-wildlife-camera-trap-dataset/)
- [11 Endangered Species, 28k](https://lote-animal.github.io/)
- [10 Generic Animals, 26k](https://www.kaggle.com/datasets/alessiocorrado99/animals10)
- [90 Generic Animals, 5.4k](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals?select=animals)
- [100 Generic Animals, 40k](https://www.scidb.cn/en/detail?dataSetId=e2ebd46cb1304a82bab54a8873cb3004)


## Further Reading
- [Graph-of-Thought Prompting](https://wandb.ai/sauravmaheshkar/prompting-techniques/reports/Chain-of-thought-tree-of-thought-and-graph-of-thought-Prompting-techniques-explained---Vmlldzo4MzQwNjMx)
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