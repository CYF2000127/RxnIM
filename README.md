# ReactionImgMLLM
This is the offical code of following paper "ReactionImgMLLM: A Multimodal Large Language Model for Reaction Image Data Extraction".

## Highlights
<p align="justify">
In this paper, we present ReactionImgMLLM, a multimodal large language model for different reaction image data extraction tasks such as reaction extraction task, condition OCR and role identification task. We first formulate these tasks into different task instructions. The model then aligns the task instructions with features extracted from reaction images. An LLM-based decoder can further make predictions based on these instructions. For the reaction extraction task, our model can achieve over 84%-92% soft match F1 score on multiple test sets, which significantly outperforms the previous works. 
The experiments also show the outstanding condition OCR and role identification abilities.
  
[comment]: <> ()
![visualization](figure/model.png)
<div align="center">
Overall Architecture of our ReactionImgMLLM.
</div> 
