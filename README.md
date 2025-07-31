## DINOv2 + UAD algorithm for assistive robot project [BRIN 2025]

# All credit to:
[Yihe Tang](https://tangyihe.com/)<sup>1</sup>, [Wenlong Huang](https://wenlong.page)<sup>1</sup>, [Yingke Wang](https://www.wykac.com/)<sup>1</sup>, [Chengshu Li](https://www.chengshuli.me/)<sup>1</sup>, [Roy Yuan](https://www.linkedin.com/in/ryuan19)<sup>1</sup>, [Ruohan Zhang](https://ai.stanford.edu/~zharu/)<sup>1</sup>, [Jiajun Wu](https://jiajunwu.com/)<sup>1</sup>, [Li Fei-Fei](https://profiles.stanford.edu/fei-fei-li)<sup>1</sup>


### Environment Setup
- (Optioal) If you are using [Omnigibson](https://behavior.stanford.edu/omnigibson/getting_started/installation.html) for rendering [Behavior1K](https://behavior.stanford.edu/behavior-1k) assets, or Blender for rendering [Objaverse](https://github.com/allenai/objaverse-xl) assets, please follow their installation guide respectively. 
  - Note the rendering libraries may have version conflicts with the data pipeline / model training code, consider using a separate env in that case. 
- (Optional) If you are using open-sourced [sentence-transformers](https://github.com/UKPLab/sentence-transformers?tab=readme-ov-file#installation) for language embedding, please follow their installation guide. 
  - We recommend installing from source
- Create your conda environment and install torch
  ```
  conda create -n uad python=3.9
  conda activate uad
  pip install torch torchvision torchaudio
  ```
- Install unsup-affordance in the same conda env
  ```
  git clone https://github.com/TangYihe/unsup-affordance.git
  cd unsup-affordance
  pip install -r requirements.txt
  pip install -e .
  ```

## Contributions
- Do some changes on the OPENAPI, now using GEMINI, as the api key is free.   
- Optimization code,as trying to change input to live camera stream.   

## To Be Change
- Everything with ```!TO CHANGE``` comment,  change it by yourself.   