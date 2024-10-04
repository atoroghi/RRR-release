
Thank you for visiting this repository!
This repository contains the implementation of our EMNLP-24 paper **"Right for Right Reasons: Large Language Models for Verifiable
Commonsense Knowledge Graph Question Answering"**.

In order to use the code, please follow these steps:

## 1- Install requirements
~~~
pip install -r requirements.txt
~~~

## 2- Running Experiments
You can run our model on Question Answering and Claim Verification tasks using commands like the following:
~~~
python -m kbc.BC_Experiments --dataset_name Creak --use_refined True --mode modified --max_depth 2 --max_depth 2 --experiment_name foo
~~~

Here, "use_refined" indicates whether [ReFined](https://github.com/amazon-science/ReFinED) should be included for entity extraction or not,  and "mode" indicates whether you are using the original or long-tail subset of the data. Arguments "max_depth" and "max_breadth" determine the depth and breadth of the search tree.

You can also use "MPR_Experiments.py" file to run the preference reasoning experiments.

For running other baselines, please refer to the "baselines" folder.



Thank you for your attention!
