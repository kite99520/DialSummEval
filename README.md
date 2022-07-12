# DialSummEval
This is the Repo for the paper [DialSummEval: Revisiting summarization evaluation for dialogues](https://aclanthology.org/2022.naacl-main.418) 


The human judgments we used are under `annotation_campus`, you can also download them through [Link](https://drive.google.com/file/d/1LW1Ii_exspexg1oJTfrU8B4Bmzf2nJl-/view).  

The annotations from AMT we did not use are under `annotation_AMT`.  

The code under `./reproduce/metrics` is used to compute the values of metrics and the files under `./reproduce/analysis/models_eval_new` record the outputs of the metrics. `analysis.py` contains some functions about analysis, such as the correlation calculation. 

# Citation
```
@inproceedings{gao-wan-2022-dialsummeval,
    title = "{D}ial{S}umm{E}val: Revisiting Summarization Evaluation for Dialogues",
    author = "Gao, Mingqi  and
      Wan, Xiaojun",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.418",
    pages = "5693--5709",
    abstract = "Dialogue summarization is receiving increasing attention from researchers due to its extraordinary difficulty and unique application value. We observe that current dialogue summarization models have flaws that may not be well exposed by frequently used metrics such as ROUGE. In our paper, we re-evaluate 18 categories of metrics in terms of four dimensions: coherence, consistency, fluency and relevance, as well as a unified human evaluation of various models for the first time. Some noteworthy trends which are different from the conventional summarization tasks are identified. We will release DialSummEval, a multi-faceted dataset of human judgments containing the outputs of 14 models on SAMSum.",
}
```

# Contact
If you have any questions, you can send emails to gaomingqi[AT]pku[DOT]edu[DOT]cn.
