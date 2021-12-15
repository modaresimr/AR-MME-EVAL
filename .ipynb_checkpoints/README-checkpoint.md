# AR-MME-eval


This tool prepare a multimodal evaluation for Activity Recognition (AR) systems

Please install Jupyter and open this notebook [AR-Compare.ipynb](AR-Compare.ipynb) 



# Usage

```
import ar_mme_eval.multi_eval
res1=ar_mme_eval.multi_eval.get_single_result(groundtruthfile,peredictionfile,None,debug=0,args={})
```

If the groundtruth,perediction are available in dataframe 
```
res1=mme_eval.multi_eval.get_single_result_df(groundtruth,perediction,None,debug=0,args={})
```

