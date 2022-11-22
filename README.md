# ulmmcm
Unsupervised Learning for Multi-Model Consensus Maximization. Based on the [single model scenario](https://openaccess.thecvf.com/content_CVPR_2019/papers/Probst_Unsupervised_Learning_of_Consensus_Maximization_for_3D_Vision_Problems_CVPR_2019_paper.pdf) developed by Thomas Probst et al.    

# setup
1. download 'data' from https://drive.google.com/drive/folders/1i_20qdSdSLnVVkvEaG6lJWl92c6HGNYQ?usp=sharing
2. place 'data' inside the repository in such a way the directory tree appears as follows:  
```
-- ulmmcm  
      |  
      | -- bin  
      | -- data  
      | -- model
      | -- results
      | -- syndalib             
      | -- utils                
      | -- losses.py             
      | -- metrics.py  
      | -- requirements.txt  
```
3. setup venv using requirements.txt file:   
`pip install -r requirements.txt`

# train
1. modify values under "user defined variables" in the script bin/train.py as you wish
2. run train.py

# test
1. run test.py script
2. choose what scenario to test among those listed in the menu  
(results are already stored in results/test)

# contacts

contact me at william.bonvini@outlook.com for any question or doubt!
