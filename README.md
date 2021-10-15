# ulmmcm
Unsupervised Learning of Multi-Model Consensus Maximization

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

