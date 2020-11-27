## [Submodular Batch Selection for Training Deep Neural Networks](https://www.ijcai.org/Proceedings/2019/0372.pdf)

IJCAI 2019 

Mini-batch gradient descent based methods are the de facto algorithms for training neural network architectures today. We introduce a mini-batch selection strategy based on submodular function maximization. Our novel submodular formulation captures the informativeness of each sample and diversity of the whole subset. We design an efficient, greedy algorithm which can give high-quality solutions to this NP-hard combinatorial optimization problem. Our extensive experiments on standard datasets show that the deep models trained using the proposed batch selection strategy provide better generalization than Stochastic Gradient Descent as well as a popular baseline sampling strategy across different learning rates, batch sizes, and distance metrics.

### Setup
Python: 2.7.6, PyTorch 0.4.1

Clone the repository and install dependencies from requirements.txt
```bash
git clone https://github.com/VamshiTeja/SMDL
cd smdl
pip install -r requirements.txt
```

### Run

All the configurations are in a single place: ./config/smdl.yml

Modify the configurations are run smdl.py

```bash
python smdl.py
```

The code is well documented and you would be able to get along. In case of troubles please do raise an issue here. We would be agile.


### Citation

```commandline
@inproceedings{joseph2019Submodular,
  title = {Submodular Batch Selection for Training Deep Neural Networks},
  author={K J, Joseph and Vamshi Teja, R and Krishnakant, Singh and Vineeth, N Balasubramanian},
  booktitle={Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI, Macao, China.},
  organization={International Joint Conferences on Artificial Intelligence Organization},
  year={2019}}
```

### Contributors

[![](https://sourcerer.io/fame/JosephKJ/VamshiTeja/SMDL/images/0)](https://sourcerer.io/fame/JosephKJ/VamshiTeja/SMDL/links/0)[![](https://sourcerer.io/fame/JosephKJ/VamshiTeja/SMDL/images/1)](https://sourcerer.io/fame/JosephKJ/VamshiTeja/SMDL/links/1)[![](https://sourcerer.io/fame/JosephKJ/VamshiTeja/SMDL/images/2)](https://sourcerer.io/fame/JosephKJ/VamshiTeja/SMDL/links/2)[![](https://sourcerer.io/fame/JosephKJ/VamshiTeja/SMDL/images/3)](https://sourcerer.io/fame/JosephKJ/VamshiTeja/SMDL/links/3)[![](https://sourcerer.io/fame/JosephKJ/VamshiTeja/SMDL/images/4)](https://sourcerer.io/fame/JosephKJ/VamshiTeja/SMDL/links/4)[![](https://sourcerer.io/fame/JosephKJ/VamshiTeja/SMDL/images/5)](https://sourcerer.io/fame/JosephKJ/VamshiTeja/SMDL/links/5)[![](https://sourcerer.io/fame/JosephKJ/VamshiTeja/SMDL/images/6)](https://sourcerer.io/fame/JosephKJ/VamshiTeja/SMDL/links/6)[![](https://sourcerer.io/fame/JosephKJ/VamshiTeja/SMDL/images/7)](https://sourcerer.io/fame/JosephKJ/VamshiTeja/SMDL/links/7)
