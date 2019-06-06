## Submodular Batch Selection for Training Deep Neural Networks

IJCAI 2019 

Mini-batch gradient descent based methods are the de facto algorithms for training neural network ar- chitectures today. We introduce a mini-batch selec- tion strategy based on submodular function maxi- mization. Our novel submodular formulation cap- tures the informativeness of each sample and diver- sity of the whole subset. We design an efficient, greedy algorithm which can give high-quality so- lutions to this NP-hard combinatorial optimization problem. Our extensive experiments on standard datasets show that the deep models trained using the proposed batch selection strategy provide better generalization than Stochastic Gradient Descent as well as a popular baseline sampling strategy across different learning rates, batch sizes, and distance metrics.

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

```bash
@inproceedings{joseph2019Submodular,
  title = {Submodular Batch Selection for Training Deep Neural Networks},
  author={K J, Joseph and Vamshi Teja, Ra and Krishnakant, Singh and Vineeth, N Balasubramanian},
  booktitle={Proceedings of the Twenty-Eighth International Joint Conference on
               Artificial Intelligence, IJCAI, Macao, China.},
  year={2019},
  organization={International Joint Conferences on Artificial Intelligence Organization}}}
```