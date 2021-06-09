Running in Docker with versions matching the book.
```
Python 3.7.5
Numpy 1.15.0
Matplotlib 3.1.1
```
```
docker build -t nnfs .
docker run -it --rm nnfs
```


### Chapter 19
MNIT Fashion Dataset Results after 10 epochs

epoch: 10
step: 0, acc: 0.930, loss: 0.214 (data_loss: 0.214, reg_loss: 0.000), lr: 0.0001915341888527102
step: 100, acc: 0.898, loss: 0.285 (data_loss: 0.285, reg_loss: 0.000), lr: 0.00018793459875963167
step: 200, acc: 0.898, loss: 0.289 (data_loss: 0.289, reg_loss: 0.000), lr: 0.00018446781036709093
step: 300, acc: 0.945, loss: 0.148 (data_loss: 0.148, reg_loss: 0.000), lr: 0.00018112660749864155
step: 400, acc: 0.891, loss: 0.243 (data_loss: 0.243, reg_loss: 0.000), lr: 0.00017790428749332856
step: 468, acc: 0.958, loss: 0.203 (data_loss: 0.203, reg_loss: 0.000), lr: 0.00017577781683951485
training, acc: 0.915, loss: 0.238, data_loss: 0.238, reg_loss: 0.000, lr: 0.000
validation, acc: 0.958, loss: 0.203