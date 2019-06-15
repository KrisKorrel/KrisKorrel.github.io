---
layout: post
title:  "Don't mix-and-match keras and tensorflow.keras imports"
---
**TL;DR: Do not use both `keras` and `tensorflow.keras` in your projects. Choose one.**

### Introduction
Sometimes, error messages can be as informative as someone trying to explain quantum mechanics in Russian.
It may sound very interesting, but I don't understand what you're trying to tell me.
Tensorflow is especially known for its lack of guidance in finding bugs in your code.
The main culprit for this used to be the fact that Tensorflow would require you to first define the computational graph.
After this, you would have to start a session and perform forward and backward propagations through your defined network.
This would make bugs in your graph definition hard to trace.

With the introduction of [eager execution](https://www.tensorflow.org/guide/eager), which will even be the default for [Tensorflow 2.0](https://www.tensorflow.org/beta/guide/eager), this luckily should now just be a painful memory be of the past.
However, there is still a common fallacy that may introduce almost untraceable bugs: mixing-and-matching `keras` and `tensorflow.keras` imports.

### Example
Let's show an example of where it could go wrong.
We define `create_model.py`, which creates and returns a sequential keras model.
```python
import keras

def create_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(50, activation='relu', input_shape=(10,)))
    model.add(keras.layers.Dense(10, activation='relu'))
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model
```
<br>
Imagine that for some reason we want to load the created model and extend it with an additional layer.
`extend_model.py` will do this, after which it performs a single prediction.
```python
import keras
import numpy as np

from create_model import create_model

model = create_model()
output_layer = model.output
new_output_layer = keras.layers.Activation('softmax')(output_layer)
extended_model = keras.models.Model(model.input, new_output_layer)

pred = extended_model.predict(np.ones((1, 10)))
print(pred)
# Prints [[0.14706095 0.13719702 0.07847318 0.07847318 0.13134125 0.07847318 0.11356177 0.07847318 0.07847318 0.07847318]]
# (depending on random initialization, of course)
```
<br>
Note that in both files we have used Keras. Now, problems arise if, in any of the two files, we replace `import keras` with `import tensorflow.keras as keras`. This will produce an error such as
```
Traceback (most recent call last):
  ...
  File "~/keras-tensorflow/venv/lib/python3.7/site-packages/tensorflow/python/keras/engine/base_layer.py", line 1742, in __init__
    layer.outbound_nodes.append(self)
AttributeError: 'Dense' object has no attribute 'outbound_nodes'
```
However, this is just one of the types of error messages that may be produced.
Depending on your application, another type of error message may be
```
FailedPreconditionError: Attempting to use uninitialized value dense_1_W
	[[Node: dense_1_W/read = IdentityT=DT_FLOAT, _class=["loc:@dense_1_W"], _device="/job:localhost/replica:0/task:0/cpu:0"]]
```

### The problem
Although I am in no way an expert in the internal workings of Tensorflow and Keras, my guess was that the problem has to do with both imports using different sessions.
Thus, the model would be created in one session in which the weights would also be initialized. Then, in a different file, with different imports, the model would be extended in another session, in which the variables are not yet initialized. 
Let's show this with the following scripts.  
`a.py`:
```python
import tensorflow
import keras as k1
import tensorflow.keras as k2


def print_sessions_a():
    print(f"a: {tensorflow.get_default_session()}")
    print(f"a: {k1.backend.get_session()}")
    print(f"a: {k2.backend.get_session()}")
```
and `b.py`:
```python
import tensorflow
import keras as k1
import tensorflow.keras as k2

from a import print_sessions_a


def print_sessions_b():
    print(f"b: {tensorflow.get_default_session()}")
    print(f"b: {k1.backend.get_session()}")
    print(f"b: {k2.backend.get_session()}")


print_sessions_a()
print_sessions_b()
```
Which prints
```
a: None
a: <tensorflow.python.client.session.Session object at 0x109167588>
a: <tensorflow.python.client.session.Session object at 0x12790c748>
b: None
b: <tensorflow.python.client.session.Session object at 0x109167588>
b: <tensorflow.python.client.session.Session object at 0x12790c748>
```
We see that unlike Tensorflow, Keras initializes a default session, which, within the same runtime, is always the same. However, the default session of `keras` is different of that of `tensorflow.keras`.
<br>
However, if this would indeed be the culprit, it would mean that we could mitigate the problem by manually fixing the difference in sessions:
```python
import keras as k1
import tensorflow.keras as keras
keras.backend.set_session(k1.backend.get_session())
```
Unfortunately, this does not solve the problem however.

### The solution
In my project, I have chosen for drastic measures: replace every single Keras import with `tensorflow.keras`.
This indeed solves the problems, and as a bonus, reduces the dependencies of my projects by one.
This strategy would be less viable if your project uses a dependency that is implemented in Keras however.
Please let me know if you know a more elegant solution.
