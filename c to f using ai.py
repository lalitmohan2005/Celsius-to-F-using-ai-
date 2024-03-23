import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

celsius = np.array([-40,-10,0,8,15,22,38,100, 10, 20, 30, 40],dtype=float)
fahrenheit = np.array([-40,14,32,46,59,72,100,212,50.0, 68.0, 86.0, 104.0], dtype=float)

layer = tf.keras.layers.Dense(units=1, input_shape=[1])
model= tf.keras.Sequential([layer])

model.compile(
    optimizer= tf.keras.optimizers.Adam(0.1),
    loss  ='mean_squared_error'

)

results =model.fit(celsius,fahrenheit, epochs=1000, verbose=False)

plt.xlabel("# epochs")
plt.ylabel("Loss")
plt.plot(results.history["loss"])

res = model.predict([100.0])
print("the result is" + str(res) + "farenheit")