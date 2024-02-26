import tensorflow as tf

"""我的第一个模型"""

# 加载并准备 MNIST 数据集，将样本数据从整数转换为浮点数
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 通过堆叠层来构建 tf.keras.Sequential 模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
tf.nn.softmax(predictions)

# 使用 losses.SparseCategoricalCrossentropy 为训练定义损失函数
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

# 使用 Keras Model.compile 配置和编译模型
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# 使用 Model.fit 方法调整模型参数并最小化损失
model.fit(x_train, y_train, epochs=20)
model.evaluate(x_test, y_test, verbose=2)

# 封装经过训练的模型，并将 softmax 附加到该模型
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
probability_model(x_test[:5])
