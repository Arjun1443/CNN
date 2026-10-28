"""Run locally: python streamlit_app/train_cloud_model.py (downloads CIFAR-10)."""
from pathlib import Path
import tensorflow as tf

tf.keras.utils.set_random_seed(42)
(images,labels),(test_images,test_labels)=tf.keras.datasets.cifar10.load_data()
model=tf.keras.Sequential([
    tf.keras.layers.Input((32,32,3)),
    tf.keras.layers.Conv2D(32,3,activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64,3,activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(images[:10000]/255.,labels[:10000],validation_split=.2,epochs=5,batch_size=64)
print('Held-out evaluation:',model.evaluate(test_images/255.,test_labels,verbose=0))
model.save(Path(__file__).parent/'model.keras')
