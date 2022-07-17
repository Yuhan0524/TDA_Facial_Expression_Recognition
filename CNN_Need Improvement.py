import tensorflow as tf
in_1 = tf.keras.layers.Input((48,48,1),name='in_1')
model = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(in_1)
model = tf.keras.layers.MaxPool2D(pool_size=(5,5), strides=(2, 2))(model)

model=tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(model)
model=tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(model)
model=tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2, 2))(model)

model=tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(model)
model=tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(model)
model=tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2, 2))(model)

model=tf.keras.layers.Flatten()(model)

model=tf.keras.layers.Dense(1024, activation='relu')(model)
model=tf.keras.layers.Dropout(0.2)(model)
model=tf.keras.layers.Dense(1024, activation='relu')(model)
model=tf.keras.layers.Dropout(0.2)(model)

model=tf.keras.layers.Dense(7, activation='softmax')(model)

#------------------------------------------------------------------------------------------------------------------
in_2 = tf.keras.layers.Input((30,100,1),name='in_2')
model_2=tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(in_2)

model_2=tf.keras.layers.MaxPool2D(pool_size=(5,5), strides=(2, 2))(model_2)

model_2=tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(model_2)
model_2=tf.keras.layers.Conv2D(64, (2, 2), activation='relu')(model_2)
model_2=tf.keras.layers.MaxPool2D(pool_size=(1,1), strides=(2, 2))(model_2)

model_2=tf.keras.layers.Flatten()(model_2)

model_2=tf.keras.layers.Dense(1024, activation='relu')(model_2)
model_2=tf.keras.layers.Dropout(0.5)(model_2)
model_2=tf.keras.layers.Dense(1024, activation='relu')(model_2)
model_2=tf.keras.layers.Dropout(0.5)(model_2)

model_2=tf.keras.layers.Dense(7, activation='softmax')(model_2)


merged = tf.keras.layers.Concatenate()([model, model_2])
output = tf.keras.layers.Dense(7, activation='softmax')(merged)

model_final = tf.keras.Model(inputs=[in_1,in_2],outputs=[output])
model_final.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss='categorical_crossentropy',metrics=['accuracy'])
#model_final.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'],loss_weights=[1,0.2])

hist = model_final.fit(
    {"in_1": train_ori,"in_2": train_data},train_labels_small,
    validation_data=({"in_1":test_ori, "in_2":private_data},private_labels_small),
    epochs=30,
    batch_size=256)