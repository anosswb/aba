import os
import tensorflow as tf
import numpy as np
from datetime import datetime

print(f"Starting dental caries detection model training: {datetime.utcnow()}")

class Config:
    IMAGE_SIZE = 96
    BATCH_SIZE = 32
    EPOCHS = 100  # Increased for better accuracy
    BASE_DIR = '/content/dental_caries_project'
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')

def create_directories():
    for directory in [Config.MODEL_DIR, Config.DATA_DIR, Config.LOGS_DIR]:
        os.makedirs(directory, exist_ok=True)
    print("Directories created successfully")

def parse_tfrecord(example_proto):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64)
    }
    
    features = tf.io.parse_single_example(example_proto, feature_description)
    
    image = tf.io.decode_jpeg(features['image/encoded'], channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [Config.IMAGE_SIZE, Config.IMAGE_SIZE])
    image = image / 255.0
    
    label = tf.sparse.to_dense(features['image/object/class/label'])
    label = tf.reduce_max(label)
    label = tf.cast(tf.equal(label, 1), tf.float32)
    
    return image, label

def create_dataset(tfrecord_path, is_training=True):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        dataset = dataset.shuffle(1000)
        dataset = dataset.repeat()
    
    dataset = dataset.batch(Config.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3)),
        
        tf.keras.layers.Conv2D(32, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(),
        
        tf.keras.layers.Conv2D(64, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(),
        
        tf.keras.layers.Conv2D(64, 3, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def train_model(train_dataset, valid_dataset, train_steps, valid_steps):
    model = create_model()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(Config.MODEL_DIR, 'best_model.keras'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=Config.EPOCHS,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def export_model(model):
    keras_path = os.path.join(Config.MODEL_DIR, 'final_model.keras')
    model.save(keras_path)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(Config.MODEL_DIR, 'model_esp32.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"\nModel export completed at: {datetime.utcnow()}")
    print(f"Keras model saved to: {keras_path}")
    print(f"TFLite model saved to: {tflite_path}")
    print(f"TFLite model size: {os.path.getsize(tflite_path) / 1024:.2f} KB")

def main():
    start_time = datetime.utcnow()
    print(f"\nStarting training pipeline at: {start_time}")
    
    create_directories()
    
    train_path = os.path.join(Config.DATA_DIR, 'train/teeth-ARWb-udNj.tfrecord')
    valid_path = os.path.join(Config.DATA_DIR, 'valid/teeth-ARWb-udNj.tfrecord')
    
    print("\nPreparing datasets...")
    train_dataset = create_dataset(train_path, True)
    valid_dataset = create_dataset(valid_path, False)
    
    print("\nCalculating steps...")
    train_samples = sum(1 for _ in tf.data.TFRecordDataset(train_path))
    valid_samples = sum(1 for _ in tf.data.TFRecordDataset(valid_path))
    
    train_steps = train_samples // Config.BATCH_SIZE
    valid_steps = (valid_samples + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE
    
    print(f"Training samples: {train_samples}")
    print(f"Validation samples: {valid_samples}")
    
    print("\nStarting model training...")
    model, history = train_model(train_dataset, valid_dataset, train_steps, valid_steps)
    
    print("\nExporting model...")
    export_model(model)
    
    end_time = datetime.utcnow()
    training_duration = end_time - start_time
    print(f"\nTotal training time: {training_duration}")
    
    return model, history

if __name__ == "__main__":
    tf.keras.backend.clear_session()
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    model, history = main()