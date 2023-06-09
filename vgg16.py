import os
import shutil
import argparse
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Set the TensorFlow backend to use the GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.metrics import f1_score

# Define the paths to your original data and the output directories for train, validation and test sets

# train_dir = '~/G077-Machine-Leaning-Practical/Data/Clean_data/vgg/train/'
# val_dir = '~/G077-Machine-Leaning-Practical/Data/Clean_data/validation/'
# test_dir = '~/G077-Machine-Leaning-Practical/Data/Clean_data/test/'
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--train_dir",
        type=str,
        default=None,
        help="The name of the training dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default=None,
        help="The name of the validation dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help="The name of the test dataset to use (via the datasets library).",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    # Define the data generators
    train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)


    test_generator = test_datagen.flow_from_directory(args.test_dir,
                                                    target_size=(512, 512),
                                                    batch_size=32,
                                                    class_mode='categorical')

    train_generator = train_datagen.flow_from_directory(args.train_dir,
                                                        target_size=(512, 512),
                                                        batch_size=32,
                                                        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(args.val_dir,
                                                            target_size=(512, 512),
                                                            batch_size=32,
                                                            class_mode='categorical')

    # Load the pre-trained VGG-16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(512, 512, 3))

    # Freeze the convolutional layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add new layers for classification
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(4, activation='softmax')(x)

    # Define the model
    model = Model(inputs=base_model.input, outputs=x)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    # Train the model
    model.fit(train_generator,
            epochs=1,
            validation_data=validation_generator)

    # Evaluate the model on the training set
    train_loss, train_acc, train_precision, train_recall = model.evaluate(train_generator)

    print('Training accuracy:', train_acc)
    print('Training loss:', train_loss)

    # Evaluate the model on the validation set
    val_loss, val_acc, val_precision, val_recall = model.evaluate(validation_generator)
    print('Validation accuracy:', val_acc)

    # Evaluate the model on the test set
    test_loss, test_acc,test_precision, test_recall = model.evaluate(test_generator)
    y_true = test_generator.classes
    y_pred = model.predict(test_generator).argmax(axis=-1)
    f1score = f1_score(y_true, y_pred, average='weighted')
    print('Test accuracy:', test_acc)
    print('Test Loss; ', test_loss)
    print('Test precision: ', test_precision)
    print('Test recall: ', test_recall)
    print('Test F1-score: ',f1score)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model.save(os.path.join(args.output_dir, "vgg16_model.h5"))


    # Use the model for prediction
    # predict_image = some_image # Replace with your own image
    # predicted_class = model.predict(predict_image)

if __name__ == "__main__":
    main()


# python3 vgg16.py --train_dir ~/G077-Machine-Learning-Practical/Data/split_data/train_0.75 --val_dir ~/G077-Machine-Learning-Practical/Data/Clean_data/validation/ --test_dir ~/G077-Machine-Learning-Practical/Data/Clean_data/test/ --output_dir ~/G077-Machine-Learning-Practical/output
# python3 vgg16.py --train_dir ~/G077-Machine-Learning-Practical/Data/Clean_data/train --val_dir ~/G077-Machine-Learning-Practical/Data/Clean_data/validation/ --test_dir ~/G077-Machine-Learning-Practical/Data/Clean_data/test/ --output_dir ~/G077-Machine-Learning-Practical/output