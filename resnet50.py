import argparse
import tensorflow as tf
import os

# Set the TensorFlow backend to use the GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--dataset_train",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_validation",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_test",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    
    args = parser.parse_args()

    return args



def main():
    args = parse_args()
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(512, 512, 3))

    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(4, activation='softmax'))

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])


    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # change data directory 

    train_generator = train_datagen.flow_from_directory(args.dataset_train, target_size=(512, 512), batch_size=32, class_mode='categorical')
    valid_generator = valid_datagen.flow_from_directory(args.dataset_validation, target_size=(512, 512), batch_size=32, class_mode='categorical')

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_dir = args.dataset_test
    batch_size =32
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(512, 512), batch_size=batch_size, class_mode='categorical', shuffle=False)

    history = model.fit(train_generator, epochs=10, validation_data=valid_generator)

    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    print('Train accuracy: ', train_acc)
    print('Validation accuracy: ', val_acc)

    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_generator)
    y_true = test_generator.classes
    y_pred = model.predict(test_generator).argmax(axis=-1)
    f1score = f1_score(y_true, y_pred, average='weighted')

    print('Test accuracy: ', test_accuracy)
    print('Test loss: ', test_loss)
    print('Test precision: ', test_precision)
    print('Test recall: ', test_recall)
    print('Test F1-score: ', f1score)

    model.save(args.output_dir)


if __name__ == "__main__":
    main()


# python3 resnet50.py --dataset_train ~/G077-Machine-Learning-Practical/Data/Clean_data/train/ --dataset_validation ~/G077-Machine-Learning-Practical/Data/Clean_data/validation/ --dataset_test ~/G077-Machine-Learning-Practical/Data/Clean_data/test/
# python3 resnet50.py --dataset_train ~/G077-Machine-Learning-Practical/Data/split_data/train_0.75 --dataset_validation ~/G077-Machine-Learning-Practical/Data/Clean_data/validation/ --dataset_test ~/G077-Machine-Learning-Practical/Data/Clean_data/test/ --output_dir ~/G077-Machine-Learning-Practical/output/
# python3 resnet50.py --dataset_train ~/G077-Machine-Learning-Practical/Data/split_data/train_0.5 --dataset_validation ~/G077-Machine-Learning-Practical/Data/Clean_data/validation/ --dataset_test ~/G077-Machine-Learning-Practical/Data/Clean_data/test/ --output_dir ~/G077-Machine-Learning-Practical/output/