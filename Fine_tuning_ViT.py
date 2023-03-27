import pathlib
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import trange
from datasets import load_dataset
import tensorflow as tf
from transformers import ViTImageProcessor
from transformers import ViTForImageClassification
from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader
import torch
import argparse
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.transforms import (CenterCrop,
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)



def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=None,
        help="log directory",
    )
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default='no',
        help="The evaluation strategy to adopt during training.",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default='no',
        help="The checkpoint save strategy to adopt during training.",
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default=None,
        help= "Use in conjunction with `load_best_model_at_end` to specify the metric to use to compare two different models.",
    )
    parser.add_argument(
        "--load_best_model_at_end",
        type=bool,
        default=False,
        help="Whether or not to load the best model found during training at the end of training.",
    )
    parser.add_argument(
        "--remove_unused_columns",
        type=bool,
        default=True,
        help=(
            "Whether or not to automatically remove the columns unused by the model forward method."
        ),
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset("imagefolder", data_dir= args.dataset_name)
    else:
        raise ValueError("Make sure the data is provided")

    splits = dataset["train"].train_test_split(test_size=0.1)

    test_ds = splits["test"]

    splits_2 = splits["train"].train_test_split(test_size=0.1)
    train_ds = splits_2["train"]

    val_ds = splits_2["test"]

    id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
    label2id = {label:id for id,label in id2label.items()}


    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]

    normalize = Normalize(mean=image_mean, std=image_std)
    _train_transforms = Compose(
            [
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )

    _val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )

    def train_transforms(examples):
        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
        return examples

    def val_transforms(examples):
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
        return examples

    # Set the transforms
    train_ds.set_transform(train_transforms)
    val_ds.set_transform(val_transforms)
    test_ds.set_transform(val_transforms)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    train_dataloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size)


#     model = ViTForImageClassification.from_pretrained(args.model_name_or_path,id2label=id2label,label2id=label2id)
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                    id2label=id2label,
                                                    label2id=label2id)

    metric_name = "accuracy"


    args = TrainingArguments(
        output_dir=f"test-brain_tumor_classification",
        # save_steps = 1,
        save_strategy = "epoch",
        save_total_limit = 1,
        evaluation_strategy="epoch",
        # eval_step=1,
        learning_rate=2e-5,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=4,
        # training epoch, could be changed later 
        num_train_epochs=1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        logging_dir='logs',
        remove_unused_columns=False,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(predictions, labels)
        print('accuracy: ', accuracy)
        # scce = tf.keras.losses.SparseCategoricalCrossentropy()
        # print('loss: ',scce(labels, predictions).numpy())
        f1 = f1_score(predictions, labels, average='weighted')
        print('f1 ', f1)
        return dict(accuracy=accuracy, f1_score=f1)

    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )
                                  
    # %load_ext tensorboard
    # %tensorboard --logdir logs/
                                  
#   training                               
    trainer.train()  
    trainer.save_model()        
    outputs = trainer.predict(test_ds)
    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)
    labels = train_ds.features['label'].names
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation=45)
                                  
#  save model 
    # trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()


# python3 Fine_tuning_ViT.py --dataset_name ~/G077-Machine-Learning-Practical/Data/Clean_data/ --output_dir ~/G077-Machine-Learning-Practical/output/