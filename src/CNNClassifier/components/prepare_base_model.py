import os
import ssl
import shutil
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from CNNClassifier.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    @staticmethod
    def _download_weights(url: str, filename: str) -> str:
        cache_dir = os.path.join(os.path.expanduser("~"), ".keras", "models")
        os.makedirs(cache_dir, exist_ok=True)
        target_path = os.path.join(cache_dir, filename)

        if not os.path.exists(target_path):
            context = ssl._create_unverified_context()
            with request.urlopen(url, context=context) as response:
                with open(target_path, "wb") as f:
                    shutil.copyfileobj(response, f)

        return target_path

    def get_base_model(self):
        weights = self.config.params_weights
        include_top = self.config.params_include_top

        if weights is None or str(weights).lower() == "none":
            self.model = tf.keras.applications.vgg16.VGG16(
                input_shape=self.config.params_image_size,
                weights=None,
                include_top=include_top
            )
        else:
            if include_top:
                url = "https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
                filename = "vgg16_weights_tf_dim_ordering_tf_kernels.h5"
            else:
                url = "https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
                filename = "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

            weights_path = self._download_weights(url=url, filename=filename)

            self.model = tf.keras.applications.vgg16.VGG16(
                input_shape=self.config.params_image_size,
                weights=None,
                include_top=include_top
            )
            self.model.load_weights(weights_path)

        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)