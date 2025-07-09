import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
import dagshub
from chestCancerClassification.entity.config_entity import EvaluationConfig
from chestCancerClassification.utils.common import read_yaml, create_directories, save_json


dagshub.init(repo_owner='abcdefya', repo_name='Chest-Cancer-Classification', mlflow=True)

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.valid_generator = None
        self.score = None

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        try:
            self.model = self.load_model(self.config.path_of_model)
            self._valid_generator()
            self.score = self.model.evaluate(self.valid_generator)
            self.save_score()
        except Exception as e:
            raise Exception(f"Error during evaluation: {str(e)}")

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        try:
            # Thiết lập tracking URI
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run(run_name="Chest_Cancer_Evaluation"):
                # Log các tham số từ config
                mlflow.log_params(self.config.all_params)
                
                # Log metrics
                mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})

                # Log mô hình Keras
                if tracking_url_type_store != "file":
                    mlflow.keras.log_model(
                        self.model,
                        "model",
                        registered_model_name="VGG16Model"
                    )
                else:
                    mlflow.keras.log_model(self.model, "model")


                mlflow.log_artifact("scores.json")
                
        except Exception as e:
            raise Exception(f"Error logging to MLflow: {str(e)}")

