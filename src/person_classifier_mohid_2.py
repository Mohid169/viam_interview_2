from typing import ClassVar, Mapping, Optional, List, Any
from typing_extensions import Self

from viam.media.video import ViamImage
from viam.proto.common import Classification, Detection, PointCloudObject
from viam.proto.service.vision import GetPropertiesResponse
from viam.resource.types import Model, ModelFamily
from viam.module.types import Reconfigurable
from viam.resource.base import ResourceBase
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.utils import ValueTypes

from vision_python import Vision
from viam.logging import getLogger

LOGGER = getLogger(__name__)

class personClassifierMohid2(Vision, Reconfigurable):
    MODEL: ClassVar[Model] = Model(
        ModelFamily("mohid-soln-interview", "person-classifier"),
        "mohid-custom-model-2"
    )

    def __init__(self, name: str):
        super().__init__(name)
        # Initialize your single classification model here
        self.model = self.load_model()

    def load_model(self):
        # Load your custom single classification model here
        # For example, you might load a pre-trained model from a file
        # Replace this with your actual model loading code
        # Example:
        # model = YourCustomModel.load('path_to_model')
        model = None  # Placeholder
        return model

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        # Initialize and configure the resource instance
        my_class = cls(config.name)
        my_class.reconfigure(config, dependencies)
        return my_class

    @classmethod
    def validate(cls, config: ComponentConfig):
        # Validate the config if necessary
        pass

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        # Reconfigure the resource instance if necessary
        self.dependencies = dependencies  # Store dependencies for later use
        pass

    """ Implement the methods the Viam RDK defines for the Vision API (rdk:service:vision) """

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        # Get the camera from dependencies
        camera = self.dependencies.get(ResourceName(name=camera_name))
        if not camera:
            LOGGER.error(f"Camera '{camera_name}' not found in dependencies")
            return []

        # Get an image from the camera
        image = await camera.get_image()
        if not image:
            LOGGER.error(f"Failed to get image from camera '{camera_name}'")
            return []

        # Get classifications from the image
        return await self.get_classifications(image, count, extra=extra, timeout=timeout)

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        # Use your single classification model to predict
        # Process the image as required by your model
        classifications = []

        # Placeholder for model prediction
        # Replace this with your actual prediction logic
        # Example:
        # confidence = self.model.predict(image)
        confidence = 0.95  # Assuming a confidence score returned by your model

        # Check if the confidence is above the threshold
        if confidence > 0.80:
            classification = Classification(
                class_name="person_detected",
                confidence=confidence
            )
            classifications.append(classification)

        # Return the top N classifications based on count
        return classifications[:count]

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> GetPropertiesResponse:
        return GetPropertiesResponse(
            classifications_supported=True,
            detections_supported=False,
            object_point_clouds_supported=False
        )

    # For irrelevant methods, write pass

    async def get_detections_from_camera(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        pass

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[Detection]:
        pass

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        pass

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> List[PointCloudObject]:
        pass

