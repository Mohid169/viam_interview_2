"""
This file registers the model with the Python SDK.
"""

from viam.services.vision import Vision
from viam.resource.registry import Registry, ResourceCreatorRegistration

from .personClassifierMohid-2 import personClassifierMohid-2

Registry.register_resource_creator(Vision.SUBTYPE, personClassifierMohid-2.MODEL, ResourceCreatorRegistration(personClassifierMohid-2.new, personClassifierMohid-2.validate))
