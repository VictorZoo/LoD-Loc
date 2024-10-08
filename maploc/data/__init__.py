from .LoD_data.dataset import UAVDataModule
from .LoD_data.dataset_swiss import SwissDataModule
from .LoD_data.dataset_demo import DEMODataModule

modules = {"UAVD4L-LoD": UAVDataModule, "Swiss-EPFL": SwissDataModule, "DEMO": DEMODataModule}
