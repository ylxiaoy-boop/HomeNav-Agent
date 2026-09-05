"""Contract checks for the AI2-THOR adapter without a Unity runtime."""

from __future__ import annotations

import unittest
from copy import deepcopy

from models.ai2thor_environment import AI2THORHomeEnvironment
from models.yolo_model import YOLOModel


class FakeEvent:
    def __init__(self, metadata):
        self.metadata = metadata
        self.frame = None


class FakeController:
    def __init__(self, **_kwargs):
        self.stopped = False
        self.last_event = self._event()

    def step(self, action: str, **_parameters):
        metadata = deepcopy(self.last_event.metadata)
        metadata["lastActionSuccess"] = True
        if action == "GetReachablePositions":
            metadata["actionReturn"] = [{"x": 0.0, "y": 0.9, "z": 0.0}]
        self.last_event = FakeEvent(metadata)
        return self.last_event

    def stop(self):
        self.stopped = True

    @staticmethod
    def _event():
        return FakeEvent(
            {
                "sceneName": "FloorPlan1",
                "lastActionSuccess": True,
                "agent": {"position": {"x": 0.0, "y": 0.9, "z": 0.0}, "rotation": {"y": 0}},
                "objects": [
                    {
                        "objectId": "Cup|0",
                        "objectType": "Cup",
                        "visible": True,
                        "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                    }
                ],
            }
        )


class AI2THORAdapterTests(unittest.TestCase):
    def test_navigation_and_perception_share_ai2thor_environment(self) -> None:
        environment = AI2THORHomeEnvironment(controller_factory=FakeController)
        try:
            result = environment.navigate("cup", max_steps=10)
            self.assertTrue(result["success"])
            self.assertEqual(result["data"]["backend"], "ai2thor")

            detection = YOLOModel(environment).detect(target="cup")
            self.assertEqual(detection["backend"], "ai2thor")
            self.assertEqual(detection["objects"][0]["class_name"], "Cup")
        finally:
            environment.close()

        self.assertTrue(environment.controller.stopped)


if __name__ == "__main__":
    unittest.main()
