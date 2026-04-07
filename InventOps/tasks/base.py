from abc import ABC, abstractmethod

class BaseTask(ABC):
    config: dict

    @abstractmethod
    def grade(self, episode_log: list[dict]) -> float:
        ...
