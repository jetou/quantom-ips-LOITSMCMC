import torch
import logging

logger = logging.getLogger(__name__)


class StatefulModule(torch.nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        logger.info(f"Loading {self.__class__.__name__} from {path}")
        if not torch.cuda.is_available():
            weights = torch.load(
                path, weights_only=True, map_location=torch.device("cpu")
            )
        else:
            weights = torch.load(path, weights_only=True)
        self.load_state_dict(weights)
