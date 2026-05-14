import logging
from quantom_ips.utils.stateful_module import StatefulModule
from tqdm.auto import trange, tqdm
from typing import Optional

logger = logging.getLogger(__name__)


class BaseOptimizer:
    def __init__(
        self,
        n_epochs: int = 1,
        progress_bar: bool = False,
        limit_batches: int = -1,
        limit_eval_batches: Optional[int] = None,
    ):
        super().__init__()
        self.n_epochs = n_epochs
        self.progress_bar = progress_bar
        self.limit_batches = limit_batches
        self.limit_eval_batches = limit_eval_batches

        # Default limit_eval_batches to limit_batches if not set
        if self.limit_eval_batches is None:
            self.limit_eval_batches = self.limit_batches

        # Initial values for loop variables
        self.current_epoch = 0
        self.inputs = None
        self.targets = None
        self.outputs = None
        self.losses = None

    def run_event(self, tag):
        for alg in self.algorithms:
            if alg.match(tag):
                alg.apply(self, tag)

    def train(self, data_parser, algorithms, val_data_parser=None):
        """Train the optimizer

        Utilizes the following event names:

        * before_fit
        * before_epoch
        * process_batch
        * before_update
        * after_batch
        * after_epoch
        * after_fit

        If using a validation dataset, additional event names are used:

        * before_eval
        * process_eval_batch
        * after_eval_batch

        Args:
            data_parser (iterable): Training dataset
            algorithms (list[Algorithm]): Algorithms to run during events
            val_data_parser (iterable, optional): Validation dataset. Defaults to None.
        """
        self.setup()
        self.algorithms = algorithms
        self.run_event("before_fit")
        for self.current_epoch in trange(
            1, self.n_epochs + 1, disable=not self.progress_bar
        ):
            self.run_event("before_epoch")
            for idx, (self.inputs, self.targets) in enumerate(
                tqdm(data_parser, disable=not self.progress_bar, leave=False)
            ):
                self.run_event("process_batch")
                self.outputs = self.predict()

                self.run_event("before_update")
                self.losses = self.update_model()

                self.run_event("after_batch")

                if self.limit_batches >= 0 and idx >= self.limit_batches:
                    break

            if val_data_parser is not None:
                self.validation_loop(val_data_parser)

            self.run_event("after_epoch")
        self.run_event("after_fit")

    def validation_loop(self, data_parser):
        self.run_event("before_eval")
        for idx, (self.inputs, self.targets) in enumerate(data_parser):
            if self.limit_eval_batches >= 0 and idx >= self.limit_eval_batches:
                break
            self.run_event("process_eval_batch")
            self.outputs = self.predict()
            self.run_event("after_eval_batch")

    def setup(self):
        pass

    def predict(self):
        pass

    def update_model(self):
        pass


class PyTorchOptimizer(BaseOptimizer, StatefulModule):
    def setup(self):
        self.device = self.parameters().__next__().device
        self.dtype = self.parameters().__next__().dtype
