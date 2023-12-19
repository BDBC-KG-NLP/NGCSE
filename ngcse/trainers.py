# import logging
import os
import sys
import pdb
import json
from pathlib import Path
from packaging import version
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Mapping

import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from transformers.utils import (
    is_apex_available,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)
from transformers.trainer_pt_utils import nested_detach
from transformers.debug_utils import DebugOption
from transformers import Trainer
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState

if is_apex_available():
    from apex import amp

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat

# Set path to SentEval
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
import numpy as np
from datetime import datetime
from filelock import FileLock

# logger = logging.getLogger(__name__)
logger = logging.get_logger()


class GenerateEmbeddingCallback(TrainerCallback):
    def _prepare_input(self, args: TrainingArguments, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(args, v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(args, v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=args.device)
            if args.deepspeed and data.dtype != torch.int64:
                # NLP models inputs are int64 and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update(dict(dtype=args.hf_deepspeed_config.dtype()))
            return data.to(**kwargs)
        return data

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.info('Generating Hyperbolic Embeddings for sentences in train_dataset.')
        
        model = kwargs.pop('model')
        train_dataloader = kwargs.pop('train_dataloader')

        hyperbolic_embeddings = []
        embeddings_num = 0
        for inputs in train_dataloader:
            inputs = self._prepare_input(args=args, data=inputs)
            outputs = model(**inputs, sent_emb=True)
            tmp_embeddings: torch.Tensor = outputs.pooler_output
            tmp_embeddings = tmp_embeddings.view(-1, tmp_embeddings.shape[-1])
            hyperbolic_embeddings.append(tmp_embeddings)
            embeddings_num += tmp_embeddings.shape[0]
            if embeddings_num >= args.dump_embeddings_num:
                break

        hyperbolic_embeddings = torch.cat(hyperbolic_embeddings, dim=0).tolist()
        with open(os.path.join(args.output_dir, 'hyperbolic_embeddings.json'), 'w', encoding='utf8') as fo:
            json.dump(hyperbolic_embeddings, fo)

        logger.info(f'Hyperbolic Embeddings for sentences in train_dataset were saved to {os.path.join(args.output_dir, "hyperbolic_embeddings.json")}')

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        metrics: Dict[str, float] = kwargs.pop('metrics')

        # Determine the new best metric / best model checkpoint
        if metrics is not None and args.metric_for_best_model is not None:
            metric_to_check = args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if args.greater_is_better else np.less
            if (
                state.best_metric is None
                or state.best_model_checkpoint is None
                or operator(metric_value, state.best_metric)
            ):
                control.should_save = True

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logger.info(f'metric_for_best_model: {args.metric_for_best_model}, best_metric: {state.best_metric}, best_model_checkpoint: {state.best_model_checkpoint}')

class NGTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_senteval_transfer: bool = False
    ) -> Dict[str, float]:

        # SentEval prepare and batcher
        def prepare(params, samples):
            return

        def batcher(params, batch):
            sentences = [' '.join(s) for s in batch]
            batch = self.tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )
            for k in batch:
                batch[k] = batch[k].to(self.args.device)
            with torch.no_grad():
                outputs = self.model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
                pooler_output = outputs.pooler_output
            return pooler_output.cpu()

        # Set params for SentEval (fastmode)
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                            'tenacity': 3, 'epoch_size': 2}

        se = senteval.engine.SE(params, batcher, prepare)
        tasks = ['STSBenchmark', 'SICKRelatedness']
        if eval_senteval_transfer and self.args.eval_transfer:
            tasks = ['STSBenchmark', 'SICKRelatedness', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
        self.model.eval()
        results = se.eval(tasks)
        
        stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
        sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]

        metrics = {"eval_stsb_spearman": stsb_spearman, "eval_sickr_spearman": sickr_spearman, "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2} 
        if eval_senteval_transfer and self.args.eval_transfer:
            avg_transfer = 0
            for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
                avg_transfer += results[task]['devacc']
                metrics['eval_{}'.format(task)] = results[task]['devacc']
            avg_transfer /= 7
            metrics['eval_avg_transfer'] = avg_transfer

        self.log(metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics
    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)