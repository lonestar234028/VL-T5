
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import collections
from pathlib import Path
from packaging import version

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import logging
import shutil
from pprint import pprint

from param import parse_args


from nlvr_data import get_loader
from utils import load_state_dict, LossMeter, set_global_logging_level
import wandb

set_global_logging_level(logging.ERROR, ["transformers"])

proj_dir = Path(__file__).resolve().parent.parent


_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast


from trainer_base import TrainerBase

class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train=train)

        if not self.verbose:
            set_global_logging_level(logging.ERROR, ["transformers"])

        from nlvr_model import VLT5NLVR, VLBartNLVR

        model_kwargs = {}
        if 't5' in args.backbone:
            model_class = VLT5NLVR
        elif 'bart' in args.backbone:
            model_class = VLBartNLVR

        config = self.create_config()
        self.tokenizer = self.create_tokenizer()
        if 'bart' in self.args.tokenizer:
            num_added_toks = 0
            if config.use_vis_order_embedding:
                additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                    [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)]
                special_tokens_dict = {
                    'additional_special_tokens': additional_special_tokens}
                num_added_toks = self.tokenizer.add_special_tokens(
                    special_tokens_dict)

                config.default_obj_order_ids = self.tokenizer.convert_tokens_to_ids(
                    [f'<vis_extra_id_{i}>' for i in range(100)])

        self.model = self.create_model(model_class, config, **model_kwargs)

        if 't5' in self.args.tokenizer:
            self.model.resize_token_embeddings(self.tokenizer.vocab_size)
        elif 'bart' in self.args.tokenizer:
            self.model.resize_token_embeddings(
                self.model.model.shared.num_embeddings + num_added_toks)
            if self.verbose:
                print(f'Vocab resize: {self.tokenizer.vocab_size} -> {self.model.model.shared.num_embeddings}')
                assert self.model.model.shared.weight is self.model.lm_head.weight
                assert self.model.model.shared.weight is self.model.model.encoder.visual_embedding.obj_order_embedding.weight


        self.model.tokenizer = self.tokenizer
        if 't5' in self.args.tokenizer or 'bart' in self.args.tokenizer:
            self.model.true_id = self.tokenizer('true', add_special_tokens=False).input_ids[0]
            self.model.false_id = self.tokenizer('false', add_special_tokens=False).input_ids[0]

        # Load Checkpoint
        self.start_epoch = None
        # if args.load is not None:
        #     ckpt_path = args.load + '.pth'
        #     self.load_checkpoint(ckpt_path)


        if self.args.from_scratch:
            self.init_weights()

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()
            elif _use_apex:
                self.model, self.optim = amp.initialize(
                    self.model, self.optim, opt_level='O1', verbosity=self.verbose)

        if args.multiGPU:
            if args.distributed:
                self.model = DDP(self.model, device_ids=[args.gpu],
                                 find_unused_parameters=True
                                 )
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

    def train(self):
        if self.verbose:
            loss_meter = LossMeter()
            # best_eval_loss = 9595.
            quesid2ans = {}
            best_valid = 0.
            best_epoch = 0

            if 't5' in self.args.backbone:
                project_name = "VLT5_NLVR"
            elif 'bart' in self.args.backbone:
                project_name = "VLBART_NLVR"

            wandb.init(project=project_name)
            wandb.run.name = self.args.run_name
            wandb.config.update(self.args)
            wandb.watch(self.model)

            src_dir = Path(__file__).resolve().parent
            base_path = str(src_dir.parent)
            src_dir = str(src_dir)
            wandb.save(os.path.join(src_dir + "/*.py"), base_path=base_path)

        if self.args.distributed:
            dist.barrier()

        global_step = 0
        if self.verbose:
            self.save("LAST")

            # Test Set
            best_path = os.path.join(self.args.output, 'BEST')
            self.load(best_path)

            log_str = 'Test set results\n'

            dump_path = os.path.join(self.args.output, 'submit.csv')
            test_score_dict = self.evaluate(self.test_loader, dump_path=dump_path)
            wandb.save(dump_path, base_path=self.args.output)

            wandb_log_dict = {}
            for score_name, score in test_score_dict.items():
                wandb_log_dict[f'Test/{score_name}'] = score * 100.
            wandb.log(wandb_log_dict, step=0)

            from pprint import pformat

            log_str += pformat(test_score_dict)

            print(log_str)


            wandb.log({'finished': True})

    def predict(self, loader, dump_path=None):
        """
        Predict the answers to questions in a data split.
        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        with torch.no_grad():
            quesid2ans = {}
            for i, batch in enumerate(tqdm(loader, ncols=150, desc="Prediction")):

                if self.args.distributed:
                    results = self.model.module.test_step(batch)
                else:
                    results = self.model.test_step(batch)

                pred_ans = results['pred_ans_id']
                ques_ids = batch['question_ids']

                for qid, ans in zip(ques_ids, pred_ans):
                    quesid2ans[qid] = ans

            if dump_path is not None:
                loader.evaluator.dump_result(quesid2ans, dump_path)
            return quesid2ans

    def evaluate(self, loader, dump_path=None):
        evaluator = loader.evaluator
        quesid2ans = self.predict(loader, dump_path)
        return evaluator.evaluate(quesid2ans)


def main_worker(gpu, args):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    print(f'Building train loader at GPU {gpu},{__file__}')
    # train_loader = get_loader(
    #     args,
    #     split=args.train, mode='train', batch_size=args.batch_size,
    #     distributed=args.distributed, gpu=args.gpu,
    #     workers=args.num_workers,
    #     topk=args.train_topk,
    # )

    print(f'Building val loader at GPU {gpu}')
    # val_loader = get_loader(
    #     args,
    #     split=args.valid, mode='val', batch_size=args.batch_size,
    #     distributed=args.distributed, gpu=args.gpu,
    #     workers=4,
    #     topk=args.valid_topk,
    # )
    print(f'Building test loader at GPU {gpu}')
    test_loader = get_loader(
        args,
        split=args.test, mode='val', batch_size=args.batch_size,
        distributed=args.distributed, gpu=args.gpu,
        workers=4,
        topk=args.valid_topk,
    )

    trainer = Trainer(args, None, None, test_loader, train=False)
    trainer.train()


if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    if args.local_rank in [0, -1]:
        print(args)

        comments = []
        if args.load is not None:
            ckpt_str = "_".join(args.load.split('/')[-3:])
            comments.append(ckpt_str)
        if args.comment != '':
            comments.append(args.comment)
        comment = '_'.join(comments)

        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M')

        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'

        args.run_name = run_name

    if args.distributed:
        main_worker(args.local_rank, args)
