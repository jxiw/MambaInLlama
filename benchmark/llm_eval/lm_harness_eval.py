import torch

import transformers
from transformers import AutoTokenizer

from torch import nn as nn

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from lm_eval.models.huggingface import HFLM
from lm_eval.utils import stop_sequences_criteria

from mamba.hybrid_wrapper import MambaTransformerHybridModelWrapper

@register_model("mamba_hybrid")
class MambaEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained, max_length=2048, batch_size=None, device="cuda",
                 dtype=torch.bfloat16):
        LM.__init__(self)
        self._model = MambaTransformerHybridModelWrapper.from_pretrained(pretrained, torch_dtype=dtype).model
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        print(self._model)
        self._model = self._model.cuda()
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self.truncation = False
        self._device = torch.device(device)

    @property
    def batch_size(self):
        return self._batch_size

    # this is copied from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py#L824-L849
    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=False,
            **generation_kwargs,
        )


if __name__ == "__main__":
    cli_evaluate()
    
