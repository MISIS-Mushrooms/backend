import copy
import warnings
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers import T5Model, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import T5Stack, T5PreTrainedModel
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map


class T5ForConditionalGeneration(T5PreTrainedModel):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.lm_head = nn.Linear(config.d_model, config.d_model, bias=False)
        self.model_dim = config.d_model

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config)


        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5ForConditionalGeneration.parallelize` is deprecated and will be removed in v5 of Transformers, you"
            " should load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also"
            " provide your own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'encoder.block.0': 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return None

    def set_input_embeddings(self, new_embeddings):
        pass

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_embeds: torch.FloatTensor,
        input_attention_mask: torch.FloatTensor,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        # if head_mask is not None and decoder_head_mask is None:
        #     if self.config.num_layers == self.config.num_decoder_layers:
        #         warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
        #         decoder_head_mask = head_mask

        # Convert encoder inputs in embeddings if needed
        encoder_outputs = self.encoder(
            attention_mask=input_attention_mask,
            inputs_embeds=input_embeds,
        )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        # if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
        #     # get decoder inputs from shifting lm labels to the right
        #     decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        # if self.model_parallel:
        #     torch.cuda.set_device(self.decoder.first_device)
        #     hidden_states = hidden_states.to(self.decoder.first_device)
        #     if decoder_input_ids is not None:
        #         decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
        #     if attention_mask is not None:
        #         attention_mask = attention_mask.to(self.decoder.first_device)
        #     if decoder_attention_mask is not None:
        #         decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=input_attention_mask,
            head_mask=decoder_head_mask,
            # cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_vec = self.lm_head(sequence_output)
        loss = None

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_vec,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past