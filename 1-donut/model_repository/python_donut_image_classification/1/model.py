import re

import numpy as np
import torch.cuda
import triton_python_backend_utils as pb_utils
from transformers import DonutProcessor, VisionEncoderDecoderModel


class TritonPythonModel:
    def initialize(self, args):
        self.feature_extractor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-rvlcdip")
        self.model.to(self.device)

    def execute(self, requests):
        responses = []
        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "pixel_values")

            input_image = np.squeeze(inp.as_numpy()).transpose((2, 0, 1))

            task_prompt = "<s_rvlcdip>"
            decoder_input_ids = self.feature_extractor.tokenizer(task_prompt, add_special_tokens=False,
                                                                 return_tensors="pt").input_ids

            pixel_values = self.feature_extractor(images=input_image,
                                                  return_tensors="pt").pixel_values

            outputs = self.model.generate(
                pixel_values.to(self.device),
                decoder_input_ids=decoder_input_ids.to(self.device),
                max_length=self.model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=self.feature_extractor.tokenizer.pad_token_id,
                eos_token_id=self.feature_extractor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[self.feature_extractor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

            sequence = self.feature_extractor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(self.feature_extractor.tokenizer.eos_token, "").replace(
                self.feature_extractor.tokenizer.pad_token, "")
            sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
            results = self.feature_extractor.token2json(sequence)

            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(
                    "output",
                    np.array(results)
                )
            ])
            responses.append(inference_response)
        return responses

    def finalize(self):
        print('Cleaning up...')