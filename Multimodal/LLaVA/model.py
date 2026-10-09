import torch
import torch.nn as nn

class LLaVA(nn.Module):
    def __init__(self, vision_encoder, vicuna, vision_encoder_out_dim, embedding_size):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.vicuna = vicuna

        self.embedding = self.vicuna.get_input_embeddings()

        self.proj = nn.Linear(vision_encoder_out_dim, embedding_size)

    def forward(self, image, question, answer):
        # 마지막 층 직전 특징을 클래스 토큰 빼고 반환
        image_feature = self.vision_encoder(image, output_hidden_states=True).hidden_states[-2][:, 1:, :] # B, patch_size, d
        image_feature = self.proj(image_feature) # 텍스트와 같은 차원으로 투영

        embedded_question = self.embedding(question)
        embedded_answer = self.embedding(answer)

        x = torch.concat([embedded_question, image_feature, embedded_answer], dim=1) # B, L+patch_size, d
        x = x.to(dtype=self.vicuna.get_input_embeddings().weight.dtype)
        attention_mask = torch.ones(x.shape[:2])

        outputs = self.vicuna(inputs_embeds=x, attention_mask=attention_mask)

        return outputs