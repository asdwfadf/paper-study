import torch
from transformers import CLIPVisionModel, AutoModelForCausalLM
from model import LLaVA
from torchinfo import summary

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
vicuna = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.3")
vision_encoder.requires_grad_(False)
vision_encoder.eval()
vicuna.requires_grad_(False)

batch_size = 1
vocab_size = vicuna.config.vocab_size
max_context_length = 1000
N = (224 / 14) ** 2
vision_encoder_out_dim = vision_encoder.config.hidden_size
embedding_size = vicuna.config.hidden_size

model = LLaVA(vision_encoder=vision_encoder, vicuna=vicuna, vision_encoder_out_dim=vision_encoder_out_dim, embedding_size=embedding_size).to(device)

human_token = torch.zeros((batch_size, 1), device=device)
assistant_token = torch.ones((batch_size, 1), device=device)
stop_token = torch.full((batch_size, 1), 2, device=device)

image = torch.randn(batch_size, 3, 224, 224, device=device)
question = torch.randint(0, vocab_size, (batch_size, 20), device=device)
answer = torch.randint(0, vocab_size, (batch_size, 30), device=device)

outputs = model(image, question, answer)

print(f'logits shape: {outputs.logits.shape}')
summary(model, input_data=(image, question, answer))